"""High-level KONASH API — the user-facing entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from konash.corpus import Corpus
from konash.agent import Agent as BaseAgent
from konash.harness.environment import Environment
from konash.harness.dispatcher import Dispatcher
from konash.harness.strategy import (
    StandardStrategy,
    ParallelThinkingStrategy,
)
from konash.plugins.compression import CompressionPlugin
from konash.plugins.control import StepBudgetPlugin
from konash.inference.parallel import ParallelThinkingEngine
from konash.inference.aggregation import GenerativeAggregator
from konash.synthesis.pipeline import SynthesisPipeline
from konash.synthesis.qa import QuestionAnswerSynthesizer, SyntheticExample
from konash.synthesis.rollouts import RolloutGenerator
from konash.training.dataset import OfflineRolloutDataset
from konash.training.oapl import OAPLTrainer
from konash.training.iteration import IterativeTrainingPipeline


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "glm-4.5-air-together": {
        "base_model": "zai-org/GLM-4.5-Air-FP8",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "description": "GLM 4.5 Air (106B MoE, 12B active) on Together AI",
        "pricing": {"input_per_m": 0.20, "output_per_m": 1.10},
    },
    "glm-4.5-air-unsloth": {
        "base_model": "unsloth/GLM-4.5-Air",
        "use_unsloth": True,
        "load_in_fp8": True,
        "temperature": 0.7,
        "description": "GLM 4.5 Air via Unsloth (local OAPL training, FP8)",
    },
    "glm-4.5-air-zhipu": {
        "base_model": "glm-4.5-air",
        "api_base": "https://api.z.ai/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY",
        "temperature": 0.7,
        "description": "GLM 4.5 Air on Zhipu (native provider)",
    },
}


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM client wrapper
# ---------------------------------------------------------------------------

class _OpenAILLMClient:
    """Thin synchronous wrapper around an OpenAI-compatible chat API.

    Calls the chat completions endpoint and returns a dict that the
    internal ``BaseAgent`` expects::

        {"role": "assistant", "content": "...", "tool_calls": [...]}
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import time
        import urllib.error
        import urllib.request

        url = f"{self.api_base}/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
        if kwargs.get("stop"):
            body["stop"] = kwargs["stop"]

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "KONASH/0.1",
            },
        )

        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in {429, 500, 502, 503, 504} and attempt < 3:
                    retry_after = int(e.headers.get("Retry-After", 0))
                    backoff = max(retry_after, 2 ** attempt)
                    time.sleep(backoff)
                    continue
                raise

        choice = result["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        # GLM 4.5 Air returns reasoning in a separate field; fall back to it
        # when the main content is empty (e.g. low max_tokens budget).
        if not content and message.get("reasoning_content"):
            content = message["reasoning_content"]
        response: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if message.get("tool_calls"):
            response["tool_calls"] = message["tool_calls"]
        return response


# ---------------------------------------------------------------------------
# High-level Agent
# ---------------------------------------------------------------------------

class Agent:
    """High-level KONASH agent for training and querying knowledge agents.

    This is the primary user-facing class.  It orchestrates corpus ingestion,
    data synthesis, OAPL training, and inference (with optional parallel
    thinking) behind a simple interface.

    Examples
    --------
    ::

        import konash

        # Zhipu API (GLM 4.5 Air — the paper's default)
        agent = konash.Agent(
            base_model="glm-4.5-air",
            corpus="./my_docs",
            api_base="https://api.z.ai/api/paas/v4",
            api_key="your-zhipu-key",  # or set ZHIPU_API_KEY env var
        )
        agent.train()
        answer = agent.solve("What caused the 2008 financial crisis?")

        # Or load locally on GPU with LoRA (for training)
        agent = konash.Agent(
            base_model="THUDM/glm-4-9b-chat",
            corpus="./my_docs",
        )
        agent.train()
        answer = agent.solve("What is KONASH?")

    Parameters
    ----------
    base_model : str
        HuggingFace model ID or path to a local model.
    corpus : str | Path | Corpus
        Path to a document directory (or a pre-built ``Corpus``).
    project : str
        Project name used for checkpoint and log organisation.
    api_base : str | None
        OpenAI-compatible API base URL for inference (e.g. a local vLLM
        server).  When set, the API client is used instead of loading the
        model locally.  Reads ``KONASH_API_BASE`` env var if not set.
    api_key : str | None
        API key for the inference server.  Reads ``KONASH_API_KEY`` env var
        if not set.  Defaults to ``"no-key"`` for local servers.
    checkpoint_dir : str | None
        Where to save/load LoRA checkpoints.  Defaults to
        ``.konash/<project>/checkpoints``.
    lora_r : int
        LoRA rank (default 16).
    lora_alpha : int
        LoRA alpha (default 32).
    load_in_4bit : bool
        Use 4-bit QLoRA quantization (saves VRAM, requires ``bitsandbytes``).
    device : str
        ``"auto"`` (default), ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.
    dtype : str
        ``"auto"`` (default), ``"fp16"``, ``"bf16"``, or ``"fp32"``.
    """

    def __init__(
        self,
        base_model: str,
        corpus: str | Path | Corpus,
        *,
        project: str = "default",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        inference_api_base: Optional[str] = None,
        inference_api_key: Optional[str] = None,
        inference_model: Optional[str] = None,
        hf_token: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        chunk_size: int = 512,
        embedding_provider: str = "gemini",
        temperature: float = 0.7,
        # LoRA / model config
        lora_r: int = 16,
        lora_alpha: int = 32,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        load_in_fp8: bool = False,
        use_unsloth: bool = False,
        gradient_checkpointing: bool = False,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        self.base_model = base_model
        self.project = project
        self.temperature = temperature
        self._use_unsloth = use_unsloth
        self._load_in_fp8 = load_in_fp8
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")

        # LLM connection
        self.api_base = api_base or os.environ.get("KONASH_API_BASE")
        self.api_key = api_key or os.environ.get("KONASH_API_KEY", "no-key")

        # Corpus — choose embedding provider
        self.embedding_provider = embedding_provider
        if isinstance(corpus, Corpus):
            self.corpus = corpus
        else:
            embed_fn = self._make_embed_fn()
            # Cache embeddings next to checkpoint dir for instant reload
            _ckpt = checkpoint_dir or os.path.join(".konash", project, "checkpoints")
            _cache_dir = os.path.join(_ckpt, "index_cache")
            self.corpus = Corpus(
                corpus, chunk_size=chunk_size, embed_fn=embed_fn,
                cache_dir=_cache_dir,
            )

        # Inference API (split mode: fast API for inference, local model for training)
        self.inference_api_base = inference_api_base or os.environ.get("KONASH_INFERENCE_API_BASE")
        self.inference_api_key = inference_api_key or os.environ.get("KONASH_INFERENCE_API_KEY")
        self.inference_model = inference_model or os.environ.get("KONASH_INFERENCE_MODEL")

        # Checkpoints
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            ".konash", project, "checkpoints"
        )

        # Model config (for local loading)
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        self._gradient_checkpointing = gradient_checkpointing
        self._device = device
        self._dtype = dtype

        # Internal state (built lazily)
        self._llm_client: Optional[_OpenAILLMClient] = None
        self._inference_client: Optional[_OpenAILLMClient] = None
        self._model_engine: Optional[Any] = None  # LocalModelEngine
        self._base_agent: Optional[BaseAgent] = None
        self._value_model: Optional[Any] = None  # ValueModel for VGS
        self._trained = False
        self._iteration = 0

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(
        cls,
        preset: str,
        corpus: str | Path | Corpus,
        **kwargs: Any,
    ) -> "Agent":
        """Create an Agent from a named model preset.

        Available presets: ``glm-4.5-air-together``, ``glm-4.5-air-zhipu``.

        Parameters
        ----------
        preset : str
            Preset name (see ``MODEL_PRESETS``).
        corpus : str | Path | Corpus
            Path to documents or a ``Corpus`` instance.
        **kwargs
            Additional overrides passed to ``Agent.__init__``.
        """
        if preset not in MODEL_PRESETS:
            available = ", ".join(sorted(MODEL_PRESETS))
            raise ValueError(
                f"Unknown preset {preset!r}. Available: {available}"
            )
        cfg = MODEL_PRESETS[preset]
        defaults: Dict[str, Any] = {
            "base_model": cfg["base_model"],
        }
        if cfg.get("api_base"):
            defaults["api_base"] = cfg["api_base"]
        if cfg.get("temperature") is not None:
            defaults["temperature"] = cfg["temperature"]
        if cfg.get("use_unsloth"):
            defaults["use_unsloth"] = True
        if cfg.get("load_in_fp8"):
            defaults["load_in_fp8"] = True

        # Resolve API key from env if not explicitly provided
        if "api_key" not in kwargs:
            env_var = cfg.get("api_key_env", "")
            key = os.environ.get(env_var, "")
            if key:
                defaults["api_key"] = key

        defaults.update(kwargs)
        return cls(corpus=corpus, **defaults)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        iterations: int = 2,
        synthesis_calls: int = 1500,
        rollouts_per_example: int = 8,
        rollout_max_steps: int = 50,
        max_examples: Optional[int] = None,
        few_shot_examples: Optional[List[SyntheticExample]] = None,
        learning_rate: float = 1e-6,
        beta_kl: float = 0.001,
        beta_value: float = 1.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the full KONASH training loop.

        1. Load the model locally (with LoRA).
        2. Ingest the corpus and build the vector search index.
        3. Synthesize QA pairs via *synthesis_calls* independent calls
           (KARL: 1,735 calls × 8 candidates each).
        4. Deduplicate and generate rollouts, filter by pass rate.
        5. Train with OAPL (real gradient updates on LoRA params).
        6. Repeat for *iterations* rounds.
        7. Save the LoRA adapter checkpoint.

        Parameters
        ----------
        iterations : int
            Number of synthesis → train cycles (default 2).
        synthesis_calls : int
            Independent synthesis calls per iteration (default 1500).
            Each call generates ~8 QA pairs with fresh random seed docs.
            KARL paper uses 1,735 calls for BrowseComp-Plus.
        rollouts_per_example : int
            Number of rollouts per training example (default 8).
        rollout_max_steps : int
            Max reasoning steps per rollout (default 50).
        max_examples : int | None
            Cap on synthesized training examples per iteration.
        few_shot_examples : list[SyntheticExample] | None
            Representative QA pairs that guide the synthesizer toward the
            expected question format and difficulty (KARL Section 4.1).
        learning_rate : float
            Learning rate for OAPL training.
        beta_kl : float
            KL regularisation coefficient.
        beta_value : float
            Temperature for soft value estimation.
        verbose : bool
            Print progress updates.

        Returns
        -------
        dict
            Training summary with per-iteration statistics.
        """
        # Step 1: Load model locally (for real OAPL training)
        # In split mode (inference_api_base set), still load the local model
        # because OAPL gradient updates require local weights.
        if verbose:
            from rich.console import Console as _Console
            _con = _Console()

        engine = None
        if self.api_base is None or self.inference_api_base is not None:
            try:
                engine = self._get_model_engine()
                if verbose:
                    _con.print(f"  [green]✓[/]  Model loaded: [bold]{self.base_model}[/]")
                    if self.inference_api_base:
                        _con.print(
                            f"    [dim]Inference via {self.inference_model or self.base_model} "
                            f"@ {self.inference_api_base}[/]"
                        )
            except (ImportError, OSError, ValueError, RuntimeError) as e:
                if verbose:
                    _con.print(f"  [yellow]–[/]  Could not load model locally [dim]({e})[/]")
                    _con.print("    [dim]Lightweight mode (no gradient updates)[/]")
                    _con.print("    [dim]Install: pip install konash[train][/]")

        # Step 2: Ingest corpus
        if not self.corpus.indexed:
            if verbose:
                from rich.live import Live as _Live
                from rich.text import Text as _Text

                _ingest_phase = ["reading"]
                _ingest_cur = [0]
                _ingest_total = [0]

                def _build_ingest_display() -> _Text:
                    labels = {
                        "reading": "Reading files",
                        "chunking": "Chunking docs",
                        "embedding": "Embedding",
                    }
                    label = labels.get(_ingest_phase[0], _ingest_phase[0])
                    cur, tot = _ingest_cur[0], _ingest_total[0]
                    if tot > 0:
                        pct = cur * 100 // tot
                        bar_w = 24
                        filled = bar_w * cur // tot
                        bar = f"[cyan]{'━' * filled}[/][dim]{'─' * (bar_w - filled)}[/]"
                        return _Text.from_markup(
                            f"  {label}  {bar}  "
                            f"[dim]{cur:,}/{tot:,}[/]  [dim]{pct}%[/]"
                        )
                    return _Text.from_markup(f"  {label}  [dim]starting...[/]")

                _ingest_live = _Live(
                    _build_ingest_display(),
                    console=_con, refresh_per_second=4, transient=True,
                )

                def _progress(phase: str, current: int, total: int) -> None:
                    _ingest_phase[0] = phase
                    _ingest_cur[0] = current
                    _ingest_total[0] = total
                    _ingest_live.update(_build_ingest_display())

                with _ingest_live:
                    self.corpus.ingest(progress_callback=_progress)
            else:
                self.corpus.ingest()
            if verbose:
                _con.print(
                    f"  [green]✓[/]  Indexed [bold]{self.corpus.num_documents:,}[/] chunks"
                )

        # Set up synthesis components (LLM-backed via inference API or local model)
        generate_fn = self._get_generate_fn()

        # Synthesis needs more tokens (thinking + many QA pairs).
        # Rollouts need fewer (short JSON reasoning steps).
        def _synthesis_fn(messages, **kwargs):
            kwargs.setdefault("max_new_tokens", 2048)
            return generate_fn(messages, **kwargs)

        def _rollout_fn(messages, **kwargs):
            kwargs.setdefault("max_new_tokens", 512)
            return generate_fn(messages, **kwargs)

        synthesizer = QuestionAnswerSynthesizer(
            vector_search_tool=self.corpus.vector_search,
            llm_fn=_synthesis_fn,
            few_shot_examples=few_shot_examples,
        )

        # Progress callback for rollouts — _current_examples is mutated
        # before each stage-two call so the closure sees the right list.
        _current_examples: List[Any] = []

        def _on_step(qa_idx, rollout_idx, step_idx, step_record):
            pass  # Progress shown by Rich status spinner

        rollout_gen = RolloutGenerator(
            search_tool=self.corpus.vector_search,
            llm_fn=_rollout_fn,
            on_step=_on_step,
            max_steps=rollout_max_steps,
        )
        pipeline = SynthesisPipeline(
            synthesizer=synthesizer,
            rollout_generator=rollout_gen,
        )

        # Set up trainer
        trainer = OAPLTrainer(
            beta_kl=beta_kl,
            beta_value=beta_value,
        )

        stats: List[Dict[str, Any]] = []
        all_value_rollouts: List[Any] = []  # Accumulate for value model
        all_value_rewards: List[float] = []

        for iteration in range(iterations):
            if verbose:
                _con.print()
                _con.rule(
                    f"[bold]Iteration {iteration + 1}/{iterations}[/]",
                    style="dim",
                )

            # Step 3: Synthesize QA pairs
            # Free transient GPU memory before generation
            import gc
            gc.collect()
            if self._model_engine is not None:
                self._model_engine._torch.cuda.empty_cache()

            # Multi-call synthesis: KARL makes ~1,735 independent calls,
            # each generating ~8 QA pairs with fresh random seed docs.
            # This volume + dedup is how diversity is achieved.
            all_raw_examples: List[SyntheticExample] = []

            if verbose:
                from rich.live import Live
                from rich.table import Table as _Table
                from rich.text import Text

                def _build_synth_display(
                    call_idx: int, total_calls: int,
                    qa_count: int, latest_q: str, latest_a: str,
                ) -> _Table:
                    outer = _Table(
                        box=None, show_header=False, pad_edge=False,
                        expand=True, padding=(0, 0),
                    )

                    # Phase + progress
                    done = call_idx + 1
                    pct = done * 100 // total_calls if total_calls else 0
                    bar_w = 32
                    filled = bar_w * done // total_calls if total_calls else 0
                    bar = f"[cyan]{'━' * filled}[/][dim]{'─' * (bar_w - filled)}[/]"

                    outer.add_row(
                        Text("  Synthesizing QA pairs", style="bold"),
                    )
                    outer.add_row(Text(""))
                    outer.add_row(
                        Text.from_markup(
                            f"    {bar}  [dim]{done}/{total_calls}[/]  "
                            f"[bold]{qa_count}[/] pairs  [dim]{pct}%[/]"
                        ),
                    )

                    # Latest QA pair
                    if latest_q:
                        outer.add_row(Text(""))
                        outer.add_row(
                            Text.from_markup(
                                f"    [dim]Q:[/]  {latest_q[:90]}"
                                + ("[dim]...[/]" if len(latest_q) > 90 else "")
                            ),
                        )
                    if latest_a:
                        outer.add_row(
                            Text.from_markup(
                                f"    [dim]A:[/]  {latest_a[:90]}"
                                + ("[dim]...[/]" if len(latest_a) > 90 else "")
                            ),
                        )

                    return outer

                with Live(
                    _build_synth_display(0, synthesis_calls, 0, "", ""),
                    console=_con, refresh_per_second=4, transient=True,
                ) as live:
                    for call_idx in range(synthesis_calls):
                        try:
                            batch = synthesizer.synthesize(
                                documents=None,
                                num_examples=8,
                            )
                            all_raw_examples.extend(batch)
                            latest_q = batch[-1].question or "" if batch else ""
                            latest_a = batch[-1].answer or "" if batch else ""
                        except (ValueError, RuntimeError):
                            latest_q, latest_a = "", ""
                            continue
                        live.update(
                            _build_synth_display(
                                call_idx, synthesis_calls,
                                len(all_raw_examples), latest_q, latest_a,
                            )
                        )

                _con.print(
                    f"  [green]✓[/]  {len(all_raw_examples)} QA pairs synthesized"
                )
            else:
                for call_idx in range(synthesis_calls):
                    try:
                        batch = synthesizer.synthesize(
                            documents=None,
                            num_examples=8,
                        )
                        all_raw_examples.extend(batch)
                    except (ValueError, RuntimeError):
                        continue

            if not all_raw_examples:
                if verbose:
                    _con.print("  [yellow]–[/]  No QA pairs synthesized — skipping.")
                continue

            # Deduplicate
            if verbose:
                with _con.status(
                    f"  [cyan]Deduplicating {len(all_raw_examples)} pairs...",
                    spinner="dots",
                ):
                    examples = pipeline.deduplicate(all_raw_examples)
            else:
                examples = pipeline.deduplicate(all_raw_examples)

            if max_examples and len(examples) > max_examples:
                examples = examples[:max_examples]

            if verbose:
                _con.print(f"  [green]✓[/]  {len(examples)} unique examples")
                for i, ex in enumerate(examples[:3]):
                    _con.print(f"    [dim]{i+1}.[/] {(ex.question or '')[:80]}")
                if len(examples) > 3:
                    _con.print(f"    [dim]... and {len(examples) - 3} more[/]")

            # Step 4: Generate rollouts + filter
            if verbose:
                with _con.status(
                    f"  [cyan]Generating rollouts  "
                    f"({len(examples)} × {rollouts_per_example})...",
                    spinner="dots",
                ):
                    _current_examples[:] = examples
                    final_examples = pipeline.run_stage_two(
                        examples=examples,
                        num_rollouts=rollouts_per_example,
                    )
                _con.print(
                    f"  [green]✓[/]  {len(final_examples)} examples after "
                    f"pass-rate filtering"
                )
            else:
                _current_examples[:] = examples
                final_examples = pipeline.run_stage_two(
                    examples=examples,
                    num_rollouts=rollouts_per_example,
                )

            if not final_examples or not pipeline.filtered_groups:
                if verbose:
                    _con.print("  [yellow]–[/]  No training data — skipping.")
                continue

            # Step 5: Build dataset and train with OAPL
            rollout_dicts = []
            for group in pipeline.filtered_groups:
                for rollout in group.rollouts:
                    rollout_dicts.append({
                        "prompt": group.prompt,
                        "rollout": rollout.steps,
                        "reward": 1.0 if rollout.passed else 0.0,
                    })
            if rollout_dicts:
                dataset = OfflineRolloutDataset.from_rollouts(rollout_dicts)

                # Use real PyTorch training when model is loaded locally
                if verbose:
                    with _con.status(
                        f"  [cyan]Training OAPL  "
                        f"[dim]({len(rollout_dicts)} rollouts)[/]...",
                        spinner="dots",
                    ):
                        if self._model_engine is not None:
                            epoch_stats = trainer.train_epoch_torch(
                                dataset, engine,
                                learning_rate=learning_rate,
                            )
                        else:
                            epoch_stats = trainer.train_epoch(
                                dataset, learning_rate=learning_rate,
                            )
                else:
                    if self._model_engine is not None:
                        epoch_stats = trainer.train_epoch_torch(
                            dataset, engine,
                            learning_rate=learning_rate,
                        )
                    else:
                        epoch_stats = trainer.train_epoch(
                            dataset, learning_rate=learning_rate,
                        )

                stats.append({
                    "iteration": iteration + 1,
                    "examples": len(final_examples),
                    "rollouts": len(rollout_dicts),
                    **epoch_stats,
                })
                if verbose:
                    _con.print(
                        f"  [green]✓[/]  Loss [bold]{epoch_stats['mean_loss']:.4f}[/]  "
                        f"[dim]{epoch_stats['num_groups']} groups  "
                        f"{epoch_stats['num_rollouts']} rollouts[/]"
                    )

                # Accumulate rollout data for value model training
                for rd in rollout_dicts:
                    all_value_rollouts.append(rd["rollout"])
                    all_value_rewards.append(rd["reward"])

            self._iteration = iteration + 1

            # Snapshot current LoRA as πref for the next iteration
            # (KARL Section 4.2: replace πref with the latest policy)
            if self._model_engine is not None and iteration < iterations - 1:
                engine.snapshot_reference()
                if verbose:
                    _con.print("  [dim]Snapshotted LoRA as πref for next iteration[/]")

        # Step 6: Train value model for VGS inference
        value_model_trained = False
        if all_value_rollouts:
            from konash.inference.value_model import ValueModel

            if verbose:
                with _con.status(
                    f"  [cyan]Training value model  "
                    f"[dim]({len(all_value_rollouts)} rollouts)[/]...",
                    spinner="dots",
                ):
                    self._value_model = ValueModel(feature_dim=64)
                    vm_stats = self._value_model.fit(
                        all_value_rollouts,
                        all_value_rewards,
                        lr=0.01,
                        epochs=20,
                    )
            else:
                self._value_model = ValueModel(feature_dim=64)
                vm_stats = self._value_model.fit(
                    all_value_rollouts,
                    all_value_rewards,
                    lr=0.01,
                    epochs=20,
                )
            value_model_trained = True
            if verbose:
                _con.print(
                    f"  [green]✓[/]  Value model trained  "
                    f"[dim]loss {vm_stats['final_loss']:.4f}[/]"
                )

        # Step 7: Save checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        meta_path = os.path.join(self.checkpoint_dir, "training_meta.json")

        # Save LoRA adapter if we did real training
        if self._model_engine is not None:
            adapter_path = os.path.join(self.checkpoint_dir, "adapter")
            engine.save_adapter(adapter_path)

        # Save value model
        if value_model_trained and self._value_model is not None:
            import numpy as _np
            vm_path = os.path.join(self.checkpoint_dir, "value_model.json")
            with open(vm_path, "w") as f:
                weights = self._value_model.weights
                if hasattr(weights, "tolist"):
                    weights = weights.tolist()
                json.dump({
                    "weights": weights,
                    "bias": self._value_model.bias,
                    "feature_dim": self._value_model.feature_dim,
                }, f)

        with open(meta_path, "w") as f:
            json.dump({
                "base_model": self.base_model,
                "project": self.project,
                "iterations": self._iteration,
                "stats": stats,
                "value_model": value_model_trained,
            }, f, indent=2)

        self._trained = True
        if verbose:
            _con.print()
            _con.print(
                f"  [bold green]Training complete.[/]  "
                f"Checkpoint saved to [dim]{self.checkpoint_dir}[/]"
            )

        return {"iterations": self._iteration, "stats": stats}

    # ------------------------------------------------------------------
    # Solve (inference)
    # ------------------------------------------------------------------

    def solve(
        self,
        query: str,
        *,
        parallel_rollouts: int = 1,
        max_steps: int = 20,
        top_k: int = 10,
        use_vgs: Optional[bool] = None,
        vgs_candidate_width: int = 2,
        vgs_max_depth: int = 10,
    ) -> str:
        """Answer a question using the trained (or base) knowledge agent.

        Parameters
        ----------
        query : str
            The question to answer.
        parallel_rollouts : int
            Number of independent rollouts to run and aggregate (default 1).
            For VGS, this is the number of parallel search trees.
            For Parallel Thinking, this is the number of rollouts.
        max_steps : int
            Maximum agent steps per rollout.
        top_k : int
            Number of documents to retrieve per search.
        use_vgs : bool | None
            Use Value-Guided Search when a value model is available.
            ``None`` (default) auto-detects: uses VGS if a value model
            is loaded, otherwise falls back to Parallel Thinking.
        vgs_candidate_width : int
            Number of candidate continuations per BFS expansion step.
        vgs_max_depth : int
            Maximum BFS depth per search tree.

        Returns
        -------
        str
            The agent's answer.
        """
        if not self.corpus.indexed:
            self.corpus.ingest()

        agent = self._make_agent(max_steps=max_steps)

        def _extract_tool_query(tool_call: Any) -> str:
            if not isinstance(tool_call, dict):
                return str(tool_call)

            query_text = tool_call.get("query", "") or tool_call.get("input", "")
            if query_text:
                return str(query_text)

            function_call = tool_call.get("function")
            if isinstance(function_call, dict):
                arguments = function_call.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        return arguments
                if isinstance(arguments, dict):
                    nested_query = arguments.get("query", "") or arguments.get("input", "")
                    if nested_query:
                        return str(nested_query)

            return str(tool_call)

        # Build environment with vector search as a tool
        def tool_executor(tool_call: Any) -> Dict[str, Any]:
            query_text = _extract_tool_query(tool_call)
            results = self.corpus.search(query_text, top_k=top_k)
            result_text = "\n\n".join(
                f"[{i+1}] (score: {r.get('score', 0):.3f}) {r.get('text', '')[:500]}"
                for i, r in enumerate(results)
            )
            observation: Dict[str, Any] = {"role": "tool", "content": result_text}
            if isinstance(tool_call, dict) and tool_call.get("id"):
                observation["tool_call_id"] = tool_call["id"]
            return observation

        def make_environment() -> Environment:
            return Environment(
                tool_executor=tool_executor,
                plugins=[
                    CompressionPlugin(threshold_tokens=8000, target_tokens=4000),
                    StepBudgetPlugin(max_steps=max_steps),
                ],
                available_tools=[{
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the knowledge base for relevant documents.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query.",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }],
            )

        if parallel_rollouts <= 1 and not (use_vgs and self._value_model):
            # Single rollout
            env = make_environment()
            env.reset(prompt=query)
            result = env.run_episode(agent, max_steps=max_steps)
            return result.get("final_answer") or ""

        # Shared agent wrapper for both VGS and Parallel Thinking
        class _EnvironmentBackedAgent:
            def __init__(self, base_agent: BaseAgent) -> None:
                self._base_agent = base_agent

            def generate_rollout(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
                env = make_environment()
                env.reset(prompt=prompt)
                return env.run_episode(
                    self._base_agent,
                    max_steps=kwargs.get("max_steps", max_steps),
                )

            def generate_step(
                self, conversation_history: Any, **kwargs: Any,
            ) -> Dict[str, Any]:
                """Single agent step for VGS expand().

                VGS calls this with the conversation history built from
                the search tree state. We run one agent step (generate +
                optional tool execution) and return a step dict.
                """
                messages = (
                    list(conversation_history)
                    if isinstance(conversation_history, list)
                    else [{"role": "user", "content": str(conversation_history)}]
                )

                # Available tools for the agent
                search_tools = [{
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the knowledge base.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string",
                                          "description": "The search query."}
                            },
                            "required": ["query"],
                        },
                    },
                }]

                step_result = self._base_agent.generate_step(
                    messages, available_tools=search_tools,
                )
                content = step_result.get("content", "")
                tool_calls = step_result.get("tool_calls")

                # Execute tool call if present
                if tool_calls:
                    obs = tool_executor(tool_calls[0])
                    return {
                        "type": "tool_call",
                        "role": "assistant",
                        "content": content,
                        "result": obs.get("content", ""),
                        "terminal": False,
                    }

                # No tool call → final answer
                return {
                    "type": "answer",
                    "role": "assistant",
                    "content": content,
                    "terminal": True,
                }

            def extract_final_answer(self, conversation_history: Any, **kwargs: Any) -> str | None:
                if isinstance(conversation_history, dict):
                    conversation_history = (
                        conversation_history.get("history")
                        or conversation_history.get("messages")
                        or conversation_history
                    )
                return self._base_agent.extract_final_answer(conversation_history, **kwargs)

        env_agent = _EnvironmentBackedAgent(agent)

        # Decide: VGS or Parallel Thinking
        _use_vgs = use_vgs if use_vgs is not None else (self._value_model is not None)

        if _use_vgs and self._value_model is not None:
            from konash.inference.value_search import ValueGuidedSearchEngine

            vgs_engine = ValueGuidedSearchEngine(
                agent=env_agent,
                value_model=self._value_model,
                aggregator=GenerativeAggregator(
                    agent=env_agent,
                    aggregation_mode="weighted_majority_vote",
                ),
                candidate_width=vgs_candidate_width,
                parallel_searches=max(parallel_rollouts, 1),
                max_depth=vgs_max_depth,
            )
            result = vgs_engine.run(query)
            return result.get("answer", "")

        # Fallback: Parallel Thinking (no value model)
        engine = ParallelThinkingEngine(
            agent=env_agent,
            aggregator=GenerativeAggregator(
                agent=env_agent,
                aggregation_mode="generative",
            ),
            num_rollouts=parallel_rollouts,
        )
        result = engine.run(query)
        return result.get("answer", "")

    # ------------------------------------------------------------------
    # Search (direct corpus search without LLM)
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the corpus directly without an LLM.

        Useful for debugging retrieval quality.
        """
        return self.corpus.search(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        project_dir: str,
        corpus: Optional[str | Path | Corpus] = None,
        **kwargs: Any,
    ) -> "Agent":
        """Load a previously trained agent from its checkpoint directory.

        Loads the LoRA adapter so the agent is ready for inference.

        Parameters
        ----------
        project_dir : str
            Path to the ``.konash/<project>`` directory.
        corpus : str | Path | Corpus | None
            Corpus to use for inference.  Required for ``solve()``.
        """
        meta_path = os.path.join(project_dir, "checkpoints", "training_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No training metadata found at {meta_path}. "
                f"Train an agent first with agent.train()."
            )

        with open(meta_path) as f:
            meta = json.load(f)

        if corpus is None:
            raise ValueError("corpus is required to load an agent for inference.")

        agent = cls(
            base_model=meta["base_model"],
            corpus=corpus,
            project=meta["project"],
            checkpoint_dir=os.path.join(project_dir, "checkpoints"),
            **kwargs,
        )
        agent._trained = True
        agent._iteration = meta.get("iterations", 0)

        # Load value model if it was trained
        if meta.get("value_model"):
            vm_path = os.path.join(project_dir, "checkpoints", "value_model.json")
            if os.path.exists(vm_path):
                from konash.inference.value_model import ValueModel
                with open(vm_path) as f:
                    vm_data = json.load(f)
                agent._value_model = ValueModel(
                    weights=vm_data["weights"],
                    bias=vm_data["bias"],
                    feature_dim=vm_data["feature_dim"],
                )

        return agent

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model_engine(self) -> Any:
        """Load the model locally with LoRA.

        Returns an ``UnslothEngine`` if ``use_unsloth=True``, otherwise
        a ``LocalModelEngine``.
        """
        if self._model_engine is not None:
            return self._model_engine

        adapter_path = os.path.join(self.checkpoint_dir, "adapter")
        has_adapter = os.path.exists(adapter_path)

        if self._use_unsloth:
            from konash.training.unsloth_engine import UnslothEngine

            self._model_engine = UnslothEngine(
                self.base_model,
                lora_r=self._lora_r,
                lora_alpha=self._lora_alpha,
                load_in_fp8=self._load_in_fp8,
                temperature=self.temperature,
            )
            # Load existing adapter if resuming
            if has_adapter:
                from peft import PeftModel
                self._model_engine.model = PeftModel.from_pretrained(
                    self._model_engine.model, adapter_path, is_trainable=True,
                )
        else:
            from konash.inference.local import LocalModelEngine

            self._model_engine = LocalModelEngine(
                self.base_model,
                device=self._device,
                dtype=self._dtype,
                lora_r=self._lora_r,
                lora_alpha=self._lora_alpha,
                load_in_4bit=self._load_in_4bit,
                load_in_8bit=self._load_in_8bit,
                gradient_checkpointing=self._gradient_checkpointing,
                temperature=self.temperature,
                adapter_path=adapter_path if has_adapter else None,
            )
        return self._model_engine

    def _get_inference_client(self) -> _OpenAILLMClient:
        """Lazily build the inference-specific API client (split mode)."""
        if self._inference_client is None:
            if self.inference_api_base is None:
                raise RuntimeError("No inference API configured.")
            self._inference_client = _OpenAILLMClient(
                api_base=self.inference_api_base,
                api_key=self.inference_api_key or "no-key",
                model=self.inference_model or self.base_model,
                temperature=self.temperature,
            )
        return self._inference_client

    def _get_llm_client(self) -> _OpenAILLMClient:
        if self._llm_client is None:
            if self.api_base is None:
                raise RuntimeError(
                    "No API backend configured. Either:\n"
                    "  1. Remove api_base to use local model loading\n"
                    "  2. Set KONASH_API_BASE env var to a vLLM server URL\n"
                    "  3. Pass api_base='http://localhost:8000/v1' to Agent()"
                )
            self._llm_client = _OpenAILLMClient(
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.base_model,
                temperature=self.temperature,
            )
        return self._llm_client

    def _make_embed_fn(self):
        """Return an embed_fn based on ``self.embedding_provider``.

        Providers:
        - ``"gemini"`` — Gemini Embedding API (fast, free tier, default)
        - ``"hf"`` — Qwen3-Embedding-8B via HuggingFace Inference API
        - ``"local"`` — Qwen3-Embedding-0.6B via sentence-transformers (CPU)
        """
        if self.embedding_provider == "gemini":
            return self._make_gemini_embed_fn()
        elif self.embedding_provider == "hf":
            return self._make_hf_embed_fn()
        elif self.embedding_provider == "local":
            from konash.retrieval.vector_search import load_embedding_model
            return load_embedding_model("Qwen/Qwen3-Embedding-0.6B")
        else:
            raise ValueError(
                f"Unknown embedding_provider={self.embedding_provider!r}. "
                "Use 'gemini', 'hf', or 'local'."
            )

    def _make_gemini_embed_fn(self):
        """Return an embed_fn using Gemini Embedding API."""
        from konash.retrieval.vector_search import load_gemini_embedding_model
        return load_gemini_embedding_model(
            model_name="gemini-embedding-2-preview",
            output_dimensionality=768,
        )

    def _make_hf_embed_fn(self):
        """Return an embed_fn using Qwen3-Embedding-8B via HuggingFace Inference API."""
        from huggingface_hub import InferenceClient

        hf_token = self._hf_token or os.environ.get("HF_TOKEN")
        client = InferenceClient(api_key=hf_token)
        model = "Qwen/Qwen3-Embedding-8B"
        batch_size = 100

        def embed_fn(texts):
            import numpy as _np
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                r = client.feature_extraction(batch, model=model)
                all_embs.append(_np.array(r, dtype=_np.float32))
            return _np.vstack(all_embs) if len(all_embs) > 1 else all_embs[0]

        return embed_fn

    def _get_generate_fn(self):
        """Return a callable that generates text from messages.

        Priority: inference API (split mode) > local model > API client > None.

        The returned function strips ``<think>...</think>`` tags from
        reasoning models (e.g. Qwen3) so downstream parsers see only the
        actual content.
        """
        import re as _re

        def _strip_think_tags(result):
            if isinstance(result, dict) and "content" in result:
                result["content"] = _re.sub(
                    r"<think>.*?</think>\s*", "", result["content"],
                    flags=_re.DOTALL,
                )
                result["content"] = _re.sub(
                    r"<think>.*", "", result["content"],
                    flags=_re.DOTALL,
                ).strip()
                # GLM 4.5 Air sometimes emits stray XML-like tags
                result["content"] = _re.sub(
                    r"</arg_value>\s*", "", result["content"],
                )
            return result

        # Inference API takes priority (split mode: API for inference, local for training)
        if self.inference_api_base is not None:
            client = self._get_inference_client()
            def generate_fn(messages, **kwargs):
                if "max_new_tokens" in kwargs and "max_tokens" not in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
                return _strip_think_tags(client.generate(messages, **kwargs))
            return generate_fn

        # Local model is loaded — use it
        if self._model_engine is not None:
            engine = self._model_engine
            def generate_fn(messages, **kwargs):
                return _strip_think_tags(engine.generate(messages, **kwargs))
            return generate_fn

        # API client configured — use it
        if self.api_base is not None:
            client = self._get_llm_client()
            def generate_fn(messages, **kwargs):
                if "max_new_tokens" in kwargs and "max_tokens" not in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
                return _strip_think_tags(client.generate(messages, **kwargs))
            return generate_fn

        # No backend — return None so synthesis uses stubs
        return None

    def _make_agent(self, max_steps: int = 20) -> BaseAgent:
        """Build an internal BaseAgent wired to the best available backend."""
        system_prompt = (
            "You are a knowledge agent. You have access to a search tool that "
            "retrieves relevant documents from a knowledge base. Use it to find "
            "evidence before answering. Search iteratively — refine your queries "
            "based on what you find. When you have enough evidence, provide a "
            "clear, well-supported answer."
        )

        # Prefer inference API (split mode), then local, then general API
        if self.inference_api_base is not None:
            llm_client = self._get_inference_client()
        elif self._model_engine is not None:
            llm_client = self._model_engine
        elif self.api_base is not None:
            llm_client = self._get_llm_client()
        else:
            llm_client = self._get_model_engine()

        return BaseAgent(
            llm_client=llm_client,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return (
            f"Agent(model={self.base_model!r}, project={self.project!r}, "
            f"corpus={self.corpus!r}, {status})"
        )
