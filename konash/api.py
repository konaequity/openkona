"""High-level KONASH API — the user-facing entry point."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_api_logger = logging.getLogger(__name__)

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
from konash.synthesis.config import SynthesisTaskConfig
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
        if kwargs.get("tool_choice"):
            body["tool_choice"] = kwargs["tool_choice"]
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

        t0 = time.monotonic()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in {429, 500, 502, 503, 504} and attempt < 3:
                    retry_after = int(e.headers.get("Retry-After", 0))
                    backoff = max(retry_after, 2 ** attempt)
                    _api_logger.warning(
                        "api_retry attempt=%d/4 code=%d backoff=%ds model=%s",
                        attempt + 1, e.code, backoff, self.model,
                    )
                    time.sleep(backoff)
                    continue
                raise

        elapsed = time.monotonic() - t0
        usage = result.get("usage", {})
        _api_logger.debug(
            "api_call model=%s latency=%.2fs tokens_in=%s tokens_out=%s",
            self.model, elapsed,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
        )

        choice = result["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        # GLM 4.5 Air uses "reasoning", other models use "reasoning_content"
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        # Fall back to reasoning when content is empty and no tool calls
        if not content and reasoning and not message.get("tool_calls"):
            content = reasoning
        response: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        # Preserve reasoning so downstream can capture thinking
        # even on tool-call steps where content is empty
        if reasoning:
            response["reasoning_content"] = reasoning
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
        ``~/.konash/projects/<project>/checkpoints``.
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
        embedding_provider: str = "local",
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
            _ckpt = checkpoint_dir or os.path.join(
                os.path.expanduser("~/.konash/projects"), project, "checkpoints"
            )
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
            os.path.expanduser("~/.konash/projects"), project, "checkpoints"
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
        self._task_name = self._infer_task_name()

    def _infer_task_name(self) -> Optional[str]:
        """Infer the KARL task name from the corpus path.

        Returns a task name recognised by ``PassRateFilter`` and
        ``QualityFilter`` (e.g. ``"BrowseCompPlus"``, ``"TRECBiogen"``),
        or ``None`` for unknown / user-supplied corpora.
        """
        corpus_path = str(self.corpus.path).lower()
        if "browsecomp" in corpus_path:
            return "BrowseCompPlus"
        if "trec" in corpus_path or "biogen" in corpus_path:
            return "TRECBiogen"
        return None

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
        if verbose:
            from rich.console import Console as _Console
            _con = _Console()

        from konash.training import checkpoint as ckpt
        from konash.training.logger import TrainingLogger, configure_file_logging
        debug_log_path = configure_file_logging(self.project)
        _api_logger.info("Debug log: %s", debug_log_path)

        self._log = TrainingLogger(self.project)
        log = self._log
        log.start(
            iterations=iterations,
            corpus=str(self.corpus.path),
            model=self.base_model,
        )

        # Client-server architecture:
        #
        # 1. CLIENT (this laptop): synthesis + rollouts via Together AI API.
        #    No GPU needed. Checkpointed after each phase for crash recovery.
        #
        # 2. SERVER (cloud GPU): OAPL gradient updates. Provisioned via
        #    Shadeform only when rollouts are ready. Stays alive across
        #    iterations for bootstrapping (trained model from iter N
        #    becomes synthesizer for iter N+1).
        #
        # For iteration 1: client generates rollouts → sends to server.
        # For iteration 2+: server runs full pipeline (it has the trained
        # model locally for bootstrapping).

        stats: List[Dict[str, Any]] = []

        # ── Iteration 1: client-side synthesis + rollouts ──
        if verbose:
            _con.print()
            _con.rule(f"[bold]Iteration 1/{iterations}[/]", style="dim")
            _con.print()
            _con.print("  [dim]Phase 1: Synthesis + rollouts (local, via API)[/]")

        rollouts_path = self._run_iteration_local(
            iteration=0,
            synthesis_calls=synthesis_calls,
            rollouts_per_example=rollouts_per_example,
            rollout_max_steps=rollout_max_steps,
            max_examples=max_examples,
            few_shot_examples=few_shot_examples,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        if not rollouts_path:
            if verbose:
                _con.print("  [yellow]–[/]  No training data produced.")
            self._trained = True
            self._iteration = 1
            return {"iterations": 1, "stats": []}

        # ── Provision GPU server, send rollouts for OAPL ──
        if verbose:
            _con.print()
            _con.print("  [dim]Phase 2: OAPL training (cloud GPU)[/]")
            _con.print(
                "  [cyan]Finding cheapest GPU via Shadeform...[/]"
            )

        from konash.cloud import train_oapl_from_rollouts, train_remote, tear_down

        multi_iter = iterations > 1

        # Iter 1: OAPL from client-generated rollouts
        # Keep GPU alive if more iterations follow
        cloud_result = train_oapl_from_rollouts(
            rollouts_path=rollouts_path,
            base_model=self.base_model,
            checkpoint_dir=self.checkpoint_dir,
            learning_rate=learning_rate,
            keep_alive=multi_iter,
            verbose=verbose,
        )
        iter_stats = (cloud_result.get("stats", [{}]) or [{}])[-1]
        if iter_stats:
            stats.append(iter_stats)

        # Iter 2+: full pipeline on the same GPU (bootstrapping)
        # The trained model from iter 1 is already loaded on the cluster
        if multi_iter:
            if verbose:
                _con.print()
                _con.rule(
                    f"[bold]Iterations 2–{iterations}[/]  (cloud, bootstrapping)",
                    style="dim",
                )
                _con.print(
                    "  [dim]Trained model becomes next synthesizer[/]"
                )

            cloud_result = train_remote(
                corpus=str(self.corpus.path),
                base_model=self.base_model,
                checkpoint_dir=self.checkpoint_dir,
                iterations=iterations - 1,
                rollouts_per_example=rollouts_per_example,
                learning_rate=learning_rate,
                keep_alive=False,  # tear down after last iteration
                verbose=verbose,
            )
            for s in (cloud_result.get("stats") or []):
                stats.append(s)
        else:
            # Single iteration — already torn down by train_oapl_from_rollouts
            pass

        self._iteration = iterations
        self._trained = True

        # Save metadata
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        meta_path = os.path.join(self.checkpoint_dir, "training_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "base_model": self.base_model,
                "corpus": str(self.corpus.path),
                "project": self.project,
                "iterations": self._iteration,
                "stats": stats,
                "value_model": False,
            }, f, indent=2)

        log.complete(
            iterations=self._iteration,
            total_seconds=time.monotonic() - log._start_time,
            stats=stats,
        )

        if verbose:
            _con.print()
            _con.print(
                f"  [bold green]Training complete.[/]  "
                f"Checkpoint saved to [dim]{self.checkpoint_dir}[/]"
            )
            _con.print(
                f"  [dim]Training log: {log.path}[/]"
            )

        return {"iterations": self._iteration, "stats": stats}

    def _run_iteration_local(
        self,
        *,
        iteration: int,
        synthesis_calls: int,
        rollouts_per_example: int,
        rollout_max_steps: int,
        max_examples: Optional[int],
        few_shot_examples: Optional[List[SyntheticExample]],
        learning_rate: float,
        verbose: bool,
    ) -> Optional[str]:
        """Run synthesis + rollouts locally, return path to rollout checkpoint.

        Checkpoints after each phase so interrupted runs can resume.
        Returns None if no training data was produced.
        """
        from konash.training import checkpoint as ckpt

        if verbose:
            from rich.console import Console as _Console
            _con = _Console()

        # Check for existing checkpoints (resume)
        latest = ckpt.find_latest_phase(self.checkpoint_dir, iteration + 1)
        if latest and verbose:
            _con.print(
                f"  [dim]Resuming from {latest.name.lower()} checkpoint[/]"
            )

        # ── Corpus ingestion ──
        if not self.corpus.indexed:
            if verbose:
                with _con.status("  [cyan]Indexing corpus...", spinner="dots"):
                    self.corpus.ingest()
                _con.print(
                    f"  [green]✓[/]  Indexed [bold]{self.corpus.num_documents:,}[/] chunks"
                )
            else:
                self.corpus.ingest()

        # ── Stage 1: Synthesis ──
        # Try to resume from checkpoint — but ignore empty ones
        examples = []
        resumed = False

        if latest and latest >= ckpt.Phase.DEDUP:
            examples_data = ckpt.load(
                self.checkpoint_dir, iteration + 1, ckpt.Phase.DEDUP
            )
            examples = [
                SyntheticExample(
                    question=e.get("question", ""),
                    answer=e.get("answer", ""),
                )
                for e in (examples_data or [])
            ]
            if examples:
                resumed = True
                if verbose:
                    _con.print(
                        f"  [green]✓[/]  {len(examples)} examples (from checkpoint)"
                    )

        if not resumed and latest and latest >= ckpt.Phase.SYNTHESIS:
            # Have raw synthesis checkpoint — check it has data
            synth_data = ckpt.load(
                self.checkpoint_dir, iteration + 1, ckpt.Phase.SYNTHESIS
            )
            raw_examples = [
                SyntheticExample(
                    question=e.get("question", ""),
                    answer=e.get("answer", ""),
                )
                for e in (synth_data.get("examples", []) if synth_data else [])
            ]
            if raw_examples:
                resumed = True
                examples = self._run_dedup(raw_examples, max_examples, verbose)
                ckpt.save(
                    self.checkpoint_dir, iteration + 1, ckpt.Phase.DEDUP,
                    [{"question": e.question, "answer": e.answer} for e in examples],
                )

        if not resumed:
            # No valid checkpoint — run synthesis from scratch
            raw_examples = self._run_synthesis(
                synthesis_calls, few_shot_examples, iteration, verbose,
            )
            if not raw_examples:
                if verbose:
                    _con.print("  [yellow]–[/]  No QA pairs synthesized.")
                return None

            examples = self._run_dedup(raw_examples, max_examples, verbose)
            ckpt.save(
                self.checkpoint_dir, iteration + 1, ckpt.Phase.DEDUP,
                [{"question": e.question, "answer": e.answer} for e in examples],
            )

        if not examples:
            if verbose:
                _con.print("  [yellow]–[/]  No examples after dedup.")
            return {}

        # ── Stage 2: Rollouts + filtering ──
        if latest and latest >= ckpt.Phase.ROLLOUTS:
            rollouts_path = os.path.join(
                ckpt.checkpoint_dir(self.checkpoint_dir, iteration + 1),
                "stage2_rollouts.json",
            )
            if verbose:
                _con.print(
                    f"  [green]✓[/]  Rollouts loaded from checkpoint"
                )
        else:
            rollouts_path = self._run_rollouts(
                examples, rollouts_per_example, rollout_max_steps,
                iteration, verbose,
            )
            if not rollouts_path:
                if verbose:
                    _con.print("  [yellow]–[/]  No training data after filtering.")
                return {}

        # Save run manifest listing all artifacts for this iteration
        try:
            from konash.training.logger import configure_file_logging  # noqa: already imported above
            debug_log = os.path.expanduser(
                f"~/.konash/projects/{self.project}/training_debug.log"
            )
            jsonl_log = os.path.expanduser(
                f"~/.konash/projects/{self.project}/training.jsonl"
            )
            ckpt.save_manifest(
                self.checkpoint_dir, iteration + 1,
                {
                    "artifacts": {
                        "rollouts_checkpoint": rollouts_path,
                        "debug_log": debug_log,
                        "training_log_jsonl": jsonl_log,
                        "corpus_index_cache": os.path.join(
                            self.checkpoint_dir, "index_cache",
                        ),
                    },
                    "counts": {
                        "deduped_examples": len(examples),
                    },
                },
            )
        except Exception:
            _api_logger.debug("Failed to save manifest", exc_info=True)

        return rollouts_path

    def _run_synthesis(
        self,
        synthesis_calls: int,
        few_shot_examples: Optional[List[SyntheticExample]],
        iteration: int,
        verbose: bool,
    ) -> List[SyntheticExample]:
        """Run QA synthesis via API with checkpointing."""
        from konash.training import checkpoint as ckpt
        from concurrent.futures import ThreadPoolExecutor

        if verbose:
            from rich.console import Console as _Console
            _con = _Console()

        generate_fn = self._get_generate_fn()

        def _synthesis_fn(messages, **kwargs):
            kwargs.setdefault("max_new_tokens", 2048)
            return generate_fn(messages, **kwargs)

        synthesizer = QuestionAnswerSynthesizer(
            vector_search_tool=self.corpus.vector_search,
            llm_fn=_synthesis_fn,
            few_shot_examples=few_shot_examples,
        )

        # Check for incremental checkpoint
        existing = ckpt.load_synthesis_incremental(
            self.checkpoint_dir, iteration + 1,
        )
        all_raw: List[SyntheticExample] = []
        start_call = 0
        if existing:
            all_raw = [
                SyntheticExample(question=e["question"], answer=e["answer"])
                for e in existing["examples"]
            ]
            start_call = existing["calls_completed"]
            if verbose:
                _con.print(
                    f"  [dim]Resuming synthesis from call "
                    f"{start_call}/{synthesis_calls} "
                    f"({len(all_raw)} pairs so far)[/]"
                )

        remaining = synthesis_calls - start_call
        if remaining <= 0:
            return all_raw

        def _synth_one(_ci):
            try:
                return synthesizer.synthesize(documents=None, num_examples=8)
            except (ValueError, RuntimeError, OSError):
                return []

        completed = [start_call]
        workers = min(remaining, 20)
        checkpoint_interval = max(remaining // 10, 10)

        if verbose:
            import threading
            from rich.live import Live
            from rich.table import Table as _Table
            from rich.text import Text

            _synth_lock = threading.Lock()
            _latest_q = [""]
            _latest_a = [""]

            def _build_display() -> _Table:
                outer = _Table(
                    box=None, show_header=False, pad_edge=False,
                    expand=True, padding=(0, 0),
                )
                done = completed[0]
                pct = done * 100 // synthesis_calls if synthesis_calls else 0
                bar_w = 32
                filled = bar_w * done // synthesis_calls if synthesis_calls else 0
                bar = f"[cyan]{'━' * filled}[/][dim]{'─' * (bar_w - filled)}[/]"

                outer.add_row(Text("  Synthesizing QA pairs", style="bold"))
                outer.add_row(Text(""))
                outer.add_row(Text.from_markup(
                    f"    {bar}  [dim]{done}/{synthesis_calls}[/]  "
                    f"[bold]{len(all_raw)}[/] pairs  [dim]{pct}%[/]"
                ))

                if _latest_q[0]:
                    q = _latest_q[0][:90]
                    outer.add_row(Text(""))
                    outer.add_row(Text.from_markup(
                        f"    [dim]Q:[/]  {q}" + ("[dim]...[/]" if len(_latest_q[0]) > 90 else "")
                    ))
                if _latest_a[0]:
                    a = _latest_a[0][:90]
                    outer.add_row(Text.from_markup(
                        f"    [dim]A:[/]  {a}" + ("[dim]...[/]" if len(_latest_a[0]) > 90 else "")
                    ))
                return outer

            with Live(
                _build_display(), console=_con,
                refresh_per_second=4, transient=True,
            ) as live:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_synth_one, ci) for ci in range(remaining)]
                    for fut in futures:
                        batch = fut.result()
                        with _synth_lock:
                            all_raw.extend(batch)
                            completed[0] += 1
                            if batch:
                                _latest_q[0] = batch[-1].question or ""
                                _latest_a[0] = batch[-1].answer or ""
                            live.update(_build_display())

                        if completed[0] % checkpoint_interval == 0:
                            ckpt.save_synthesis_incremental(
                                self.checkpoint_dir, iteration + 1,
                                [{"question": e.question, "answer": e.answer} for e in all_raw],
                                completed[0], synthesis_calls,
                            )
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_synth_one, ci) for ci in range(remaining)]
                for fut in futures:
                    batch = fut.result()
                    all_raw.extend(batch)
                    completed[0] += 1

                    if completed[0] % checkpoint_interval == 0:
                        ckpt.save_synthesis_incremental(
                            self.checkpoint_dir, iteration + 1,
                            [{"question": e.question, "answer": e.answer} for e in all_raw],
                            completed[0], synthesis_calls,
                        )

        # Final checkpoint
        ckpt.save_synthesis_incremental(
            self.checkpoint_dir, iteration + 1,
            [{"question": e.question, "answer": e.answer} for e in all_raw],
            synthesis_calls, synthesis_calls,
        )

        if verbose:
            _con.print(
                f"  [green]✓[/]  {len(all_raw)} QA pairs synthesized"
            )

        if hasattr(self, "_log"):
            self._log.synthesis(
                iteration=iteration + 1,
                calls_completed=synthesis_calls,
                calls_total=synthesis_calls,
                raw_pairs=len(all_raw),
                deduped=0,  # dedup happens separately
            )

        return all_raw

    def _run_dedup(
        self,
        raw_examples: List[SyntheticExample],
        max_examples: Optional[int],
        verbose: bool,
    ) -> List[SyntheticExample]:
        """Deduplicate synthesized examples."""
        if verbose:
            from rich.console import Console as _Console
            _con = _Console()

        generate_fn = self._get_generate_fn()

        def _synthesis_fn(messages, **kwargs):
            kwargs.setdefault("max_new_tokens", 2048)
            return generate_fn(messages, **kwargs)

        def _rollout_fn(messages, **kwargs):
            kwargs.setdefault("max_new_tokens", 512)
            return generate_fn(messages, **kwargs)

        pipeline = SynthesisPipeline(
            config=SynthesisTaskConfig(task_name=self._task_name),
            synthesizer=QuestionAnswerSynthesizer(
                vector_search_tool=self.corpus.vector_search,
                llm_fn=_synthesis_fn,
            ),
            rollout_generator=RolloutGenerator(
                search_tool=self.corpus.vector_search,
                llm_fn=_rollout_fn,
            ),
        )

        if verbose:
            with _con.status(
                f"  [cyan]Deduplicating {len(raw_examples)} pairs...",
                spinner="dots",
            ):
                examples = pipeline.deduplicate(raw_examples)
        else:
            examples = pipeline.deduplicate(raw_examples)

        if max_examples and len(examples) > max_examples:
            examples = examples[:max_examples]

        if verbose:
            _con.print(f"  [green]✓[/]  {len(examples)} unique examples")

        return examples

    def _run_rollouts(
        self,
        examples: List[SyntheticExample],
        rollouts_per_example: int,
        rollout_max_steps: int,
        iteration: int,
        verbose: bool,
    ) -> Optional[str]:
        """Generate rollouts, filter, checkpoint, return path to rollout JSON."""
        from konash.training import checkpoint as ckpt

        if verbose:
            from rich.console import Console as _Console
            _con = _Console()

        generate_fn = self._get_generate_fn()

        def _rollout_fn(messages, **kwargs):
            kwargs.setdefault("max_new_tokens", 512)
            return generate_fn(messages, **kwargs)

        def _judge_fn(messages, **kwargs):
            kwargs.setdefault("max_new_tokens", 1024)
            return generate_fn(messages, **kwargs)

        rollout_gen = RolloutGenerator(
            search_tool=self.corpus.vector_search,
            llm_fn=_rollout_fn,
            max_steps=rollout_max_steps,
        )
        pipeline = SynthesisPipeline(
            config=SynthesisTaskConfig(task_name=self._task_name),
            synthesizer=QuestionAnswerSynthesizer(
                vector_search_tool=self.corpus.vector_search,
                llm_fn=_rollout_fn,
            ),
            rollout_generator=rollout_gen,
            judge_fn=_judge_fn,
        )

        # Incremental rollout checkpointing — pass checkpoint_dir so
        # completed groups are saved every 50 QA pairs
        rollout_ckpt_dir = self.checkpoint_dir

        # Progress callback → JSONL log
        import time as _time
        _rollout_t0 = _time.monotonic()

        def _on_progress(completed: int, total: int) -> None:
            if hasattr(self, "_log"):
                self._log.rollout_progress(
                    iteration=iteration + 1,
                    completed=completed,
                    total=total,
                    elapsed_seconds=_time.monotonic() - _rollout_t0,
                )

        stage_two_kwargs = dict(
            examples=examples,
            num_rollouts=rollouts_per_example,
            checkpoint_dir=rollout_ckpt_dir,
            checkpoint_iteration=iteration + 1,
            on_rollout_progress=_on_progress,
        )

        if verbose:
            with _con.status(
                f"  [cyan]Generating rollouts "
                f"({len(examples)} × {rollouts_per_example})...",
                spinner="dots",
            ):
                final_examples = pipeline.run_stage_two(**stage_two_kwargs)
        else:
            final_examples = pipeline.run_stage_two(**stage_two_kwargs)

        if verbose:
            _con.print(
                f"  [green]✓[/]  {len(final_examples)} examples after "
                f"pass-rate filtering"
            )

        if not final_examples or not pipeline.filtered_groups:
            return None

        # Build rollout data in the format expected by train_oapl_unsloth.py
        groups = []
        for group in pipeline.filtered_groups:
            groups.append({
                "prompt": group.prompt,
                "question": group.prompt,
                "rollouts": [
                    {
                        "steps": r.steps,
                        "final_answer": r.final_answer,
                        "passed": r.passed,
                    }
                    for r in group.rollouts
                ],
            })

        # Save checkpoint
        rollouts_path = ckpt.save(
            self.checkpoint_dir, iteration + 1, ckpt.Phase.ROLLOUTS,
            {"groups": groups},
        )

        if verbose:
            _con.print(
                f"  [green]✓[/]  Rollouts checkpointed to {rollouts_path}"
            )

        if hasattr(self, "_log"):
            total_rollouts = sum(len(g["rollouts"]) for g in groups)
            self._log.rollouts(
                iteration=iteration + 1,
                examples=len(examples),
                rollouts=total_rollouts,
                filtered=len(groups),
            )
            # Log filter summaries
            self._log.filter_summary(
                iteration=iteration + 1,
                phase="pass_rate",
                input_count=len(pipeline.rollout_groups),
                output_count=len(pipeline.filtered_groups),
            )
            self._log.filter_summary(
                iteration=iteration + 1,
                phase="quality",
                input_count=len(pipeline.filtered_groups),
                output_count=len(final_examples),
            )

        return rollouts_path

    # ------------------------------------------------------------------
    # Solve (inference)
    # ------------------------------------------------------------------

    def solve(
        self,
        query: str,
        *,
        parallel_rollouts: int = 1,
        max_steps: int = 20,
        top_k: int = 20,
        use_vgs: Optional[bool] = None,
        vgs_candidate_width: int = 2,
        vgs_max_depth: int = 10,
        return_trace: bool = False,
    ) -> str | Dict[str, Any]:
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
        return_trace : bool
            If True, return a dict with ``answer`` and ``trajectory``
            (list of step records) instead of just the answer string.

        Returns
        -------
        str | dict
            The agent's answer, or ``{"answer": str, "trajectory": list}``
            when *return_trace* is True.
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
                f"[{i+1}] (score: {r.get('score', 0):.3f}) {r.get('text', '')}"
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
            answer = result.get("final_answer") or ""
            if return_trace:
                return {"answer": answer, "trajectory": result.get("trajectory", [])}
            return answer

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
            answer = result.get("answer", "")
            if return_trace:
                return {"answer": answer, "trajectory": result.get("trajectories", [])}
            return answer

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
        answer = result.get("answer", "")
        if return_trace:
            return {"answer": answer, "trajectory": result.get("rollouts", [])}
        return answer

    # ------------------------------------------------------------------
    # Search (direct corpus search without LLM)
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
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
            Path to the ``~/.konash/projects/<project>`` directory.
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
        - ``"gte-large"`` — GTE-large via sentence-transformers (1024-dim, best retrieval)
        - ``"gemini"`` — Gemini Embedding API (fast, free tier)
        - ``"hf"`` — Qwen3-Embedding-8B via HuggingFace Inference API
        - ``"local"`` — Qwen3-Embedding-0.6B via sentence-transformers (CPU)
        """
        if self.embedding_provider == "gte-large":
            return self._make_gte_large_embed_fn()
        elif self.embedding_provider == "gemini":
            return self._make_gemini_embed_fn()
        elif self.embedding_provider == "hf":
            return self._make_hf_embed_fn()
        elif self.embedding_provider == "local":
            from konash.retrieval.vector_search import load_embedding_model
            return load_embedding_model("Qwen/Qwen3-Embedding-0.6B")
        else:
            raise ValueError(
                f"Unknown embedding_provider={self.embedding_provider!r}. "
                "Use 'gte-large', 'gemini', 'hf', or 'local'."
            )

    def _make_gte_large_embed_fn(self):
        """Return an embed_fn using GTE-large via sentence-transformers."""
        from konash.retrieval.vector_search import load_embedding_model
        return load_embedding_model("thenlper/gte-large")

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
