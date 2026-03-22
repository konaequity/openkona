"""High-level KONASH API — the user-facing entry point."""

from __future__ import annotations

import json
import logging
import os
import time
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
from konash.plugins.compression import RLTrainableCompressionPlugin
from konash.plugins.control import StepBudgetPlugin
from konash.inference.parallel import ParallelThinkingEngine
from konash.inference.aggregation import GenerativeAggregator
from konash.synthesis.qa import SyntheticExample
from konash.training.project_state import build_dataset_spec, suggest_project_name
from konash.models import ModelPreset, get_model_presets


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

MODEL_PRESETS: Dict[str, ModelPreset] = get_model_presets()


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
        max_retries = 8
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                # Log response body for debugging API errors
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if e.code == 400:
                    _api_logger.warning(
                        "api_400 model=%s body=%s err=%s",
                        self.model, json.dumps(body)[:500], err_body[:500],
                    )
                if e.code in {400, 429, 500, 502, 503, 504} and attempt < max_retries - 1:
                    retry_after = int(e.headers.get("Retry-After", 0))
                    # 429 gets longer backoff with jitter to spread out retries
                    if e.code == 429:
                        import random
                        backoff = max(retry_after, 2 ** attempt + random.random() * 2)
                    else:
                        backoff = max(retry_after, 2 ** attempt)
                    _api_logger.warning(
                        "api_retry attempt=%d/%d code=%d backoff=%.1fs model=%s",
                        attempt + 1, max_retries, e.code, backoff, self.model,
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
        project: str | None = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        inference_api_base: Optional[str] = None,
        inference_api_key: Optional[str] = None,
        inference_model: Optional[str] = None,
        hf_token: Optional[str] = None,
        remote_corpus_name: Optional[str] = None,
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
        corpus_path = corpus.path if isinstance(corpus, Corpus) else Path(corpus)
        auto_project = suggest_project_name(base_model, build_dataset_spec([str(corpus_path)]))
        self.project = project or auto_project
        self.temperature = temperature
        self._use_unsloth = use_unsloth
        self._load_in_fp8 = load_in_fp8
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.remote_corpus_name = remote_corpus_name

        # LLM connection
        self.api_base = api_base or os.environ.get("KONASH_API_BASE")
        self.api_key = api_key or os.environ.get("KONASH_API_KEY", "no-key")

        # Billing clock — set by CLI when Shadeform instance is provisioned
        self.billing_started_at: str = ""

        # Corpus — choose embedding provider
        self.embedding_provider = embedding_provider
        if isinstance(corpus, Corpus):
            self.corpus = corpus
        else:
            embed_fn = None
            if not self._should_defer_embed_init(corpus):
                embed_fn = self._make_embed_fn()
            # Cache embeddings next to checkpoint dir for instant reload
            _ckpt = checkpoint_dir or os.path.join(
                os.path.expanduser("~/.konash/projects"), self.project, "checkpoints"
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
            os.path.expanduser("~/.konash/projects"), self.project, "checkpoints"
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

    def _should_defer_embed_init(self, corpus: str | Path | Corpus) -> bool:
        """Skip eager embedding-model loading when a bundled index exists.

        Downloaded benchmark corpora ship with ``prebuilt_index.npz`` and
        ``Corpus.ingest()`` already knows how to align the query embedder
        from that metadata. Avoiding eager local model init keeps eval from
        pulling in heavy native dependencies unnecessarily.
        """
        if self.embedding_provider != "local":
            return False
        corpus_path = Path(corpus)
        if corpus_path.is_file():
            candidates = [corpus_path.parent]
        else:
            candidates = [corpus_path, corpus_path.parent]
        return any((candidate / "prebuilt_index.npz").exists() for candidate in candidates)

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
            "base_model": cfg.base_model,
        }
        if cfg.api_base:
            defaults["api_base"] = cfg.api_base
        defaults["temperature"] = cfg.temperature
        if cfg.use_unsloth:
            defaults["use_unsloth"] = True
        if cfg.load_in_fp8:
            defaults["load_in_fp8"] = True

        # Resolve API key from env if not explicitly provided
        if "api_key" not in kwargs:
            env_var = cfg.api_key_env or ""
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
        synthesis_rollout_backend: str = "remote_full",
        keep_alive: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the full KONASH training loop.

        Training now runs as a single remote Shadeform pipeline. The remote
        worker handles synthesis, rollouts, and OAPL on GPU and writes the
        resulting checkpoint back to the project directory.

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
        synthesis_rollout_backend : str
            Training execution mode. Only ``"remote_full"`` is supported.
        keep_alive : bool
            Keep the remote Shadeform instance alive after training or failure.
        verbose : bool
            Print progress updates.

        Returns
        -------
        dict
            Training summary with per-iteration statistics.
        """
        from konash.training.execution import plan_training_execution
        console = self._make_training_console(verbose)
        log = self._start_training_run(iterations)

        plan = plan_training_execution(
            iterations=iterations,
            synthesis_rollout_backend=synthesis_rollout_backend,
        )
        return self._train_with_remote_full_pipeline(
            log=log,
            plan=plan,
            iterations=iterations,
            synthesis_calls=synthesis_calls,
            rollouts_per_example=rollouts_per_example,
            rollout_max_steps=rollout_max_steps,
            max_examples=max_examples,
            learning_rate=learning_rate,
            keep_alive=keep_alive,
            verbose=verbose,
            console=console,
        )

    def _make_training_console(self, verbose: bool) -> Any:
        """Return a Rich console for verbose training output, or None."""
        if not verbose:
            return None
        from rich.console import Console as _Console
        return _Console()

    def _start_training_run(self, iterations: int) -> Any:
        """Configure logging and emit the training start event."""
        from konash.training.logger import TrainingLogger, configure_file_logging

        debug_log_path = configure_file_logging(self.project)
        _api_logger.info("Debug log: %s", debug_log_path)
        self._log = TrainingLogger(self.project)
        self._log.start(
            iterations=iterations,
            corpus=self.remote_corpus_name or str(self.corpus.path),
            model=self.base_model,
            billing_started_at=self.billing_started_at,
        )
        return self._log

    def _save_training_metadata(
        self,
        *,
        iterations: int,
        stats: List[Dict[str, Any]],
        value_model: bool,
        synthesis_rollout_backend: str,
    ) -> None:
        """Persist training metadata for later load/inference flows."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        meta_path = os.path.join(self.checkpoint_dir, "training_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "base_model": self.base_model,
                "corpus": str(self.corpus.path),
                "project": self.project,
                "iterations": iterations,
                "stats": stats,
                "value_model": value_model,
                "synthesis_rollout_backend": synthesis_rollout_backend,
            }, f, indent=2)

    def _complete_training_run(
        self,
        *,
        log: Any,
        iterations: int,
        stats: List[Dict[str, Any]],
        verbose: bool,
        console: Any,
    ) -> Dict[str, Any]:
        """Finalize agent state, emit completion logs, and return summary."""
        self._iteration = iterations
        self._trained = True
        log.complete(
            iterations=self._iteration,
            total_seconds=time.monotonic() - log._start_time,
            stats=stats,
        )
        if verbose and console is not None:
            console.print()
            console.print(
                f"  [bold green]Training complete.[/]  "
                f"Checkpoint saved to [dim]{self.checkpoint_dir}[/]"
            )
            console.print(f"  [dim]Training log: {log.path}[/]")
        return {"iterations": self._iteration, "stats": stats}

    def _train_with_remote_full_pipeline(
        self,
        *,
        log: Any,
        plan: Any,
        iterations: int,
        synthesis_calls: int,
        rollouts_per_example: int,
        rollout_max_steps: int,
        max_examples: Optional[int],
        learning_rate: float,
        keep_alive: bool,
        verbose: bool,
        console: Any,
    ) -> Dict[str, Any]:
        """Run synthesis, rollouts, and OAPL fully on the remote GPU path."""
        from konash.cloud import train_remote

        if verbose and console is not None:
            console.print()
            console.rule("[bold]Cloud Training[/]", style="dim")
            console.print()
            console.print("  [dim]Synthesis + rollouts + OAPL will run on Shadeform.[/]")
            console.print("  [cyan]Finding cheapest GPU via Shadeform...[/]")

        cloud_result = train_remote(
            corpus=self.remote_corpus_name or str(self.corpus.path),
            base_model=self.base_model,
            checkpoint_dir=self.checkpoint_dir,
            iterations=iterations,
            synthesis_calls=synthesis_calls,
            rollouts_per_example=rollouts_per_example,
            rollout_max_steps=rollout_max_steps,
            max_examples=max_examples,
            learning_rate=learning_rate,
            keep_alive=keep_alive,
            verbose=verbose,
        )
        stats = list(cloud_result.get("stats") or [])
        self._save_training_metadata(
            iterations=iterations,
            stats=stats,
            value_model=cloud_result.get("value_model", False),
            synthesis_rollout_backend=plan.synthesis_rollout_backend,
        )
        return self._complete_training_run(
            log=log,
            iterations=iterations,
            stats=stats,
            verbose=verbose,
            console=console,
        )

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
        formatted_query = self._format_solver_prompt(query)
        environment_factory = self._make_environment_factory(
            agent=agent,
            max_steps=max_steps,
            top_k=top_k,
        )

        if parallel_rollouts <= 1 and not (use_vgs and self._value_model):
            return self._solve_with_single_rollout(
                agent=agent,
                formatted_query=formatted_query,
                environment_factory=environment_factory,
                max_steps=max_steps,
                return_trace=return_trace,
            )

        env_agent = self._make_environment_backed_agent(
            base_agent=agent,
            environment_factory=environment_factory,
            max_steps=max_steps,
            top_k=top_k,
        )

        # Decide: VGS or Parallel Thinking
        _use_vgs = use_vgs if use_vgs is not None else (self._value_model is not None)

        if _use_vgs and self._value_model is not None:
            return self._solve_with_value_guided_search(
                env_agent=env_agent,
                formatted_query=formatted_query,
                parallel_rollouts=parallel_rollouts,
                vgs_candidate_width=vgs_candidate_width,
                vgs_max_depth=vgs_max_depth,
                return_trace=return_trace,
            )

        return self._solve_with_parallel_thinking(
            env_agent=env_agent,
            formatted_query=formatted_query,
            parallel_rollouts=parallel_rollouts,
            return_trace=return_trace,
        )

    def _extract_tool_query(self, tool_call: Any) -> str:
        """Extract a search query string from a tool call payload."""
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

    def _build_search_tool_schema(self, description: str) -> Dict[str, Any]:
        """Return the standard search tool schema used by solve-time agents."""
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": description,
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
        }

    def _search_tool_observation(self, tool_call: Any, top_k: int) -> Dict[str, Any]:
        """Execute a search tool call against the corpus and format the observation."""
        query_text = self._extract_tool_query(tool_call)
        results = self.corpus.search(query_text, top_k=top_k)
        result_text = "\n\n".join(
            f"[{i+1}] (score: {r.get('score', 0):.3f}) [{os.path.basename(r.get('source', ''))}] {r.get('text', '')}"
            for i, r in enumerate(results)
        )
        observation: Dict[str, Any] = {"role": "tool", "content": result_text}
        if isinstance(tool_call, dict) and tool_call.get("id"):
            observation["tool_call_id"] = tool_call["id"]
        return observation

    def _make_environment_factory(
        self,
        *,
        agent: BaseAgent,
        max_steps: int,
        top_k: int,
    ) -> Any:
        """Return a factory that builds solve-time environments with search enabled."""
        available_tools = [self._build_search_tool_schema(
            "Search the knowledge base for relevant documents."
        )]

        def make_environment() -> Environment:
            return Environment(
                tool_executor=lambda tool_call: self._search_tool_observation(tool_call, top_k),
                plugins=[
                    RLTrainableCompressionPlugin(
                        threshold_chars=150_000,
                        target_chars=2_000,
                        agent_fn=agent.generate,
                    ),
                    StepBudgetPlugin(max_steps=max_steps),
                ],
                available_tools=available_tools,
            )

        return make_environment

    def _extract_full_response_from_history(self, history: List[Dict[str, Any]]) -> str:
        """Return the final assistant response content from an episode history."""
        for msg in reversed(history):
            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                return (
                    msg.get("content", "")
                    or msg.get("reasoning_content", "")
                    or msg.get("reasoning", "")
                    or ""
                )
        return ""

    def _solve_with_single_rollout(
        self,
        *,
        agent: BaseAgent,
        formatted_query: str,
        environment_factory: Any,
        max_steps: int,
        return_trace: bool,
    ) -> str | Dict[str, Any]:
        """Run one environment-backed rollout and optionally return its trace."""
        env = environment_factory()
        env.reset(prompt=formatted_query)
        result = env.run_episode(agent, max_steps=max_steps)
        answer = result.get("final_answer") or ""
        if return_trace:
            return {
                "answer": answer,
                "full_response": self._extract_full_response_from_history(result.get("history", [])),
                "trajectory": result.get("trajectory", []),
            }
        return answer

    def _make_environment_backed_agent(
        self,
        *,
        base_agent: BaseAgent,
        environment_factory: Any,
        max_steps: int,
        top_k: int,
    ) -> Any:
        """Wrap a BaseAgent so VGS and parallel thinking can use search environments."""
        search_tools = [self._build_search_tool_schema("Search the knowledge base.")]
        search_tool_observation = lambda tool_call: self._search_tool_observation(tool_call, top_k)

        class _EnvironmentBackedAgent:
            def __init__(self, wrapped_agent: BaseAgent) -> None:
                self._wrapped_agent = wrapped_agent

            def generate_rollout(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
                env = environment_factory()
                env.reset(prompt=prompt)
                return env.run_episode(
                    self._wrapped_agent,
                    max_steps=kwargs.get("max_steps", max_steps),
                )

            def generate_step(
                self, conversation_history: Any, **kwargs: Any,
            ) -> Dict[str, Any]:
                messages = (
                    list(conversation_history)
                    if isinstance(conversation_history, list)
                    else [{"role": "user", "content": str(conversation_history)}]
                )
                step_result = self._wrapped_agent.generate_step(
                    messages,
                    available_tools=search_tools,
                )
                content = step_result.get("content", "")
                tool_calls = step_result.get("tool_calls")
                if tool_calls:
                    observation = search_tool_observation(tool_calls[0])
                    return {
                        "type": "tool_call",
                        "role": "assistant",
                        "content": content,
                        "result": observation.get("content", ""),
                        "terminal": False,
                    }
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
                return self._wrapped_agent.extract_final_answer(conversation_history, **kwargs)

        return _EnvironmentBackedAgent(base_agent)

    def _solve_with_value_guided_search(
        self,
        *,
        env_agent: Any,
        formatted_query: str,
        parallel_rollouts: int,
        vgs_candidate_width: int,
        vgs_max_depth: int,
        return_trace: bool,
    ) -> str | Dict[str, Any]:
        """Solve with Value-Guided Search when a value model is available."""
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
        result = vgs_engine.run(formatted_query)
        answer = result.get("answer", "")
        if return_trace:
            return {"answer": answer, "trajectory": result.get("trajectories", [])}
        return answer

    def _solve_with_parallel_thinking(
        self,
        *,
        env_agent: Any,
        formatted_query: str,
        parallel_rollouts: int,
        return_trace: bool,
    ) -> str | Dict[str, Any]:
        """Solve with parallel rollouts and generative aggregation."""
        engine = ParallelThinkingEngine(
            agent=env_agent,
            aggregator=GenerativeAggregator(
                agent=env_agent,
                aggregation_mode="generative",
            ),
            num_rollouts=parallel_rollouts,
        )
        result = engine.run(formatted_query)
        answer = result.get("answer", "")
        full_response = result.get("full_response", "") or answer
        if return_trace:
            return {
                "answer": answer,
                "full_response": full_response,
                "trajectory": result.get("rollouts", []),
            }
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
            try:
                return load_embedding_model("Qwen/Qwen3-Embedding-0.6B")
            except RuntimeError:
                return None  # Prebuilt index will handle queries
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

    _solver_prompt_template: str | None = None

    @classmethod
    def _get_solver_prompt_template(cls) -> str:
        """Return the KARL Figure 34 solver prompt template (cached)."""
        if cls._solver_prompt_template is None:
            from konash.prompts.registry import PromptRegistry
            cls._solver_prompt_template = PromptRegistry.prompts["figure_34_solver_rollout"].template
        return cls._solver_prompt_template

    def _format_solver_prompt(self, question: str) -> str:
        """Build the full solver prompt with the question substituted."""
        return self._get_solver_prompt_template().replace("{question}", question)

    def _make_agent(self, max_steps: int = 20) -> BaseAgent:
        """Build an internal BaseAgent wired to the best available backend."""

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
            system_prompt=None,
            max_steps=max_steps,
        )

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return (
            f"Agent(model={self.base_model!r}, project={self.project!r}, "
            f"corpus={self.corpus!r}, {status})"
        )
