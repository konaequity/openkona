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
                if e.code == 429 and attempt < 3:
                    retry_after = int(e.headers.get("Retry-After", 10))
                    time.sleep(retry_after)
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
        checkpoint_dir: Optional[str] = None,
        chunk_size: int = 512,
        temperature: float = 0.7,
        # LoRA / model config
        lora_r: int = 16,
        lora_alpha: int = 32,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        gradient_checkpointing: bool = False,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        self.base_model = base_model
        self.project = project
        self.temperature = temperature

        # Corpus
        if isinstance(corpus, Corpus):
            self.corpus = corpus
        else:
            self.corpus = Corpus(corpus, chunk_size=chunk_size)

        # LLM connection
        self.api_base = api_base or os.environ.get("KONASH_API_BASE")
        self.api_key = api_key or os.environ.get("KONASH_API_KEY", "no-key")

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
        self._trained = False
        self._iteration = 0

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        iterations: int = 2,
        rollouts_per_example: int = 8,
        max_examples: Optional[int] = None,
        learning_rate: float = 1e-5,
        beta_kl: float = 0.1,
        beta_value: float = 0.05,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the full KONASH training loop.

        1. Load the model locally (with LoRA).
        2. Ingest the corpus and build the vector search index.
        3. Synthesize QA pairs from the corpus using the model.
        4. Generate rollouts and filter by pass rate.
        5. Train with OAPL (real gradient updates on LoRA params).
        6. Repeat for *iterations* rounds.
        7. Save the LoRA adapter checkpoint.

        Parameters
        ----------
        iterations : int
            Number of synthesis → train cycles (default 2).
        rollouts_per_example : int
            Number of rollouts per training example (default 8).
        max_examples : int | None
            Cap on synthesized training examples per iteration.
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
        engine = None
        if self.api_base is None or self.inference_api_base is not None:
            try:
                engine = self._get_model_engine()
                if verbose:
                    print(f"Model loaded: {self.base_model}")
                    if self.inference_api_base:
                        print(f"Inference via: {self.inference_model or self.base_model} "
                              f"@ {self.inference_api_base}")
            except (ImportError, OSError, ValueError, RuntimeError) as e:
                if verbose:
                    print(f"  Note: Could not load model locally ({e}).")
                    print("  Using lightweight mode (no gradient updates).")
                    print("  Install: pip install openkona[train]")

        # Step 2: Ingest corpus
        if not self.corpus.indexed:
            if verbose:
                print(f"Ingesting corpus from {self.corpus.path} ...")
            self.corpus.ingest()
            if verbose:
                print(f"  Indexed {self.corpus.num_documents} chunks.")

        # Set up synthesis and rollout generate functions.
        # KARL paper Section 7.2: rollouts must use the MODEL BEING TRAINED
        # so there's variance at the learning frontier.  A stronger model
        # (e.g. 32B via Groq) aces or fails everything → no learning signal.
        #
        # Split mode: inference API (Groq) for synthesis, local model for rollouts.
        synthesis_generate_fn = self._get_generate_fn()

        # Rollout function: always prefer the local model engine
        import re as _re_local

        def _strip_think(result):
            if isinstance(result, dict) and "content" in result:
                result["content"] = _re_local.sub(
                    r"<think>.*?</think>\s*", "", result["content"],
                    flags=_re_local.DOTALL,
                )
                result["content"] = _re_local.sub(
                    r"<think>.*", "", result["content"],
                    flags=_re_local.DOTALL,
                ).strip()
            return result

        if engine is not None:
            # Local model for rollouts (matches KARL: same model for both)
            def rollout_generate_fn(messages, **kwargs):
                return _strip_think(engine.generate(messages, **kwargs))
            if verbose and self.inference_api_base:
                print("  Rollouts: local model (learning frontier)")
                print("  Synthesis: inference API (strong model)")
        else:
            # No local engine — fall back to inference API for everything
            rollout_generate_fn = synthesis_generate_fn

        # Synthesis needs more tokens (thinking + many QA pairs).
        # Rollouts need fewer (short JSON reasoning steps).
        def _synthesis_fn(messages, **kwargs):
            kwargs.setdefault("max_new_tokens", 2048)
            return synthesis_generate_fn(messages, **kwargs)

        def _rollout_fn(messages, **kwargs):
            kwargs.setdefault("max_new_tokens", 256)
            return rollout_generate_fn(messages, **kwargs)

        synthesizer = QuestionAnswerSynthesizer(
            vector_search_tool=self.corpus.vector_search,
            llm_fn=_synthesis_fn,
        )

        # Progress callback for rollouts — _current_examples is mutated
        # before each stage-two call so the closure sees the right list.
        _current_examples: List[Any] = []

        def _on_step(qa_idx, rollout_idx, step_idx, step_record):
            if verbose and step_idx == 0 and rollout_idx == 0:
                q_text = ""
                if qa_idx < len(_current_examples):
                    q_text = (_current_examples[qa_idx].question or "")[:80]
                print(f"    [{qa_idx + 1}] {q_text}")

        rollout_gen = RolloutGenerator(
            search_tool=self.corpus.vector_search,
            llm_fn=_rollout_fn,
            on_step=_on_step,
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

        for iteration in range(iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{iterations}")
                print(f"{'='*60}")

            # Step 3: Synthesize QA pairs
            # Free transient GPU memory before generation
            import gc
            gc.collect()
            if self._model_engine is not None:
                self._model_engine._torch.cuda.empty_cache()
            # Sample a subset of documents to keep the prompt manageable
            # on small GPUs (full corpus can be hundreds of chunks).
            import random as _random
            all_docs = [d["text"] for d in self.corpus.documents]
            sample_size = min(5, len(all_docs))
            documents = _random.sample(all_docs, sample_size)
            documents = [d[:500] for d in documents]  # truncate for VRAM
            if verbose:
                previews = [d[:60].replace("\n", " ") for d in documents]
                print(f"  Synthesizing from {len(documents)} chunks:")
                for p in previews:
                    print(f"    - {p}...")
                print("  Generating QA pairs ...")
            try:
                examples = pipeline.run_stage_one(
                    documents=documents,
                    num_examples=max_examples,
                )
            except (ValueError, RuntimeError) as e:
                if verbose:
                    print(f"  Synthesis failed: {e}")
                    print("  Skipping this iteration.")
                continue
            if verbose:
                print(f"  Generated {len(examples)} examples:")
                for i, ex in enumerate(examples):
                    print(f"    [{i+1}] {(ex.question or '')[:80]}")

            # Step 4: Generate rollouts + filter
            if verbose:
                print("  Generating rollouts ...")
            _current_examples[:] = examples
            final_examples = pipeline.run_stage_two(
                examples=examples,
                num_rollouts=rollouts_per_example,
            )
            if verbose:
                print(f"  {len(final_examples)} examples after filtering.")

            if not final_examples or not pipeline.filtered_groups:
                if verbose:
                    print("  No training data — skipping training step.")
                continue

            # Step 5: Build dataset and train with OAPL
            if verbose:
                print("  Training with OAPL ...")
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
                    print(f"  Loss: {epoch_stats['mean_loss']:.4f}  "
                          f"Groups: {epoch_stats['num_groups']}  "
                          f"Rollouts: {epoch_stats['num_rollouts']}")

            self._iteration = iteration + 1

        # Step 6: Save checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        meta_path = os.path.join(self.checkpoint_dir, "training_meta.json")

        # Save LoRA adapter if we did real training
        if self._model_engine is not None:
            adapter_path = os.path.join(self.checkpoint_dir, "adapter")
            engine.save_adapter(adapter_path)

        with open(meta_path, "w") as f:
            json.dump({
                "base_model": self.base_model,
                "project": self.project,
                "iterations": self._iteration,
                "stats": stats,
            }, f, indent=2)

        self._trained = True
        if verbose:
            print(f"\nTraining complete. Checkpoint saved to {self.checkpoint_dir}")

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
    ) -> str:
        """Answer a question using the trained (or base) knowledge agent.

        Parameters
        ----------
        query : str
            The question to answer.
        parallel_rollouts : int
            Number of independent rollouts to run and aggregate (default 1).
            Set to 10-20 for Parallel Thinking.
        max_steps : int
            Maximum agent steps per rollout.
        top_k : int
            Number of documents to retrieve per search.

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

        if parallel_rollouts <= 1:
            # Single rollout
            env = make_environment()
            env.reset(prompt=query)
            result = env.run_episode(agent, max_steps=max_steps)
            return result.get("final_answer") or ""

        # Parallel Thinking
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

            def extract_final_answer(self, conversation_history: Any, **kwargs: Any) -> str | None:
                if isinstance(conversation_history, dict):
                    conversation_history = (
                        conversation_history.get("history")
                        or conversation_history.get("messages")
                        or conversation_history
                    )
                return self._base_agent.extract_final_answer(conversation_history, **kwargs)

        env_agent = _EnvironmentBackedAgent(agent)
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
        return agent

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model_engine(self) -> Any:
        """Load the model locally with LoRA. Returns a LocalModelEngine."""
        if self._model_engine is not None:
            return self._model_engine

        from konash.inference.local import LocalModelEngine

        # Check for existing adapter
        adapter_path = os.path.join(self.checkpoint_dir, "adapter")
        has_adapter = os.path.exists(adapter_path)

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
