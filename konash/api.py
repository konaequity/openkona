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
        import urllib.request

        url = f"{self.api_base}/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
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
            },
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())

        choice = result["choices"][0]
        message = choice["message"]
        response: Dict[str, Any] = {
            "role": "assistant",
            "content": message.get("content", ""),
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

        agent = konash.Agent(
            base_model="Qwen/Qwen3.5-7B",
            corpus="./my_docs",
            project="my-knowledge-agent",
        )
        agent.train()
        answer = agent.solve("What caused the 2008 financial crisis?")

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
        server).  Reads ``KONASH_API_BASE`` env var if not set.
    api_key : str | None
        API key for the inference server.  Reads ``KONASH_API_KEY`` env var
        if not set.  Defaults to ``"no-key"`` for local servers.
    checkpoint_dir : str | None
        Where to save/load LoRA checkpoints.  Defaults to
        ``.konash/<project>/checkpoints``.
    """

    def __init__(
        self,
        base_model: str,
        corpus: str | Path | Corpus,
        *,
        project: str = "default",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        chunk_size: int = 512,
        temperature: float = 0.7,
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

        # Checkpoints
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            ".konash", project, "checkpoints"
        )

        # Internal state (built lazily)
        self._llm_client: Optional[_OpenAILLMClient] = None
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

        1. Ingest the corpus and build the vector search index.
        2. Synthesize QA pairs from the corpus.
        3. Generate rollouts and filter by pass rate.
        4. Train with OAPL.
        5. Repeat for *iterations* rounds.

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
        # Step 1: Ingest corpus
        if not self.corpus.indexed:
            if verbose:
                print(f"Ingesting corpus from {self.corpus.path} ...")
            self.corpus.ingest()
            if verbose:
                print(f"  Indexed {self.corpus.num_documents} chunks.")

        # Set up synthesis components
        generate_fn = self._get_generate_fn()
        synthesizer = QuestionAnswerSynthesizer(
            vector_search_tool=self.corpus.vector_search,
            llm_fn=generate_fn,
        )
        rollout_gen = RolloutGenerator(
            search_tool=self.corpus.vector_search,
            llm_fn=generate_fn,
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

            # Step 2: Synthesize QA pairs
            if verbose:
                print("  Synthesizing training examples ...")
            documents = [d["text"] for d in self.corpus.documents]
            examples = pipeline.run_stage_one(
                documents=documents,
                num_examples=max_examples,
            )
            if verbose:
                print(f"  Generated {len(examples)} examples.")

            # Step 3: Generate rollouts + filter
            if verbose:
                print("  Generating rollouts ...")
            final_examples = pipeline.run_stage_two(
                examples=examples,
                num_rollouts=rollouts_per_example,
            )
            if verbose:
                print(f"  {len(final_examples)} examples after filtering.")

            if not final_examples and not pipeline.rollout_groups:
                if verbose:
                    print("  No training data — skipping training step.")
                continue

            # Step 4: Build dataset and train
            if verbose:
                print("  Training with OAPL ...")
            rollout_dicts = []
            for group in pipeline.rollout_groups:
                for rollout in group.rollouts:
                    rollout_dicts.append({
                        "prompt": group.prompt,
                        "rollout": rollout.steps,
                        "reward": 1.0 if rollout.passed else 0.0,
                    })
            if rollout_dicts:
                dataset = OfflineRolloutDataset.from_rollouts(rollout_dicts)
                epoch_stats = trainer.train_epoch(
                    dataset, learning_rate=learning_rate
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

        # Save checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        meta_path = os.path.join(self.checkpoint_dir, "training_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "base_model": self.base_model,
                "project": self.project,
                "iterations": self._iteration,
                "stats": stats,
            }, f, indent=2)

        self._trained = True
        if verbose:
            print(f"\nTraining complete. Metadata saved to {meta_path}")

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

        # Build environment with vector search as a tool
        def tool_executor(tool_call: Any) -> Dict[str, Any]:
            query_text = ""
            if isinstance(tool_call, dict):
                query_text = tool_call.get("query", "") or tool_call.get("input", "") or str(tool_call)
            else:
                query_text = str(tool_call)

            results = self.corpus.search(query_text, top_k=top_k)
            result_text = "\n\n".join(
                f"[{i+1}] (score: {r.get('score', 0):.3f}) {r.get('text', '')[:500]}"
                for i, r in enumerate(results)
            )
            return {"role": "tool", "content": result_text}

        env = Environment(
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
            env.reset(prompt=query)
            result = env.run_episode(agent, max_steps=max_steps)
            return result.get("final_answer") or ""

        # Parallel Thinking
        engine = ParallelThinkingEngine(
            agent=agent,
            aggregator=GenerativeAggregator(
                agent=agent,
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

    def _get_llm_client(self) -> _OpenAILLMClient:
        if self._llm_client is None:
            if self.api_base is None:
                raise RuntimeError(
                    "No LLM backend configured. Either:\n"
                    "  1. Set KONASH_API_BASE env var to a vLLM server URL\n"
                    "  2. Pass api_base='http://localhost:8000/v1' to Agent()\n"
                    "  3. Start a vLLM server: vllm serve {self.base_model}"
                )
            self._llm_client = _OpenAILLMClient(
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.base_model,
                temperature=self.temperature,
            )
        return self._llm_client

    def _get_generate_fn(self):
        """Return a callable that generates text from messages, or None if no LLM backend."""
        if self.api_base is None:
            return None
        client = self._get_llm_client()
        def generate_fn(messages, **kwargs):
            return client.generate(messages, **kwargs)
        return generate_fn

    def _make_agent(self, max_steps: int = 20) -> BaseAgent:
        """Build an internal BaseAgent wired to the LLM backend."""
        system_prompt = (
            "You are a knowledge agent. You have access to a search tool that "
            "retrieves relevant documents from a knowledge base. Use it to find "
            "evidence before answering. Search iteratively — refine your queries "
            "based on what you find. When you have enough evidence, provide a "
            "clear, well-supported answer."
        )
        return BaseAgent(
            llm_client=self._get_llm_client(),
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return (
            f"Agent(model={self.base_model!r}, project={self.project!r}, "
            f"corpus={self.corpus!r}, {status})"
        )
