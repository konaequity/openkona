from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from konash.eval.metrics import EvaluationReport


class EvaluationRunner:
    """Orchestrates evaluation across benchmarks and inference modes.

    Supports three inference strategies:
    1. **Single rollout**: standard single-pass generation.
    2. **Parallel thinking**: N independent rollouts + aggregation (TTC).
    3. **Value-guided search**: BFS with a learned value model (TTC).

    After running evaluations, :meth:`summarize` produces an
    :class:`EvaluationReport` with quality, cost, latency, and
    in-/out-of-distribution breakdowns.
    """

    def __init__(
        self,
        agent=None,
        scorer=None,
        benchmarks: Optional[Sequence[str]] = None,
        *,
        parallel_engine=None,
        vgs_engine=None,
        training_tasks: Optional[set] = None,
        held_out_tasks: Optional[set] = None,
    ):
        """
        Args:
            agent: Agent instance with ``generate_rollout`` method.
            scorer: A :class:`NuggetScorer` (or compatible) for evaluation.
            benchmarks: List of benchmark names to evaluate on.
            parallel_engine: A :class:`ParallelThinkingEngine` instance.
            vgs_engine: A :class:`ValueGuidedSearchEngine` instance.
            training_tasks: Set of benchmark names that are in-distribution.
            held_out_tasks: Set of benchmark names that are out-of-distribution.
        """
        self.agent = agent
        self.scorer = scorer
        self.benchmarks = list(benchmarks) if benchmarks else []
        self.parallel_engine = parallel_engine
        self.vgs_engine = vgs_engine
        self.training_tasks = training_tasks or {"BrowseCompPlus", "TRECBiogen"}
        self.held_out_tasks = held_out_tasks or {
            "FinanceBench",
            "QAMPARI",
            "FreshStack",
            "PMBench",
        }
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Inference modes
    # ------------------------------------------------------------------

    def run_single_rollout(
        self,
        query: str,
        *,
        reference: Any = None,
        benchmark: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single rollout and optionally score the result.

        Returns a dict with ``answer``, ``score`` (if scorer + reference
        are available), ``latency_seconds``, and ``mode``.
        """
        t0 = time.monotonic()

        if self.agent is not None:
            rollout = self.agent.generate_rollout(query, context=context)
            answer = self._extract_answer(rollout)
        else:
            rollout = {"query": query, "final_answer": ""}
            answer = ""

        latency = time.monotonic() - t0

        result: Dict[str, Any] = {
            "mode": "single_rollout",
            "answer": answer,
            "latency_seconds": latency,
            "benchmark": benchmark,
        }

        if reference is not None and self.scorer is not None:
            score_result = self.scorer.score(answer, reference)
            result["score"] = score_result.get("score", 0.0)
            result["nugget_scores"] = score_result.get("nugget_scores", [])
        else:
            result["score"] = None

        self._results.append(result)
        return result

    def run_parallel_thinking(
        self,
        query: str,
        *,
        num_rollouts: Optional[int] = None,
        reference: Any = None,
        benchmark: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run parallel thinking (N rollouts + aggregation) and optionally score.

        Returns a dict with ``answer``, ``candidates``, ``score``,
        ``latency_seconds``, ``num_rollouts``, and ``mode``.
        """
        t0 = time.monotonic()

        if self.parallel_engine is not None:
            pt_result = self.parallel_engine.run(
                query, num_rollouts=num_rollouts, context=context
            )
            answer = pt_result.get("answer", "")
            candidates = pt_result.get("candidates", [])
            n = pt_result.get("num_rollouts", num_rollouts)
        else:
            # Fallback: run multiple single rollouts manually.
            n = num_rollouts or 1
            candidates = []
            for _ in range(n):
                sr = self.run_single_rollout(
                    query, benchmark=benchmark, context=context
                )
                candidates.append(sr.get("answer", ""))
                # Remove from results -- we'll record the PT result instead.
                if self._results and self._results[-1] is sr:
                    self._results.pop()
            answer = candidates[0] if candidates else ""

        latency = time.monotonic() - t0

        result: Dict[str, Any] = {
            "mode": "parallel_thinking",
            "answer": answer,
            "candidates": candidates,
            "latency_seconds": latency,
            "num_rollouts": n,
            "benchmark": benchmark,
        }

        if reference is not None and self.scorer is not None:
            score_result = self.scorer.score(answer, reference)
            result["score"] = score_result.get("score", 0.0)
        else:
            result["score"] = None

        self._results.append(result)
        return result

    def run_value_guided_search(
        self,
        query: str,
        *,
        parallel_searches: Optional[int] = None,
        reference: Any = None,
        benchmark: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run value-guided search and optionally score the result.

        Returns a dict with ``answer``, ``candidates``, ``scores``,
        ``score``, ``latency_seconds``, and ``mode``.
        """
        t0 = time.monotonic()

        if self.vgs_engine is not None:
            vgs_result = self.vgs_engine.run(
                query, parallel_searches=parallel_searches, context=context
            )
            answer = vgs_result.get("answer", "")
            candidates = vgs_result.get("candidates", [])
            tree_scores = vgs_result.get("scores", [])
        else:
            answer = ""
            candidates = []
            tree_scores = []

        latency = time.monotonic() - t0

        result: Dict[str, Any] = {
            "mode": "value_guided_search",
            "answer": answer,
            "candidates": candidates,
            "tree_scores": tree_scores,
            "latency_seconds": latency,
            "parallel_searches": parallel_searches,
            "benchmark": benchmark,
        }

        if reference is not None and self.scorer is not None:
            score_result = self.scorer.score(answer, reference)
            result["score"] = score_result.get("score", 0.0)
        else:
            result["score"] = None

        self._results.append(result)
        return result

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def summarize(self) -> EvaluationReport:
        """Produce an :class:`EvaluationReport` from all recorded results.

        The report includes:
        - **quality**: average score across all scored results.
        - **cost_per_query**: estimated as average num_rollouts (or 1 for
          single rollout mode).
        - **latency_seconds**: average latency across all results.
        - **in_distribution**: average score on training benchmarks.
        - **out_of_distribution**: average score on held-out benchmarks.
        """
        if not self._results:
            return EvaluationReport(
                quality=0.0,
                cost_per_query=0.0,
                latency_seconds=0.0,
                in_distribution=0.0,
                out_of_distribution=0.0,
            )

        # Gather scored results.
        scored = [r for r in self._results if r.get("score") is not None]
        all_scores = [r["score"] for r in scored] if scored else [0.0]
        quality = sum(all_scores) / len(all_scores)

        # Latency.
        latencies = [
            r.get("latency_seconds", 0.0) for r in self._results
        ]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Cost estimate: num_rollouts for PT, parallel_searches for VGS, 1 for single.
        costs: List[float] = []
        for r in self._results:
            mode = r.get("mode", "single_rollout")
            if mode == "parallel_thinking":
                costs.append(float(r.get("num_rollouts", 1)))
            elif mode == "value_guided_search":
                costs.append(float(r.get("parallel_searches", 1) or 1))
            else:
                costs.append(1.0)
        cost_per_query = sum(costs) / len(costs) if costs else 1.0

        # In- / out-of-distribution splits.
        id_scores: List[float] = []
        ood_scores: List[float] = []
        for r in scored:
            bench = r.get("benchmark")
            if bench in self.training_tasks:
                id_scores.append(r["score"])
            elif bench in self.held_out_tasks:
                ood_scores.append(r["score"])

        in_dist = (
            sum(id_scores) / len(id_scores) if id_scores else None
        )
        out_dist = (
            sum(ood_scores) / len(ood_scores) if ood_scores else None
        )

        return EvaluationReport(
            quality=quality,
            cost_per_query=cost_per_query,
            latency_seconds=avg_latency,
            in_distribution=in_dist,
            out_of_distribution=out_dist,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_answer(self, rollout: Any) -> str:
        """Extract the final answer string from a rollout."""
        if self.agent is not None and hasattr(self.agent, "extract_final_answer"):
            result = self.agent.extract_final_answer(rollout)
            if result is not None:
                return str(result)
        if isinstance(rollout, dict):
            ans = rollout.get("final_answer") or rollout.get("answer")
            if ans:
                return str(ans)
        if isinstance(rollout, str):
            return rollout
        return ""
