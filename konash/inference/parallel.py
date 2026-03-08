from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence


class ParallelThinkingEngine:
    """Parallel Thinking (Test-Time Compute) engine.

    Generates N independent rollouts for a given query and then aggregates
    their final answers into a single, higher-quality response.  This is the
    primary TTC strategy described in the KARL paper (Section 5).

    Attributes:
        num_rollouts: Number of independent rollouts to generate per query.
            ``None`` means the caller must supply the count at invocation time.
    """

    num_rollouts = None

    def __init__(
        self,
        agent=None,
        aggregator=None,
        num_rollouts: Optional[int] = None,
    ):
        """
        Args:
            agent: An object that exposes ``generate_rollout(query, **kw)``
                and ``extract_final_answer(rollout)`` methods.  When *None*
                the engine can still be instantiated (e.g. for testing), but
                ``run`` / ``generate_parallel_rollouts`` will raise.
            aggregator: An object that exposes ``aggregate(candidates, query)``
                (e.g. :class:`GenerativeAggregator`).  When *None* the engine
                falls back to majority-vote aggregation.
            num_rollouts: Override the class-level default.
        """
        if num_rollouts is not None:
            self.num_rollouts = num_rollouts
        self.agent = agent
        self.aggregator = aggregator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        *,
        num_rollouts: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full parallel-thinking pipeline for a single query.

        1. Generate *N* independent rollouts via :meth:`generate_parallel_rollouts`.
        2. Extract an answer string from each rollout via :meth:`extract_answers`.
        3. Aggregate the candidate answers via :meth:`aggregate`.

        Returns a dict with keys ``answer``, ``candidates``, ``rollouts``,
        and ``num_rollouts``.
        """
        n = num_rollouts or self.num_rollouts
        if n is None:
            raise ValueError(
                "num_rollouts must be set either at construction time or "
                "passed to run()"
            )

        rollouts = self.generate_parallel_rollouts(
            query, num_rollouts=n, context=context
        )
        candidates = self.extract_answers(rollouts)
        answer = self.aggregate(candidates, query=query, context=context)

        return {
            "answer": answer,
            "candidates": candidates,
            "rollouts": rollouts,
            "num_rollouts": n,
        }

    def generate_parallel_rollouts(
        self,
        query: str,
        *,
        num_rollouts: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run *N* independent rollouts for *query*.

        Each rollout is produced by calling ``self.agent.generate_rollout``.
        If no agent is configured a lightweight stub rollout is returned so
        that downstream methods still receive well-typed data.
        """
        n = num_rollouts or self.num_rollouts
        if n is None:
            raise ValueError("num_rollouts is required")

        rollouts: List[Dict[str, Any]] = []
        for i in range(n):
            if self.agent is not None:
                rollout = self.agent.generate_rollout(
                    query, rollout_index=i, context=context
                )
            else:
                # Stub rollout when no agent is wired up (useful for unit tests
                # and dry-run evaluations).
                rollout = {
                    "query": query,
                    "rollout_index": i,
                    "steps": [],
                    "final_answer": None,
                }
            rollouts.append(rollout)
        return rollouts

    def extract_answers(
        self, rollouts: Sequence[Dict[str, Any]]
    ) -> List[str]:
        """Extract the final answer string from each rollout.

        Uses the agent's ``extract_final_answer`` helper when available;
        otherwise falls back to reading the ``"final_answer"`` key from the
        rollout dict, and finally tries pulling the last assistant message.
        """
        answers: List[str] = []
        for rollout in rollouts:
            answer = self._extract_single_answer(rollout)
            answers.append(answer)
        return answers

    def aggregate(
        self,
        candidates: Sequence[str],
        *,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Combine candidate answers into one final answer.

        If an :class:`GenerativeAggregator` (or compatible object) was
        provided at init time, delegates to it.  Otherwise performs a
        simple majority-vote: the most frequently occurring answer string
        wins.  Ties are broken by first occurrence.
        """
        if not candidates:
            return ""

        if self.aggregator is not None:
            return self.aggregator.aggregate(
                candidates, query=query, context=context
            )

        # Fallback: majority vote
        return self._majority_vote(candidates)

    def run_aggregation_rollout(
        self,
        candidates: Sequence[str],
        query: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run one additional *aggregation rollout*.

        This gives the LLM a chance to see all candidate answers and produce
        a synthesis that may include tool calls (retrieval, calculation, etc.)
        to verify or refine the answer.

        If no agent is available, falls back to :meth:`aggregate`.
        """
        if self.agent is not None:
            # Build an aggregation prompt that includes all candidates.
            agg_prompt = self._build_aggregation_query(candidates, query)
            rollout = self.agent.generate_rollout(
                agg_prompt, context=context
            )
            return self._extract_single_answer(rollout)

        # No agent -- fall back to normal aggregation.
        return self.aggregate(candidates, query=query, context=context)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _majority_vote(candidates: Sequence[str]) -> str:
        """Return the most common answer string, ties broken by first appearance."""
        if not candidates:
            return ""
        # Normalise whitespace for comparison, but return original form.
        normalised: Dict[str, str] = {}
        for c in candidates:
            key = " ".join(c.split()).strip().lower()
            if key not in normalised:
                normalised[key] = c  # keep first occurrence's form
        counts = Counter(" ".join(c.split()).strip().lower() for c in candidates)
        winner_key, _ = counts.most_common(1)[0]
        return normalised[winner_key]

    def _extract_single_answer(self, rollout: Dict[str, Any]) -> str:
        """Best-effort answer extraction from a single rollout dict."""
        # 1. Delegate to agent if possible.
        if self.agent is not None and hasattr(self.agent, "extract_final_answer"):
            history = rollout
            if isinstance(rollout, dict):
                history = rollout.get("history") or rollout.get("messages") or rollout
            result = self.agent.extract_final_answer(history)
            if result is not None:
                return str(result)

        # 2. Explicit key in the rollout dict.
        if isinstance(rollout.get("final_answer"), str) and rollout["final_answer"]:
            return rollout["final_answer"]

        # 3. Last assistant message in a messages/steps list.
        for key in ("steps", "messages"):
            items = rollout.get(key, [])
            if items:
                last = items[-1]
                if isinstance(last, dict):
                    text = last.get("content") or last.get("text") or ""
                    if text:
                        return str(text)
                elif isinstance(last, str):
                    return last

        return ""

    @staticmethod
    def _build_aggregation_query(
        candidates: Sequence[str], original_query: str
    ) -> str:
        """Format an aggregation prompt for the LLM."""
        parts = [
            "Below are multiple candidate answers to the same question.",
            "Synthesize them into one high-quality final answer.\n",
            f"Question: {original_query}\n",
        ]
        for idx, ans in enumerate(candidates, 1):
            parts.append(f"Candidate {idx}: {ans}")
        parts.append(
            "\nProvide a single, well-reasoned final answer that draws on "
            "the best information from all candidates."
        )
        return "\n".join(parts)
