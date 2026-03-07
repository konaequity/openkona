from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


class GenerativeAggregator:
    """Generative aggregation of candidate answers.

    Unlike simple majority voting, the generative aggregator builds a prompt
    containing all candidate answers and asks an LLM to synthesize a single
    high-quality response.  Because the aggregation rollout itself may include
    tool calls (retrieval, calculation, etc.), this strategy is strictly more
    expressive than counting-based approaches.

    Attributes:
        tool_access_enabled: When *True* the aggregation rollout is allowed
            to invoke tools (e.g. retrieval, calculator) during synthesis.
    """

    tool_access_enabled = True

    def __init__(
        self,
        agent=None,
        *,
        tool_access_enabled: bool = True,
        aggregation_mode: str = "generative",
    ):
        """
        Args:
            agent: An object exposing ``generate_rollout(prompt, **kw)`` and
                optionally ``extract_final_answer(rollout)``.  When *None* the
                aggregator falls back to majority vote.
            tool_access_enabled: Whether the agent is permitted to call tools
                during the aggregation rollout.
            aggregation_mode: One of ``"generative"``, ``"majority_vote"``,
                ``"weighted_majority_vote"``, or ``"best_of_n"``.
        """
        self.agent = agent
        self.tool_access_enabled = tool_access_enabled
        self.aggregation_mode = aggregation_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        candidates: Sequence[str],
        *,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> str:
        """Synthesize a final answer from *candidates*.

        Dispatches to the appropriate strategy based on ``aggregation_mode``:
        - ``"generative"``: build a prompt and run an LLM aggregation rollout.
        - ``"majority_vote"``: simple frequency-based selection.
        - ``"weighted_majority_vote"``: weight-aware frequency selection.
        - ``"best_of_n"``: pick the candidate with the highest weight.

        If no agent is available for generative mode the method silently
        degrades to majority vote.
        """
        if not candidates:
            return ""

        mode = self.aggregation_mode

        if mode == "generative" and self.agent is not None:
            return self._generative_aggregate(candidates, query, context)
        elif mode == "weighted_majority_vote" and weights is not None:
            return self._weighted_majority_vote(candidates, weights)
        elif mode == "best_of_n" and weights is not None:
            return self._best_of_n(candidates, weights)
        else:
            # Default / fallback: plain majority vote.
            return self._majority_vote(candidates)

    def build_aggregation_prompt(
        self,
        candidates: Sequence[str],
        query: Optional[str] = None,
        *,
        task_prompt: Optional[str] = None,
    ) -> str:
        """Format candidate answers into a prompt for the aggregation LLM.

        The returned string is a self-contained prompt that instructs the
        model to read all candidates and produce one well-reasoned final
        answer.
        """
        sections: List[str] = []

        # System-level framing
        sections.append(
            "You are an expert answer aggregator. You will be given multiple "
            "candidate answers to the same question. Your job is to synthesize "
            "them into one high-quality, accurate, and complete final answer."
        )

        if task_prompt:
            sections.append(f"\nTask instructions:\n{task_prompt}")

        if query:
            sections.append(f"\nQuestion:\n{query}")

        sections.append("\nCandidate answers:")
        for idx, candidate in enumerate(candidates, 1):
            sections.append(f"\n--- Candidate {idx} ---\n{candidate}")

        sections.append(
            "\n--- End of candidates ---\n\n"
            "Instructions:\n"
            "1. Identify the information that appears consistently across "
            "candidates.\n"
            "2. Resolve any conflicts by preferring the most detailed and "
            "well-supported claims.\n"
            "3. If candidates provide complementary information, merge them.\n"
            "4. Output a single, concise final answer."
        )

        if self.tool_access_enabled:
            sections.append(
                "\nYou may use available tools (retrieval, calculation, etc.) "
                "to verify or refine your answer if needed."
            )

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Aggregation strategies
    # ------------------------------------------------------------------

    def _generative_aggregate(
        self,
        candidates: Sequence[str],
        query: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Run an LLM rollout to aggregate candidates."""
        prompt = self.build_aggregation_prompt(candidates, query)
        rollout = self.agent.generate_rollout(
            prompt,
            context=context,
            tool_access=self.tool_access_enabled,
        )
        # Extract answer from rollout
        if hasattr(self.agent, "extract_final_answer"):
            result = self.agent.extract_final_answer(rollout)
            if result is not None:
                return str(result)
        # Fallback: read from rollout dict
        if isinstance(rollout, dict):
            ans = rollout.get("final_answer") or rollout.get("answer")
            if ans:
                return str(ans)
        if isinstance(rollout, str):
            return rollout
        return self._majority_vote(candidates)

    @staticmethod
    def _majority_vote(candidates: Sequence[str]) -> str:
        """Simple majority vote -- most common answer string wins."""
        if not candidates:
            return ""
        normalised: Dict[str, str] = {}
        for c in candidates:
            key = " ".join(c.split()).strip().lower()
            if key not in normalised:
                normalised[key] = c
        counts = Counter(
            " ".join(c.split()).strip().lower() for c in candidates
        )
        winner_key, _ = counts.most_common(1)[0]
        return normalised[winner_key]

    @staticmethod
    def _weighted_majority_vote(
        candidates: Sequence[str], weights: Sequence[float]
    ) -> str:
        """Majority vote weighted by value-model scores."""
        if not candidates:
            return ""
        weighted: Dict[str, float] = {}
        first_form: Dict[str, str] = {}
        for cand, w in zip(candidates, weights):
            key = " ".join(cand.split()).strip().lower()
            weighted[key] = weighted.get(key, 0.0) + w
            if key not in first_form:
                first_form[key] = cand
        best_key = max(weighted, key=weighted.get)  # type: ignore[arg-type]
        return first_form[best_key]

    @staticmethod
    def _best_of_n(
        candidates: Sequence[str], weights: Sequence[float]
    ) -> str:
        """Return the candidate with the highest weight."""
        if not candidates:
            return ""
        best_idx = max(range(len(weights)), key=lambda i: weights[i])
        return candidates[best_idx]
