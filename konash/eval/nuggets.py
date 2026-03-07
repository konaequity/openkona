from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


class NuggetScorer:
    """Nugget-based evaluation scorer.

    The KARL paper uses nugget-based evaluation across all benchmarks.
    A *nugget* is an atomic unit of information that a correct answer should
    contain.  The scorer checks which nuggets from the reference answer(s)
    are covered by a candidate response.

    Different tasks have different nuggetisation strategies (see
    :class:`NuggetEvaluationPolicy`):

    - **single_nugget**: The entire reference is one nugget.
    - **entity_per_nugget**: Each entity in the reference list is a nugget
      (QAMPARI-style).
    - **fixed_nuggets**: Nuggets are pre-defined in the dataset.
    - **report_nuggets**: Nuggets are extracted from multi-reference reports.
    """

    def __init__(
        self,
        judge=None,
        *,
        policy: Optional[NuggetEvaluationPolicy] = None,
    ):
        """
        Args:
            judge: An LLM-based judge object with a ``judge(candidate, nugget)``
                method that returns a float in [0, 1].  When *None* the scorer
                falls back to substring matching.
            policy: A :class:`NuggetEvaluationPolicy` that controls
                nuggetisation and scoring behaviour.
        """
        self.judge = judge
        self.policy = policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        candidate: str,
        reference: Any,
        *,
        policy: Optional[NuggetEvaluationPolicy] = None,
        nuggets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Score a candidate answer against a reference.

        Steps:
        1. Nuggetise the reference (or use pre-supplied nuggets).
        2. Judge each nugget against the candidate.
        3. Aggregate per-nugget scores into a final score.

        Returns:
            A dict with ``score`` (float in [0, 1]), ``nugget_scores``
            (list of per-nugget scores), and ``nuggets`` (the nugget list).
        """
        pol = policy or self.policy

        # Step 1: obtain nuggets.
        if nuggets is None:
            nuggets = self.nuggetize_reference(reference, policy=pol)

        if not nuggets:
            return {"score": 0.0, "nugget_scores": [], "nuggets": []}

        # Step 2: judge each nugget.
        nugget_scores: List[float] = []
        for nugget in nuggets:
            ns = self.judge_nugget(candidate, nugget)
            nugget_scores.append(ns)

        # Step 3: aggregate.
        final_score = self.aggregate_scores(nugget_scores, policy=pol)

        return {
            "score": final_score,
            "nugget_scores": nugget_scores,
            "nuggets": nuggets,
        }

    def judge_nugget(self, candidate: str, nugget: str) -> float:
        """Judge whether *candidate* contains the information in *nugget*.

        When an LLM judge is available, delegates to it.  Otherwise uses
        case-insensitive substring matching with a soft partial-match
        heuristic based on word overlap.

        Returns a float in [0, 1].
        """
        if self.judge is not None:
            result = self.judge.judge(candidate, nugget)
            if isinstance(result, (int, float)):
                return float(min(max(result, 0.0), 1.0))
            return 1.0 if result else 0.0

        # Fallback: substring / word-overlap matching.
        return self._substring_judge(candidate, nugget)

    def aggregate_scores(
        self,
        nugget_scores: Sequence[float],
        *,
        policy: Optional[NuggetEvaluationPolicy] = None,
    ) -> float:
        """Aggregate per-nugget scores into a single scalar.

        The aggregation mode depends on the policy:
        - **single_nugget / entity_per_nugget**: arithmetic mean (nugget
          completion rate).
        - **fixed_nuggets / report_nuggets**: arithmetic mean.
        - Default: arithmetic mean.
        """
        if not nugget_scores:
            return 0.0
        return sum(nugget_scores) / len(nugget_scores)

    def nuggetize_reference(
        self,
        reference: Any,
        *,
        policy: Optional[NuggetEvaluationPolicy] = None,
    ) -> List[str]:
        """Break a reference answer into nuggets.

        The strategy is determined by the policy's ``mode``:
        - ``single_nugget``: return the entire reference as one nugget.
        - ``entity_per_nugget``: split by common delimiters (commas,
          semicolons, newlines) and treat each entity as a nugget.
        - ``fixed_nuggets``: expect the reference to already be a list.
        - ``report_nuggets``: split into sentence-level nuggets.
        """
        mode = (policy.mode if policy else None) or "single_nugget"

        if isinstance(reference, (list, tuple)):
            # Already a list of nuggets.
            return [str(n).strip() for n in reference if str(n).strip()]

        ref_str = str(reference).strip()
        if not ref_str:
            return []

        if mode == "single_nugget":
            return [ref_str]

        if mode == "entity_per_nugget":
            # Split on commas, semicolons, and newlines.
            entities = re.split(r"[,;\n]+", ref_str)
            return [e.strip() for e in entities if e.strip()]

        if mode == "fixed_nuggets":
            # Expect newline-delimited or numbered nuggets.
            lines = ref_str.split("\n")
            nuggets: List[str] = []
            for line in lines:
                cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
                if cleaned:
                    nuggets.append(cleaned)
            return nuggets if nuggets else [ref_str]

        if mode == "report_nuggets":
            # Sentence-level splitting for report-style references.
            sentences = re.split(r"(?<=[.!?])\s+", ref_str)
            return [s.strip() for s in sentences if s.strip()]

        # Default: treat as single nugget.
        return [ref_str]

    def consolidate_references(
        self,
        references: Sequence[Any],
        *,
        policy: Optional[NuggetEvaluationPolicy] = None,
    ) -> List[str]:
        """Consolidate multiple reference answers into a single nugget list.

        Used for tasks with ``multi_reference_consolidation`` (e.g.
        TRECBiogen) where multiple assessors provide reference nuggets and
        they must be deduplicated.
        """
        all_nuggets: List[str] = []
        for ref in references:
            nuggets = self.nuggetize_reference(ref, policy=policy)
            all_nuggets.extend(nuggets)

        # Deduplicate while preserving order.
        seen: set = set()
        consolidated: List[str] = []
        for nugget in all_nuggets:
            key = nugget.strip().lower()
            if key not in seen:
                seen.add(key)
                consolidated.append(nugget)

        return consolidated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _substring_judge(candidate: str, nugget: str) -> float:
        """Heuristic substring / word-overlap judge.

        Returns 1.0 for exact substring match (case-insensitive), otherwise
        computes Jaccard-like word overlap and returns a fractional score.
        """
        cand_lower = candidate.lower()
        nugget_lower = nugget.lower()

        # Exact substring match.
        if nugget_lower in cand_lower:
            return 1.0

        # Word-level overlap.
        nugget_words = set(re.findall(r"\w+", nugget_lower))
        if not nugget_words:
            return 0.0
        cand_words = set(re.findall(r"\w+", cand_lower))
        overlap = nugget_words & cand_words
        recall = len(overlap) / len(nugget_words)

        # Threshold: require at least 60% word recall to count at all.
        if recall < 0.6:
            return 0.0
        return recall


class NuggetEvaluationPolicy:
    """Per-task policy that controls how nugget-based evaluation behaves.

    Attributes:
        task_name: Name of the benchmark task.
        mode: Nuggetisation mode -- one of ``"single_nugget"``,
            ``"entity_per_nugget"``, ``"fixed_nuggets"``, or
            ``"report_nuggets"``.
        reference_handling: How references are provided -- either
            ``"single_reference"`` or ``"multi_reference_consolidation"``.
        requires_task_prompt: Whether the task provides a specialised prompt
            that should be included during evaluation.
    """

    task_name = None
    mode = None
    reference_handling = None
    requires_task_prompt = False

    def __init__(
        self,
        task_name=None,
        mode=None,
        reference_handling=None,
        requires_task_prompt=False,
    ):
        self.task_name = task_name
        self.mode = mode
        self.reference_handling = reference_handling
        self.requires_task_prompt = requires_task_prompt

    def __repr__(self) -> str:
        return (
            f"NuggetEvaluationPolicy(task_name={self.task_name!r}, "
            f"mode={self.mode!r}, "
            f"reference_handling={self.reference_handling!r}, "
            f"requires_task_prompt={self.requires_task_prompt!r})"
        )


class NuggetPolicyRegistry:
    """Registry of per-task nugget evaluation policies.

    Maps benchmark names to :class:`NuggetEvaluationPolicy` instances
    with the exact settings described in the KARL paper.
    """

    policies = {
        "QAMPARI": NuggetEvaluationPolicy(
            task_name="QAMPARI",
            mode="entity_per_nugget",
            reference_handling="single_reference",
        ),
        "FreshStack": NuggetEvaluationPolicy(
            task_name="FreshStack",
            mode="fixed_nuggets",
            reference_handling="single_reference",
            requires_task_prompt=True,
        ),
        "PMBench": NuggetEvaluationPolicy(
            task_name="PMBench",
            mode="fixed_nuggets",
            reference_handling="single_reference",
            requires_task_prompt=True,
        ),
        "TRECBiogen": NuggetEvaluationPolicy(
            task_name="TRECBiogen",
            mode="report_nuggets",
            reference_handling="multi_reference_consolidation",
            requires_task_prompt=True,
        ),
        "BrowseCompPlus": NuggetEvaluationPolicy(
            task_name="BrowseCompPlus",
            mode="single_nugget",
            reference_handling="single_reference",
        ),
        "FinanceBench": NuggetEvaluationPolicy(
            task_name="FinanceBench",
            mode="single_nugget",
            reference_handling="single_reference",
        ),
    }

    @classmethod
    def get(cls, task_name: str) -> NuggetEvaluationPolicy:
        """Look up the policy for a given task name."""
        if task_name not in cls.policies:
            raise KeyError(
                f"No nugget evaluation policy registered for {task_name!r}. "
                f"Available: {sorted(cls.policies.keys())}"
            )
        return cls.policies[task_name]

    @classmethod
    def list_policies(cls) -> List[str]:
        """Return sorted list of registered task names."""
        return sorted(cls.policies.keys())
