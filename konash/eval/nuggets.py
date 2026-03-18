from __future__ import annotations

import ast as _ast
import json as _json
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


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
        # Use batch judging when the judge supports it (e.g. LLMNuggetJudge)
        # for efficiency — the paper's prompt evaluates all nuggets at once.
        if self.judge is not None and hasattr(self.judge, "judge_batch"):
            nugget_scores = self.judge.judge_batch(candidate, nuggets)
        else:
            nugget_scores = []
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


class LLMNuggetJudge:
    """LLM-backed nugget judge using the KARL paper's Nugget-Completeness
    Prompt (Appendix D.1, Figure 31).

    For each nugget (decompositional fact), the LLM assigns one of three
    labels: ``support`` (1.0), ``partial_support`` (0.5), ``not_support`` (0.0).

    The prompt evaluates all nuggets in a single LLM call for efficiency,
    matching the paper's batch evaluation approach.

    Parameters
    ----------
    llm_fn : callable
        ``(messages) -> {"role": "assistant", "content": "..."}``.
    question_context : str | None
        The question being evaluated. Included in the prompt for context.
    """

    LABEL_SCORES: Dict[str, float] = {
        "support": 1.0,
        "partial_support": 0.5,
        "not_support": 0.0,
    }

    def __init__(
        self,
        llm_fn: Callable,
        question_context: Optional[str] = None,
    ):
        self.llm_fn = llm_fn
        self.question_context = question_context
        self.last_raw_response: str = ""  # Raw judge output for debugging

    def judge(self, candidate: str, nugget: str) -> float:
        """Judge a single nugget against the candidate answer.

        Wraps :meth:`judge_batch` for compatibility with the
        ``NuggetScorer.judge_nugget`` interface.
        """
        scores = self.judge_batch(candidate, [nugget])
        return scores[0] if scores else 0.0

    def judge_batch(self, candidate: str, nuggets: List[str]) -> List[float]:
        """Judge all nuggets at once using the paper's Nugget-Completeness
        Prompt (Figure 31, Appendix D.1).

        Parameters
        ----------
        candidate : str
            The candidate answer to evaluate.
        nuggets : list[str]
            Decompositional facts to check.

        Returns
        -------
        list[float]
            Per-nugget scores: 1.0 (support), 0.5 (partial_support),
            0.0 (not_support).  Returns all zeros for empty/blank candidates.
        """
        if not nuggets:
            return []

        # Empty or blank candidate cannot support any nugget
        if not candidate or not candidate.strip():
            return [0.0] * len(nuggets)

        nuggets_formatted = "\n".join(
            f"- {n}" for n in nuggets
        )
        question = self.question_context or "(not provided)"

        # Exact prompt from KARL paper Appendix D.1, Figure 31
        prompt = (
            "Nugget-Completeness Prompt\n\n"
            "Your Role: You will evaluate whether an answer to a question "
            "(which can include a code snippet or documentation) sufficiently "
            "supports each decompositional fact.\n\n"
            "Process:\n"
            "1. Read the question and the answer.\n"
            f"2. Read each of the {len(nuggets)} decompositional facts "
            "carefully one by one.\n"
            "3. Based on the question and answer, judge whether the answer "
            "supports, partially supports, or does not support each "
            "decompositional fact. Read every fact and document pair carefully "
            "as you would when proofreading.\n\n"
            'It may be helpful to ask yourself: "Does the answer provide '
            "sufficient evidence required to support the decompositional "
            'fact?" Be sure to check all of the information in the answer.\n\n'
            "Label Definitions:\n"
            "- support: The answer fully captures and entails all necessary "
            "parts of the decompositional fact.\n"
            "- partial_support: The answer partially captures the "
            "decompositional fact, but does not fully capture all necessary "
            "parts.\n"
            "- not_support: The answer does not capture or does not provide "
            "information entailing the decompositional fact.\n\n"
            "Output Format: First provide a brief reasoning for each fact, "
            "then return the labels as a Python list of strings "
            "(List[str]), in the same order as the decompositional facts.\n"
            "Format:\n"
            "Reasoning: <one line per fact explaining your judgment>\n"
            'Labels: ["support", "not_support", "partial_support", ...]\n\n'
            "Input:\n"
            f"Question: {question}\n"
            f"Answer: {candidate}\n"
            f"Decompositional Facts: {nuggets_formatted}\n"
            "Labels:"
        )

        try:
            response = self.llm_fn([{"role": "user", "content": prompt}])
            content = (
                response.get("content", "")
                if isinstance(response, dict)
                else str(response)
            )
            self.last_prompt = prompt  # Store input for debugging
            self.last_raw_response = content  # Store output for debugging

            # Parse the Python list from the response
            match = re.search(r"\[.*?\]", content, re.DOTALL)
            if match:
                labels = _ast.literal_eval(match.group())
                scores: List[float] = []
                for label in labels:
                    label_str = str(label).strip().lower()
                    scores.append(self.LABEL_SCORES.get(label_str, 0.0))
                # Pad to match nuggets length if LLM returned fewer labels
                while len(scores) < len(nuggets):
                    scores.append(0.0)
                return scores[: len(nuggets)]
        except Exception:
            pass

        # Fallback: return 0.0 for all nuggets
        return [0.0] * len(nuggets)


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
