from __future__ import annotations

import json as _json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PassRateFilter:
    """Filters rollout groups by their pass rate, rejecting groups where
    the solver succeeds too often (trivially easy) or too rarely
    (potentially unanswerable or ambiguous).

    Supports per-task, per-iteration adaptive thresholds matching the KARL
    paper (Section 7.2):
    - Multi-task iterations use binarization thresholds 0.6 / 0.7
    - TREC-Biogen expert iterations use 0.6 / 0.75 / 0.9
    - BrowseComp-Plus uses binary pass/fail filtering

    Attributes
    ----------
    min_pass_rate : float or None
        Minimum pass rate to keep a group (inclusive).
    max_pass_rate : float or None
        Maximum pass rate to keep a group (inclusive).
    task_name : str or None
        Task identifier for adaptive threshold lookup.
    iteration : int
        Current training iteration (0-indexed) for adaptive thresholds.
    """

    # Paper-derived per-task, per-iteration thresholds.
    # Keys: (task_name, iteration) -> (min_pass_rate, max_pass_rate)
    # From Section 7.2.1 and 7.2.2:
    #   Multi-task: binarization at 0.6 (iter 0) and 0.7 (iter 1)
    #   TREC-Biogen expert: 0.6, 0.75, 0.9 across 3 iterations
    #   BrowseComp-Plus: binary scores, so pass-rate filter keeps partial
    ADAPTIVE_THRESHOLDS: Dict[tuple, tuple] = {
        # Multi-task training
        ("TRECBiogen", 0): (0.1, 0.9),
        ("TRECBiogen", 1): (0.1, 0.9),
        ("BrowseCompPlus", 0): (0.1, 0.9),
        ("BrowseCompPlus", 1): (0.1, 0.9),
        # TREC-Biogen expert (single-task), increasing difficulty per iteration
        ("TRECBiogen_expert", 0): (0.1, 0.9),
        ("TRECBiogen_expert", 1): (0.1, 0.9),
        ("TRECBiogen_expert", 2): (0.1, 0.9),
    }

    # Paper binarization thresholds: continuous scores are binarized at this
    # cutoff before computing pass rates (Section 7.2.1).
    BINARIZATION_THRESHOLDS: Dict[tuple, float] = {
        # Multi-task
        ("TRECBiogen", 0): 0.6,
        ("TRECBiogen", 1): 0.7,
        # TREC-Biogen expert
        ("TRECBiogen_expert", 0): 0.6,
        ("TRECBiogen_expert", 1): 0.75,
        ("TRECBiogen_expert", 2): 0.9,
        # BrowseComp-Plus uses binary scores natively (0 or 1)
        ("BrowseCompPlus", 0): 0.5,
        ("BrowseCompPlus", 1): 0.5,
    }

    min_pass_rate = None
    max_pass_rate = None

    def __init__(
        self,
        min_pass_rate: Optional[float] = None,
        max_pass_rate: Optional[float] = None,
        task_name: Optional[str] = None,
        iteration: int = 0,
    ):
        self.task_name = task_name
        self.iteration = iteration

        # Use adaptive thresholds if task_name is specified and no explicit
        # overrides are given.
        if task_name and min_pass_rate is None and max_pass_rate is None:
            key = (task_name, iteration)
            if key in self.ADAPTIVE_THRESHOLDS:
                min_pass_rate, max_pass_rate = self.ADAPTIVE_THRESHOLDS[key]

        self.min_pass_rate = min_pass_rate
        self.max_pass_rate = max_pass_rate

    @property
    def binarization_threshold(self) -> float:
        """The score threshold at which continuous rewards are binarized.

        For nugget-based evaluation (TREC-Biogen), scores in [0, 1] are
        binarized before computing pass rates.  The threshold increases
        across iterations to focus on harder examples.
        """
        if self.task_name:
            key = (self.task_name, self.iteration)
            if key in self.BINARIZATION_THRESHOLDS:
                return self.BINARIZATION_THRESHOLDS[key]
        return 0.5  # default binary threshold

    def binarize_scores(self, scores: List[float]) -> List[float]:
        """Binarize continuous scores using the current threshold.

        Parameters
        ----------
        scores : list[float]
            Continuous scores in [0, 1].

        Returns
        -------
        list[float]
            Binary scores (0.0 or 1.0).
        """
        thresh = self.binarization_threshold
        return [1.0 if s >= thresh else 0.0 for s in scores]

    def apply(self, groups: List[Any]) -> List[Any]:
        """Filter a list of rollout groups by their pass rate.

        Each group must expose a ``pass_rate`` attribute or property
        (float in [0, 1]).

        Parameters
        ----------
        groups : list
            Rollout groups to filter. Each must have a ``.pass_rate`` attribute.

        Returns
        -------
        list
            Groups whose pass rate falls within [min_pass_rate, max_pass_rate].
        """
        result: List[Any] = []
        for group in groups:
            rate = self._get_pass_rate(group)
            if rate is None:
                continue

            if self.min_pass_rate is not None and rate < self.min_pass_rate:
                continue
            if self.max_pass_rate is not None and rate > self.max_pass_rate:
                continue

            result.append(group)
        return result

    @staticmethod
    def _get_pass_rate(group: Any) -> Optional[float]:
        """Extract the pass rate from a group object or dict."""
        if hasattr(group, "pass_rate"):
            return float(group.pass_rate)
        if isinstance(group, dict) and "pass_rate" in group:
            return float(group["pass_rate"])
        return None


class QualityFilter:
    """Quality filter matching the KARL paper (Sections 7.2.1-7.2.2).

    Uses the exact prompts from Figures 35 (BrowseComp-Plus) and 36
    (TREC-Biogen) to evaluate whether synthetic QA pairs are valid for
    training.  The paper's approach is a single unified check that combines
    ambiguity and reference accuracy into one LLM call, using solver rollout
    attempts as evidence.

    When ``task_name`` is set and ``rollout_attempts`` are provided to
    :meth:`apply`, the paper's structured prompts are used.  Otherwise
    falls back to heuristic checks.

    Parameters
    ----------
    judge_fn : callable | None
        LLM function ``(messages) -> {"role": ..., "content": ...}``.
    judge_model : str | None
        Model name (informational / for logging).
    task_name : str | None
        Task identifier (``"BrowseCompPlus"`` or ``"TRECBiogen"``).
        Determines which paper prompt to use.
    checks_ambiguity : bool
        Whether to run the ambiguity check (heuristic fallback path).
    checks_reference_accuracy : bool
        Whether to run the reference-accuracy check (heuristic fallback path).
    """

    def __init__(
        self,
        judge_fn: Any = None,
        judge_model: Optional[str] = None,
        task_name: Optional[str] = None,
        checks_ambiguity: bool = True,
        checks_reference_accuracy: bool = True,
    ):
        self.judge_fn = judge_fn
        self.judge_model = judge_model
        self.task_name = task_name
        self.checks_ambiguity = checks_ambiguity
        self.checks_reference_accuracy = checks_reference_accuracy

    def apply(
        self,
        examples: List[Any],
        reference_documents: Optional[List[str]] = None,
        rollout_attempts: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[Any]:
        """Filter examples by quality, removing ambiguous or inaccurate ones.

        Parameters
        ----------
        examples : list
            QA examples to evaluate. Each should have ``question``, ``answer``,
            and optionally ``nuggets`` attributes.
        reference_documents : list[str] | None
            Source documents for reference-accuracy checking (heuristic path).
        rollout_attempts : list[list[dict]] | None
            Per-example lists of solver rollout attempts.  Each attempt dict
            should have ``answer`` (str) and ``score`` (float).  When provided
            alongside ``judge_fn`` and ``task_name``, the paper's structured
            prompts (Figures 35-36) are used.

        Returns
        -------
        list
            Examples that pass all enabled quality checks.
        """
        passed: List[Any] = []
        for idx, example in enumerate(examples):
            question = getattr(example, "question", None) or (
                example.get("question") if isinstance(example, dict) else None
            )
            answer = getattr(example, "answer", None) or (
                example.get("answer") if isinstance(example, dict) else None
            )

            if question is None or answer is None:
                continue

            # Paper path: unified quality check with solver attempts
            attempts = (
                rollout_attempts[idx]
                if rollout_attempts is not None and idx < len(rollout_attempts)
                else None
            )
            if self.judge_fn is not None and attempts:
                result = self._llm_judge_quality(question, answer, example, attempts)
                if result is not None:
                    if result.get("is_valid", True):
                        passed.append(example)
                    continue

            # Fallback: separate heuristic checks
            if self.checks_ambiguity:
                ambiguity_result = self.judge_ambiguity(question, answer)
                if ambiguity_result.get("is_ambiguous", False):
                    continue

            if self.checks_reference_accuracy and reference_documents:
                accuracy_result = self.judge_reference_accuracy(
                    question, answer, reference_documents
                )
                if not accuracy_result.get("is_accurate", True):
                    continue

            passed.append(example)
        return passed

    # -- Paper-faithful unified quality judge (Figures 35-36) ---------------

    def _llm_judge_quality(
        self,
        question: str,
        answer: str,
        example: Any,
        attempts: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Unified quality check using the KARL paper's exact prompts.

        Prompt selection logic:

        1. If ``task_name`` explicitly matches a KARL task, use that task's
           prompt (BrowseComp-Plus → Figure 35, TREC-Biogen → Figure 36).
        2. Otherwise, inspect the example: if it has ``nuggets``, use the
           TREC prompt (Figure 36) since nugget-based evaluation applies.
        3. Default to the BrowseComp-Plus prompt (Figure 35) — it handles
           the common case of a single ground-truth answer with binary
           correct/incorrect solver attempts.

        Returns
        -------
        dict or None
            ``{"is_valid": bool, "reason": str}`` on success, ``None`` on
            failure (caller falls back to heuristics).
        """
        task = (self.task_name or "").lower()

        # Explicit task match
        if "browsecomp" in task:
            return self._judge_browsecomp(question, answer, attempts)
        if "trec" in task or "biogen" in task:
            return self._judge_trec(question, answer, example, attempts)

        # Infer from example data: nuggets → TREC, otherwise BrowseComp
        nuggets = getattr(example, "nuggets", None) or (
            example.get("nuggets") if isinstance(example, dict) else None
        )
        if nuggets:
            return self._judge_trec(question, answer, example, attempts)

        return self._judge_browsecomp(question, answer, attempts)

    def _judge_browsecomp(
        self,
        question: str,
        ground_truth: str,
        attempts: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """BrowseComp-Plus quality filter (Figure 35).

        Uses binary correct/incorrect labels for each solver attempt.
        """
        # Format attempts: truncate each to 1000 chars, label correct/incorrect
        attempt_lines: List[str] = []
        for i, att in enumerate(attempts, 1):
            att_answer = str(att.get("answer", ""))[:1000]
            score = att.get("score", 0.0)
            label = "Correct" if score >= 0.5 else "Incorrect"
            attempt_lines.append(f"Attempt {i} ({label}): {att_answer}")
        attempts_text = "\n".join(attempt_lines)

        template = self._get_prompt_template("figure_35_browsecomp_quality_filter")
        if template is None:
            logger.warning(
                "Prompt 'figure_35_browsecomp_quality_filter' not found in "
                "PromptRegistry; falling back to heuristic quality check."
            )
            return None
        prompt = template.format(
            question=question,
            ground_truth=ground_truth,
            attempts=attempts_text,
        )
        return self._call_quality_judge(prompt)

    def _judge_trec(
        self,
        question: str,
        answer: str,
        example: Any,
        attempts: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """TREC-Biogen quality filter (Figure 36).

        Uses nugget coverage percentages and computes avg/max/min stats.
        """
        # Extract nuggets from example
        nuggets = getattr(example, "nuggets", None) or (
            example.get("nuggets") if isinstance(example, dict) else None
        )
        if nuggets is None:
            nuggets = answer  # Fall back to answer as nugget source

        nuggets_text = nuggets if isinstance(nuggets, str) else "\n".join(str(n) for n in nuggets)

        # Format attempts with nugget coverage scores
        attempt_lines: List[str] = []
        scores: List[float] = []
        for i, att in enumerate(attempts, 1):
            att_answer = str(att.get("answer", ""))[:1000]
            score = float(att.get("score", 0.0))
            scores.append(score)
            pct = int(score * 100)
            attempt_lines.append(f"Attempt {i} (Coverage: {pct}%): {att_answer}")
        attempts_text = "\n".join(attempt_lines)

        # Compute stats
        avg_score = int((sum(scores) / len(scores)) * 100) if scores else 0
        max_score = int(max(scores) * 100) if scores else 0
        min_score = int(min(scores) * 100) if scores else 0

        template = self._get_prompt_template("figure_36_trec_quality_filter")
        if template is None:
            logger.warning(
                "Prompt 'figure_36_trec_quality_filter' not found in "
                "PromptRegistry; falling back to heuristic quality check."
            )
            return None
        prompt = template.format(
            question=question,
            nuggets=nuggets_text,
            attempts=attempts_text,
            avg=avg_score,
            max=max_score,
            min=min_score,
        )
        return self._call_quality_judge(prompt)

    @staticmethod
    def _get_prompt_template(name: str) -> Optional[str]:
        """Safely retrieve a prompt template from the PromptRegistry.

        Returns the template string, or None if the registry is unavailable
        or the prompt is not registered.
        """
        try:
            from konash.prompts.registry import PromptRegistry
            entry = PromptRegistry.get(name)
            if entry is not None and hasattr(entry, "template"):
                return entry.template
        except (ImportError, KeyError, AttributeError) as exc:
            logger.debug("PromptRegistry lookup failed for %r: %s", name, exc)
        return None

    def _call_quality_judge(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Send a quality-filter prompt to the LLM and parse the response.

        Expects ``<valid>yes/no</valid>`` in the response (paper format).
        """
        try:
            resp = self.judge_fn([{"role": "user", "content": prompt}])
            text = resp.get("content", "") if isinstance(resp, dict) else str(resp)

            # Parse <valid>yes/no</valid>
            valid_match = re.search(r"<valid>\s*(yes|no)\s*</valid>", text, re.IGNORECASE)
            if valid_match:
                is_valid = valid_match.group(1).strip().lower() == "yes"
            else:
                # Fallback: look for "yes" or "no" at end of response
                is_valid = "yes" in text.lower().split()[-3:]

            # Extract reasoning if present
            reason = ""
            reason_match = re.search(
                r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE
            )
            if reason_match:
                reason = reason_match.group(1).strip()

            return {"is_valid": is_valid, "reason": reason}
        except Exception:
            return None

    # -- Legacy separate checks (heuristic fallback) ------------------------

    def judge_ambiguity(
        self, question: str, answer: str
    ) -> Dict[str, Any]:
        """Evaluate whether a question is ambiguous.

        When ``judge_fn`` is set and no ``task_name`` is configured, calls
        the LLM with a generic prompt.  Falls back to heuristics if no LLM
        is available or the call fails.

        Returns
        -------
        dict
            Keys: ``is_ambiguous`` (bool), ``confidence`` (float), ``reason`` (str).
        """
        if self.judge_fn is not None and not self.task_name:
            result = self._llm_judge_ambiguity(question, answer)
            if result is not None:
                return result

        return self._heuristic_judge_ambiguity(question, answer)

    def _llm_judge_ambiguity(
        self, question: str, answer: str
    ) -> Optional[Dict[str, Any]]:
        """LLM-backed ambiguity check (generic fallback when no task_name)."""
        prompt = (
            "You are evaluating a synthetic question-answer pair for ambiguity.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "An ambiguous question has multiple valid interpretations that "
            "would lead to meaningfully different answers. Minor phrasing "
            "variations do NOT count as ambiguity.\n\n"
            'Respond with JSON: {"is_ambiguous": true/false, "reason": "..."}'
        )
        try:
            resp = self.judge_fn([{"role": "user", "content": prompt}])
            text = resp.get("content", "") if isinstance(resp, dict) else str(resp)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                parsed = _json.loads(match.group())
                return {
                    "is_ambiguous": parsed.get("is_ambiguous", False),
                    "confidence": 0.9,
                    "reason": parsed.get("reason", ""),
                }
        except Exception:
            pass
        return None

    def _heuristic_judge_ambiguity(
        self, question: str, answer: str
    ) -> Dict[str, Any]:
        """Heuristic fallback for ambiguity detection."""
        ambiguity_markers = [
            r"\bor\b", r"\beither\b", r"\bcould be\b",
            r"\bmight\b", r"\bpossibly\b", r"\bwhich one\b",
        ]
        question_lower = question.lower()
        marker_count = sum(
            1 for pat in ambiguity_markers if re.search(pat, question_lower)
        )
        word_count = len(question.split())
        too_short = word_count < 4
        has_interrogative = any(
            w in question_lower
            for w in ["what", "who", "when", "where", "why", "how", "which",
                       "is", "are", "does", "do", "did", "can"]
        )

        is_ambiguous = (marker_count >= 2) or too_short or not has_interrogative
        confidence = min(1.0, 0.3 + marker_count * 0.2 + (0.3 if too_short else 0.0))

        reason = ""
        if is_ambiguous:
            reasons = []
            if marker_count >= 2:
                reasons.append("multiple ambiguity markers detected")
            if too_short:
                reasons.append("question too short for specificity")
            if not has_interrogative:
                reasons.append("no clear interrogative focus")
            reason = "; ".join(reasons)

        return {
            "is_ambiguous": is_ambiguous,
            "confidence": confidence if is_ambiguous else 1.0 - confidence,
            "reason": reason,
        }

    def judge_reference_accuracy(
        self,
        question: str,
        answer: str,
        reference_documents: List[str],
    ) -> Dict[str, Any]:
        """Evaluate whether an answer is accurate with respect to the
        reference documents.

        When ``judge_fn`` is set, calls the LLM (e.g. gpt-4o-mini) as the
        judge, matching the KARL paper (Table 10).  Falls back to token-overlap
        heuristic if no LLM is available or the call fails.

        Returns
        -------
        dict
            Keys: ``is_accurate`` (bool), ``confidence`` (float),
            ``grounding_score`` (float), ``reason`` (str).
        """
        if not reference_documents:
            return {
                "is_accurate": False, "confidence": 0.0,
                "grounding_score": 0.0, "reason": "no reference documents provided",
            }

        if self.judge_fn is not None:
            result = self._llm_judge_reference_accuracy(
                question, answer, reference_documents
            )
            if result is not None:
                return result

        return self._heuristic_judge_reference_accuracy(
            question, answer, reference_documents
        )

    def _llm_judge_reference_accuracy(
        self, question: str, answer: str, reference_documents: List[str],
    ) -> Optional[Dict[str, Any]]:
        """LLM-backed reference accuracy check (KARL paper Table 10)."""
        docs_text = "\n\n".join(
            f"[Doc {i+1}]: {d[:500]}"
            for i, d in enumerate(reference_documents[:5])
        )
        prompt = (
            "You are evaluating whether a synthetic answer is factually "
            "accurate and grounded in the reference documents.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Reference Documents:\n{docs_text}\n\n"
            "Is the answer factually accurate and supported by the documents? "
            "Minor omissions are acceptable; outright factual errors are not.\n\n"
            'Respond with JSON: {"is_accurate": true/false, "reason": "..."}'
        )
        try:
            resp = self.judge_fn([{"role": "user", "content": prompt}])
            text = resp.get("content", "") if isinstance(resp, dict) else str(resp)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                parsed = _json.loads(match.group())
                return {
                    "is_accurate": parsed.get("is_accurate", True),
                    "confidence": 0.9,
                    "grounding_score": 1.0 if parsed.get("is_accurate") else 0.0,
                    "reason": parsed.get("reason", ""),
                }
        except Exception:
            pass
        return None

    def _heuristic_judge_reference_accuracy(
        self, question: str, answer: str, reference_documents: List[str],
    ) -> Dict[str, Any]:
        """Token-overlap heuristic fallback for reference accuracy."""
        answer_tokens = set(self._normalize_tokens(answer))
        if not answer_tokens:
            return {
                "is_accurate": False, "confidence": 0.0,
                "grounding_score": 0.0, "reason": "empty answer",
            }

        corpus_text = " ".join(reference_documents).lower()
        corpus_tokens = set(corpus_text.split())

        grounded = answer_tokens & corpus_tokens
        grounding_score = len(grounded) / len(answer_tokens)

        is_accurate = grounding_score >= 0.3
        return {
            "is_accurate": is_accurate,
            "confidence": grounding_score,
            "grounding_score": grounding_score,
            "reason": "" if is_accurate else f"only {grounding_score:.0%} of answer tokens found in references",
        }

    @staticmethod
    def _normalize_tokens(text: str) -> List[str]:
        """Lowercase and split into tokens, removing stopwords and punctuation."""
        stopwords: Set[str] = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
            "both", "either", "neither", "each", "every", "all", "any", "few",
            "more", "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "because", "if", "when", "where",
            "how", "what", "which", "who", "whom", "this", "that", "these",
            "those", "it", "its", "i", "me", "my", "we", "us", "our", "you",
            "your", "he", "him", "his", "she", "her", "they", "them", "their",
        }
        text_clean = re.sub(r"[^\w\s]", "", text.lower())
        return [t for t in text_clean.split() if t not in stopwords and len(t) > 1]


class GroundingFilter:
    """Checks that answer tokens are grounded in retrieved documents.

    This is a lighter-weight alternative to ``QualityFilter.judge_reference_accuracy``
    that operates at the token level without calling an LLM.
    """

    def __init__(self, min_grounding_ratio: float = 0.3):
        self.min_grounding_ratio = min_grounding_ratio

    def apply(
        self,
        examples: List[Any],
        retrieved_documents: Optional[List[List[str]]] = None,
    ) -> List[Any]:
        """Filter examples by checking that answer tokens appear in the
        corresponding retrieved documents.

        Parameters
        ----------
        examples : list
            QA examples to check. Each should have ``answer`` and optionally
            ``citations`` attributes.
        retrieved_documents : list[list[str]] | None
            Per-example lists of retrieved document texts. If ``None``,
            the filter falls back to using the ``citations`` field on each
            example.

        Returns
        -------
        list
            Examples whose answers are sufficiently grounded.
        """
        result: List[Any] = []
        for i, example in enumerate(examples):
            answer = getattr(example, "answer", None) or (
                example.get("answer") if isinstance(example, dict) else None
            )
            if answer is None:
                continue

            # Determine the reference docs for this example
            if retrieved_documents is not None and i < len(retrieved_documents):
                docs = retrieved_documents[i]
            else:
                docs = getattr(example, "citations", None) or (
                    example.get("citations", []) if isinstance(example, dict) else []
                )

            if not docs:
                # No documents to ground against -- skip
                continue

            grounding = self._compute_grounding(answer, docs)
            if grounding >= self.min_grounding_ratio:
                result.append(example)

        return result

    @staticmethod
    def _compute_grounding(answer: str, documents: List[str]) -> float:
        """Compute the fraction of answer tokens found in documents."""
        answer_clean = re.sub(r"[^\w\s]", "", answer.lower())
        answer_tokens = set(answer_clean.split())
        if not answer_tokens:
            return 0.0

        # Build corpus token set from all documents
        corpus = " ".join(str(d) for d in documents).lower()
        corpus_clean = re.sub(r"[^\w\s]", "", corpus)
        corpus_tokens = set(corpus_clean.split())

        grounded = answer_tokens & corpus_tokens
        return len(grounded) / len(answer_tokens)
