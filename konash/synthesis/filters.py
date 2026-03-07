from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set


class PassRateFilter:
    """Filters rollout groups by their pass rate, rejecting groups where
    the solver succeeds too often (trivially easy) or too rarely
    (potentially unanswerable or ambiguous).

    Attributes
    ----------
    min_pass_rate : float or None
        Minimum pass rate to keep a group (inclusive). Groups below this
        threshold are considered too hard or broken.
    max_pass_rate : float or None
        Maximum pass rate to keep a group (inclusive). Groups above this
        threshold are considered trivially easy.
    """

    min_pass_rate = None
    max_pass_rate = None

    def __init__(
        self,
        min_pass_rate: Optional[float] = None,
        max_pass_rate: Optional[float] = None,
    ):
        self.min_pass_rate = min_pass_rate
        self.max_pass_rate = max_pass_rate

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
    """Multi-dimensional quality filter that checks for ambiguity and
    reference accuracy using an LLM judge.

    In a full deployment, ``judge_ambiguity`` and ``judge_reference_accuracy``
    call an LLM (e.g. gpt-4o-mini) to evaluate each QA pair. The base
    implementation uses heuristic checks.
    """

    def __init__(
        self,
        judge_model: Optional[str] = None,
        checks_ambiguity: bool = True,
        checks_reference_accuracy: bool = True,
    ):
        self.judge_model = judge_model
        self.checks_ambiguity = checks_ambiguity
        self.checks_reference_accuracy = checks_reference_accuracy

    def apply(
        self,
        examples: List[Any],
        reference_documents: Optional[List[str]] = None,
    ) -> List[Any]:
        """Filter examples by quality, removing ambiguous or inaccurate ones.

        Parameters
        ----------
        examples : list
            QA examples to evaluate. Each should have ``question``, ``answer``,
            and optionally ``citations`` attributes.
        reference_documents : list[str] | None
            Source documents for reference-accuracy checking.

        Returns
        -------
        list
            Examples that pass all enabled quality checks.
        """
        passed: List[Any] = []
        for example in examples:
            question = getattr(example, "question", None) or (
                example.get("question") if isinstance(example, dict) else None
            )
            answer = getattr(example, "answer", None) or (
                example.get("answer") if isinstance(example, dict) else None
            )

            if question is None or answer is None:
                continue

            # Ambiguity check
            if self.checks_ambiguity:
                ambiguity_result = self.judge_ambiguity(question, answer)
                if ambiguity_result.get("is_ambiguous", False):
                    continue

            # Reference accuracy check
            if self.checks_reference_accuracy and reference_documents:
                accuracy_result = self.judge_reference_accuracy(
                    question, answer, reference_documents
                )
                if not accuracy_result.get("is_accurate", True):
                    continue

            passed.append(example)
        return passed

    def judge_ambiguity(
        self, question: str, answer: str
    ) -> Dict[str, Any]:
        """Evaluate whether a question is ambiguous.

        An ambiguous question has multiple valid interpretations that would
        lead to different answers.

        Parameters
        ----------
        question : str
            The question text.
        answer : str
            The proposed answer.

        Returns
        -------
        dict
            Keys: ``is_ambiguous`` (bool), ``confidence`` (float), ``reason`` (str).
        """
        # Heuristic ambiguity signals
        ambiguity_markers = [
            r"\bor\b",
            r"\beither\b",
            r"\bcould be\b",
            r"\bmight\b",
            r"\bpossibly\b",
            r"\bwhich one\b",
        ]
        question_lower = question.lower()

        marker_count = sum(
            1 for pat in ambiguity_markers if re.search(pat, question_lower)
        )

        # Questions that are too short are often ambiguous
        word_count = len(question.split())
        too_short = word_count < 4

        # Questions with no clear interrogative focus
        has_interrogative = any(
            w in question_lower
            for w in ["what", "who", "when", "where", "why", "how", "which", "is", "are", "does", "do", "did", "can"]
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

        Parameters
        ----------
        question : str
            The question text.
        answer : str
            The proposed answer.
        reference_documents : list[str]
            Source documents to check against.

        Returns
        -------
        dict
            Keys: ``is_accurate`` (bool), ``confidence`` (float),
            ``grounding_score`` (float), ``reason`` (str).
        """
        if not reference_documents:
            return {
                "is_accurate": False,
                "confidence": 0.0,
                "grounding_score": 0.0,
                "reason": "no reference documents provided",
            }

        # Tokenize the answer and check how many tokens appear in the corpus
        answer_tokens = set(self._normalize_tokens(answer))
        if not answer_tokens:
            return {
                "is_accurate": False,
                "confidence": 0.0,
                "grounding_score": 0.0,
                "reason": "empty answer",
            }

        # Combine all reference documents
        corpus_text = " ".join(reference_documents).lower()
        corpus_tokens = set(corpus_text.split())

        # Compute grounding score: fraction of answer tokens found in corpus
        grounded = answer_tokens & corpus_tokens
        grounding_score = len(grounded) / len(answer_tokens) if answer_tokens else 0.0

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
