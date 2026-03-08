from __future__ import annotations

import hashlib
import logging
import re as _re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class EmbeddingDeduplicator:
    """Embedding-based deduplication using cosine similarity.

    Computes pairwise similarity between question embeddings and removes
    near-duplicates above a configurable threshold.

    Attributes
    ----------
    threshold : float
        Cosine-similarity threshold above which two questions are considered
        duplicates (default 0.85).
    """

    threshold = 0.85

    def __init__(self, threshold: float = 0.85, embedding_fn: Any = None):
        self.threshold = threshold
        self.embedding_fn = embedding_fn

    def score_pairs(
        self,
        embeddings_a: Any,
        embeddings_b: Optional[Any] = None,
    ) -> Any:
        """Compute pairwise cosine similarity between two sets of embeddings.

        Parameters
        ----------
        embeddings_a : array-like, shape (n, d)
            First set of embedding vectors.
        embeddings_b : array-like, shape (m, d) | None
            Second set of embedding vectors. If ``None``, computes the
            self-similarity matrix for ``embeddings_a``.

        Returns
        -------
        numpy.ndarray, shape (n, m)
            Cosine similarity matrix.
        """
        if np is None:
            raise RuntimeError("numpy is required for embedding deduplication")

        a = np.asarray(embeddings_a, dtype=np.float64)
        if embeddings_b is None:
            b = a
        else:
            b = np.asarray(embeddings_b, dtype=np.float64)

        # Normalize rows to unit vectors
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)

        return a_norm @ b_norm.T

    def deduplicate(
        self,
        questions: List[str],
        embeddings: Optional[Any] = None,
    ) -> List[str]:
        """Remove near-duplicate questions based on embedding similarity.

        Parameters
        ----------
        questions : list[str]
            Input question strings.
        embeddings : array-like, shape (n, d) | None
            Pre-computed embeddings. If ``None``, embeddings are computed
            using ``self.embedding_fn``.

        Returns
        -------
        list[str]
            Deduplicated question list with near-duplicates removed.
        """
        if not questions:
            return []

        if embeddings is None:
            if self.embedding_fn is not None:
                embeddings = self.embedding_fn(questions)
            else:
                # Fallback: use character-level hash-based pseudo-embeddings
                embeddings = self._pseudo_embeddings(questions)

        sim_matrix = self.score_pairs(embeddings)
        n = len(questions)
        removed: Set[int] = set()
        kept: List[str] = []

        for i in range(n):
            if i in removed:
                continue
            kept.append(questions[i])
            # Mark all subsequent similar questions as duplicates
            for j in range(i + 1, n):
                if j not in removed and sim_matrix[i, j] >= self.threshold:
                    removed.add(j)

        return kept

    def find_exact_duplicates(self, questions: List[str]) -> List[Tuple[int, int]]:
        """Find pairs of questions that are exact duplicates after normalization.

        Parameters
        ----------
        questions : list[str]
            Input question strings.

        Returns
        -------
        list[tuple[int, int]]
            Pairs of indices (i, j) where i < j and questions[i] == questions[j]
            after whitespace/case normalization.
        """
        normalized: Dict[str, int] = {}
        duplicates: List[Tuple[int, int]] = []

        for idx, q in enumerate(questions):
            key = self._normalize(q)
            if key in normalized:
                duplicates.append((normalized[key], idx))
            else:
                normalized[key] = idx

        return duplicates

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for exact-match comparison."""
        return " ".join(text.lower().split())

    @staticmethod
    def _pseudo_embeddings(texts: List[str]) -> Any:
        """Generate simple pseudo-embeddings based on character trigrams.

        This is a fallback for when no embedding model is available.
        """
        if np is None:
            raise RuntimeError("numpy is required")

        dim = 256
        vectors = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float64)
            normalized = " ".join(text.lower().split())
            # Character trigram hashing
            for i in range(len(normalized) - 2):
                trigram = normalized[i : i + 3]
                h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
                idx = h % dim
                vec[idx] += 1.0
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors)


class DeduplicationAgent:
    """Full two-stage deduplication pipeline.

    Stage 1: Remove exact duplicates (normalized string match).
    Stage 2: Remove near-duplicates using embedding retrieval + paraphrase judge.

    The agent tracks evaluation questions (held-out test set) and synthetic
    questions separately to prevent contamination.

    Attributes
    ----------
    evaluation_questions : list[str] | None
        Held-out evaluation questions to check for contamination.
    synthetic_questions : list[str] | None
        Synthetic questions to deduplicate.
    removed_exact_matches : list | None
        Records of exact duplicates removed.
    removed_near_duplicates : list | None
        Records of near-duplicates removed.
    embedding_model : str | None
        Name of the embedding model to use for similarity retrieval.
    similarity_top_k : int | None
        Number of candidates to retrieve for each near-duplicate check.
    paraphrase_judge : str or callable | None
        LLM judge for paraphrase detection.
    """

    evaluation_questions = None
    synthetic_questions = None
    removed_exact_matches = None
    removed_near_duplicates = None
    embedding_model = None
    similarity_top_k = None
    paraphrase_judge = None

    def __init__(
        self,
        evaluation_questions: Optional[List[str]] = None,
        synthetic_questions: Optional[List[str]] = None,
        embedding_model: Optional[str] = None,
        similarity_top_k: Optional[int] = None,
        paraphrase_judge: Any = None,
        embedding_fn: Any = None,
    ):
        self.evaluation_questions = evaluation_questions or []
        self.synthetic_questions = synthetic_questions or []
        self.embedding_model = embedding_model
        self.similarity_top_k = similarity_top_k or 20
        self.paraphrase_judge = paraphrase_judge
        self.embedding_fn = embedding_fn
        self.removed_exact_matches = []
        self.removed_near_duplicates = []

        self._deduplicator = EmbeddingDeduplicator(
            threshold=0.85,
            embedding_fn=embedding_fn,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        synthetic_questions: Optional[List[str]] = None,
        evaluation_questions: Optional[List[str]] = None,
    ) -> List[str]:
        """Execute the full two-stage deduplication pipeline.

        1. Remove exact duplicates within the synthetic set and against eval.
        2. Remove near-duplicates using embedding similarity + paraphrase judge.

        Parameters
        ----------
        synthetic_questions : list[str] | None
            Override the synthetic questions to process.
        evaluation_questions : list[str] | None
            Override the evaluation questions to check against.

        Returns
        -------
        list[str]
            Cleaned synthetic questions.
        """
        if synthetic_questions is not None:
            self.synthetic_questions = list(synthetic_questions)
        if evaluation_questions is not None:
            self.evaluation_questions = list(evaluation_questions)

        self.removed_exact_matches = []
        self.removed_near_duplicates = []

        # Stage 1: Exact deduplication
        self.synthetic_questions = self.remove_exact_duplicates(self.synthetic_questions)

        # Stage 1b: Remove exact matches against evaluation set
        if self.evaluation_questions:
            self.synthetic_questions = self.deduplicate_against_eval_set(
                self.synthetic_questions, self.evaluation_questions
            )

        # Stage 2: Near-duplicate removal within synthetic set
        self.synthetic_questions = self.deduplicate_within_synthetic_set(
            self.synthetic_questions
        )

        # Stage 2b: Near-duplicate removal against eval set
        if self.evaluation_questions:
            self.synthetic_questions = self.remove_near_duplicates(
                self.synthetic_questions, self.evaluation_questions
            )

        return self.synthetic_questions

    def remove_exact_duplicates(self, questions: List[str]) -> List[str]:
        """Remove exact duplicates from a list of questions.

        Parameters
        ----------
        questions : list[str]

        Returns
        -------
        list[str]
            Deduplicated list.
        """
        seen: Dict[str, int] = {}
        result: List[str] = []

        for idx, q in enumerate(questions):
            key = self._normalize(q)
            if key in seen:
                self.removed_exact_matches.append({
                    "removed_index": idx,
                    "duplicate_of": seen[key],
                    "question": q,
                })
            else:
                seen[key] = idx
                result.append(q)

        return result

    def remove_near_duplicates(
        self,
        candidates: List[str],
        reference_set: List[str],
    ) -> List[str]:
        """Remove candidates that are near-duplicates of reference questions.

        Uses embedding retrieval to find top-k most similar reference questions,
        then applies the paraphrase judge.

        Parameters
        ----------
        candidates : list[str]
            Questions to filter.
        reference_set : list[str]
            Reference questions to check against (e.g., eval set).

        Returns
        -------
        list[str]
            Candidates with near-duplicates of reference questions removed.
        """
        if not reference_set or not candidates:
            return list(candidates)

        kept: List[str] = []
        for candidate in candidates:
            similar = self.retrieve_similar_questions(candidate, reference_set)
            is_duplicate = False

            for ref_q, sim_score in similar:
                if sim_score >= self._deduplicator.threshold:
                    is_paraphrase = self.judge_paraphrase(candidate, ref_q)
                    if is_paraphrase:
                        self.removed_near_duplicates.append({
                            "candidate": candidate,
                            "reference": ref_q,
                            "similarity": sim_score,
                        })
                        is_duplicate = True
                        break

            if not is_duplicate:
                kept.append(candidate)

        return kept

    def deduplicate_against_eval_set(
        self,
        synthetic: List[str],
        eval_set: List[str],
    ) -> List[str]:
        """Remove synthetic questions that are exact matches of eval questions.

        Parameters
        ----------
        synthetic : list[str]
            Synthetic questions.
        eval_set : list[str]
            Evaluation questions to check for contamination.

        Returns
        -------
        list[str]
            Synthetic questions with eval-set contamination removed.
        """
        eval_normalized: Set[str] = {self._normalize(q) for q in eval_set}
        result: List[str] = []

        for q in synthetic:
            if self._normalize(q) in eval_normalized:
                self.removed_exact_matches.append({
                    "question": q,
                    "reason": "exact_match_eval_set",
                })
            else:
                result.append(q)

        return result

    def deduplicate_within_synthetic_set(
        self, questions: List[str]
    ) -> List[str]:
        """Remove near-duplicates within the synthetic question set.

        Parameters
        ----------
        questions : list[str]

        Returns
        -------
        list[str]
            Deduplicated synthetic questions.
        """
        if not questions:
            return []

        # Compute embeddings
        if self.embedding_fn is not None:
            embeddings = self.embedding_fn(questions)
        else:
            embeddings = self._deduplicator._pseudo_embeddings(questions)

        sim_matrix = self._deduplicator.score_pairs(embeddings)
        n = len(questions)
        removed: Set[int] = set()
        kept: List[str] = []

        for i in range(n):
            if i in removed:
                continue
            kept.append(questions[i])
            for j in range(i + 1, n):
                if j in removed:
                    continue
                if sim_matrix[i, j] >= self._deduplicator.threshold:
                    # Apply paraphrase judge for borderline cases
                    if self.paraphrase_judge is not None:
                        is_para = self.judge_paraphrase(questions[i], questions[j])
                        if not is_para:
                            continue
                    self.removed_near_duplicates.append({
                        "candidate": questions[j],
                        "reference": questions[i],
                        "similarity": float(sim_matrix[i, j]),
                    })
                    removed.add(j)

        return kept

    def judge_paraphrase(
        self,
        question_a: str,
        question_b: str,
        answer_a: Optional[str] = None,
        answer_b: Optional[str] = None,
    ) -> bool:
        """Determine whether two questions are paraphrases of each other.

        When ``paraphrase_judge`` is an :class:`LLMParaphraseJudge` (or
        any callable), delegates to it using the exact prompts from the
        KARL paper (Figure 32 for TREC-Biogen, Figure 33 for
        BrowseComp-Plus).  Falls back to token-overlap heuristic when
        no judge is configured.

        Parameters
        ----------
        question_a : str
            First question (typically the generated/synthetic question).
        question_b : str
            Second question (typically the validation-set question).
        answer_a : str | None
            Answer for question_a (used in BrowseComp-Plus mode).
        answer_b : str | None
            Answer for question_b (used in BrowseComp-Plus mode).

        Returns
        -------
        bool
            True if the questions are judged to be paraphrases.
        """
        if self.paraphrase_judge is not None:
            if isinstance(self.paraphrase_judge, LLMParaphraseJudge):
                return self.paraphrase_judge.judge(
                    question_a, question_b,
                    answer_a=answer_a, answer_b=answer_b,
                )
            if callable(self.paraphrase_judge):
                return bool(self.paraphrase_judge(question_a, question_b))

        # Heuristic fallback: high token overlap indicates paraphrase
        return _heuristic_paraphrase_check(question_a, question_b)

    def retrieve_similar_questions(
        self,
        query: str,
        corpus: List[str],
    ) -> List[Tuple[str, float]]:
        """Retrieve the most similar questions from a corpus using embeddings.

        Parameters
        ----------
        query : str
            The query question.
        corpus : list[str]
            Corpus of questions to search.

        Returns
        -------
        list[tuple[str, float]]
            Top-k (question, similarity_score) pairs, sorted by descending
            similarity.
        """
        if not corpus:
            return []

        top_k = self.similarity_top_k or 20

        # Compute embeddings for query and corpus
        all_texts = [query] + corpus
        if self.embedding_fn is not None:
            all_embeddings = self.embedding_fn(all_texts)
        else:
            all_embeddings = self._deduplicator._pseudo_embeddings(all_texts)

        query_emb = all_embeddings[:1]
        corpus_emb = all_embeddings[1:]

        similarities = self._deduplicator.score_pairs(query_emb, corpus_emb)[0]

        # Get top-k indices
        if np is not None:
            top_indices = np.argsort(-similarities)[:top_k]
        else:
            indexed = sorted(enumerate(similarities), key=lambda x: -x[1])
            top_indices = [idx for idx, _ in indexed[:top_k]]

        return [(corpus[i], float(similarities[i])) for i in top_indices]

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for exact-match comparison."""
        return " ".join(text.lower().split())


class LLMParaphraseJudge:
    """LLM-backed paraphrase judge using the KARL paper's deduplication
    prompts (Appendix D.2, Figures 32–33).

    Two modes match the two tasks in the paper:

    - **question_only** (Figure 32, TREC-Biogen): compares questions only.
      Uses ``gpt-4o-mini`` at temperature 0.
    - **question_and_answer** (Figure 33, BrowseComp-Plus): compares
      question-answer pairs, including "inverse" detection where one
      pair's answer appears in the other's question.

    Parameters
    ----------
    llm_fn : callable
        ``(messages) -> {"role": "assistant", "content": "..."}``.
    mode : str
        ``"question_only"`` or ``"question_and_answer"``.
    """

    def __init__(
        self,
        llm_fn: Callable,
        mode: str = "question_only",
    ):
        self.llm_fn = llm_fn
        self.mode = mode

    def judge(
        self,
        question_a: str,
        question_b: str,
        answer_a: Optional[str] = None,
        answer_b: Optional[str] = None,
    ) -> bool:
        """Judge whether two questions (or QA pairs) are duplicates.

        Returns True if the LLM determines they are duplicates.
        Falls back to heuristic if the LLM call fails.
        """
        if self.mode == "question_and_answer" and answer_a and answer_b:
            return self._judge_qa_pairs(
                question_a, answer_a, question_b, answer_b,
            )
        return self._judge_questions(question_a, question_b)

    def _judge_questions(self, q1: str, q2: str) -> bool:
        """TREC-Biogen dedup prompt (Figure 32, Appendix D.2).

        Question-only comparison. The judge determines if two questions
        ask for the SAME information, even if phrased differently.
        """
        prompt = (
            "Question Deduplication Judge Prompt for TREC-Biogen\n\n"
            "Your Role: You are judging whether two questions are "
            "semantically equivalent or duplicate.\n\n"
            f"Question 1: {q1}\n"
            f"Question 2: {q2}\n\n"
            "Your Task: Determine if Question 1 and Question 2 are asking "
            "for the SAME information, even if phrased differently.\n\n"
            "Guidelines:\n"
            '- "What is the capital of France?" and "Which city is the '
            'capital of France?" are duplicates (same question).\n'
            '- "What is the capital of France?" and "What is the population '
            'of France?" are NOT duplicates (different questions).\n'
            '- "Who invented the telephone?" and "Who created the '
            'telephone?" are duplicates (same question).\n'
            "- Minor differences in wording are acceptable if the core "
            "question is the same.\n"
            "- Consider paraphrasing -- different words can ask the same "
            "question.\n\n"
            "Output Format:\n"
            "<reasoning>[Brief explanation of judgment]</reasoning>\n"
            "<duplicate>[yes or no]</duplicate>"
        )
        return self._call_and_parse(prompt)

    def _judge_qa_pairs(
        self, q1: str, a1: str, q2: str, a2: str,
    ) -> bool:
        """BrowseComp-Plus dedup prompt (Figure 33, Appendix D.2).

        Question-answer pair comparison. Handles "inverse" questions
        where one pair's answer appears in the other's question.
        """
        prompt = (
            "Deduplication Judge Prompt for BrowseComp-Plus\n\n"
            "You are judging whether two question-answer pairs are "
            "duplicates.\n\n"
            "Question-Answer Pair 1 (Generated):\n"
            f"Question 1: {q1}\n"
            f"Answer 1: {a1}\n\n"
            "Question-Answer Pair 2 (Validation Set):\n"
            f"Question 2: {q2}\n"
            f"Answer 2: {a2}\n\n"
            "Your Task: Determine if these question-answer pairs are about "
            "the same underlying fact or relationship. Two pairs are "
            "duplicates if:\n"
            "1. They are about the same underlying fact, relationship, or "
            "piece of knowledge.\n"
            "2. This includes \"inverse\" questions where Q1's answer "
            "appears in Q2's question and vice versa.\n\n"
            "Examples:\n"
            '- Q1: "Who is the CEO of Apple?" A1: "Tim Cook" vs '
            'Q2: "Who leads Apple Inc?" A2: "Tim Cook"\n'
            "  -> DUPLICATE (same fact)\n"
            '- Q1: "Who is the CEO of Apple?" A1: "Tim Cook" vs '
            'Q2: "Who is Tim Cook?" A2: "CEO of Apple"\n'
            "  -> DUPLICATE (same fact, inverse framing)\n"
            '- Q1: "What year was Obama born?" A1: "1961" vs '
            'Q2: "When did Obama become president?" A2: "2009"\n'
            "  -> NOT DUPLICATE (different facts about the same person)\n"
            '- Q1: "Capital of France?" A1: "Paris" vs '
            'Q2: "Largest city in France?" A2: "Paris"\n'
            "  -> NOT DUPLICATE (different facts, answer happens to be "
            "the same)\n"
            '- Q1: "Who directed Inception?" A1: "Christopher Nolan" vs '
            'Q2: "Who directed The Dark Knight?" A2: "Christopher Nolan"\n'
            "  -> NOT DUPLICATE (different facts, same answer)\n\n"
            "Output Format:\n"
            "<reasoning>Analyze whether both pairs encode the same "
            "underlying fact or relationship</reasoning>\n"
            "<duplicate>yes or no</duplicate>"
        )
        return self._call_and_parse(prompt)

    def _call_and_parse(self, prompt: str) -> bool:
        """Call the LLM and parse the <duplicate> tag from the response."""
        try:
            response = self.llm_fn([{"role": "user", "content": prompt}])
            content = (
                response.get("content", "")
                if isinstance(response, dict)
                else str(response)
            )

            # Parse <duplicate>yes/no</duplicate>
            match = _re.search(
                r"<duplicate>\s*(yes|no)\s*</duplicate>",
                content,
                _re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().lower() == "yes"

            # Fallback: look for "yes" or "no" in the last line
            lines = content.strip().split("\n")
            last_line = lines[-1].strip().lower() if lines else ""
            if "yes" in last_line:
                return True
            if "no" in last_line:
                return False
        except Exception:
            pass

        # LLM failed — fall back to heuristic
        return False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _heuristic_paraphrase_check(question_a: str, question_b: str) -> bool:
    """Token-overlap heuristic for paraphrase detection.

    Uses Jaccard similarity >= 0.6 as threshold.  This is the fallback
    when no LLM judge is configured.
    """
    tokens_a = set(question_a.lower().split())
    tokens_b = set(question_b.lower().split())

    if not tokens_a or not tokens_b:
        return False

    overlap = tokens_a & tokens_b
    jaccard = len(overlap) / len(tokens_a | tokens_b)
    return jaccard >= 0.6


def _load_dedup_embedding_fn(model_name: str) -> Optional[Callable]:
    """Attempt to load a real embedding model for deduplication.

    Tries the retrieval module's :func:`load_embedding_model` first.
    Returns None if no ML library is available (caller should use
    pseudo-embeddings as fallback).
    """
    try:
        from konash.retrieval.vector_search import (
            load_embedding_model,
            resolve_embedding_model_name,
        )
        resolved = resolve_embedding_model_name(model_name)
        return load_embedding_model(resolved)
    except Exception as exc:
        logger.debug("Could not load embedding model %r for dedup: %s", model_name, exc)
        return None


class TRECBiogenDedupPolicy:
    """Deduplication policy tuned for the TREC Biogen benchmark.

    Uses a large embedding model (Qwen3-8B-Embedding) and retrieves more
    candidates (top_k=20) to catch subtle paraphrases in biomedical text.

    The paraphrase judge uses the Figure 32 prompt (question-only
    comparison) with ``gpt-4o-mini`` at temperature 0.

    Attributes
    ----------
    embedding_model : str
        ``"Qwen3-8B-Embedding"``
    similarity_top_k : int
        ``20``
    exact_match_scope : str or None
        Scope for exact match checks (e.g., "question_only", "question_and_answer").
    paraphrase_judge_model : str or None
        LLM model name for paraphrase judgments.
    """

    embedding_model = "Qwen3-8B-Embedding"
    similarity_top_k = 20
    exact_match_scope = None
    paraphrase_judge_model = None

    def __init__(
        self,
        exact_match_scope: Optional[str] = None,
        paraphrase_judge_model: Optional[str] = None,
    ):
        self.exact_match_scope = exact_match_scope or "question_and_answer"
        self.paraphrase_judge_model = paraphrase_judge_model

    def create_agent(self, llm_fn: Any = None, **kwargs: Any) -> DeduplicationAgent:
        """Create a DeduplicationAgent configured with this policy.

        Automatically loads the Qwen3-8B-Embedding model for semantic
        similarity.  Falls back to pseudo-embeddings if the model cannot
        be loaded.

        Parameters
        ----------
        llm_fn : callable | None
            LLM function for paraphrase judging.  When provided, creates
            an ``LLMParaphraseJudge`` using the TREC-Biogen prompt
            (Figure 32, question-only mode).
        """
        judge = None
        if llm_fn is not None:
            judge = LLMParaphraseJudge(llm_fn=llm_fn, mode="question_only")

        embedding_fn = kwargs.pop("embedding_fn", None)
        if embedding_fn is None:
            embedding_fn = _load_dedup_embedding_fn(self.embedding_model)

        return DeduplicationAgent(
            embedding_model=self.embedding_model,
            similarity_top_k=self.similarity_top_k,
            paraphrase_judge=judge,
            embedding_fn=embedding_fn,
            **kwargs,
        )


class BrowseCompDedupPolicy:
    """Deduplication policy tuned for the BrowseComp benchmark.

    Uses a smaller embedding model (Qwen3-0.6B-Embedding) and fewer candidates
    (top_k=10) since BrowseComp questions tend to be more distinct.

    The paraphrase judge uses the Figure 33 prompt (question+answer
    comparison with inverse detection) with ``gpt-4o-mini`` at
    temperature 0.

    Attributes
    ----------
    embedding_model : str
        ``"Qwen3-0.6B-Embedding"``
    similarity_top_k : int
        ``10``
    exact_answer_blocklist : list or None
        Answers that should be blocked (e.g., overly common / trivial answers).
    paraphrase_judge_model : str or None
        LLM model name for paraphrase judgments.
    """

    embedding_model = "Qwen3-0.6B-Embedding"
    similarity_top_k = 10
    exact_answer_blocklist = None
    paraphrase_judge_model = None

    def __init__(
        self,
        exact_answer_blocklist: Optional[List[str]] = None,
        paraphrase_judge_model: Optional[str] = None,
    ):
        self.exact_answer_blocklist = exact_answer_blocklist or []
        self.paraphrase_judge_model = paraphrase_judge_model

    def create_agent(self, llm_fn: Any = None, **kwargs: Any) -> DeduplicationAgent:
        """Create a DeduplicationAgent configured with this policy.

        Automatically loads the Qwen3-0.6B-Embedding model for semantic
        similarity.  Falls back to pseudo-embeddings if the model cannot
        be loaded.

        Parameters
        ----------
        llm_fn : callable | None
            LLM function for paraphrase judging.  When provided, creates
            an ``LLMParaphraseJudge`` using the BrowseComp-Plus prompt
            (Figure 33, question+answer mode with inverse detection).
        """
        judge = None
        if llm_fn is not None:
            judge = LLMParaphraseJudge(
                llm_fn=llm_fn, mode="question_and_answer",
            )

        embedding_fn = kwargs.pop("embedding_fn", None)
        if embedding_fn is None:
            embedding_fn = _load_dedup_embedding_fn(self.embedding_model)

        return DeduplicationAgent(
            embedding_model=self.embedding_model,
            similarity_top_k=self.similarity_top_k,
            paraphrase_judge=judge,
            embedding_fn=embedding_fn,
            **kwargs,
        )
