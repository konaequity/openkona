from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


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

    def judge_paraphrase(self, question_a: str, question_b: str) -> bool:
        """Determine whether two questions are paraphrases of each other.

        In a full deployment, this calls an LLM judge. The base
        implementation uses token-overlap heuristics.

        Parameters
        ----------
        question_a : str
        question_b : str

        Returns
        -------
        bool
            True if the questions are judged to be paraphrases.
        """
        if self.paraphrase_judge is not None and callable(self.paraphrase_judge):
            return bool(self.paraphrase_judge(question_a, question_b))

        # Heuristic: high token overlap indicates paraphrase
        tokens_a = set(question_a.lower().split())
        tokens_b = set(question_b.lower().split())

        if not tokens_a or not tokens_b:
            return False

        overlap = tokens_a & tokens_b
        jaccard = len(overlap) / len(tokens_a | tokens_b)
        return jaccard >= 0.6

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


class TRECBiogenDedupPolicy:
    """Deduplication policy tuned for the TREC Biogen benchmark.

    Uses a large embedding model (Qwen3-8B-Embedding) and retrieves more
    candidates (top_k=20) to catch subtle paraphrases in biomedical text.

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

    def create_agent(self, **kwargs: Any) -> DeduplicationAgent:
        """Create a DeduplicationAgent configured with this policy."""
        return DeduplicationAgent(
            embedding_model=self.embedding_model,
            similarity_top_k=self.similarity_top_k,
            paraphrase_judge=self.paraphrase_judge_model,
            **kwargs,
        )


class BrowseCompDedupPolicy:
    """Deduplication policy tuned for the BrowseComp benchmark.

    Uses a smaller embedding model (Qwen3-0.6B-Embedding) and fewer candidates
    (top_k=10) since BrowseComp questions tend to be more distinct.

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

    def create_agent(self, **kwargs: Any) -> DeduplicationAgent:
        """Create a DeduplicationAgent configured with this policy."""
        return DeduplicationAgent(
            embedding_model=self.embedding_model,
            similarity_top_k=self.similarity_top_k,
            paraphrase_judge=self.paraphrase_judge_model,
            **kwargs,
        )
