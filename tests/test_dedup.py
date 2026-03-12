"""Tests for EmbeddingDeduplicator — cosine similarity, dedup logic,
and exact duplicate detection without requiring any API calls."""

import numpy as np
import pytest

from konash.synthesis.dedup import EmbeddingDeduplicator


class TestScorePairs:

    def test_identical_vectors_score_one(self):
        dedup = EmbeddingDeduplicator()
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = dedup.score_pairs(vecs)
        assert abs(sim[0, 0] - 1.0) < 1e-6
        assert abs(sim[1, 1] - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self):
        dedup = EmbeddingDeduplicator()
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = dedup.score_pairs(vecs)
        assert abs(sim[0, 1]) < 1e-6

    def test_cross_similarity(self):
        dedup = EmbeddingDeduplicator()
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = dedup.score_pairs(a, b)
        assert sim.shape == (1, 2)
        assert abs(sim[0, 0] - 1.0) < 1e-6
        assert abs(sim[0, 1]) < 1e-6

    def test_self_similarity_is_symmetric(self):
        dedup = EmbeddingDeduplicator()
        rng = np.random.default_rng(42)
        vecs = rng.normal(size=(5, 10))
        sim = dedup.score_pairs(vecs)
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)


class TestDeduplicate:

    def test_removes_near_duplicates(self):
        dedup = EmbeddingDeduplicator(threshold=0.9)
        # Two nearly identical vectors, one different
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],  # near-duplicate of first
            [0.0, 0.0, 1.0],    # distinct
        ])
        questions = ["Q1", "Q2", "Q3"]
        result = dedup.deduplicate(questions, embeddings=embeddings)
        assert "Q1" in result
        assert "Q3" in result
        assert "Q2" not in result

    def test_keeps_all_when_distinct(self):
        dedup = EmbeddingDeduplicator(threshold=0.9)
        embeddings = np.eye(3)  # all orthogonal
        questions = ["Q1", "Q2", "Q3"]
        result = dedup.deduplicate(questions, embeddings=embeddings)
        assert result == ["Q1", "Q2", "Q3"]

    def test_empty_input(self):
        dedup = EmbeddingDeduplicator()
        assert dedup.deduplicate([]) == []

    def test_single_question(self):
        dedup = EmbeddingDeduplicator()
        embeddings = np.array([[1.0, 0.0]])
        result = dedup.deduplicate(["Q1"], embeddings=embeddings)
        assert result == ["Q1"]

    def test_all_identical(self):
        dedup = EmbeddingDeduplicator(threshold=0.9)
        embeddings = np.array([[1.0, 0.0]] * 5)
        questions = [f"Q{i}" for i in range(5)]
        result = dedup.deduplicate(questions, embeddings=embeddings)
        assert result == ["Q0"]

    def test_uses_pseudo_embeddings_when_no_fn(self):
        """Fallback to pseudo-embeddings when no embedding_fn provided."""
        dedup = EmbeddingDeduplicator(threshold=0.99)
        questions = [
            "What is the capital of France?",
            "What is the capital of France?",  # exact dup
            "How tall is Mount Everest?",       # different
        ]
        result = dedup.deduplicate(questions)
        assert len(result) <= 2  # at least the exact dup should be removed


class TestExactDuplicates:

    def test_finds_exact_duplicates(self):
        dedup = EmbeddingDeduplicator()
        questions = ["Hello world", "hello  world", "Different question"]
        pairs = dedup.find_exact_duplicates(questions)
        assert (0, 1) in pairs

    def test_no_duplicates(self):
        dedup = EmbeddingDeduplicator()
        questions = ["Q1", "Q2", "Q3"]
        pairs = dedup.find_exact_duplicates(questions)
        assert pairs == []

    def test_case_insensitive(self):
        dedup = EmbeddingDeduplicator()
        questions = ["Hello World", "hello world"]
        pairs = dedup.find_exact_duplicates(questions)
        assert len(pairs) == 1
