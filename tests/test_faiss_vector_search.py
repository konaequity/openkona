"""Tests for FAISS-backed vector search in VectorSearchTool."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from konash.retrieval.vector_search import (
    VectorSearchTool,
    _FAISS_AVAILABLE,
    _normalize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_embed_fn(texts, dim=64):
    """Deterministic pseudo-embeddings for testing."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(len(texts), dim).astype(np.float32)
    return _normalize(vecs)


def _make_documents(n=100):
    return [{"text": f"Document number {i} with content.", "id": i} for i in range(n)]


# ---------------------------------------------------------------------------
# Tests that run with both backends
# ---------------------------------------------------------------------------

class TestFaissBackend:
    """Tests that verify FAISS backend produces correct results."""

    @pytest.mark.skipif(not _FAISS_AVAILABLE, reason="faiss not installed")
    def test_faiss_index_builds(self):
        tool = VectorSearchTool(embed_fn=_dummy_embed_fn, use_faiss=True)
        docs = _make_documents(50)
        tool.index(docs)
        assert tool._faiss_index is not None
        assert tool._faiss_index.ntotal == 50

    @pytest.mark.skipif(not _FAISS_AVAILABLE, reason="faiss not installed")
    def test_faiss_search_returns_results(self):
        tool = VectorSearchTool(embed_fn=_dummy_embed_fn, use_faiss=True)
        docs = _make_documents(50)
        tool.index(docs)
        results = tool.search("test query", top_k=5)
        assert len(results) == 5
        assert all("score" in r for r in results)
        # Scores should be descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.skipif(not _FAISS_AVAILABLE, reason="faiss not installed")
    def test_faiss_batch_search(self):
        tool = VectorSearchTool(embed_fn=_dummy_embed_fn, use_faiss=True)
        docs = _make_documents(50)
        tool.index(docs)
        results = tool.batch_search(["query 1", "query 2"], top_k=3)
        assert len(results) == 2
        assert len(results[0]) == 3
        assert len(results[1]) == 3

    @pytest.mark.skipif(not _FAISS_AVAILABLE, reason="faiss not installed")
    def test_faiss_and_numpy_agree(self):
        """FAISS and numpy backends should return the same results."""
        docs = _make_documents(50)

        tool_faiss = VectorSearchTool(embed_fn=_dummy_embed_fn, use_faiss=True)
        tool_faiss.index(docs)

        tool_numpy = VectorSearchTool(embed_fn=_dummy_embed_fn, use_faiss=False)
        tool_numpy.index(docs)

        faiss_results = tool_faiss.search("test query", top_k=10)
        numpy_results = tool_numpy.search("test query", top_k=10)

        # Same document IDs in the same order
        faiss_ids = [r["id"] for r in faiss_results]
        numpy_ids = [r["id"] for r in numpy_results]
        assert faiss_ids == numpy_ids

        # Scores should be close (float precision differences)
        for fr, nr in zip(faiss_results, numpy_results):
            assert abs(fr["score"] - nr["score"]) < 1e-5

    @pytest.mark.skipif(not _FAISS_AVAILABLE, reason="faiss not installed")
    def test_faiss_batch_and_numpy_batch_agree(self):
        docs = _make_documents(50)
        queries = ["query a", "query b", "query c"]

        tool_faiss = VectorSearchTool(embed_fn=_dummy_embed_fn, use_faiss=True)
        tool_faiss.index(docs)

        tool_numpy = VectorSearchTool(embed_fn=_dummy_embed_fn, use_faiss=False)
        tool_numpy.index(docs)

        faiss_batch = tool_faiss.batch_search(queries, top_k=5)
        numpy_batch = tool_numpy.batch_search(queries, top_k=5)

        for f_results, n_results in zip(faiss_batch, numpy_batch):
            f_ids = [r["id"] for r in f_results]
            n_ids = [r["id"] for r in n_results]
            assert f_ids == n_ids


class TestFaissCachePersistence:
    """Tests for saving and loading FAISS indices."""

    @pytest.mark.skipif(not _FAISS_AVAILABLE, reason="faiss not installed")
    def test_save_and_load_faiss_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = _make_documents(30)

            # Build and save
            tool = VectorSearchTool(
                embed_fn=_dummy_embed_fn, cache_dir=tmpdir, use_faiss=True
            )
            tool.index(docs)
            saved_path = tool.save_index()
            assert saved_path is not None

            # Verify FAISS files were created
            index_id = tool._index_id
            assert Path(tmpdir, f"{index_id}.faiss").exists()
            assert Path(tmpdir, f"{index_id}.meta.npz").exists()
            # Legacy npz also saved
            assert Path(tmpdir, f"{index_id}.npz").exists()

            # Load into a fresh tool
            tool2 = VectorSearchTool(
                embed_fn=_dummy_embed_fn, cache_dir=tmpdir, use_faiss=True
            )
            loaded = tool2.load_cached_index(index_id=index_id)
            assert loaded
            assert tool2._faiss_index is not None
            assert tool2._faiss_index.ntotal == 30

            # Search results should match
            r1 = tool.search("test", top_k=5)
            r2 = tool2.search("test", top_k=5)
            assert [r["id"] for r in r1] == [r["id"] for r in r2]

    @pytest.mark.skipif(not _FAISS_AVAILABLE, reason="faiss not installed")
    def test_from_cache_classmethod(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = _make_documents(20)

            # Build and save
            tool = VectorSearchTool(
                embed_fn=_dummy_embed_fn, cache_dir=tmpdir, use_faiss=True
            )
            tool.index(docs)
            saved_path = tool.save_index()

            # Load via from_cache
            tool2 = VectorSearchTool.from_cache(
                path=saved_path,
                embed_fn=_dummy_embed_fn,
                use_faiss=True,
            )
            assert tool2._faiss_index is not None or tool2._vectors is not None
            results = tool2.search("query", top_k=3)
            assert len(results) == 3

    @pytest.mark.skipif(not _FAISS_AVAILABLE, reason="faiss not installed")
    def test_from_cache_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            VectorSearchTool.from_cache(
                path="/nonexistent/path/index.faiss",
                embed_fn=_dummy_embed_fn,
            )

    @pytest.mark.skipif(not _FAISS_AVAILABLE, reason="faiss not installed")
    def test_numpy_can_load_faiss_saved_index(self):
        """A numpy-only tool can load the legacy .npz saved alongside FAISS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = _make_documents(20)

            # Save with FAISS
            tool = VectorSearchTool(
                embed_fn=_dummy_embed_fn, cache_dir=tmpdir, use_faiss=True
            )
            tool.index(docs)
            tool.save_index()

            # Load with numpy-only
            tool2 = VectorSearchTool(
                embed_fn=_dummy_embed_fn, cache_dir=tmpdir, use_faiss=False
            )
            loaded = tool2.load_cached_index(index_id=tool._index_id)
            assert loaded
            assert tool2._faiss_index is None
            results = tool2.search("query", top_k=5)
            assert len(results) == 5


class TestFaissFallback:
    """Tests for graceful fallback when FAISS is not available."""

    def test_numpy_fallback_works(self):
        """Even with use_faiss=False, search works via numpy."""
        tool = VectorSearchTool(embed_fn=_dummy_embed_fn, use_faiss=False)
        docs = _make_documents(20)
        tool.index(docs)
        assert tool._faiss_index is None
        results = tool.search("test", top_k=5)
        assert len(results) == 5

    def test_use_faiss_true_without_install_raises(self):
        if _FAISS_AVAILABLE:
            pytest.skip("FAISS is installed, can't test missing-faiss path")
        with pytest.raises(ImportError, match="faiss is not installed"):
            VectorSearchTool(embed_fn=_dummy_embed_fn, use_faiss=True)

    def test_auto_detect_default(self):
        tool = VectorSearchTool(embed_fn=_dummy_embed_fn)
        assert tool._use_faiss == _FAISS_AVAILABLE
