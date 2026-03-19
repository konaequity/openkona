"""Tests for Corpus — chunking, file reading, ingestion, and search
without requiring any external API calls."""

import json

import numpy as np
import pytest

from konash.corpus import Corpus, _chunk_text, _read_file, _char_trigram_embed


class TestChunkText:

    def test_short_text_single_chunk(self):
        text = "Hello world this is a test"
        chunks = _chunk_text(text, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_long_text(self):
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        chunks = _chunk_text(text, max_tokens=30, overlap=5)
        assert len(chunks) > 1
        # Each chunk should have at most 30 words
        for chunk in chunks:
            assert len(chunk.split()) <= 30

    def test_overlap_creates_shared_words(self):
        words = [f"w{i}" for i in range(60)]
        text = " ".join(words)
        chunks = _chunk_text(text, max_tokens=30, overlap=10)
        assert len(chunks) >= 2
        # Last words of chunk 0 should appear in chunk 1
        words_0 = set(chunks[0].split()[-10:])
        words_1 = set(chunks[1].split()[:10])
        assert len(words_0 & words_1) > 0

    def test_empty_text(self):
        chunks = _chunk_text("", max_tokens=100)
        assert chunks == [""]


class TestReadFile:

    def test_read_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        assert _read_file(f) == "Hello world"

    def test_read_json_string(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text(json.dumps("plain string"))
        assert _read_file(f) == "plain string"

    def test_read_json_object(self, tmp_path):
        f = tmp_path / "test.json"
        data = {"key": "value"}
        f.write_text(json.dumps(data))
        result = _read_file(f)
        assert "key" in result
        assert "value" in result

    def test_read_html_strips_tags(self, tmp_path):
        f = tmp_path / "test.html"
        f.write_text("<html><body><p>Hello</p></body></html>")
        result = _read_file(f)
        assert "<p>" not in result
        assert "Hello" in result

    def test_read_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nContent here")
        result = _read_file(f)
        assert "Title" in result

    def test_read_python(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    return 'world'")
        result = _read_file(f)
        assert "def hello" in result


class TestCharTrigramEmbed:

    def test_output_shape(self):
        texts = ["hello", "world", "test"]
        vecs = _char_trigram_embed(texts, dim=128)
        assert vecs.shape == (3, 128)

    def test_normalized(self):
        texts = ["hello world"]
        vecs = _char_trigram_embed(texts, dim=128)
        norm = np.linalg.norm(vecs[0])
        assert abs(norm - 1.0) < 1e-5

    def test_similar_texts_close(self):
        vecs = _char_trigram_embed(["hello world", "hello worlds"], dim=256)
        sim = np.dot(vecs[0], vecs[1])
        assert sim > 0.8  # similar texts should have high cosine similarity

    def test_different_texts_less_similar(self):
        vecs = _char_trigram_embed(["hello world", "xyz quantum"], dim=256)
        sim = np.dot(vecs[0], vecs[1])
        assert sim < 0.5

    def test_deterministic(self):
        vecs1 = _char_trigram_embed(["test"], dim=128)
        vecs2 = _char_trigram_embed(["test"], dim=128)
        np.testing.assert_array_equal(vecs1, vecs2)


class TestCorpusIngestion:

    def test_ingest_txt_files(self, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "a.txt").write_text("Alpha bravo charlie delta echo foxtrot " * 20)
        (doc_dir / "b.txt").write_text("Golf hotel india juliet kilo lima " * 20)

        corpus = Corpus(str(doc_dir))
        corpus.ingest()

        assert corpus.indexed
        assert corpus.num_documents > 0

    def test_search_returns_results(self, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "france.txt").write_text("The capital of France is Paris. " * 10)
        (doc_dir / "germany.txt").write_text("The capital of Germany is Berlin. " * 10)

        corpus = Corpus(str(doc_dir))
        corpus.ingest()
        results = corpus.search("Paris France capital", top_k=5)

        assert len(results) > 0
        assert "text" in results[0]

    def test_ignores_unsupported_extensions(self, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "valid.txt").write_text("Valid content here " * 20)
        (doc_dir / "image.png").write_bytes(b"\x89PNG fake image data")

        corpus = Corpus(str(doc_dir))
        corpus.ingest()
        assert corpus.num_documents > 0

    def test_empty_directory_raises(self, tmp_path):
        doc_dir = tmp_path / "empty"
        doc_dir.mkdir()
        corpus = Corpus(str(doc_dir))
        with pytest.raises(ValueError, match="No documents found"):
            corpus.ingest()

    def test_num_documents_before_ingest(self, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        corpus = Corpus(str(doc_dir))
        assert corpus.num_documents == 0
        assert not corpus.indexed

    def test_prebuilt_qwen3_index_prefers_hf_query_fn_when_token_exists(self, tmp_path, monkeypatch):
        corpus_dir = tmp_path / "financebench"
        corpus_dir.mkdir()
        np.savez(
            corpus_dir / "prebuilt_index.npz",
            vectors=np.array([[1.0, 0.0]], dtype=np.float32),
            doc_ids=np.array(["doc1"]),
            embed_model="Qwen/Qwen3-Embedding-0.6B",
        )
        (corpus_dir / "pages").mkdir()
        (corpus_dir / "pages" / "doc1.txt").write_text("alpha facts")

        monkeypatch.setenv("HF_TOKEN", "hf_test_token")

        def fail_load_embedding_model(*args, **kwargs):
            raise AssertionError("local embedding model should not be loaded")

        called = {"hf": False}

        def fake_set_qwen3_query_fn(self, model_name=""):
            called["hf"] = True
            self.vector_search.embed_fn = lambda texts: np.array([[1.0, 0.0]], dtype=np.float32)
            self.embed_fn = self.vector_search.embed_fn

        monkeypatch.setattr("konash.retrieval.vector_search.load_embedding_model", fail_load_embedding_model)
        monkeypatch.setattr(Corpus, "_set_qwen3_query_fn", fake_set_qwen3_query_fn)

        corpus = Corpus(str(corpus_dir))
        corpus.ingest()

        assert called["hf"] is True


class TestCorpusBatchSearch:

    def test_batch_search(self, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "doc.txt").write_text("Cats and dogs are pets. " * 20)

        corpus = Corpus(str(doc_dir))
        corpus.ingest()
        results = corpus.batch_search(["cats", "dogs"], top_k=3)

        assert len(results) == 2
        for r in results:
            assert isinstance(r, list)
