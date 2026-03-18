"""Corpus ingestion: read documents, chunk, embed, and index for vector search."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from konash.retrieval.bm25 import BM25
from konash.retrieval.vector_search import VectorSearchTool


# ---------------------------------------------------------------------------
# Lightweight default embeddings (no ML deps required)
# ---------------------------------------------------------------------------

def _char_trigram_embed(texts: List[str], dim: int = 512) -> np.ndarray:
    """Deterministic bag-of-character-trigrams embedding.

    Good enough for cosine-similarity retrieval during development and
    small-corpus use.  Replace with a real embedding model for production.
    """
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        text = text.lower()
        for j in range(len(text) - 2):
            trigram = text[j : j + 3]
            idx = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % dim
            vecs[i, idx] += 1.0
    # L2-normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs /= norms
    return vecs


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_tokens: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping chunks of roughly *max_tokens* words."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunks.append(" ".join(words[start:end]))
        start += max_tokens - overlap
    return chunks


def _read_file(path: Path) -> str:
    """Read a single file and return its text content."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".rst", ".csv", ".log"):
        return path.read_text(errors="replace")
    if suffix == ".json":
        data = json.loads(path.read_text(errors="replace"))
        if isinstance(data, str):
            return data
        return json.dumps(data, indent=2)
    if suffix in (".html", ".htm"):
        text = path.read_text(errors="replace")
        # Strip tags (best-effort, no external dep)
        import re
        return re.sub(r"<[^>]+>", " ", text)
    # Fallback: try reading as text
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class Corpus:
    """Manages document ingestion from a directory, chunking, embedding,
    and building a ``VectorSearchTool`` index.

    Parameters
    ----------
    path : str | Path
        Path to a directory of documents (or a single file).
    embed_fn : callable | None
        ``(list[str]) -> np.ndarray`` embedding function.  If *None* a
        lightweight character-trigram embedding is used (no GPU needed).
    chunk_size : int
        Maximum words per chunk (default 512).
    chunk_overlap : int
        Overlap between consecutive chunks (default 64 words).
    extensions : list[str] | None
        File extensions to include.  ``None`` = all supported text formats.
    cache_dir : str | None
        Directory to cache the built index for faster reload.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".rst", ".csv", ".log",
        ".json", ".html", ".htm",
        ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h",
    }

    def __init__(
        self,
        path: str | Path,
        *,
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        extensions: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        self.embed_fn = embed_fn or _char_trigram_embed
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extensions = (
            set(extensions) if extensions else self.SUPPORTED_EXTENSIONS
        )
        self.cache_dir = cache_dir

        self.documents: List[Dict[str, Any]] = []
        self.vector_search = VectorSearchTool(
            embed_fn=self.embed_fn, cache_dir=cache_dir
        )
        self.bm25 = BM25()
        self._indexed = False
        self._lazy_text = False  # True for prebuilt indexes (text loaded on demand)
        self._docs_dir: Optional[Path] = None  # base dir for lazy text loading

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _cache_path(self) -> Optional[Path]:
        """Return the index cache file path, or None if caching is disabled."""
        if self.cache_dir is None:
            return None
        # Deterministic cache key from corpus path
        corpus_id = hashlib.md5(str(self.path.resolve()).encode()).hexdigest()[:12]
        return Path(self.cache_dir) / f"index_{corpus_id}.npz"

    def _bundled_index_path(self) -> Optional[Path]:
        """Check for a pre-built index shipped alongside the corpus.

        Looks for ``prebuilt_index.npz`` in the corpus directory itself
        and its parent (handles ``browsecomp-plus/documents/``).
        """
        for candidate in (self.path, self.path.parent):
            p = candidate / "prebuilt_index.npz"
            if p.exists():
                return p
        return None

    def _load_prebuilt(
        self,
        index_path: Path,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> bool:
        """Load a slim pre-built index (vectors + doc_ids) and map text from disk."""
        try:
            data = np.load(str(index_path), allow_pickle=True)
        except Exception:
            return False

        # Slim format: vectors + doc_ids (no text stored)
        if "doc_ids" in data:
            vectors = data["vectors"].astype(np.float32)
            doc_ids = data["doc_ids"].tolist()
            docs_dir = self.path if self.path.is_dir() else self.path.parent

            # Check if documents live in a subdirectory.
            # Prefer pages/ (page-level indexes) over documents/ (whole-doc).
            pages_sub = docs_dir / "pages"
            docs_sub = docs_dir / "documents"
            if pages_sub.is_dir():
                self._docs_dir = pages_sub
            elif docs_sub.is_dir():
                self._docs_dir = docs_sub
            else:
                self._docs_dir = docs_dir

            # Create lightweight stub documents (no file I/O — text loaded on demand)
            documents = []
            for docid in doc_ids:
                name = str(docid)
                # If doc_id is already an absolute path that exists, use it directly
                if os.path.isabs(name) and os.path.exists(name):
                    fpath = Path(name)
                else:
                    # Try multiple filename mappings:
                    # 1. Direct: docid with / → _ (FreshStack style)
                    # 2. URL tail: last path segment with _ → space (QAMPARI/Wikipedia style)
                    # 3. Basename: just the filename from the path
                    direct = name.replace("/", "_").replace("\\", "_")[:100]
                    fpath = self._docs_dir / f"{direct}.txt"
                    if not fpath.exists() and "/" in name:
                        # Try basename first (handles absolute paths stored as doc_ids)
                        basename = name.rsplit("/", 1)[-1]
                        candidate = self._docs_dir / basename
                        if candidate.exists():
                            fpath = candidate
                        else:
                            tail = basename.replace("_", " ")
                            fpath = self._docs_dir / f"{tail}.txt"
                documents.append({"text": "", "source": str(fpath), "chunk_index": 0})

            self.documents = documents
            self.vector_search.index(documents, embeddings=vectors, text_key="text")
            self._indexed = True
            self._lazy_text = True  # Skip BM25, load text on search
            self._align_embed_fn(index_path)
            if progress_callback:
                progress_callback("embedding", len(documents), len(documents))
            return True

        # Legacy format: full documents stored in npz
        if self.vector_search.load_cached_index(path=str(index_path)):
            self.documents = self.vector_search._documents
            self._indexed = True
            self.bm25.index(self.documents, text_key="text")
            self._align_embed_fn(index_path)
            if progress_callback:
                progress_callback("embedding", len(self.documents), len(self.documents))
            return True

        return False

    def _align_embed_fn(self, index_path: Path) -> None:
        """Swap the query embed function to match a pre-built index's model.

        Pre-built indexes store an ``embed_model`` key so we know which
        model produced the vectors.  If it differs from the currently
        configured embed function we load the matching one.
        """
        try:
            meta = np.load(str(index_path), allow_pickle=True)
            model = str(meta.get("embed_model", ""))
        except Exception:
            return

        if not model or model == "":
            return

        if "0.6B" in model or "0.6b" in model:
            # Small model — load on CPU to avoid GPU memory conflicts
            # (vLLM or FAISS may already be using GPU)
            try:
                from konash.retrieval.vector_search import load_embedding_model
                local_fn = load_embedding_model(model, device="cpu")
                # Verify dimensions match the index
                test = local_fn(["test"])
                index_dim = (
                    self.vector_search._vectors.shape[-1]
                    if hasattr(self.vector_search, "_vectors") and self.vector_search._vectors is not None
                    else None
                )
                if index_dim is None or test.shape[-1] == index_dim:
                    self.vector_search.embed_fn = local_fn
                    self.embed_fn = local_fn
            except Exception:
                pass
        elif "qwen3" in model.lower():
            try:
                self._set_qwen3_query_fn()
            except Exception:
                pass  # Fall back to whatever was configured

    def _set_qwen3_query_fn(self) -> None:
        """Set query embed_fn to Qwen3-Embedding-8B via HF Inference API."""
        import json as _json
        from huggingface_hub import InferenceClient

        config_path = os.path.expanduser("~/.konash/config.json")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token and os.path.exists(config_path):
            with open(config_path) as f:
                hf_token = _json.load(f).get("hf_token")

        client = InferenceClient(api_key=hf_token)
        hf_model = "Qwen/Qwen3-Embedding-8B"

        def query_fn(texts):
            all_embs = []
            for i in range(0, len(texts), 100):
                batch = texts[i : i + 100]
                r = client.feature_extraction(batch, model=hf_model)
                all_embs.append(np.array(r, dtype=np.float32))
            return np.vstack(all_embs) if len(all_embs) > 1 else all_embs[0]

        self.vector_search.embed_fn = query_fn
        self.embed_fn = query_fn

    def ingest(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> "Corpus":
        """Read, chunk, embed, and index all documents under ``self.path``.

        Parameters
        ----------
        progress_callback : callable | None
            Called with ``(phase, current, total)`` to report progress.
            Phases: ``"reading"``, ``"chunking"``, ``"embedding"``.

        Returns *self* for chaining.
        """
        # Try loading from project-specific cache first
        cache_file = self._cache_path()
        if cache_file is not None and cache_file.exists():
            if self.vector_search.load_cached_index(path=str(cache_file)):
                self.documents = self.vector_search._documents
                self._indexed = True
                self.bm25.index(self.documents, text_key="text")
                if progress_callback:
                    total = len(self.documents)
                    progress_callback("embedding", total, total)
                return self

        # Try bundled pre-built index (shipped with corpus download)
        bundled = self._bundled_index_path()
        if bundled is not None:
            if self._load_prebuilt(bundled, progress_callback):
                return self

        raw_docs = self._read_all(progress_callback)
        self.documents = self._chunk_all(raw_docs, progress_callback)

        if not self.documents:
            raise ValueError(
                f"No documents found under {self.path}. "
                f"Supported extensions: {sorted(self.extensions)}"
            )

        # Embed in batches for progress reporting and memory efficiency
        embed_batch_size = 512
        total = len(self.documents)
        if progress_callback:
            progress_callback("embedding", 0, total)

        if self.embed_fn is not None and total > embed_batch_size:
            texts = [doc["text"] for doc in self.documents]
            all_vecs = []
            for i in range(0, total, embed_batch_size):
                batch = texts[i : i + embed_batch_size]
                vecs = np.array(self.embed_fn(batch), dtype=np.float32)
                all_vecs.append(vecs)
                if progress_callback:
                    progress_callback("embedding", min(i + embed_batch_size, total), total)
            embeddings = np.vstack(all_vecs)
            self.vector_search.index(
                self.documents, embeddings=embeddings, text_key="text"
            )
        else:
            self.vector_search.index(self.documents, text_key="text")
            if progress_callback:
                progress_callback("embedding", total, total)

        self._indexed = True
        self.bm25.index(self.documents, text_key="text")

        # Save index to cache for instant reload next time
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.vector_search.save_index(str(cache_file))

        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Search the indexed corpus.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int
            Number of results to return.
        mode : str
            ``"hybrid"`` (default) — reciprocal rank fusion of BM25 + vector.
            ``"vector"`` — embedding similarity only.
            ``"bm25"`` — keyword matching only.
        """
        if not self._indexed:
            self.ingest()

        # Prebuilt indexes: build BM25 on first hybrid search call
        if self._lazy_text:
            if mode == "vector":
                results = self.vector_search.search(query, top_k=top_k, **kwargs)
                self._resolve_text(results)
                return results
            # Load text into BM25 on demand for hybrid/bm25 search
            if self.bm25._n_docs == 0:
                self._build_lazy_bm25()
            if mode == "bm25":
                results = self.bm25.search(query, top_k=top_k)
                self._resolve_text(results)
                return results
            # Hybrid: reciprocal rank fusion
            results = self._hybrid_search(query, top_k=top_k, **kwargs)
            self._resolve_text(results)
            return results

        if mode == "vector":
            return self.vector_search.search(query, top_k=top_k, **kwargs)
        if mode == "bm25":
            return self.bm25.search(query, top_k=top_k)
        # Hybrid: reciprocal rank fusion
        return self._hybrid_search(query, top_k=top_k, **kwargs)

    def _hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Combine BM25 and vector search with reciprocal rank fusion."""
        # Fetch more candidates from each source for better fusion
        n_candidates = min(top_k * 3, len(self.documents))
        vec_results = self.vector_search.search(query, top_k=n_candidates, **kwargs)
        bm25_results = self.bm25.search(query, top_k=n_candidates)

        # Build RRF scores keyed by (source, chunk_index) to deduplicate
        rrf_scores: Dict[tuple, float] = {}
        doc_map: Dict[tuple, Dict[str, Any]] = {}

        for rank, result in enumerate(vec_results):
            key = (result.get("source", ""), result.get("chunk_index", 0))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = result

        for rank, result in enumerate(bm25_results):
            key = (result.get("source", ""), result.get("chunk_index", 0))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            if key not in doc_map:
                doc_map[key] = result

        # Sort by fused score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for key, score in ranked:
            result = dict(doc_map[key])
            result["score"] = score
            results.append(result)
        return results

    def batch_search(
        self,
        queries: Sequence[str],
        top_k: int = 10,
        mode: str = "hybrid",
        **kwargs: Any,
    ) -> List[List[Dict[str, Any]]]:
        """Run multiple searches."""
        if not self._indexed:
            self.ingest()
        if mode == "vector":
            return self.vector_search.batch_search(queries, top_k=top_k, **kwargs)
        return [self.search(q, top_k=top_k, mode=mode, **kwargs) for q in queries]

    def _build_lazy_bm25(self) -> None:
        """Load text from disk for all documents and build BM25 index.

        Called on demand when hybrid search is requested on a prebuilt index.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Building BM25 index for prebuilt corpus (%d docs)...", len(self.documents))
        for doc in self.documents:
            if doc.get("text"):
                continue
            source = doc.get("source", "")
            if source:
                try:
                    with open(source, errors="replace") as f:
                        doc["text"] = f.read()
                except OSError:
                    if source.endswith(".txt.txt"):
                        try:
                            with open(source[:-4], errors="replace") as f:
                                doc["text"] = f.read()
                            continue
                        except OSError:
                            pass
                    doc["text"] = ""
        self.bm25.index(self.documents, text_key="text")
        logger.info("BM25 index built: %d documents", len(self.documents))

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    @property
    def indexed(self) -> bool:
        return self._indexed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_text(self, results: List[Dict[str, Any]]) -> None:
        """Lazy-load document text for search results only.

        For prebuilt indexes, documents are stored as stubs (no text).
        This reads the actual file content on demand for just the
        returned results (~10 files instead of 67K).
        """
        for r in results:
            if r.get("text"):
                continue
            source = r.get("source", "")
            if source:
                try:
                    with open(source, errors="replace") as f:
                        r["text"] = f.read(2048)
                except OSError:
                    # Handle double-extension bug in prebuilt indexes
                    # (doc_id stored as "file.txt.txt" but actual file is "file.txt")
                    if source.endswith(".txt.txt"):
                        try:
                            with open(source[:-4], errors="replace") as f:
                                r["text"] = f.read(2048)
                            continue
                        except OSError:
                            pass
                    r["text"] = ""

    def _read_all(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[Dict[str, str]]:
        """Read all matching files under ``self.path``."""
        docs: List[Dict[str, str]] = []
        if self.path.is_file():
            text = _read_file(self.path)
            if text.strip():
                docs.append({"text": text, "source": str(self.path)})
            return docs

        # Count matching files first for progress reporting
        all_files: List[Path] = []
        for root, _dirs, files in os.walk(self.path):
            for fname in sorted(files):
                fpath = Path(root) / fname
                if fpath.suffix.lower() in self.extensions:
                    all_files.append(fpath)

        total = len(all_files)
        for i, fpath in enumerate(all_files):
            if progress_callback:
                progress_callback("reading", i, total)
            text = _read_file(fpath)
            if text.strip():
                docs.append({"text": text, "source": str(fpath)})

        if progress_callback:
            progress_callback("reading", total, total)
        return docs

    def _chunk_all(
        self,
        raw_docs: List[Dict[str, str]],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk all documents."""
        total = len(raw_docs)
        chunked: List[Dict[str, Any]] = []
        for i, doc in enumerate(raw_docs):
            if progress_callback:
                progress_callback("chunking", i, total)
            chunks = _chunk_text(
                doc["text"],
                max_tokens=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            for idx, chunk in enumerate(chunks):
                chunked.append({
                    "text": chunk,
                    "source": doc["source"],
                    "chunk_index": idx,
                })
        if progress_callback:
            progress_callback("chunking", total, total)
        return chunked

    def __repr__(self) -> str:
        status = f"{self.num_documents} chunks, indexed" if self._indexed else "not ingested"
        return f"Corpus({self.path!s}, {status})"
