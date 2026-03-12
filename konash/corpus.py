"""Corpus ingestion: read documents, chunk, embed, and index for vector search."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

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
        self._indexed = False

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
        # Try loading from cache first
        cache_file = self._cache_path()
        if cache_file is not None and cache_file.exists():
            if self.vector_search.load_cached_index(path=str(cache_file)):
                self.documents = self.vector_search._documents
                self._indexed = True
                if progress_callback:
                    total = len(self.documents)
                    progress_callback("embedding", total, total)
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

        # Save index to cache for instant reload next time
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.vector_search.save_index(str(cache_file))

        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Search the indexed corpus. Calls ``ingest()`` automatically if needed."""
        if not self._indexed:
            self.ingest()
        return self.vector_search.search(query, top_k=top_k, **kwargs)

    def batch_search(
        self,
        queries: Sequence[str],
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[List[Dict[str, Any]]]:
        """Run multiple searches in one vectorised pass."""
        if not self._indexed:
            self.ingest()
        return self.vector_search.batch_search(queries, top_k=top_k, **kwargs)

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    @property
    def indexed(self) -> bool:
        return self._indexed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
