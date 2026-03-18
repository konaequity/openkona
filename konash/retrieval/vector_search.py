from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

faiss = None
_FAISS_AVAILABLE = False


def _ensure_faiss():
    """Lazy-load faiss to avoid conflicts with embedding models."""
    global faiss, _FAISS_AVAILABLE
    if _FAISS_AVAILABLE:
        return True
    try:
        import faiss as _faiss
        faiss = _faiss
        _FAISS_AVAILABLE = True
        return True
    except ImportError:
        return False

logger = logging.getLogger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise each row in *vectors* (in-place safe)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


# ---------------------------------------------------------------------------
# Embedding model loader — provides real semantic embeddings
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL_CACHE: Dict[str, Any] = {}


def load_embedding_model(
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    device: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Callable[[List[str]], np.ndarray]:
    """Load a HuggingFace sentence-transformers / transformer embedding model
    and return a callable ``(texts: list[str]) -> np.ndarray``.

    Supported model families (matching the KARL paper):
    - ``sentence-transformers/*`` via sentence-transformers library
    - ``Qwen/*-Embedding*`` via transformers + mean-pooling
    - ``BAAI/bge-*``, ``thenlper/gte-*`` via sentence-transformers

    Falls back to a lightweight trigram hash if no ML library is available.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.  The paper uses:
        - ``"Qwen/Qwen3-Embedding-0.6B"`` for BrowseComp-Plus & dedup
        - ``"Qwen/Qwen3-Embedding-8B"`` for TREC-Biogen dedup
    device : str or None
        ``"cuda"``, ``"cpu"``, ``"mps"``, or ``None`` for auto-detection.
    trust_remote_code : bool
        Whether to trust remote code for Qwen models.

    Returns
    -------
    callable
        ``(texts: list[str]) -> np.ndarray`` of shape ``(len(texts), dim)``.
    """
    cache_key = f"{model_name}:{device}"
    if cache_key in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE[cache_key]

    embed_fn = _try_load_sentence_transformers(model_name, device)
    if embed_fn is None:
        embed_fn = _try_load_transformers(model_name, device, trust_remote_code)
    if embed_fn is None:
        raise RuntimeError(
            f"Could not load embedding model {model_name!r}. "
            f"Install sentence-transformers or transformers+torch."
        )

    _EMBEDDING_MODEL_CACHE[cache_key] = embed_fn
    return embed_fn


def _try_load_sentence_transformers(
    model_name: str, device: Optional[str]
) -> Optional[Callable[[List[str]], np.ndarray]]:
    """Try loading via sentence-transformers (preferred path)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    # Qwen3 bf16 crashes on MPS — force CPU on Mac when device is unset
    if device is None:
        import platform
        if platform.system() == "Darwin":
            device = "cpu"

    try:
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        logger.info("Loaded embedding model via sentence-transformers: %s", model_name)

        def embed_fn(texts: List[str]) -> np.ndarray:
            return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        return embed_fn
    except Exception as exc:
        logger.debug("sentence-transformers failed for %s: %s", model_name, exc)
        return None


def _try_load_transformers(
    model_name: str, device: Optional[str], trust_remote_code: bool
) -> Optional[Callable[[List[str]], np.ndarray]]:
    """Try loading via transformers + torch with mean-pooling."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        model = model.to(device).eval()
        logger.info("Loaded embedding model via transformers: %s → %s", model_name, device)

        def embed_fn(texts: List[str]) -> np.ndarray:
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**encoded)
            # Mean pooling over non-padding tokens
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy()

        return embed_fn
    except Exception as exc:
        logger.debug("transformers loading failed for %s: %s", model_name, exc)
        return None


def _trigram_embed_fn(texts: List[str], dim: int = 384) -> np.ndarray:
    """Fallback: character-trigram pseudo-embeddings (no ML deps)."""
    vectors = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        normalized = " ".join(text.lower().split())
        for i in range(len(normalized) - 2):
            trigram = normalized[i : i + 3]
            h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


# Paper model name aliases for convenience
EMBEDDING_MODELS = {
    "Qwen3-0.6B-Embedding": "Qwen/Qwen3-Embedding-0.6B",
    "Qwen3-8B-Embedding": "Qwen/Qwen3-Embedding-8B",
    "GTE-large": "thenlper/gte-large",
    "BGE-large": "BAAI/bge-large-en-v1.5",
    "gemini-embedding-001": "gemini-embedding-001",
    "gemini-embedding-2": "gemini-embedding-2-preview",
}


def resolve_embedding_model_name(name: str) -> str:
    """Resolve short alias to full HuggingFace model ID."""
    return EMBEDDING_MODELS.get(name, name)


# ---------------------------------------------------------------------------
# Gemini Embedding via Google GenAI API
# ---------------------------------------------------------------------------

# Gemini Embedding 2 rate limits by tier
# See: https://ai.google.dev/gemini-api/docs/rate-limits
GEMINI_TIERS = {
    "free":   {"rpm": 100,    "target_rpm": 80,     "workers": 3},
    "tier-1": {"rpm": 3_000,  "target_rpm": 2_500,  "workers": 25},
    "tier-2": {"rpm": 5_000,  "target_rpm": 4_000,  "workers": 40},
    "tier-3": {"rpm": 20_000, "target_rpm": 16_000,  "workers": 80},
}


def _load_gemini_tier() -> Optional[dict]:
    """Load Gemini tier from ``~/.konash/config.json``."""
    config_path = os.path.expanduser("~/.konash/config.json")
    try:
        if os.path.exists(config_path):
            import json as _json
            with open(config_path) as f:
                tier_name = _json.load(f).get("google_tier")
            if tier_name and tier_name in GEMINI_TIERS:
                return {"tier": tier_name, **GEMINI_TIERS[tier_name]}
    except Exception:
        pass
    return None


def load_gemini_embedding_model(
    model_name: str = "gemini-embedding-2-preview",
    api_key: Optional[str] = None,
    output_dimensionality: int = 768,
    batch_size: int = 100,
    max_workers: Optional[int] = None,
    target_rpm: Optional[int] = None,
) -> Callable[[List[str]], np.ndarray]:
    """Load a Gemini embedding model and return a callable.

    Parameters
    ----------
    model_name : str
        ``"gemini-embedding-001"`` (text-only) or
        ``"gemini-embedding-2-preview"`` (multimodal).
    api_key : str or None
        Google API key.  Falls back to ``GOOGLE_API_KEY`` env var.
    output_dimensionality : int
        Embedding vector size.  Recommended: 768, 1536, or 3072.
        Smaller = faster downstream search, minimal quality loss (MRL).
    batch_size : int
        Max texts per API call (default 100, Gemini max).
    max_workers : int or None
        Concurrent API requests.  Auto-detected from tier if ``None``.
    target_rpm : int or None
        Max requests per minute.  Auto-detected from tier if ``None``.
        Set to 0 to disable pacing.

    Returns
    -------
    callable
        ``(texts: list[str]) -> np.ndarray`` of shape ``(len(texts), dim)``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from google import genai
    from google.genai import types

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        # Check ~/.konash/config.json (set by `konash setup`)
        config_path = os.path.expanduser("~/.konash/config.json")
        if os.path.exists(config_path):
            import json as _json
            try:
                with open(config_path) as f:
                    key = _json.load(f).get("google_api_key")
            except Exception:
                pass
    if not key:
        raise ValueError(
            "No Google API key found. Run `konash setup` or set GOOGLE_API_KEY env var."
        )

    client = genai.Client(api_key=key)
    resolved = resolve_embedding_model_name(model_name)

    # Resolve tier → workers/rpm if not explicitly set
    if max_workers is None or target_rpm is None:
        tier = _load_gemini_tier()
        if tier is None:
            # Default to tier-1 (most common paid tier)
            tier = {"tier": "tier-1", **GEMINI_TIERS["tier-1"]}
        if max_workers is None:
            max_workers = tier["workers"]
        if target_rpm is None:
            target_rpm = tier["target_rpm"]
        logger.info(
            "Gemini tier: %s (RPM limit %d → target %d, %d workers)",
            tier["tier"], tier.get("rpm", "?"), target_rpm, max_workers,
        )
    else:
        logger.info(
            "Gemini embedding: model=%s, dim=%d, batch=%d, workers=%d, rpm=%d",
            resolved, output_dimensionality, batch_size, max_workers, target_rpm,
        )

    def _embed_batch(batch: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        import time as _time

        for attempt in range(8):
            try:
                result = client.models.embed_content(
                    model=resolved,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=output_dimensionality,
                    ),
                )
                vecs = np.array(
                    [e.values for e in result.embeddings], dtype=np.float32
                )
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                return vecs / norms
            except Exception as exc:
                if attempt < 7:
                    wait = min(2 ** attempt, 30)
                    logger.debug(
                        "Gemini embed attempt %d failed (%s), retry in %ds",
                        attempt + 1, type(exc).__name__, wait,
                    )
                    _time.sleep(wait)
                else:
                    raise

    # Token-bucket rate limiter: allows bursts up to max_workers but
    # sustains at most target_rpm requests per minute.
    import threading as _threading

    class _RateLimiter:
        """Sliding-window rate limiter using a token bucket."""

        def __init__(self, rpm: int):
            self._interval = 60.0 / rpm if rpm > 0 else 0.0
            self._lock = _threading.Lock()
            self._last = 0.0

        def wait(self) -> None:
            if self._interval <= 0:
                return
            import time as _t
            with self._lock:
                now = _t.monotonic()
                earliest = self._last + self._interval
                if now < earliest:
                    _t.sleep(earliest - now)
                self._last = _t.monotonic()

    _limiter = _RateLimiter(target_rpm)

    def _rate_limited_embed(batch: List[str], task_type: str) -> np.ndarray:
        _limiter.wait()
        return _embed_batch(batch, task_type)

    def _run_parallel(texts: List[str], task_type: str) -> np.ndarray:
        batches = [
            (i, texts[i : i + batch_size])
            for i in range(0, len(texts), batch_size)
        ]

        if len(batches) == 1:
            return _embed_batch(batches[0][1], task_type)

        results: Dict[int, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for idx, batch in batches:
                futures[pool.submit(_rate_limited_embed, batch, task_type)] = idx
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        ordered = [results[i] for i, _ in batches]
        return np.vstack(ordered)

    def embed_fn(texts: List[str]) -> np.ndarray:
        """Embed documents (for indexing)."""
        return _run_parallel(texts, "RETRIEVAL_DOCUMENT")

    def query_fn(texts: List[str]) -> np.ndarray:
        """Embed queries (for searching)."""
        return _run_parallel(texts, "RETRIEVAL_QUERY")

    # Attach query_fn so VectorSearchTool can use it
    embed_fn.query_fn = query_fn  # type: ignore[attr-defined]

    return embed_fn


class VectorSearchTool:
    """Embedded vector search tool for retrieval-augmented agent rollouts.

    Uses FAISS (if installed) for high-throughput inner-product search, with
    automatic fallback to a pure-numpy implementation.  All search is
    in-process — no client-server network I/O — matching the KARL paper's
    design for >500 QPS per host.

    Each worker process can instantiate its own index from shared storage
    via :meth:`from_cache`.

    Class Attributes
    ----------------
    embedded_index : optional pre-loaded index (numpy array or ``None``).
    target_qps_per_host : throughput target used for capacity planning.
    """

    embedded_index: Optional[np.ndarray] = None
    target_qps_per_host: int = 500

    def __init__(
        self,
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_faiss: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        embed_fn:
            A callable that maps a list of text strings to a 2-D numpy array
            of shape ``(n, dim)``.  If ``None`` and ``model_name`` is given,
            loads the specified HuggingFace embedding model.  If both are
            ``None``, the caller must supply pre-computed embeddings to
            :meth:`index`.
        model_name:
            HuggingFace model ID or short alias (e.g. ``"Qwen3-0.6B-Embedding"``).
            Resolved via :func:`resolve_embedding_model_name` and loaded
            via :func:`load_embedding_model`.
        cache_dir:
            Directory where serialised indexes can be stored / loaded.
        device:
            Device for the embedding model (``"cuda"``, ``"cpu"``, ``"auto"``).
        use_faiss:
            Force FAISS on/off.  ``None`` (default) auto-detects: uses FAISS
            when the library is installed, falls back to numpy otherwise.
        """
        if embed_fn is not None:
            self.embed_fn = embed_fn
        elif model_name is not None:
            resolved = resolve_embedding_model_name(model_name)
            self.embed_fn = load_embedding_model(resolved, device=device)
        else:
            self.embed_fn = None
        self.cache_dir = cache_dir

        # FAISS or numpy backend (lazy-loaded to avoid conflicts)
        if use_faiss is None:
            self._use_faiss = _ensure_faiss()
        else:
            if use_faiss and not _ensure_faiss():
                raise ImportError(
                    "use_faiss=True but faiss is not installed. "
                    "Install with: pip install faiss-cpu"
                )
            self._use_faiss = use_faiss

        # Internal state
        self._vectors: Optional[np.ndarray] = None  # (N, dim) normalised
        self._faiss_index: Optional[Any] = None      # faiss.IndexFlatIP
        self._documents: List[Dict[str, Any]] = []   # parallel metadata list
        self._index_id: Optional[str] = None

    # -- indexing -------------------------------------------------------------

    def index(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        text_key: str = "text",
        **kwargs: Any,
    ) -> None:
        """Build (or replace) the in-memory index.

        Parameters
        ----------
        documents:
            List of document dicts.  Each must contain at least a field
            named *text_key* with the text to embed.
        embeddings:
            Pre-computed embedding matrix of shape ``(len(documents), dim)``.
            If ``None``, ``self.embed_fn`` is used to compute embeddings
            from the text field.
        text_key:
            Key in each document dict that holds the text content.
        """
        if embeddings is not None:
            vecs = np.array(embeddings, dtype=np.float32)
        elif self.embed_fn is not None:
            texts = [doc[text_key] for doc in documents]
            vecs = np.array(self.embed_fn(texts), dtype=np.float32)
        else:
            raise ValueError(
                "Either pre-computed embeddings or an embed_fn must be provided."
            )

        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        self._vectors = _normalize(vecs)
        self._documents = list(documents)

        # Build FAISS index if available
        if self._use_faiss:
            dim = self._vectors.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            # FAISS expects contiguous float32
            vecs_c = np.ascontiguousarray(self._vectors, dtype=np.float32)
            self._faiss_index.add(vecs_c)
            logger.info(
                "Built FAISS IndexFlatIP: %d vectors, dim=%d",
                self._faiss_index.ntotal, dim,
            )

        # Compute a stable ID for cache keying
        content_hash = hashlib.sha256(
            json.dumps([d.get(text_key, "") for d in documents], sort_keys=True).encode()
        ).hexdigest()[:16]
        self._index_id = content_hash

        # Update the class-level reference for quick access
        self.embedded_index = self._vectors

    # -- searching ------------------------------------------------------------

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Return the *top_k* most similar documents to *query*.

        Parameters
        ----------
        query:
            Either a text string (requires ``embed_fn``) or a 1-D numpy
            embedding vector.
        top_k:
            Number of results to return.

        Returns
        -------
        A list of dicts, each containing the original document fields plus
        a ``"score"`` key with the cosine similarity.
        """
        if self._vectors is None or len(self._documents) == 0:
            return []

        query_vec = self._encode_query(query)

        # FAISS path: single-query inner-product search
        if self._use_faiss and self._faiss_index is not None:
            k = min(top_k, self._faiss_index.ntotal)
            query_mat = query_vec.reshape(1, -1).astype(np.float32)
            scores_arr, indices_arr = self._faiss_index.search(query_mat, k)
            results: List[Dict[str, Any]] = []
            for j in range(k):
                idx = int(indices_arr[0, j])
                if idx < 0:
                    continue
                result = dict(self._documents[idx])
                result["score"] = float(scores_arr[0, j])
                results.append(result)
            return results

        # NumPy fallback
        scores = self._vectors @ query_vec  # cosine similarity (vectors are normalised)

        k = min(top_k, len(scores))
        # argpartition is O(n) vs O(n log n) for full sort
        if k < len(scores):
            top_indices = np.argpartition(scores, -k)[-k:]
        else:
            top_indices = np.arange(len(scores))
        # Sort the top-k by descending score
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            result = dict(self._documents[idx])
            result["score"] = float(scores[idx])
            results.append(result)
        return results

    def batch_search(
        self,
        queries: Sequence[Union[str, np.ndarray]],
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[List[Dict[str, Any]]]:
        """Run multiple searches in a single vectorised pass.

        Parameters
        ----------
        queries:
            A sequence of query strings or embedding vectors.
        top_k:
            Number of results per query.

        Returns
        -------
        A list of result lists, one per query.
        """
        if self._vectors is None or len(self._documents) == 0:
            return [[] for _ in queries]

        # Encode all queries into a matrix (Q, dim)
        query_vecs = np.stack([self._encode_query(q) for q in queries])  # (Q, dim)

        # FAISS path: batch inner-product search
        if self._use_faiss and self._faiss_index is not None:
            k = min(top_k, self._faiss_index.ntotal)
            query_mat = np.ascontiguousarray(query_vecs, dtype=np.float32)
            all_scores, all_indices = self._faiss_index.search(query_mat, k)

            results: List[List[Dict[str, Any]]] = []
            for q_idx in range(len(queries)):
                query_results: List[Dict[str, Any]] = []
                for j in range(k):
                    idx = int(all_indices[q_idx, j])
                    if idx < 0:
                        continue
                    result = dict(self._documents[idx])
                    result["score"] = float(all_scores[q_idx, j])
                    query_results.append(result)
                results.append(query_results)
            return results

        # NumPy fallback
        all_scores_np = query_vecs @ self._vectors.T

        results = []
        for q_idx in range(len(queries)):
            scores = all_scores_np[q_idx]
            k = min(top_k, len(scores))
            if k < len(scores):
                top_indices = np.argpartition(scores, -k)[-k:]
            else:
                top_indices = np.arange(len(scores))
            top_indices = top_indices[np.argsort(-scores[top_indices])]

            query_results = []
            for idx in top_indices:
                result = dict(self._documents[idx])
                result["score"] = float(scores[idx])
                query_results.append(result)
            results.append(query_results)
        return results

    # -- cache management -----------------------------------------------------

    def load_cached_index(
        self,
        index_id: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Attempt to load a previously saved index from disk.

        Tries FAISS format first (``*.faiss`` + ``*.meta.npz``), then falls
        back to the legacy numpy-only format (``*.npz``).

        Parameters
        ----------
        index_id:
            Identifier for the cached index.  Used to construct the
            filename inside ``self.cache_dir``.
        path:
            Explicit path to an index file.  Overrides *index_id*.

        Returns
        -------
        ``True`` if the index was loaded successfully, ``False`` otherwise.
        """
        # Resolve base path (without extension) for multi-file FAISS cache
        if path is not None:
            base = Path(path).with_suffix("")
        elif index_id is not None and self.cache_dir is not None:
            base = Path(self.cache_dir) / index_id
        else:
            return False

        # Try FAISS format first
        faiss_path = base.with_suffix(".faiss")
        meta_path = base.with_suffix(".meta.npz")
        if self._use_faiss and faiss_path.exists() and meta_path.exists():
            try:
                self._faiss_index = faiss.read_index(str(faiss_path))
                meta = np.load(str(meta_path), allow_pickle=True)
                self._documents = meta["documents"].tolist()
                self._vectors = meta.get("vectors")
                if self._vectors is not None:
                    self._vectors = self._vectors.astype(np.float32)
                self._index_id = index_id or base.stem
                self.embedded_index = self._vectors
                logger.info(
                    "Loaded FAISS index from cache: %d vectors",
                    self._faiss_index.ntotal,
                )
                return True
            except Exception as exc:
                logger.debug("FAISS cache load failed: %s", exc)

        # Fall back to legacy numpy format
        npz_path = base.with_suffix(".npz")
        if path is not None:
            npz_path = Path(path)

        if not npz_path.exists():
            return False

        try:
            data = np.load(str(npz_path), allow_pickle=True)
            self._vectors = data["vectors"].astype(np.float32)
            self._documents = data["documents"].tolist()
            self._index_id = index_id or npz_path.stem
            self.embedded_index = self._vectors

            # Rebuild FAISS index from loaded vectors
            if self._use_faiss:
                dim = self._vectors.shape[1]
                self._faiss_index = faiss.IndexFlatIP(dim)
                vecs_c = np.ascontiguousarray(self._vectors, dtype=np.float32)
                self._faiss_index.add(vecs_c)

            return True
        except Exception:
            return False

    def save_index(self, path: Optional[str] = None) -> Optional[str]:
        """Persist the current index to disk.

        When FAISS is active, saves the FAISS index (``*.faiss``) and
        document metadata (``*.meta.npz``) separately.  Also saves the
        legacy ``*.npz`` format for numpy-only environments.

        Returns the path written, or ``None`` if there is nothing to save.
        """
        if self._vectors is None:
            return None

        if path is not None:
            base = Path(path).with_suffix("")
        elif self.cache_dir is not None and self._index_id is not None:
            base = Path(self.cache_dir) / self._index_id
        else:
            return None

        base.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index + metadata
        if self._use_faiss and self._faiss_index is not None:
            faiss_path = base.with_suffix(".faiss")
            meta_path = base.with_suffix(".meta.npz")
            faiss.write_index(self._faiss_index, str(faiss_path))
            np.savez(
                str(meta_path),
                documents=np.array(self._documents, dtype=object),
                vectors=self._vectors,
            )
            logger.info("Saved FAISS index to %s", faiss_path)

        # Always save legacy numpy format for compatibility
        npz_path = base.with_suffix(".npz")
        np.savez(
            str(npz_path),
            vectors=self._vectors,
            documents=np.array(self._documents, dtype=object),
        )
        return str(npz_path)

    @classmethod
    def from_cache(
        cls,
        path: str,
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_faiss: Optional[bool] = None,
    ) -> "VectorSearchTool":
        """Instantiate a search tool from a previously cached index.

        This is the intended entry point for per-worker instantiation:
        the corpus is processed offline, indexed, and cached to shared
        storage.  Each worker then calls ``from_cache()`` to load its
        own in-process copy — no network I/O required.

        Parameters
        ----------
        path:
            Path to the cached index (FAISS or npz file).
        embed_fn:
            Embedding function for runtime queries.
        model_name:
            HuggingFace model name (resolved if ``embed_fn`` is None).
        device:
            Device for the embedding model.
        use_faiss:
            Force FAISS on/off.

        Raises
        ------
        FileNotFoundError
            If no cached index is found at *path*.
        """
        tool = cls(
            embed_fn=embed_fn,
            model_name=model_name,
            device=device,
            cache_dir=str(Path(path).parent),
            use_faiss=use_faiss,
        )
        if not tool.load_cached_index(path=path):
            raise FileNotFoundError(
                f"No cached index found at {path}. "
                "Run indexing first and save with save_index()."
            )
        return tool

    # -- internal helpers -----------------------------------------------------

    def _encode_query(self, query: Union[str, np.ndarray]) -> np.ndarray:
        """Convert a query to a normalised 1-D embedding vector.

        Uses ``embed_fn.query_fn`` (RETRIEVAL_QUERY task type) when available
        for better search quality with Gemini embeddings.
        """
        if isinstance(query, str):
            if self.embed_fn is None:
                raise ValueError(
                    "Cannot search with a text query without an embed_fn."
                )
            # Prefer query-optimized embedding when available (Gemini)
            qfn = getattr(self.embed_fn, "query_fn", None)
            fn = qfn if qfn is not None else self.embed_fn
            vec = np.array(fn([query]), dtype=np.float32).flatten()
        else:
            vec = np.array(query, dtype=np.float32).flatten()

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class RetrievalBudgetPolicy:
    """Policy that decides how many chunks to retrieve (top_k) based on a
    target token budget and the average chunk length.

    The idea is that shorter chunks allow more of them to fit within the
    token budget, so ``top_k`` scales inversely with average chunk length.

    Class Attributes
    ----------------
    max_top_k : hard cap on ``top_k`` (default 20).
    target_token_budget : the total number of tokens we want retrieval
        results to fit within.  ``None`` means the policy must be
        configured at init time.
    """

    max_top_k: int = 20
    target_token_budget: Optional[int] = None

    def __init__(
        self,
        target_token_budget: Optional[int] = None,
        max_top_k: Optional[int] = None,
    ) -> None:
        if target_token_budget is not None:
            self.target_token_budget = target_token_budget
        if max_top_k is not None:
            self.max_top_k = max_top_k

    def compute_top_k(
        self,
        avg_chunk_length: float = 200.0,
        **kwargs: Any,
    ) -> int:
        """Compute the optimal ``top_k`` for a given average chunk length.

        The formula is::

            top_k = min(max_top_k, floor(target_token_budget / avg_chunk_length))

        This ensures that ``top_k * avg_chunk_length <= target_token_budget``
        while never exceeding ``max_top_k``.

        Parameters
        ----------
        avg_chunk_length:
            Average number of tokens per chunk in the corpus.

        Returns
        -------
        An integer ``top_k`` value, always >= 1.
        """
        if self.target_token_budget is None:
            return self.max_top_k

        if avg_chunk_length <= 0:
            return self.max_top_k

        raw_k = int(self.target_token_budget / avg_chunk_length)
        return max(1, min(raw_k, self.max_top_k))
