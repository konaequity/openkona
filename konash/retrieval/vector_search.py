from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

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
        logger.warning(
            "Could not load embedding model %r (install sentence-transformers "
            "or transformers+torch). Falling back to trigram pseudo-embeddings.",
            model_name,
        )
        embed_fn = _trigram_embed_fn

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
}


def resolve_embedding_model_name(name: str) -> str:
    """Resolve short alias to full HuggingFace model ID."""
    return EMBEDDING_MODELS.get(name, name)


class VectorSearchTool:
    """Numpy-based vector search tool for retrieval-augmented agent rollouts.

    The tool maintains an in-memory matrix of document embeddings and
    performs cosine-similarity search entirely with numpy -- no torch or
    external vector-DB dependency.

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
        """
        if embed_fn is not None:
            self.embed_fn = embed_fn
        elif model_name is not None:
            resolved = resolve_embedding_model_name(model_name)
            self.embed_fn = load_embedding_model(resolved, device=device)
        else:
            self.embed_fn = None
        self.cache_dir = cache_dir

        # Internal state
        self._vectors: Optional[np.ndarray] = None  # (N, dim) normalised
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
        scores = self._vectors @ query_vec  # cosine similarity (vectors are normalised)

        k = min(top_k, len(scores))
        # argpartition is O(n) vs O(n log n) for full sort
        if k < len(scores):
            top_indices = np.argpartition(scores, -k)[-k:]
        else:
            top_indices = np.arange(len(scores))
        # Sort the top-k by descending score
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[Dict[str, Any]] = []
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
        # Compute all similarities in one matmul: (Q, N)
        all_scores = query_vecs @ self._vectors.T

        results: List[List[Dict[str, Any]]] = []
        for q_idx in range(len(queries)):
            scores = all_scores[q_idx]
            k = min(top_k, len(scores))
            if k < len(scores):
                top_indices = np.argpartition(scores, -k)[-k:]
            else:
                top_indices = np.arange(len(scores))
            top_indices = top_indices[np.argsort(-scores[top_indices])]

            query_results: List[Dict[str, Any]] = []
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

        Parameters
        ----------
        index_id:
            Identifier for the cached index.  Used to construct the
            filename inside ``self.cache_dir``.
        path:
            Explicit path to an ``.npz`` file.  Overrides *index_id*.

        Returns
        -------
        ``True`` if the index was loaded successfully, ``False`` otherwise.
        """
        if path is not None:
            target = Path(path)
        elif index_id is not None and self.cache_dir is not None:
            target = Path(self.cache_dir) / f"{index_id}.npz"
        else:
            return False

        if not target.exists():
            return False

        try:
            data = np.load(str(target), allow_pickle=True)
            self._vectors = data["vectors"].astype(np.float32)
            self._documents = data["documents"].tolist()
            self._index_id = index_id or target.stem
            self.embedded_index = self._vectors
            return True
        except Exception:
            return False

    def save_index(self, path: Optional[str] = None) -> Optional[str]:
        """Persist the current index to disk.

        Returns the path written, or ``None`` if there is nothing to save.
        """
        if self._vectors is None:
            return None

        if path is not None:
            target = Path(path)
        elif self.cache_dir is not None and self._index_id is not None:
            target = Path(self.cache_dir) / f"{self._index_id}.npz"
        else:
            return None

        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(target),
            vectors=self._vectors,
            documents=np.array(self._documents, dtype=object),
        )
        return str(target)

    # -- internal helpers -----------------------------------------------------

    def _encode_query(self, query: Union[str, np.ndarray]) -> np.ndarray:
        """Convert a query to a normalised 1-D embedding vector."""
        if isinstance(query, str):
            if self.embed_fn is None:
                raise ValueError(
                    "Cannot search with a text query without an embed_fn."
                )
            vec = np.array(self.embed_fn([query]), dtype=np.float32).flatten()
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
