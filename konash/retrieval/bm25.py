"""Lightweight BM25 scorer for hybrid search (no external dependencies)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


def _tokenize(text: str) -> List[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25 ranking over a list of documents.

    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter (default 1.5).
    b : float
        Length normalisation parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs: List[Counter] = []
        self._doc_lens: List[int] = []
        self._avg_dl: float = 0.0
        self._idf: Dict[str, float] = {}
        self._n_docs: int = 0
        self._documents: List[Dict[str, Any]] = []

    def index(self, documents: Sequence[Dict[str, Any]], text_key: str = "text") -> None:
        """Build the BM25 index from a list of document dicts."""
        self._documents = list(documents)
        self._n_docs = len(documents)

        # Tokenise and count term frequencies per document
        self._doc_freqs = []
        self._doc_lens = []
        df: Counter = Counter()  # document frequency per term

        for doc in documents:
            tokens = _tokenize(doc.get(text_key, ""))
            tf = Counter(tokens)
            self._doc_freqs.append(tf)
            self._doc_lens.append(len(tokens))
            # Each unique term in this doc increments its df
            for term in tf:
                df[term] += 1

        self._avg_dl = sum(self._doc_lens) / max(self._n_docs, 1)

        # Pre-compute IDF using the standard BM25 formula
        self._idf = {}
        for term, freq in df.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            self._idf[term] = math.log(
                (self._n_docs - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document against query tokens."""
        tf = self._doc_freqs[doc_idx]
        dl = self._doc_lens[doc_idx]
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            f = tf[term]
            idf = self._idf.get(term, 0.0)
            # BM25 term score
            num = f * (self.k1 + 1.0)
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / self._avg_dl)
            score += idf * num / denom
        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return top-k documents by BM25 score."""
        if not self._documents:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        # Score all documents
        scores = [self._score_doc(tokens, i) for i in range(self._n_docs)]

        # Get top-k indices
        k = min(top_k, self._n_docs)
        # Simple sort (fast enough for <100k docs)
        ranked = sorted(range(self._n_docs), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            result = dict(self._documents[idx])
            result["score"] = scores[idx]
            results.append(result)
        return results

    def batch_search(
        self,
        queries: Sequence[str],
        top_k: int = 10,
    ) -> List[List[Dict[str, Any]]]:
        """Run multiple BM25 searches."""
        return [self.search(q, top_k=top_k) for q in queries]
