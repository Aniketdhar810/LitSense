"""Exact inner-product search over L2-normalized vectors (CLAUDE.md s10.2).

FAISS ``IndexFlatIP`` when faiss is importable, else ``matrix @ q`` with ``argpartition``; both
return the same ids. At ~20K x 384 either costs a few milliseconds.
"""

import numpy as np

try:  # faiss-cpu is a core dependency, but the numpy path keeps tests and odd hosts working
    import faiss
except ImportError:  # pragma: no cover - exercised only where faiss is missing
    faiss = None


class VectorIndex:
    """Search ``matrix`` rows (float32, L2-normalized by the caller) by inner product."""

    def __init__(self, matrix: np.ndarray, use_faiss: bool = True) -> None:
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self._faiss = None
        if use_faiss and faiss is not None and len(self.matrix):
            self._faiss = faiss.IndexFlatIP(self.matrix.shape[1])
            self._faiss.add(self.matrix)

    def __len__(self) -> int:
        return len(self.matrix)

    def search(
        self,
        q: np.ndarray,
        k: int,
        exclude: np.ndarray | set[int] | None = None,
        allow: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top ``k`` (ids, scores), best first. ``exclude``: ids never returned; ``allow``:
        boolean mask over rows, only those may be returned."""
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        banned = np.zeros(len(self.matrix), dtype=bool)
        if exclude is not None and len(exclude):
            banned[np.fromiter(exclude, dtype=np.int64)] = True
        if allow is not None:
            banned |= ~allow
        k = min(k, int((~banned).sum()))
        if k <= 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        if self._faiss is not None and allow is None:
            fetch = min(len(self.matrix), k + int(banned.sum()))
            scores, ids = self._faiss.search(q[None], fetch)
            keep = ~banned[ids[0]]
            return ids[0][keep][:k].astype(np.int64), scores[0][keep][:k]
        scores = self.matrix @ q
        scores[banned] = -np.inf
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return top.astype(np.int64), scores[top]
