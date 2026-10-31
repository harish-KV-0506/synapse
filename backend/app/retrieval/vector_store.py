"""Vector index: FAISS if installed, otherwise exact numpy cosine search."""
from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self):
        self.matrix: np.ndarray | None = None
        self._faiss = None
        self._index = None
        try:
            import faiss  # type: ignore

            self._faiss = faiss
        except Exception:
            pass

    def build(self, vectors: np.ndarray) -> None:
        self.matrix = np.ascontiguousarray(vectors.astype(np.float32))
        if self._faiss is not None and len(vectors):
            self._index = self._faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(self.matrix)

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.matrix is None or len(self.matrix) == 0:
            return []
        k = min(k, len(self.matrix))
        q = qvec.reshape(1, -1).astype(np.float32)
        if self._index is not None:
            sims, idx = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(idx[0], sims[0]) if i >= 0]
        sims = self.matrix @ q[0]
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]
