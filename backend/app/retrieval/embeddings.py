"""Dense embeddings.

Primary: Sentence Transformers (all-MiniLM-L6-v2 by default).
Fallback: signed feature-hashing of unigrams+bigrams (deterministic, offline, no model download)
so the full pipeline and tests run anywhere. Swap is automatic.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np

from ..config import settings
from .text import tokenize

log = logging.getLogger("synapse.embeddings")


class HashingEmbedder:
    name = "hashing-ngram"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _h(self, tok: str) -> tuple[int, float]:
        d = hashlib.md5(tok.encode()).digest()
        return int.from_bytes(d[:4], "little") % self.dim, 1.0 if d[4] & 1 else -1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            toks = tokenize(t)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                i, s = self._h(f)
                out[r, i] += s * (1.0 if "_" not in f else 0.7)
            n = np.linalg.norm(out[r])
            if n > 0:
                out[r] /= n
        return out


class STEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    if settings.use_neural:
        try:
            _embedder = STEmbedder(settings.embedding_model)
            log.info("Using sentence-transformers embedder %s", settings.embedding_model)
            return _embedder
        except Exception as e:  # missing lib, no network, etc.
            log.warning("Neural embedder unavailable (%s); using hashing fallback.", e)
    _embedder = HashingEmbedder()
    return _embedder


def reset_embedder():
    global _embedder
    _embedder = None
