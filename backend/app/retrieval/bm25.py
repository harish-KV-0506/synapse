"""Dependency-free Okapi BM25."""
from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def fit(self, texts: list[str]) -> "BM25":
        self.doc_tf, self.doc_len, self.df = [], [], Counter()
        for t in texts:
            toks = tokenize(t)
            tf = Counter(toks)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            self.df.update(tf.keys())
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        return self

    def idf(self, term: str) -> float:
        n = len(self.doc_tf)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * len(self.doc_tf)
        for term in set(q):
            idf = self.idf(term)
            for i, tf in enumerate(self.doc_tf):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        sc = self.scores(query)
        idx = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:k]
        return [(i, sc[i]) for i in idx if sc[i] > 0]
