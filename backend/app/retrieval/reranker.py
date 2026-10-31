"""Reranking: cross-encoder when available, otherwise a lexical-proximity scorer."""
from __future__ import annotations

import logging

from ..config import settings
from .text import tokenize

log = logging.getLogger("synapse.reranker")


class LexicalReranker:
    """Fallback: query-term coverage + bigram overlap, length-normalised."""
    name = "lexical-coverage"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = tokenize(query)
        qset = set(q)
        qbi = set(zip(q, q[1:]))
        out = []
        for p in passages:
            t = tokenize(p)
            if not t:
                out.append(0.0)
                continue
            tset = set(t)
            cov = len(qset & tset) / max(1, len(qset))
            bi = len(qbi & set(zip(t, t[1:]))) / max(1, len(qbi))
            out.append(0.7 * cov + 0.3 * bi)
        return out


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.name = model_name

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, p) for p in passages])]


_reranker = None


def get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    if settings.use_neural:
        try:
            _reranker = CrossEncoderReranker(settings.reranker_model)
            return _reranker
        except Exception as e:
            log.warning("Cross-encoder unavailable (%s); using lexical reranker.", e)
    _reranker = LexicalReranker()
    return _reranker


def reset_reranker():
    global _reranker
    _reranker = None
