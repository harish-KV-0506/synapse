"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion, then reranked."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import settings
from ..schemas import Chunk, Paper, RetrievedChunk
from .bm25 import BM25
from .embeddings import get_embedder
from .reranker import get_reranker
from .vector_store import VectorStore

RRF_K = 60


def rrf(rankings: dict[str, list[int]], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranks in rankings.values():
        for r, idx in enumerate(ranks):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + r + 1)
    return fused


class Corpus:
    """Holds papers + chunks + indices. Persists to JSON; indices are rebuilt on load."""

    def __init__(self):
        self.papers: dict[str, Paper] = {}
        self.chunks: list[Chunk] = []
        self.bm25 = BM25()
        self.vs = VectorStore()
        self._dirty = True

    # ---- mutation -------------------------------------------------------
    def add(self, paper: Paper, chunks: list[Chunk]) -> None:
        self.papers[paper.paper_id] = paper
        self.chunks = [c for c in self.chunks if c.paper_id != paper.paper_id] + chunks
        self._dirty = True

    def remove(self, paper_id: str) -> None:
        self.papers.pop(paper_id, None)
        self.chunks = [c for c in self.chunks if c.paper_id != paper_id]
        self._dirty = True

    def clear(self) -> None:
        self.papers, self.chunks, self._dirty = {}, [], True

    # ---- persistence ----------------------------------------------------
    def save(self, path: Path | None = None) -> None:
        path = path or settings.index_path
        path.write_text(json.dumps({
            "papers": [p.model_dump() for p in self.papers.values()],
            "chunks": [c.model_dump() for c in self.chunks],
        }))

    def load(self, path: Path | None = None) -> None:
        path = path or settings.index_path
        if not path.exists():
            return
        d = json.loads(path.read_text())
        self.papers = {p["paper_id"]: Paper(**p) for p in d["papers"]}
        self.chunks = [Chunk(**c) for c in d["chunks"]]
        self._dirty = True

    # ---- indexing -------------------------------------------------------
    def build(self) -> None:
        if not self._dirty:
            return
        texts = [c.text for c in self.chunks]
        self.bm25 = BM25().fit(texts)
        if texts:
            self.vs.build(get_embedder().encode(texts))
        self._dirty = False

    # ---- search ---------------------------------------------------------
    def search(self, query: str, k: int | None = None, paper_ids: list[str] | None = None,
               rerank: bool = True) -> list[RetrievedChunk]:
        self.build()
        if not self.chunks:
            return []
        k = k or settings.top_k_final
        allowed = set(paper_ids) if paper_ids else None

        bm = [(i, s) for i, s in self.bm25.top_k(query, settings.top_k_bm25 * 3)
              if allowed is None or self.chunks[i].paper_id in allowed][: settings.top_k_bm25]
        qv = get_embedder().encode([query])[0]
        dn = [(i, s) for i, s in self.vs.search(qv, settings.top_k_dense * 3)
              if allowed is None or self.chunks[i].paper_id in allowed][: settings.top_k_dense]

        fused = rrf({"bm25": [i for i, _ in bm], "dense": [i for i, _ in dn]})
        bm_s, dn_s = dict(bm), dict(dn)
        cand = sorted(fused, key=lambda i: fused[i], reverse=True)[: max(k * 3, 20)]
        if not cand:
            return []

        if rerank:
            rr = get_reranker()
            rscores = rr.score(query, [self.chunks[i].text for i in cand])
            order = sorted(zip(cand, rscores), key=lambda t: t[1], reverse=True)[:k]
            return [RetrievedChunk(chunk=self.chunks[i], score=float(s),
                                   sources={"rrf": fused[i], "bm25": bm_s.get(i, 0.0), "dense": dn_s.get(i, 0.0), "rerank": float(s)})
                    for i, s in order]
        return [RetrievedChunk(chunk=self.chunks[i], score=fused[i],
                               sources={"rrf": fused[i], "bm25": bm_s.get(i, 0.0), "dense": dn_s.get(i, 0.0)})
                for i in cand[:k]]

    def chunks_for_paper(self, paper_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.paper_id == paper_id]


corpus = Corpus()
