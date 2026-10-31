"""Literature Agent: retrieves and ranks papers/passages for the question from the uploaded corpus."""
from __future__ import annotations

from collections import defaultdict

from ..config import settings
from ..retrieval.hybrid import corpus
from ..schemas import RetrievedChunk
from .state import WorkflowState, traced


def literature_agent(state: WorkflowState) -> WorkflowState:
    with traced(state, "Literature Agent") as t:
        if not corpus.papers:
            state["retrieved"], state["ranked_papers"] = {}, []
            state.setdefault("limitations", []).append("No papers have been uploaded.")
            t["summary"] = "Corpus is empty."
            return state
        # Search wider than the display top-k so a relevant paper is not lost to chunk-level competition.
        k = max((state.get("top_k") or 8) * 3, 24)
        best: dict[str, dict[str, RetrievedChunk]] = defaultdict(dict)
        paper_score: dict[str, float] = defaultdict(float)
        for qi, q in enumerate(state["sub_queries"]):
            weight = 1.0 if qi == 0 else 0.5  # the user's own question counts most
            for hit in corpus.search(q, k=k, paper_ids=state.get("paper_ids")):
                cid = hit.chunk.chunk_id
                pid = hit.chunk.paper_id
                if cid not in best[pid] or hit.score > best[pid][cid].score:
                    best[pid][cid] = hit
                paper_score[pid] += weight * max(hit.sources.get("rrf", 0.0), 0.0)
        ranked = sorted(paper_score, key=lambda p: paper_score[p], reverse=True)
        if ranked:
            floor = settings.paper_relevance_floor * paper_score[ranked[0]]
            ranked = [p for p in ranked if paper_score[p] >= floor][: settings.max_papers]
        state["ranked_papers"] = ranked
        state["retrieved"] = {p: sorted(best[p].values(), key=lambda h: h.score, reverse=True) for p in ranked}
        n_pass = sum(len(v) for v in state["retrieved"].values())
        t["summary"] = f"Ranked {len(ranked)} papers; {n_pass} relevant passages retrieved via BM25+dense+RRF+rerank."
        if len(ranked) < 3:
            state.setdefault("limitations", []).append(
                f"Only {len(ranked)} paper(s) matched; conclusions about gaps are weak with such a small corpus.")
    return state
