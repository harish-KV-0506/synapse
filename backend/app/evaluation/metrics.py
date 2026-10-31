"""Evaluation: retrieval quality (Recall@k, MRR, nDCG) and citation/answer quality (citation validity, span
faithfulness). RAGAS can be layered on top (see README) when an LLM key is available; these metrics need none.

Ground-truth format (JSON): [{"question": "...", "relevant": [{"paper_filename": "x.pdf", "must_contain": "phrase"}]}]
A retrieved chunk is relevant if it comes from that paper and contains the phrase (case-insensitive).
"""
from __future__ import annotations

import math

from ..retrieval.hybrid import Corpus
from ..retrieval.text import normalize
from ..schemas import ResearchReport
from ..verification.span_match import span_score


def _is_rel(chunk, paper_filename: str, phrase: str, corpus: Corpus) -> bool:
    return corpus.papers[chunk.paper_id].filename == paper_filename and normalize(phrase) in normalize(chunk.text)


def retrieval_metrics(corpus: Corpus, gt: list[dict], k: int = 5, rerank: bool = True) -> dict:
    rec, mrr, ndcg = [], [], []
    for item in gt:
        hits = corpus.search(item["question"], k=k, rerank=rerank)
        rel = item["relevant"]
        flags = [any(_is_rel(h.chunk, r["paper_filename"], r["must_contain"], corpus) for r in rel) for h in hits]
        found = sum(any(_is_rel(h.chunk, r["paper_filename"], r["must_contain"], corpus) for h in hits) for r in rel)
        rec.append(found / len(rel))
        mrr.append(next((1 / (i + 1) for i, f in enumerate(flags) if f), 0.0))
        dcg = sum(f / math.log2(i + 2) for i, f in enumerate(flags))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
        ndcg.append(dcg / idcg if idcg else 0.0)
    n = max(1, len(gt))
    return {f"recall@{k}": round(sum(rec) / n, 3), "mrr": round(sum(mrr) / n, 3), f"ndcg@{k}": round(sum(ndcg) / n, 3), "n": len(gt)}


def ablation(corpus: Corpus, gt: list[dict], k: int = 5) -> dict:
    """Compare retrieval configurations: hybrid vs hybrid+rerank (and show BM25-only / dense-only contribution)."""
    from ..config import settings
    out = {"hybrid": retrieval_metrics(corpus, gt, k, rerank=False), "hybrid+rerank": retrieval_metrics(corpus, gt, k, rerank=True)}
    old = (settings.top_k_dense, settings.top_k_bm25)
    try:
        settings.top_k_dense = 0
        out["bm25_only"] = retrieval_metrics(corpus, gt, k, rerank=False)
        settings.top_k_dense, settings.top_k_bm25 = old[0], 0
        out["dense_only"] = retrieval_metrics(corpus, gt, k, rerank=False)
    finally:
        settings.top_k_dense, settings.top_k_bm25 = old
    return out


def citation_metrics(report: ResearchReport, corpus: Corpus) -> dict:
    """Every citation must (a) point to a real paper+page and (b) have a quote present on that page."""
    total = valid_loc = faithful = 0
    for sp in report.sources.values():
        total += 1
        page_text = " ".join(c.text for c in corpus.chunks if c.paper_id == sp.paper_id and c.page == sp.page)
        if sp.paper_id in corpus.papers and page_text:
            valid_loc += 1
            if span_score(sp.quote, page_text) >= 0.95:
                faithful += 1
    return {"citations": total, "valid_location_rate": round(valid_loc / total, 3) if total else None,
            "span_faithfulness": round(faithful / total, 3) if total else None,
            "claims_unverified": report.verification_summary.get("unverified", 0)}
