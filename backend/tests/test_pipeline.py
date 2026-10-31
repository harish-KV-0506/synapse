import json
from pathlib import Path

from app.agents.workflow import run_workflow
from app.evaluation.metrics import citation_metrics, retrieval_metrics
from app.retrieval.hybrid import rrf

Q = "How can machine learning improve satellite communication link performance under heavy rainfall?"
GT = Path(__file__).resolve().parents[2] / "eval" / "ground_truth.json"


def test_ingestion_keeps_title_authors_year_and_pages(loaded_corpus):
    assert len(loaded_corpus.papers) == 5
    a = next(p for p in loaded_corpus.papers.values() if p.filename == "demo_lstm_rain_fade.pdf")
    assert a.title.startswith("Demo Study A") and a.authors == ["A. Rao", "B. Iyer"] and a.year == 2021
    assert all(c.page >= 1 for c in loaded_corpus.chunks)


def test_rrf_rewards_agreement_between_retrievers():
    fused = rrf({"bm25": [1, 2, 3], "dense": [3, 1, 4]})
    assert max(fused, key=fused.get) == 1


def test_retrieval_quality_on_ground_truth(loaded_corpus):
    m = retrieval_metrics(loaded_corpus, json.loads(GT.read_text()), k=5)
    assert m["recall@5"] >= 0.8 and m["mrr"] >= 0.6, m


def test_workflow_covers_all_relevant_papers_and_every_citation_resolves(loaded_corpus):
    r = run_workflow(Q)
    assert len(r.profiles) == 5, "a relevant paper was dropped"
    assert [t.agent for t in r.trace] == ["Research Director", "Literature Agent", "Methodology Agent",
                                          "Evidence Agent", "Research Gap Agent", "Report Writer"]
    cm = citation_metrics(r, loaded_corpus)
    assert cm["citations"] > 20 and cm["valid_location_rate"] == 1.0 and cm["span_faithfulness"] == 1.0
    import re
    used = set(re.findall(r"\[(S\d+)\]", r.literature_review))
    assert used and used <= set(r.sources), "review cites a source that does not exist"


def test_gap_agent_flags_author_acknowledged_gaps_with_evidence(loaded_corpus):
    r = run_workflow(Q)
    by = {g.title: g for g in r.gaps}
    for title in ("Generalisation to unseen sites/domains", "Extreme / peak events"):
        assert title in by, f"{title} should be a gap (most papers say it was not evaluated)"
        assert by[title].supporting_evidence, "gap must carry verified source evidence"
    assert "Uncertainty quantification" not in {g.title for g in r.gaps} or by["Uncertainty quantification"].coverage["addressed"] >= 1
    # Bayesian *optimisation* (hyper-parameter tuning) must not count as uncertainty quantification
    xgb = next(i for i, t in enumerate(r.gap_matrix.papers) if "Gradient Boosting" in t)
    ui = r.gap_matrix.aspects.index("Uncertainty quantification")
    assert r.gap_matrix.cells[xgb][ui] == "not_found"


def test_llm_synthesis_cannot_smuggle_in_invented_citations(loaded_corpus, monkeypatch):
    from app.agents import report as rep

    class FakeLLM:
        available = True
        def complete(self, system, prompt, max_tokens=0):
            return "LSTM beats the baseline [S1]. Everything works everywhere [S999, S1]. Unsourced claim [S404]."

    monkeypatch.setattr(rep, "llm", FakeLLM())
    r = run_workflow(Q)  # heuristic profiles, then force LLM synthesis path
    reg = rep.SourceRegistry()
    prof = r.profiles
    reg.cite(prof[0].objective)
    text = rep._llm_review(Q, prof, reg)
    assert "S999" not in text and "S404" not in text and "S1" in text


def test_markdown_export_contains_every_required_evidence_field(loaded_corpus):
    from app.export import report_to_markdown
    md = report_to_markdown(run_workflow(Q))
    for needle in ("# Research Intelligence Report", "## Research gap matrix", "## Evidence ledger", "## Sources",
                   "SUPPORTED", "p.1", "Rao et al., 2021", "Limitations of this analysis"):
        assert needle in md, needle
