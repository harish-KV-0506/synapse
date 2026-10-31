"""Research Gap Agent: coverage matrix (paper x research aspect) and evidence-linked gap proposals.

"not_found" means the aspect was not detected in the *uploaded corpus*; it does not prove the topic is
unexplored in the wider literature. Reports state this explicitly.
"""
from __future__ import annotations

import re

from ..llm import llm
from ..retrieval.hybrid import corpus
from ..retrieval.text import normalize
from ..schemas import GapMatrix, ResearchGap
from ..verification.span_match import split_sentences, verify_claim
from .state import WorkflowState, traced

# aspect -> (regex, evidence to investigate, proposed direction template)
ASPECTS: dict[str, tuple[str, list[str], str]] = {
    "Generalisation to unseen sites/domains": (
        r"unseen (?:site|location|station|domain)|held-?out (?:site|location|station)|cross-(?:site|location|region|domain)|"
        r"leave-one-(?:site|location)-out|transfer(?:ability)?|generali[sz]\w+ (?:to|across|beyond)",
        ["Differences in conditions between training and deployment locations", "Dataset coverage across zones/regions",
         "Performance drop on held-out sites vs. random splits", "Generalisation beyond training locations"],
        "Design a site-aware model and evaluate it with leave-one-site-out / held-out-region splits."),
    "Extreme / peak events": (
        r"extreme|peak (?:event|attenuation|error)|rare event|heavy[- ]rain|tail|outlier|worst[- ]case|high[- ]attenuation",
        ["Error conditioned on the top percentile of events", "Class imbalance handling for rare events",
         "Event-level (not just average) metrics", "Performance under extreme conditions"],
        "Evaluate with event-level and tail-conditioned metrics, and train with imbalance-aware objectives."),
    "Uncertainty quantification": (
        r"uncertaint\w+|confidence interval|probabilistic|prediction interval|calibrat\w+|bayesian (?:neural|inference|deep|model\w*|posterior)|quantile",
        ["Calibration of predictive intervals", "Probabilistic vs. point forecasts", "Decision impact of forecast uncertainty"],
        "Produce calibrated probabilistic forecasts and report interval coverage alongside point error."),
    "Physics-informed / hybrid modelling": (
        r"physics-?informed|physics-?based|hybrid (?:model|approach|forecast)|itu-r|empirical model|semi-?empirical|domain knowledge",
        ["Which physical priors improve sample efficiency", "Failure modes where pure ML departs from physics",
         "Benchmark against standard physical/empirical models"],
        "Fuse a physical/empirical baseline with a learned residual model and ablate the physical term."),
    "Real-time / on-board deployment": (
        r"real-?time|on-?board|latency|inference time|edge|embedded|computational (?:cost|complexity)|lightweight|online learning",
        ["Latency and memory budgets", "Online adaptation to drift", "Cost of retraining in operation"],
        "Profile latency/compute and test streaming or online-adaptation variants under realistic constraints."),
    "Long-term / seasonal validation": (
        r"multi-?year|long-?term|seasonal\w*|annual|year-?round|several years|climatolog\w+|inter-?annual",
        ["Seasonal and inter-annual drift", "Duration of measurement campaigns", "Stability of accuracy over time"],
        "Validate on multi-year data with temporal (forward-chaining) splits and drift analysis."),
    "Multi-band / multi-link evaluation": (
        r"multi-?(?:band|frequency|link|satellite|beam)|ka-?band|ku-?band|q/?v-?band|frequency scaling|several frequencies",
        ["Frequency scaling of the learned model", "Transfer between bands", "Cross-link consistency"],
        "Train jointly across bands/links and test frequency-transfer."),
    "Open data & reproducibility": (
        r"publicly available|open[- ]source|open data|code (?:is|are) available|github|reproducib\w+|benchmark dataset",
        ["Availability of data/code", "Standardised benchmarks and splits", "Reproducibility of reported scores"],
        "Release a benchmark with fixed splits and reference implementations."),
    "Interpretability": (
        r"interpretab\w+|explainab\w+|shap|feature importance|attention weights|saliency",
        ["Which inputs drive predictions", "Physical plausibility of learned relationships"],
        "Apply feature-attribution analysis and check agreement with domain understanding."),
}


# A sentence that mentions an aspect *while saying it was missing/weak* is evidence of a gap, not of coverage.
ABSENCE_RX = re.compile(
    r"\b(?:not|no|never|neither|without|cannot|lack\w*|untested|unexplored|underrepresented|under-represented|poorly|"
    r"remain\w* (?:high|poor|unclear|open)|only (?:one|a single|two|a few)|single (?:site|station|terminal|location)|"
    r"scarce|rare|limited|future work)\b", re.I)


def _cell(paper_id: str, rx: re.Pattern) -> tuple[str, list[tuple[str, int]], list[tuple[str, int]]]:
    """Returns (state, affirmative_hits, absence_hits). Absence hits are the authors' own statements that the
    aspect was not covered; they never count towards 'addressed'."""
    pos, neg = [], []
    title = normalize(corpus.papers[paper_id].title)[:40]
    chunks = corpus.chunks_for_paper(paper_id)
    for c in chunks:
        for s in split_sentences(c.text):
            if title and title in normalize(s):  # title/header line, not evidence
                continue
            if rx.search(s):
                (neg if ABSENCE_RX.search(s) else pos).append((s, c.page))
    sect = [1 for c in chunks if (c.section or "").lower().startswith(("method", "result", "experiment", "evaluation"))
            and any(rx.search(x) and not ABSENCE_RX.search(x) for x in split_sentences(c.text))]
    state = "addressed" if (len(pos) >= 3 or (pos and sect)) else "partial" if pos else "not_found"
    return state, pos, neg


def gap_agent(state: WorkflowState) -> WorkflowState:
    with traced(state, "Research Gap Agent") as t:
        profiles = state.get("profiles", [])
        aspects = list(ASPECTS)
        rxs = {a: re.compile(ASPECTS[a][0], re.I) for a in aspects}
        cells: list[list[str]] = []
        evidence_hits: dict[str, list[tuple[str, str, int]]] = {a: [] for a in aspects}
        absence_hits: dict[str, list[tuple[str, str, int]]] = {a: [] for a in aspects}
        for p in profiles:
            row = []
            for a in aspects:
                st, hits, neg = _cell(p.paper_id, rxs[a])
                row.append(st)
                for s, pg in hits[:1]:
                    evidence_hits[a].append((p.paper_id, s, pg))
                for s, pg in neg[:1]:
                    absence_hits[a].append((p.paper_id, s, pg))
            cells.append(row)
        state["gap_matrix"] = GapMatrix(aspects=aspects, papers=[p.title for p in profiles], cells=cells)  # type: ignore[arg-type]

        n = max(1, len(profiles))
        gaps: list[ResearchGap] = []
        for ai, a in enumerate(aspects):
            addressed = sum(1 for r in cells if r[ai] == "addressed")
            partial = sum(1 for r in cells if r[ai] == "partial")
            not_found = sum(1 for r in cells if r[ai] == "not_found")
            # gap if few papers address it in depth
            if addressed / n > 0.34 or not profiles:
                continue
            # attach verified limitation claims that touch this aspect (real evidence for the gap)
            support = []
            for p in profiles:
                for lim in p.limitations:
                    if lim.supporting and rxs[a].search(lim.supporting[0].quote):
                        support.append(lim)
            for pid, s, pg in absence_hits[a][:3]:  # authors' own "not evaluated / limited" statements: strongest evidence
                paper = corpus.papers[pid]
                vc = verify_claim(f"Authors state this was not covered or is limited in '{paper.title}'", s, paper,
                                  corpus.chunks_for_paper(pid), key_terms=[])
                if vc.supporting:
                    support.append(vc)
            for pid, s, pg in evidence_hits[a][:2]:  # partial mentions are also evidence of shallow treatment
                paper = corpus.papers[pid]
                chunks = corpus.chunks_for_paper(pid)
                vc = verify_claim(f"Only briefly touched in '{paper.title}'", s, paper, chunks, key_terms=[])
                if vc.supporting:
                    support.append(vc)
            _, to_inv, direction = ASPECTS[a]
            n_abs = len({pid for pid, _, _ in absence_hits[a]})
            desc = (f"Potential gap: {a.lower()} is addressed in depth by {addressed} of {len(profiles)} retrieved paper(s) "
                    f"({partial} mention it only in passing, {not_found} not detected)."
                    + (f" {n_abs} paper(s) explicitly state it was not covered or is limited." if n_abs else ""))
            gaps.append(ResearchGap(title=a, description=desc, evidence_to_investigate=to_inv, proposed_direction=direction,
                                    supporting_evidence=support[:4],
                                    coverage={"addressed": addressed, "partial": partial, "not_found": not_found}))
        gaps.sort(key=lambda g: (g.coverage.get("addressed", 0), -len(g.supporting_evidence), g.coverage.get("partial", 0)))
        if llm.available and gaps:
            try:
                out = llm.json(
                    "Refine research-gap proposals. Do not add facts or citations; only improve wording and make the proposed "
                    "direction specific to the research question.",
                    f"Question: {state['question']}\nGaps: " + str([{"title": g.title, "desc": g.description, "dir": g.proposed_direction} for g in gaps[:6]]) +
                    '\nReturn {"gaps":[{"title":"","proposed_direction":""}]}')
                for r in (out or {}).get("gaps", []):
                    for g in gaps:
                        if g.title == r.get("title") and r.get("proposed_direction"):
                            g.proposed_direction = r["proposed_direction"]
            except Exception:
                pass
        state["gaps"] = gaps[:6]
        state.setdefault("limitations", []).append(
            "Gap analysis reflects only the uploaded corpus: 'not detected' does not prove an aspect is unexplored in the wider literature.")
        t["summary"] = f"Built {len(profiles)}x{len(aspects)} coverage matrix; proposed {len(state['gaps'])} evidence-linked gaps."
    return state
