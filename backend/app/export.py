"""Render a ResearchReport as a self-contained Markdown document (the 'Research Intelligence Report')."""
from __future__ import annotations

from .schemas import ResearchReport, SourceSpan

ICON = {"supported": "SUPPORTED", "partially_supported": "PARTIAL", "unverified": "UNVERIFIED", "contradicted": "CONTRADICTED"}


def _cite(sp: SourceSpan) -> str:
    who = (sp.authors[0].split()[-1] + (" et al." if len(sp.authors) > 1 else "")) if sp.authors else "Unknown"
    return f"{who}, {sp.year or 'n.d.'}, p.{sp.page}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def report_to_markdown(r: ResearchReport) -> str:
    out = ["# Research Intelligence Report", "", f"**Question:** {r.question}", "",
           f"*Mode: {r.mode} - evidence-backed synthesis. Every [S#] resolves to a verbatim passage in an uploaded paper.*", "",
           r.literature_review, ""]

    if r.comparison:
        out += ["## Comparison of studies", "", "| Paper | Methods | Datasets | Metrics |", "|---|---|---|---|"]
        m = {k: {row["paper"]: row["items"] for row in v} for k, v in r.comparison.items()}
        for p in r.profiles:
            out.append("| " + " | ".join([_cell(p.title), _cell(", ".join(m.get("methods", {}).get(p.title, []))),
                                          _cell("; ".join(m.get("datasets", {}).get(p.title, []))),
                                          _cell(", ".join(m.get("metrics", {}).get(p.title, [])))]) + " |")
        out.append("")

    if r.gap_matrix:
        gm = r.gap_matrix
        sym = {"addressed": "●", "partial": "◐", "not_found": "○"}
        out += ["## Research gap matrix", "", "● addressed in depth · ◐ mentioned in passing · ○ not detected", "",
                "| Paper | " + " | ".join(gm.aspects) + " |", "|---|" + "---|" * len(gm.aspects)]
        for t, row in zip(gm.papers, gm.cells):
            out.append(f"| {_cell(t)} | " + " | ".join(sym[c] for c in row) + " |")
        out.append("")

    if r.gaps:
        out += ["## Research gap analysis", ""]
        for g in r.gaps:
            out += [f"### {g.title}", "", g.description, "", "**Evidence to investigate:**"]
            out += [f"- {e}" for e in g.evidence_to_investigate]
            out += ["", f"**Proposed research direction:** {g.proposed_direction}", ""]
            for ev in g.supporting_evidence:
                for sp in ev.supporting[:1]:
                    out.append(f"> “{sp.quote}” - {sp.paper_title} ({_cite(sp)})")
            out.append("")

    out += ["## Evidence ledger", "", f"Verification summary: {r.verification_summary}", "",
            "| Status | Conf. | Claim | Source | Passage |", "|---|---|---|---|---|"]
    for c in r.claims:
        sp = c.supporting[0] if c.supporting else None
        out.append(f"| {ICON[c.status]} | {c.confidence:.2f} | {_cell(c.claim[:120])} | "
                   f"{_cell(_cite(sp)) if sp else '-'} | {_cell('“' + sp.quote[:160] + '”') if sp else '-'} |")
    conflicts = [c for c in r.claims if c.contradicting]
    if conflicts:
        out += ["", "### Potential cross-paper conflicts (heuristic - review manually)", ""]
        for c in conflicts:
            out.append(f"- {c.claim[:140]}")
            out += [f"  - vs. “{x.quote}” ({x.paper_title}, p.{x.page})" for x in c.contradicting]

    out += ["", "## Sources", ""]
    for sid, sp in r.sources.items():
        out.append(f"- **[{sid}]** {sp.paper_title} - {', '.join(sp.authors) or 'n/a'} ({sp.year or 'n.d.'}), p.{sp.page}: “{sp.quote}”")
    out += ["", "## Limitations of this analysis", ""] + [f"- {x}" for x in r.limitations]
    out += ["", "## Agent trace", ""] + [f"- **{t.agent}** ({t.duration_ms} ms): {t.summary}" for t in r.trace]
    return "\n".join(out) + "\n"
