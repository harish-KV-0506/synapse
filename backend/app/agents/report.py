"""Assembles the Research Intelligence Report. The review only states things backed by a verified source span;
each statement carries an [S#] marker that resolves to paper/author/year/page/passage in `sources`."""
from __future__ import annotations

import re

from ..llm import llm
from ..schemas import PaperProfile, ResearchReport, SourceSpan, VerifiedClaim
from .evidence import verification_summary
from .state import WorkflowState, traced


class SourceRegistry:
    def __init__(self):
        self.spans: dict[str, SourceSpan] = {}
        self._key: dict[tuple, str] = {}

    def cite(self, claim: VerifiedClaim | None) -> str:
        """Returns ' [S3]' for a verified claim, '' if it has no usable evidence."""
        if not claim or claim.status not in ("supported", "partially_supported") or not claim.supporting:
            return ""
        sp = claim.supporting[0]
        key = (sp.paper_id, sp.page, re.sub(r"\W+", "", sp.quote.lower())[:80])
        if key not in self._key:
            sid = f"S{len(self._key) + 1}"
            self._key[key] = sid
            self.spans[sid] = sp
        return f" [{self._key[key]}]"


def _who(p: PaperProfile) -> str:
    first = p.authors[0].split()[-1] if p.authors else "Unknown"
    et = " et al." if len(p.authors) > 1 else ""
    return f"{first}{et}, {p.year or 'n.d.'}"


def _names(claims: list[VerifiedClaim]) -> list[str]:
    return [re.sub(r"^(Uses|Evaluates with)\s+", "", c.claim) for c in claims if c.status in ("supported", "partially_supported")]


def _heuristic_review(q: str, profiles: list[PaperProfile], reg: SourceRegistry, gaps) -> str:
    lines = [f"## Literature review: {q}", "",
             f"This synthesis covers {len(profiles)} paper(s) from the uploaded corpus. Every statement below is tied to a "
             "source passage ([S#]); statements without a marker could not be verified and are intentionally omitted.", ""]
    for p in profiles:
        parts = [f"**{p.title}** ({_who(p)})"]
        if p.objective and reg.cite(p.objective):
            parts.append(f"states its objective as: “{p.objective.supporting[0].quote}”{reg.cite(p.objective)}")
        para = " ".join(parts) + "."
        m = [f"{c.claim.replace('Uses ', '')}{reg.cite(c)}" for c in p.methods if reg.cite(c)]
        if m:
            para += " Methods detected: " + ", ".join(m) + "."
        d = [f"“{c.supporting[0].quote}”{reg.cite(c)}" for c in p.datasets[:1] if reg.cite(c)]
        if d:
            para += " Data: " + "; ".join(d) + "."
        mt = [f"{c.claim.replace('Evaluates with ', '')}{reg.cite(c)}" for c in p.metrics if reg.cite(c)]
        if mt:
            para += " Metrics: " + ", ".join(mt) + "."
        r = [f"“{c.supporting[0].quote}”{reg.cite(c)}" for c in p.results[:2] if reg.cite(c)]
        if r:
            para += " Reported results: " + " ".join(r)
        l = [f"“{c.supporting[0].quote}”{reg.cite(c)}" for c in p.limitations[:2] if reg.cite(c)]
        if l:
            para += " Stated limitations: " + " ".join(l)
        lines += [para, ""]
    if gaps:
        lines += ["## Emerging gaps", ""]
        for g in gaps[:4]:
            cites = "".join(reg.cite(e) for e in g.supporting_evidence[:2])
            lines.append(f"- **{g.title}** - {g.description}{cites}")
    return "\n".join(lines)


def _llm_review(q: str, profiles: list[PaperProfile], reg: SourceRegistry) -> str | None:
    facts = []
    for p in profiles:
        for label, items in [("objective", [p.objective] if p.objective else []), ("method", p.methods), ("dataset", p.datasets),
                             ("metric", p.metrics), ("result", p.results), ("limitation", p.limitations)]:
            for c in items:
                tag = reg.cite(c).strip()
                if tag:
                    facts.append(f"{tag} [{_who(p)}] {label}: {c.supporting[0].quote}")
    if not facts:
        return None
    text = llm.complete(
        "You write a literature review using ONLY the numbered evidence passages provided. Every sentence must end with one or "
        "more markers like [S3] taken from the evidence. Do not add facts, numbers or papers that are not in the evidence. "
        "If evidence conflicts, say so. Use markdown with short sections.",
        f"Research question: {q}\n\nEVIDENCE:\n" + "\n".join(facts)[:30000], max_tokens=3000)
    valid = set(reg.spans)
    # strip any marker that does not resolve - never allow an invented citation through
    def fix(m):
        ids = [i for i in re.findall(r"S\d+", m.group(0)) if i in valid]
        return "[" + ", ".join(ids) + "]" if ids else ""
    return re.sub(r"\[(?:S\d+(?:\s*,\s*)?)+\]", fix, text)


def comparison_tables(profiles: list[PaperProfile]) -> dict[str, list[dict]]:
    return {
        "methods": [{"paper": p.title, "items": _names(p.methods)} for p in profiles],
        "datasets": [{"paper": p.title, "items": [c.supporting[0].quote[:140] for c in p.datasets if c.supporting]} for p in profiles],
        "metrics": [{"paper": p.title, "items": _names(p.metrics)} for p in profiles],
    }


def report_agent(state: WorkflowState) -> WorkflowState:
    from ..knowledge_graph import build_graph, to_json
    with traced(state, "Report Writer") as t:
        profiles = state.get("profiles", [])
        reg = SourceRegistry()
        review = None
        if state.get("mode") == "llm":
            try:
                review = _llm_review(state["question"], profiles, reg)
            except Exception as e:
                state.setdefault("limitations", []).append(f"LLM synthesis failed ({e}); used verified-fact template instead.")
        if review is None:
            review = _heuristic_review(state["question"], profiles, reg, state.get("gaps", []))
        # register evidence cited by gaps so [S#] markers there resolve too
        for g in state.get("gaps", []):
            for e in g.supporting_evidence:
                reg.cite(e)
        claims = state.get("claims", [])
        g = build_graph(profiles, state.get("gap_matrix"))
        state["knowledge_graph"] = to_json(g)
        state["report"] = ResearchReport(
            question=state["question"], mode=state.get("mode", "heuristic"), literature_review=review, profiles=profiles,
            comparison=comparison_tables(profiles), gap_matrix=state.get("gap_matrix"), gaps=state.get("gaps", []),
            claims=claims, knowledge_graph=state["knowledge_graph"], sources=reg.spans, trace=state.get("trace", []),
            limitations=list(dict.fromkeys(state.get("limitations", []))), verification_summary=verification_summary(claims))
        t["summary"] = f"Report assembled with {len(reg.spans)} resolvable source citations."
        state["report"].trace = state.get("trace", [])
    return state
