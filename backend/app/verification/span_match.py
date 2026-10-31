"""Evidence verification layer.

Principle: a citation is only ever built from text that exists in an uploaded paper.
Nothing here asks a model "is this true?" - it checks whether a quoted span is physically present
(source-span matching) and whether the claim's key terms are covered by that span. A generated
summary is never treated as evidence.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..config import settings
from ..retrieval.text import normalize, tokenize
from ..schemas import Chunk, Paper, RetrievedChunk, SourceSpan, VerifiedClaim

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

POS_CUES = ("improv", "outperform", "better", "higher accuracy", "reduc", "enhanc", "increase", "gain", "superior", "robust", "effective")
NEG_CUES = ("degrad", "worse", "fail", "poor", "unable", "underperform", "no improvement", "did not improve", "does not improve",
            "not significantly", "decline", "lower accuracy", "ineffective", "no significant", "limited")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(re.sub(r"\s+", " ", text)) if s.strip()]


def polarity(text: str) -> int:
    t = text.lower()
    pos = sum(c in t for c in POS_CUES)
    neg = sum(c in t for c in NEG_CUES)
    if "not" in t.split() or "no " in t:
        neg += 0  # negation handled via explicit NEG_CUES phrases
    return 1 if pos > neg else -1 if neg > pos else 0


def span_score(quote: str, text: str) -> float:
    """Fraction of the quote found as contiguous token runs in `text` (1.0 = verbatim)."""
    nq, nt = normalize(quote), normalize(text)
    if not nq:
        return 0.0
    if nq in nt:
        return 1.0
    qt, tt = nq.split(), nt.split()
    sm = SequenceMatcher(None, qt, tt, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks() if b.size >= 3)
    return min(1.0, matched / len(qt))


def best_span(quote: str, chunks: list[Chunk]) -> tuple[float, Chunk | None, str]:
    """Locate `quote` in chunks. Returns (score, chunk, verbatim_sentence_from_source)."""
    best: tuple[float, Chunk | None, str] = (0.0, None, "")
    for c in chunks:
        sc = span_score(quote, c.text)
        if sc > best[0]:
            # return the actual source sentence(s) that contain the quote, not the model's rephrasing
            nq = normalize(quote)
            sent = next((s for s in split_sentences(c.text) if nq and nq in normalize(s)), None)
            if sent is None:
                sent = max(split_sentences(c.text), key=lambda s: span_score(s, quote) if s else 0, default=quote)
            best = (sc, c, sent)
            if sc == 1.0:
                break
    return best


def make_span(paper: Paper, chunk: Chunk, quote: str) -> SourceSpan:
    return SourceSpan(paper_id=paper.paper_id, paper_title=paper.title, authors=paper.authors, year=paper.year,
                      page=chunk.page, quote=quote, chunk_id=chunk.chunk_id)


def term_coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 1.0
    toks = set(tokenize(text))
    nt = normalize(text)
    hit = sum(1 for t in terms if t.lower() in toks or t.lower() in nt)
    return hit / len(terms)


def verify_claim(claim: str, quote: str | None, paper: Paper, chunks: list[Chunk],
                 key_terms: list[str] | None = None, note: str | None = None) -> VerifiedClaim:
    """Verify a claim extracted from `paper` against that paper's own chunks."""
    if not quote:
        return VerifiedClaim(claim=claim, status="unverified", confidence=0.0, note=note or "No source passage provided.")
    sc, chunk, sentence = best_span(quote, chunks)
    if chunk is None or sc < 0.3:
        return VerifiedClaim(claim=claim, status="unverified", confidence=round(sc * 0.5, 2),
                             note="Quoted passage not found in the source document; discarded as evidence.")
    terms = key_terms if key_terms is not None else tokenize(claim)
    cov = term_coverage(terms, sentence)
    conf = round(0.7 * sc + 0.3 * cov, 2)
    status = ("supported" if conf >= settings.verify_threshold_supported
              else "partially_supported" if conf >= settings.verify_threshold_partial else "unverified")
    return VerifiedClaim(claim=claim, status=status, confidence=conf,
                         supporting=[make_span(paper, chunk, sentence)] if status != "unverified" else [], note=note)


def find_contradictions(claims: list[tuple[VerifiedClaim, str]], min_overlap: float = 0.3) -> int:
    """Cross-paper heuristic: similar topic + opposite polarity => flag both as conflicting.
    `claims` is a list of (claim, paper_id). Mutates claims; returns number of conflicts.
    Heuristic only: flagged items need human review."""
    n = 0
    prepared = []
    for vc, pid in claims:
        if not vc.supporting:
            continue
        sp = vc.supporting[0]
        prepared.append((vc, pid, set(tokenize(sp.quote)), polarity(sp.quote), sp))
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            a, b = prepared[i], prepared[j]
            if a[1] == b[1] or a[3] == 0 or b[3] == 0 or a[3] == b[3]:
                continue
            inter = len(a[2] & b[2])
            union = len(a[2] | b[2]) or 1
            if inter / union >= min_overlap:
                a[0].contradicting.append(b[4])
                b[0].contradicting.append(a[4])
                msg = "Conflicting evidence in another paper (heuristic polarity check - review manually)."
                a[0].note = (a[0].note + " " if a[0].note else "") + msg
                b[0].note = (b[0].note + " " if b[0].note else "") + msg
                n += 1
    return n


def verify_free_claim(claim: str, hits: list[RetrievedChunk], papers: dict[str, Paper]) -> VerifiedClaim:
    """Check a user-supplied claim against retrieved passages (supporting vs contradicting)."""
    terms = tokenize(claim)
    cpol = polarity(claim)
    sup: list[tuple[float, SourceSpan]] = []
    con: list[tuple[float, SourceSpan]] = []
    for h in hits:
        paper = papers[h.chunk.paper_id]
        for s in split_sentences(h.chunk.text):
            cov = term_coverage(terms, s)
            if cov < 0.45:
                continue
            spol = polarity(s)
            span = make_span(paper, h.chunk, s)
            if cpol and spol and cpol != spol:
                con.append((cov, span))
            else:
                sup.append((cov, span))
    sup.sort(key=lambda t: t[0], reverse=True)
    con.sort(key=lambda t: t[0], reverse=True)
    best_sup = sup[0][0] if sup else 0.0
    best_con = con[0][0] if con else 0.0
    if con and best_con >= best_sup:
        status, conf = "contradicted", round(best_con, 2)
    elif best_sup >= settings.verify_threshold_supported:
        status, conf = "supported", round(best_sup, 2)
    elif best_sup >= settings.verify_threshold_partial:
        status, conf = "partially_supported", round(best_sup, 2)
    else:
        status, conf = "unverified", round(best_sup, 2)
    return VerifiedClaim(claim=claim, status=status, confidence=conf,
                         supporting=[s for _, s in sup[:3]] if status != "contradicted" else [],
                         contradicting=[s for _, s in con[:3]],
                         note="Term-coverage + polarity heuristic over retrieved passages; not proof. Read the cited passages.")
