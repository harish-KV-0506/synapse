"""PDF intelligence: page-aware text extraction, metadata inference, chunking.

Every chunk keeps its 1-based page number so citations can point to a page.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

try:  # PyMuPDF is preferred (font-aware title detection); pypdf is the portable fallback.
    import fitz  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    fitz = None

from ..config import settings
from ..schemas import Chunk, Paper

SECTION_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(abstract|introduction|related work|background|methodology|methods?|"
    r"materials and methods|data(?:sets?)?|experiments?|experimental setup|results?(?: and discussion)?|"
    r"discussion|limitations?|conclusions?|future work|references)\s*$",
    re.I,
)
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


@dataclass
class ParsedPDF:
    paper: Paper
    chunks: list[Chunk]


def _clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line wraps
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _paper_id(path: Path, data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:10]


def _guess_title(doc, first_page_text: str, filename: str, meta_title: str = "") -> str:
    if len(meta_title.strip()) > 8:
        return meta_title.strip()
    # Largest-font text on page 1 is usually the title (PyMuPDF only).
    try:
        if doc is None:
            raise RuntimeError("no font info")
        page = doc[0]
        best: list[tuple[float, str]] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = " ".join(s["text"] for s in line["spans"]).strip()
                size = max((s["size"] for s in line["spans"]), default=0)
                if text:
                    best.append((size, text))
        if best:
            top = max(s for s, _ in best)
            title = " ".join(t for s, t in best if abs(s - top) < 0.5)[:200].strip()
            if len(title) > 8:
                return title
    except Exception:
        pass
    first = next((ln.strip() for ln in first_page_text.splitlines() if len(ln.strip()) > 8), filename)
    return first[:200]


def _guess_authors_year(first_page_text: str, title: str, meta_author: str | None) -> tuple[list[str], int | None]:
    authors: list[str] = []
    if meta_author and len(meta_author) > 3:
        authors = [a.strip() for a in re.split(r",|;| and ", meta_author) if a.strip()]
    if not authors:
        lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines[:12]):
            if ln in title or title.startswith(ln):
                continue
            if re.search(r"abstract|university|institute|department", ln, re.I):
                continue
            if 1 <= len(ln.split()) <= 14 and re.match(r"^[A-Z][A-Za-z.\- ,&]+$", ln) and "," in ln or " and " in ln:
                authors = [a.strip() for a in re.split(r",|;| and | & ", ln) if a.strip()]
                break
    m = YEAR_RE.findall(first_page_text[:2500])
    year = int(m[0]) if m else None
    return authors[:8], year


def _extract_abstract(pages: list[str]) -> str:
    full = "\n".join(pages[:2])
    m = re.search(r"abstract[\s:.\-—]*\n?(.{150,1800}?)(?:\n\s*\n|\n\s*(?:1\.?\s+)?introduction|\n\s*keywords)", full, re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def chunk_page(text: str, paper_id: str, page: int, section: str | None, start_idx: int) -> list[Chunk]:
    """Sentence-aware sliding window chunks (~chunk_words words, overlapping)."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", re.sub(r"\s+", " ", text))
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_words = 0
    idx = start_idx

    def flush():
        nonlocal cur, cur_words, idx
        if cur:
            body = " ".join(cur).strip()
            if len(body.split()) >= 12:
                chunks.append(Chunk(chunk_id=f"{paper_id}-p{page}-c{idx}", paper_id=paper_id, page=page, text=body, section=section))
                idx += 1
        # keep overlap
        keep: list[str] = []
        words = 0
        for s in reversed(cur):
            words += len(s.split())
            keep.insert(0, s)
            if words >= settings.chunk_overlap:
                break
        cur = keep if len(cur) > len(keep) else []
        cur_words = sum(len(s.split()) for s in cur)

    for s in sentences:
        w = len(s.split())
        if cur_words + w > settings.chunk_words and cur:
            flush()
        cur.append(s)
        cur_words += w
    if cur:
        body = " ".join(cur).strip()
        if len(body.split()) >= 12:
            chunks.append(Chunk(chunk_id=f"{paper_id}-p{page}-c{idx}", paper_id=paper_id, page=page, text=body, section=section))
    return chunks


def _read_pages(data: bytes):
    """Return (pages_text, metadata_dict, fitz_doc_or_None). PyMuPDF if installed, else pypdf."""
    if fitz is not None:
        doc = fitz.open(stream=data, filetype="pdf")
        return [_clean(p.get_text("text")) for p in doc], dict(doc.metadata or {}), doc
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    meta = reader.metadata or {}
    md = {"title": str(meta.get("/Title") or ""), "author": str(meta.get("/Author") or "")}
    return [_clean(p.extract_text() or "") for p in reader.pages], md, None


def parse_pdf(path: Path) -> ParsedPDF:
    data = path.read_bytes()
    pid = _paper_id(path, data)
    pages, meta, doc = _read_pages(data)
    if not any(pages):
        raise ValueError(f"No extractable text in {path.name} (scanned PDF? run OCR first).")
    first = pages[0] if pages else ""
    title = _guess_title(doc, first, path.stem, meta.get("title") or "")
    authors, year = _guess_authors_year(first, title, meta.get("author"))
    paper = Paper(
        paper_id=pid, title=title, authors=authors, year=year, filename=path.name,
        num_pages=len(pages), abstract=_extract_abstract(pages),
    )
    chunks: list[Chunk] = []
    section: str | None = None
    counter = 0
    for pno, text in enumerate(pages, start=1):
        # track the section heading as we go (heading lines are short, standalone)
        lines = text.splitlines()
        buf: list[str] = []
        for ln in lines:
            m = SECTION_RE.match(ln)
            if m:
                if buf:
                    new = chunk_page("\n".join(buf), pid, pno, section, counter)
                    counter += len(new)
                    chunks.extend(new)
                    buf = []
                section = m.group(1).title()
            else:
                buf.append(ln)
        if buf:
            new = chunk_page("\n".join(buf), pid, pno, section, counter)
            counter += len(new)
            chunks.extend(new)
        # stop indexing at references to avoid retrieving bibliography noise
        if section and section.lower() == "references":
            chunks = [c for c in chunks if c.section is None or c.section.lower() != "references"]
    return ParsedPDF(paper=paper, chunks=chunks)
