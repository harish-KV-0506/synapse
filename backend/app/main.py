"""FastAPI app. Run: uvicorn app.main:app --reload --port 8000"""
from __future__ import annotations

import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from .agents.workflow import run_workflow
from .config import settings
from .export import report_to_markdown
from .ingest.pdf_parser import parse_pdf
from .knowledge_graph import build_graph, to_json
from .llm import llm
from .retrieval.embeddings import get_embedder
from .retrieval.hybrid import corpus
from .retrieval.reranker import get_reranker
from .schemas import QueryRequest, ResearchReport, SearchHit, VerifiedClaim
from .verification.span_match import verify_free_claim

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("synapse.api")

SAMPLE_DIR = Path(os.getenv("SYNAPSE_SAMPLE_DIR", Path(__file__).resolve().parents[2] / "sample_papers"))
REPORTS: dict[str, ResearchReport] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    corpus.load()
    yield


app = FastAPI(title="SYNAPSE", version="1.0.0", lifespan=lifespan,
              description="Multi-agent scientific research intelligence & evidence verification")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SYNAPSE_CORS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["*"], allow_headers=["*"])


def _ingest(path: Path, errors: list, out: list) -> None:
    try:
        parsed = parse_pdf(path)
    except Exception as e:  # corrupt / scanned PDFs must not break the batch
        errors.append({"file": path.name, "error": str(e)})
        return
    corpus.add(parsed.paper, parsed.chunks)
    out.append({**parsed.paper.model_dump(), "chunks": len(parsed.chunks)})


@app.get("/api/health")
def health():
    emb, rr = get_embedder(), get_reranker()
    corpus.build()
    return {"status": "ok", "papers": len(corpus.papers), "chunks": len(corpus.chunks),
            "mode": "llm" if llm.available else "heuristic", "embedder": emb.name, "reranker": rr.name,
            "vector_backend": corpus.vs.backend, "langfuse": settings.langfuse_enabled}


@app.post("/api/papers")
async def upload_papers(files: list[UploadFile] = File(...)):
    out, errors = [], []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            errors.append({"file": f.filename, "error": "Only PDF files are supported."})
            continue
        dest = settings.uploads_dir / Path(f.filename).name
        with dest.open("wb") as fh:
            shutil.copyfileobj(f.file, fh)
        _ingest(dest, errors, out)
    corpus.save()
    return {"added": out, "errors": errors}


@app.post("/api/demo/load")
def load_demo():
    """Load the bundled SYNTHETIC demo papers (fictional; for trying the platform without uploads)."""
    pdfs = sorted(SAMPLE_DIR.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(404, f"No demo PDFs in {SAMPLE_DIR}. Run: python scripts/make_sample_pdfs.py ../sample_papers")
    out, errors = [], []
    for p in pdfs:
        _ingest(p, errors, out)
    corpus.save()
    return {"added": out, "errors": errors}


@app.get("/api/papers")
def list_papers():
    return [{**p.model_dump(), "chunks": len(corpus.chunks_for_paper(p.paper_id))} for p in corpus.papers.values()]


@app.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: str):
    if paper_id not in corpus.papers:
        raise HTTPException(404, "Paper not found")
    corpus.remove(paper_id)
    corpus.save()
    return {"deleted": paper_id}


@app.delete("/api/papers")
def clear_papers():
    corpus.clear()
    corpus.save()
    return {"cleared": True}


@app.post("/api/search", response_model=list[SearchHit])
def search(req: QueryRequest):
    hits = corpus.search(req.question, k=req.top_k, paper_ids=req.paper_ids)
    return [SearchHit(chunk=h.chunk, paper_title=corpus.papers[h.chunk.paper_id].title, score=h.score, sources=h.sources)
            for h in hits]


@app.post("/api/analyze", response_model=ResearchReport)
def analyze(req: QueryRequest):
    if not corpus.papers:
        raise HTTPException(400, "Upload at least one PDF first (or POST /api/demo/load).")
    report = run_workflow(req.question, req.paper_ids, req.top_k)
    report.report_id = uuid.uuid4().hex[:8]
    REPORTS[report.report_id] = report
    return report


@app.get("/api/reports/{report_id}", response_model=ResearchReport)
def get_report(report_id: str):
    if report_id not in REPORTS:
        raise HTTPException(404, "Report not found (reports are kept in memory until the server restarts).")
    return REPORTS[report_id]


@app.get("/api/reports/{report_id}/markdown", response_class=PlainTextResponse)
def get_report_markdown(report_id: str):
    if report_id not in REPORTS:
        raise HTTPException(404, "Report not found")
    return report_to_markdown(REPORTS[report_id])


@app.post("/api/verify", response_model=VerifiedClaim)
def verify(req: QueryRequest):
    """Check a free-text claim against the corpus (supporting vs contradicting passages)."""
    hits = corpus.search(req.question, k=req.top_k or 8, paper_ids=req.paper_ids)
    return verify_free_claim(req.question, hits, corpus.papers)
