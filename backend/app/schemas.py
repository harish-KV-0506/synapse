"""Pydantic models shared by agents, API and frontend."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    year: int | None = None
    filename: str
    num_pages: int = 0
    abstract: str = ""


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    page: int  # 1-based
    text: str
    section: str | None = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    sources: dict[str, float] = {}  # per-retriever scores/ranks for transparency


class SourceSpan(BaseModel):
    """A verbatim passage in a specific page of a specific paper."""
    paper_id: str
    paper_title: str
    authors: list[str] = []
    year: int | None = None
    page: int
    quote: str
    chunk_id: str | None = None


VerificationStatus = Literal["supported", "partially_supported", "unverified", "contradicted"]


class VerifiedClaim(BaseModel):
    claim: str
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    supporting: list[SourceSpan] = []
    contradicting: list[SourceSpan] = []
    note: str | None = None


class PaperProfile(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    year: int | None = None
    objective: VerifiedClaim | None = None
    methods: list[VerifiedClaim] = []
    datasets: list[VerifiedClaim] = []
    metrics: list[VerifiedClaim] = []
    results: list[VerifiedClaim] = []
    limitations: list[VerifiedClaim] = []


class ResearchGap(BaseModel):
    title: str
    description: str
    evidence_to_investigate: list[str] = []
    proposed_direction: str
    supporting_evidence: list[VerifiedClaim] = []
    coverage: dict[str, int] = {}  # aspect -> number of papers addressing it


class GapMatrix(BaseModel):
    aspects: list[str]
    papers: list[str]  # paper titles in row order
    cells: list[list[Literal["addressed", "partial", "not_found"]]]


class AgentTrace(BaseModel):
    agent: str
    summary: str
    duration_ms: int = 0


class ResearchReport(BaseModel):
    report_id: str | None = None
    question: str
    mode: Literal["llm", "heuristic"]
    literature_review: str
    profiles: list[PaperProfile]
    comparison: dict[str, list[dict]] = {}
    gap_matrix: GapMatrix | None = None
    gaps: list[ResearchGap] = []
    claims: list[VerifiedClaim] = []
    knowledge_graph: dict = {}
    sources: dict[str, SourceSpan] = {}  # "S1" -> exact passage; every [S#] in the review resolves here
    trace: list[AgentTrace] = []
    limitations: list[str] = []
    verification_summary: dict[str, int] = {}


class QueryRequest(BaseModel):
    question: str
    paper_ids: list[str] | None = None
    top_k: int | None = None


class SearchHit(BaseModel):
    chunk: Chunk
    paper_title: str
    score: float
    sources: dict[str, float] = {}
