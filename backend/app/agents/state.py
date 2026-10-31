"""Shared workflow state passed between agents (LangGraph state dict)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, TypedDict

from ..schemas import (AgentTrace, GapMatrix, PaperProfile, ResearchGap, RetrievedChunk, VerifiedClaim)


class WorkflowState(TypedDict, total=False):
    question: str
    paper_ids: list[str] | None
    top_k: int
    mode: str  # "llm" | "heuristic"
    sub_queries: list[str]
    focus_terms: list[str]
    retrieved: dict[str, list[RetrievedChunk]]  # paper_id -> chunks ranked for the question
    ranked_papers: list[str]
    profiles: list[PaperProfile]
    claims: list[VerifiedClaim]
    gap_matrix: GapMatrix
    gaps: list[ResearchGap]
    knowledge_graph: dict
    report: Any
    trace: list[AgentTrace]
    limitations: list[str]


@contextmanager
def traced(state: WorkflowState, agent: str):
    t0 = time.perf_counter()
    box = {"summary": ""}
    yield box
    state.setdefault("trace", []).append(
        AgentTrace(agent=agent, summary=box["summary"], duration_ms=int((time.perf_counter() - t0) * 1000)))
