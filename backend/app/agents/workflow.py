"""LangGraph workflow: Director -> Literature -> Methodology -> Evidence -> Gap -> Report.

Falls back to a plain sequential runner if langgraph isn't installed, so behaviour is identical.
Optional Langfuse tracing wraps the whole run.
"""
from __future__ import annotations

import logging

from ..config import settings
from ..schemas import ResearchReport
from .director import director_agent
from .evidence import evidence_agent
from .gap import gap_agent
from .literature import literature_agent
from .methodology import methodology_agent
from .report import report_agent
from .state import WorkflowState

log = logging.getLogger("synapse.workflow")

NODES = [
    ("director", director_agent),
    ("literature", literature_agent),
    ("methodology", methodology_agent),
    ("evidence", evidence_agent),
    ("gap", gap_agent),
    ("report", report_agent),
]


def _build_langgraph():
    from langgraph.graph import END, StateGraph

    g = StateGraph(WorkflowState)
    for name, fn in NODES:
        g.add_node(name, fn)
    g.set_entry_point(NODES[0][0])
    for (a, _), (b, _) in zip(NODES, NODES[1:]):
        g.add_edge(a, b)
    g.add_edge(NODES[-1][0], END)
    return g.compile()


_compiled = None


def run_workflow(question: str, paper_ids: list[str] | None = None, top_k: int | None = None) -> ResearchReport:
    global _compiled
    state: WorkflowState = {"question": question, "paper_ids": paper_ids, "top_k": top_k or settings.top_k_final,
                            "trace": [], "limitations": []}
    trace_ctx = None
    if settings.langfuse_enabled:
        try:
            from langfuse import Langfuse  # type: ignore
            trace_ctx = Langfuse().trace(name="synapse-run", input={"question": question})
        except Exception as e:
            log.warning("Langfuse disabled: %s", e)
    try:
        try:
            if _compiled is None:
                _compiled = _build_langgraph()
            state = _compiled.invoke(state)
        except ImportError:
            for _, fn in NODES:
                state = fn(state)
    finally:
        if trace_ctx is not None:
            try:
                trace_ctx.update(output={"agents": [t.agent for t in state.get("trace", [])]})
            except Exception:
                pass
    return state["report"]
