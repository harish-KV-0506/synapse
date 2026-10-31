"""Research Director: understands the question, plans sub-queries, sets the mode."""
from __future__ import annotations

from ..llm import llm
from ..retrieval.text import tokenize
from .state import WorkflowState, traced

ASPECT_QUERIES = [
    "research objective aim of this study",
    "dataset measurements data collection used for training and testing",
    "machine learning model algorithm architecture method",
    "evaluation metrics RMSE MAE accuracy results performance",
    "limitations future work generalization",
]


def director_agent(state: WorkflowState) -> WorkflowState:
    with traced(state, "Research Director") as t:
        q = state["question"].strip()
        state["mode"] = "llm" if llm.available else "heuristic"
        focus = [w for w in dict.fromkeys(tokenize(q))][:12]
        subs = [q] + [f"{q} {a}" for a in ASPECT_QUERIES]
        if llm.available:
            try:
                out = llm.json(
                    "You plan literature searches. Decompose the research question into 4-6 short, distinct retrieval "
                    "queries (objective, data, methods, metrics, limitations). Do not answer the question.",
                    f'Question: {q}\nReturn {{"queries": ["..."]}}')
                if out and out.get("queries"):
                    subs = [q] + [s for s in out["queries"] if isinstance(s, str)][:6]
            except Exception:
                pass
        state["sub_queries"], state["focus_terms"] = subs, focus
        state.setdefault("limitations", [])
        t["summary"] = f"Mode={state['mode']}; planned {len(subs)} retrieval queries."
    return state
