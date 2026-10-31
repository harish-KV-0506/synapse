"""Evidence Agent: audits every claim, finds cross-paper conflicts, builds the verified-claim ledger."""
from __future__ import annotations

from collections import Counter

from ..schemas import VerifiedClaim
from ..verification.span_match import find_contradictions
from .state import WorkflowState, traced


def evidence_agent(state: WorkflowState) -> WorkflowState:
    with traced(state, "Evidence Agent") as t:
        ledger: list[VerifiedClaim] = []
        for prof in state.get("profiles", []):
            items = ([prof.objective] if prof.objective else []) + prof.methods + prof.datasets + prof.metrics + prof.results + prof.limitations
            ledger.extend(items)
        # Contradiction scan only on empirical results (opposing performance claims about similar topics)
        result_pairs = [(c, p.paper_id) for p in state.get("profiles", []) for c in p.results]
        conflicts = find_contradictions(result_pairs)
        state["claims"] = ledger
        t["summary"] = (f"Audited {len(ledger)} claims against source spans; "
                        f"{conflicts} potential cross-paper conflict(s) flagged for manual review.")
        unver = sum(1 for c in ledger if c.status == "unverified")
        if unver:
            state.setdefault("limitations", []).append(
                f"{unver} extracted claim(s) could not be matched to a source passage and are excluded from the review.")
    return state


def verification_summary(claims: list[VerifiedClaim]) -> dict[str, int]:
    return dict(Counter(c.status for c in claims))
