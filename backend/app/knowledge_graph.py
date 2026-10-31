"""Knowledge graph (NetworkX): Paper -USES-> Method, -EVALUATED_ON-> Dataset, -MEASURED_BY-> Metric, -ADDRESSES-> Aspect.
Swap to Neo4j later by writing the same triples with the neo4j driver."""
from __future__ import annotations

import hashlib
import re

import networkx as nx

from .schemas import GapMatrix, PaperProfile


def _label(claim: str) -> str:
    return re.sub(r"^(Uses|Evaluates with)\s+", "", claim).strip()[:60]


def build_graph(profiles: list[PaperProfile], matrix: GapMatrix | None) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for p in profiles:
        g.add_node(f"paper:{p.paper_id}", kind="paper", label=p.title, year=p.year)
        for m in p.methods:
            n = f"method:{_label(m.claim)}"
            g.add_node(n, kind="method", label=_label(m.claim))
            g.add_edge(f"paper:{p.paper_id}", n, rel="USES", status=m.status)
        for d in p.datasets:
            n = f"dataset:{p.paper_id}:{hashlib.md5(d.claim.encode()).hexdigest()[:6]}"
            g.add_node(n, kind="dataset", label=d.claim[:70])
            g.add_edge(f"paper:{p.paper_id}", n, rel="EVALUATED_ON", status=d.status)
        for m in p.metrics:
            n = f"metric:{_label(m.claim)}"
            g.add_node(n, kind="metric", label=_label(m.claim))
            g.add_edge(f"paper:{p.paper_id}", n, rel="MEASURED_BY", status=m.status)
    if matrix:
        for pi, p in enumerate(profiles):
            for ai, a in enumerate(matrix.aspects):
                if matrix.cells[pi][ai] != "not_found":
                    n = f"aspect:{a}"
                    g.add_node(n, kind="aspect", label=a)
                    g.add_edge(f"paper:{p.paper_id}", n, rel="ADDRESSES", level=matrix.cells[pi][ai])
    return g


def to_json(g: nx.MultiDiGraph) -> dict:
    return {
        "nodes": [{"id": n, **d} for n, d in g.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d} for u, v, d in g.edges(data=True)],
        "stats": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()},
    }


def shared_methods(g: nx.MultiDiGraph) -> dict[str, list[str]]:
    """Methods used by 2+ papers - useful for spotting method clusters."""
    out: dict[str, list[str]] = {}
    for n, d in g.nodes(data=True):
        if d.get("kind") == "method":
            papers = [g.nodes[u]["label"] for u, _, e in g.in_edges(n, data=True) if e.get("rel") == "USES"]
            if len(papers) > 1:
                out[d["label"]] = papers
    return out
