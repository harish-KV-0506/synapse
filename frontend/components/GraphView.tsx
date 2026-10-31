"use client";
import type { ResearchReport } from "@/lib/types";

const COLORS: Record<string, string> = { paper: "#5aa9ff", method: "#4cc9b0", dataset: "#d29922", metric: "#bc8cff", aspect: "#f0883e" };
const ORDER = ["paper", "method", "dataset", "metric", "aspect"];

/** Dependency-free concentric layout: one ring per node kind. */
export function GraphView({ g }: { g: ResearchReport["knowledge_graph"] }) {
  const W = 760, H = 520, cx = W / 2, cy = H / 2;
  const pos = new Map<string, { x: number; y: number }>();
  const kinds = ORDER.filter((k) => g.nodes.some((n) => n.kind === k));
  kinds.forEach((k, ring) => {
    const ns = g.nodes.filter((n) => n.kind === k);
    const radius = ring === 0 ? 70 : 70 + ring * 60;
    ns.forEach((n, i) => {
      const a = (2 * Math.PI * i) / ns.length - Math.PI / 2;
      pos.set(n.id, { x: cx + radius * Math.cos(a), y: cy + radius * Math.sin(a) });
    });
  });
  return (
    <div>
      <div className="row small" style={{ marginBottom: 6 }}>
        {kinds.map((k) => <span key={k} className="chip"><span style={{ color: COLORS[k] }}>●</span> {k}</span>)}
        <span className="muted">{g.stats.nodes} nodes · {g.stats.edges} edges</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Knowledge graph">
        {g.edges.map((e, i) => {
          const a = pos.get(e.source), b = pos.get(e.target);
          return a && b ? <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#8b99a8" strokeOpacity={0.25} /> : null;
        })}
        {g.nodes.map((n) => {
          const p = pos.get(n.id);
          if (!p) return null;
          return (
            <g key={n.id}>
              <circle cx={p.x} cy={p.y} r={n.kind === "paper" ? 7 : 5} fill={COLORS[n.kind] ?? "#888"}><title>{n.label}</title></circle>
              {n.kind !== "dataset" && <text x={p.x + 8} y={p.y + 3}>{n.label.slice(0, 22)}</text>}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
