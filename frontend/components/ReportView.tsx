"use client";
import { Fragment, useState } from "react";
import { api } from "@/lib/api";
import type { ResearchReport, SourceSpan, VerifiedClaim } from "@/lib/types";
import { GraphView } from "./GraphView";

const CITE = /(\*\*[^*]+\*\*|\[S\d+(?:\s*,\s*S\d+)*\])/g;

export function who(s: SourceSpan): string {
  const first = s.authors.length ? s.authors[0].split(" ").slice(-1)[0] : "Unknown";
  return `${first}${s.authors.length > 1 ? " et al." : ""}, ${s.year ?? "n.d."}, p.${s.page}`;
}

function Inline({ text, sel, onCite }: { text: string; sel: string | null; onCite: (id: string) => void }) {
  return (
    <>
      {text.split(CITE).map((part, i) => {
        if (part.startsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (part.startsWith("[S")) {
          return (
            <Fragment key={i}>
              {part.match(/S\d+/g)!.map((id) => (
                <button key={id} className={`cite ${sel === id ? "sel" : ""}`} onClick={() => onCite(id)} title="Show source passage">
                  {id}
                </button>
              ))}
            </Fragment>
          );
        }
        return <Fragment key={i}>{part}</Fragment>;
      })}
    </>
  );
}

function Review({ md, sel, onCite }: { md: string; sel: string | null; onCite: (id: string) => void }) {
  return (
    <div className="review">
      {md.split(/\n\s*\n/).map((block, i) => {
        const t = block.trim();
        if (!t) return null;
        if (t.startsWith("## ")) return <h2 key={i}>{t.slice(3)}</h2>;
        if (t.split("\n").every((l) => l.trim().startsWith("- "))) {
          return (
            <ul key={i}>
              {t.split("\n").map((l, k) => (
                <li key={k}><Inline text={l.trim().slice(2)} sel={sel} onCite={onCite} /></li>
              ))}
            </ul>
          );
        }
        return <p key={i}><Inline text={t} sel={sel} onCite={onCite} /></p>;
      })}
    </div>
  );
}

function Badge({ c }: { c: VerifiedClaim }) {
  return <span className={`badge ${c.status}`}>{c.status.replace("_", " ")} · {c.confidence.toFixed(2)}</span>;
}

const SYM = { addressed: "●", partial: "◐", not_found: "○" } as const;

export function ReportView({ r }: { r: ResearchReport }) {
  const [sel, setSel] = useState<string | null>(null);
  const [tab, setTab] = useState<"review" | "gaps" | "ledger" | "compare" | "graph" | "trace">("review");
  const src = sel ? r.sources[sel] : null;
  const vs = r.verification_summary;
  const conflicts = r.claims.filter((c) => c.contradicting.length);

  return (
    <div>
      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>
            <strong>{r.question}</strong>
            <div className="small muted">
              {r.profiles.length} papers · {Object.keys(r.sources).length} citations ·{" "}
              {Object.entries(vs).map(([k, v]) => `${v} ${k.replace("_", " ")}`).join(" · ")}
            </div>
          </div>
          <div className="row">
            <span className={`pill ${r.mode === "llm" ? "ok" : "warn"}`}>{r.mode === "llm" ? "LLM + verified spans" : "heuristic mode"}</span>
            {r.report_id && <a className="pill" href={api.markdownUrl(r.report_id)} target="_blank" rel="noreferrer">Download Markdown</a>}
          </div>
        </div>
        {r.limitations.map((l, i) => <div key={i} className="small muted">⚠ {l}</div>)}
      </div>

      <nav className="tabs">
        {(["review", "gaps", "ledger", "compare", "graph", "trace"] as const).map((t) => (
          <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>
            {{ review: "Literature review", gaps: "Research gaps", ledger: "Evidence ledger", compare: "Comparison", graph: "Knowledge graph", trace: "Agent trace" }[t]}
          </button>
        ))}
      </nav>

      <div className="grid2">
        <div>
          {tab === "review" && <div className="card"><Review md={r.literature_review} sel={sel} onCite={setSel} /></div>}

          {tab === "gaps" && (
            <>
              {r.gap_matrix && (
                <div className="card scroll">
                  <h3>Research gap matrix</h3>
                  <div className="small muted">● addressed in depth · ◐ mentioned in passing · ○ not detected (in this corpus)</div>
                  <table>
                    <thead><tr><th>Paper</th>{r.gap_matrix.aspects.map((a) => <th key={a}>{a}</th>)}</tr></thead>
                    <tbody>
                      {r.gap_matrix.papers.map((p, i) => (
                        <tr key={p}><td>{p}</td>{r.gap_matrix!.cells[i].map((c, j) => <td key={j}><span className={`dot ${c}`}>{SYM[c]}</span></td>)}</tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {r.gaps.map((g) => (
                <div className="card" key={g.title}>
                  <h3>{g.title}</h3>
                  <p>{g.description}</p>
                  <div className="small muted">Evidence to investigate</div>
                  <ul>{g.evidence_to_investigate.map((e) => <li key={e}>{e}</li>)}</ul>
                  <p><strong>Proposed direction:</strong> {g.proposed_direction}</p>
                  {g.supporting_evidence.flatMap((e) => e.supporting.slice(0, 1)).map((sp, i) => (
                    <blockquote key={i}>“{sp.quote}”<div className="small">{sp.paper_title} - {who(sp)}</div></blockquote>
                  ))}
                </div>
              ))}
            </>
          )}

          {tab === "ledger" && (
            <div className="card scroll">
              <h3>Evidence ledger - every extracted claim with its source span</h3>
              <table>
                <thead><tr><th>Status</th><th>Claim</th><th>Source</th><th>Passage</th></tr></thead>
                <tbody>
                  {r.claims.map((c, i) => (
                    <tr key={i}>
                      <td><Badge c={c} /></td>
                      <td>{c.claim}</td>
                      <td>{c.supporting[0] ? <>{c.supporting[0].paper_title}<div className="small muted">{who(c.supporting[0])}</div></> : "-"}</td>
                      <td>{c.supporting[0] ? `“${c.supporting[0].quote}”` : <span className="muted">{c.note}</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {conflicts.length > 0 && (
                <>
                  <h3 style={{ marginTop: 16 }}>Potential cross-paper conflicts (heuristic - review manually)</h3>
                  {conflicts.map((c, i) => (
                    <blockquote key={i}>{c.claim}{c.contradicting.map((x, k) => <div key={k} className="small">vs. “{x.quote}” - {x.paper_title}, p.{x.page}</div>)}</blockquote>
                  ))}
                </>
              )}
            </div>
          )}

          {tab === "compare" && (
            <div className="card scroll">
              <h3>Methods, datasets and metrics by paper</h3>
              <table>
                <thead><tr><th>Paper</th><th>Methods</th><th>Datasets</th><th>Metrics</th></tr></thead>
                <tbody>
                  {r.profiles.map((p) => {
                    const get = (k: string) => r.comparison[k]?.find((x) => x.paper === p.title)?.items ?? [];
                    return (
                      <tr key={p.paper_id}>
                        <td>{p.title}<div className="small muted">{p.authors.join(", ")} ({p.year ?? "n.d."})</div></td>
                        <td>{get("methods").map((m) => <span className="chip" key={m}>{m}</span>)}</td>
                        <td className="small">{get("datasets").join(" · ")}</td>
                        <td>{get("metrics").map((m) => <span className="chip" key={m}>{m}</span>)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {tab === "graph" && <div className="card"><GraphView g={r.knowledge_graph} /></div>}

          {tab === "trace" && (
            <div className="card">
              {r.trace.map((t, i) => (
                <div key={i} style={{ marginBottom: 8 }}>
                  <strong>{i + 1}. {t.agent}</strong> <span className="muted small">({t.duration_ms} ms)</span>
                  <div className="small muted">{t.summary}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <aside className="card sticky">
          <h3>Source passage</h3>
          {src ? (
            <>
              <div className="small muted">{sel}</div>
              <blockquote>“{src.quote}”</blockquote>
              <strong>{src.paper_title}</strong>
              <div className="small muted">{src.authors.join(", ") || "Unknown authors"} · {src.year ?? "n.d."} · page {src.page}</div>
            </>
          ) : (
            <div className="small muted">Click a citation chip like <span className="cite">S1</span> to see the exact passage, paper, authors, year and page it comes from.</div>
          )}
        </aside>
      </div>
    </div>
  );
}
