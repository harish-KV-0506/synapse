"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { Health, Paper, ResearchReport, SearchHit, VerifiedClaim } from "@/lib/types";
import { ReportView, who } from "@/components/ReportView";

const EXAMPLE = "How can machine learning improve satellite communication link performance under heavy rainfall?";
type Tab = "corpus" | "search" | "analyze" | "verify";

export default function Home() {
  const [tab, setTab] = useState<Tab>("corpus");
  const [health, setHealth] = useState<Health | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [q, setQ] = useState(EXAMPLE);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [claim, setClaim] = useState("The Transformer improves over the LSTM in heavy rain");
  const [verdict, setVerdict] = useState<VerifiedClaim | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try { setHealth(await api.health()); setPapers(await api.papers()); setErr(null); }
    catch (e) { setErr(`Backend not reachable: ${(e as Error).message}. Start it with: uvicorn app.main:app --port 8000`); }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  async function run<T>(label: string, fn: () => Promise<T>, then?: (v: T) => void) {
    setBusy(label); setErr(null);
    try { const v = await fn(); then?.(v); await refresh(); }
    catch (e) { setErr((e as Error).message); }
    finally { setBusy(null); }
  }

  return (
    <div className="wrap">
      <header className="top">
        <h1>SYNAPSE</h1>
        <span className="sub">Multi-agent research intelligence &amp; evidence verification</span>
        {health && (
          <>
            <span className="pill ok">{health.papers} papers · {health.chunks} passages</span>
            <span className={`pill ${health.mode === "llm" ? "ok" : "warn"}`} title="LLM mode needs ANTHROPIC_API_KEY">{health.mode} mode</span>
            <span className="pill" title="Embedder / reranker / vector store">{health.embedder} · {health.reranker} · {health.vector_backend}</span>
          </>
        )}
      </header>
      {err && <div className="err">{err}</div>}

      <nav className="tabs">
        {([["corpus", "1 · Corpus"], ["search", "2 · Search"], ["analyze", "3 · Analyze"], ["verify", "4 · Verify a claim"]] as [Tab, string][]).map(([t, l]) => (
          <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{l}</button>
        ))}
      </nav>

      {tab === "corpus" && (
        <>
          <div className="card">
            <h3>Upload research papers (PDF)</h3>
            <div className="row">
              <input ref={fileRef} type="file" accept="application/pdf" multiple />
              <button className="btn" disabled={!!busy} onClick={() => {
                const f = fileRef.current?.files;
                if (f?.length) run("upload", () => api.upload(f), (r) => { if (r.errors.length) setErr(r.errors.map((e) => `${e.file}: ${e.error}`).join("; ")); });
              }}>{busy === "upload" ? "Parsing…" : "Upload & index"}</button>
              <button className="btn ghost" disabled={!!busy} onClick={() => run("demo", api.loadDemo)}>Load synthetic demo papers</button>
              {papers.length > 0 && <button className="btn ghost" disabled={!!busy} onClick={() => run("clear", api.clear)}>Clear all</button>}
            </div>
            <div className="small muted" style={{ marginTop: 8 }}>
              Demo papers are <strong>fictional</strong> documents for testing the software. Do not cite them.
            </div>
          </div>
          {papers.map((p) => (
            <div className="card" key={p.paper_id}>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <div>
                  <strong>{p.title}</strong>
                  <div className="small muted">{p.authors.join(", ") || "authors not detected"} · {p.year ?? "year not detected"} · {p.num_pages} pages · {p.chunks} passages · {p.filename}</div>
                </div>
                <button className="btn ghost" onClick={() => run("rm", () => api.remove(p.paper_id))}>Remove</button>
              </div>
            </div>
          ))}
        </>
      )}

      {tab === "search" && (
        <>
          <div className="card">
            <div className="row">
              <input type="text" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search passages…" />
              <button className="btn" disabled={!!busy || !papers.length} onClick={() => run("search", () => api.search(q), setHits)}>Hybrid search</button>
            </div>
            <div className="small muted" style={{ marginTop: 6 }}>BM25 + dense embeddings fused with RRF, then cross-encoder reranked.</div>
          </div>
          {hits.map((h) => (
            <div className="card" key={h.chunk.chunk_id}>
              <strong>{h.paper_title}</strong> <span className="small muted">p.{h.chunk.page}{h.chunk.section ? ` · ${h.chunk.section}` : ""} · score {h.score.toFixed(3)}</span>
              <p style={{ margin: "6px 0 0" }}>{h.chunk.text}</p>
              <div className="small muted">{Object.entries(h.sources).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(" · ")}</div>
            </div>
          ))}
        </>
      )}

      {tab === "analyze" && (
        <>
          <div className="card">
            <h3>Research question</h3>
            <textarea value={q} onChange={(e) => setQ(e.target.value)} />
            <div className="row" style={{ marginTop: 8 }}>
              <button className="btn" disabled={!!busy || !papers.length} onClick={() => run("analyze", () => api.analyze(q), setReport)}>
                {busy === "analyze" ? "Agents working…" : "Run research team"}
              </button>
              {!papers.length && <span className="small muted">Upload papers first (tab 1).</span>}
            </div>
            <div className="small muted" style={{ marginTop: 6 }}>Director → Literature → Methodology → Evidence → Research Gap → Report</div>
          </div>
          {report && <ReportView r={report} />}
        </>
      )}

      {tab === "verify" && (
        <>
          <div className="card">
            <h3>Check a claim against the corpus</h3>
            <textarea value={claim} onChange={(e) => setClaim(e.target.value)} />
            <div className="row" style={{ marginTop: 8 }}>
              <button className="btn" disabled={!!busy || !papers.length} onClick={() => run("verify", () => api.verify(claim), setVerdict)}>Verify</button>
            </div>
          </div>
          {verdict && (
            <div className="card">
              <span className={`badge ${verdict.status}`}>{verdict.status.replace("_", " ")} · {verdict.confidence.toFixed(2)}</span>
              <p className="small muted">{verdict.note}</p>
              {verdict.supporting.map((s, i) => <blockquote key={`s${i}`}>Supports: “{s.quote}”<div className="small">{s.paper_title} - {who(s)}</div></blockquote>)}
              {verdict.contradicting.map((s, i) => <blockquote key={`c${i}`}>Contradicts: “{s.quote}”<div className="small">{s.paper_title} - {who(s)}</div></blockquote>)}
            </div>
          )}
        </>
      )}
    </div>
  );
}
