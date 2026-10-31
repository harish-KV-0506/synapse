import type { Health, Paper, ResearchReport, SearchHit, VerifiedClaim } from "./types";

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail ?? msg; } catch { /* non-JSON error body */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.json() as Promise<T>;
}
const post = (url: string, body: unknown) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  health: () => fetch("/api/health").then(j<Health>),
  papers: () => fetch("/api/papers").then(j<Paper[]>),
  upload: (files: FileList | File[]) => {
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    return fetch("/api/papers", { method: "POST", body: fd }).then(j<{ added: Paper[]; errors: { file: string; error: string }[] }>);
  },
  loadDemo: () => fetch("/api/demo/load", { method: "POST" }).then(j<{ added: Paper[]; errors: unknown[] }>),
  remove: (id: string) => fetch(`/api/papers/${id}`, { method: "DELETE" }).then(j<unknown>),
  clear: () => fetch("/api/papers", { method: "DELETE" }).then(j<unknown>),
  search: (question: string, top_k = 8) => post("/api/search", { question, top_k }).then(j<SearchHit[]>),
  analyze: (question: string) => post("/api/analyze", { question }).then(j<ResearchReport>),
  verify: (question: string) => post("/api/verify", { question }).then(j<VerifiedClaim>),
  markdownUrl: (id: string) => `/api/reports/${id}/markdown`,
};
