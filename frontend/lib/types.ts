// Mirrors backend/app/schemas.py
export type Status = "supported" | "partially_supported" | "unverified" | "contradicted";

export interface Paper { paper_id: string; title: string; authors: string[]; year: number | null; filename: string; num_pages: number; chunks: number; }
export interface Chunk { chunk_id: string; paper_id: string; page: number; text: string; section: string | null; }
export interface SearchHit { chunk: Chunk; paper_title: string; score: number; sources: Record<string, number>; }
export interface SourceSpan { paper_id: string; paper_title: string; authors: string[]; year: number | null; page: number; quote: string; chunk_id?: string | null; }
export interface VerifiedClaim { claim: string; status: Status; confidence: number; supporting: SourceSpan[]; contradicting: SourceSpan[]; note?: string | null; }
export interface ResearchGap { title: string; description: string; evidence_to_investigate: string[]; proposed_direction: string; supporting_evidence: VerifiedClaim[]; coverage: Record<string, number>; }
export interface GapMatrix { aspects: string[]; papers: string[]; cells: ("addressed" | "partial" | "not_found")[][]; }
export interface AgentTrace { agent: string; summary: string; duration_ms: number; }
export interface PaperProfile { paper_id: string; title: string; authors: string[]; year: number | null; objective: VerifiedClaim | null; methods: VerifiedClaim[]; datasets: VerifiedClaim[]; metrics: VerifiedClaim[]; results: VerifiedClaim[]; limitations: VerifiedClaim[]; }
export interface GraphNode { id: string; kind: string; label: string; }
export interface GraphEdge { source: string; target: string; rel: string; }
export interface ResearchReport {
  report_id: string | null; question: string; mode: "llm" | "heuristic"; literature_review: string;
  profiles: PaperProfile[]; comparison: Record<string, { paper: string; items: string[] }[]>;
  gap_matrix: GapMatrix | null; gaps: ResearchGap[]; claims: VerifiedClaim[];
  knowledge_graph: { nodes: GraphNode[]; edges: GraphEdge[]; stats: { nodes: number; edges: number } };
  sources: Record<string, SourceSpan>; trace: AgentTrace[]; limitations: string[]; verification_summary: Record<string, number>;
}
export interface Health { status: string; papers: number; chunks: number; mode: "llm" | "heuristic"; embedder: string; reranker: string; vector_backend: string; langfuse: boolean; }
