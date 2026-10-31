# SYNAPSE
**A multi-agent scientific research intelligence and evidence-verification platform.**

Upload papers → ask a research question → a team of agents retrieves passages, extracts methods/datasets/metrics/limitations, audits every claim against the source text, builds a research-gap matrix, and writes an evidence-backed literature review in which **every citation resolves to a verbatim passage on a specific page**.

> Design rule: *the AI may not invent citations, and a generated summary is never treated as proof.*
> Citations are only ever built from text that physically exists in an uploaded PDF (source-span matching). If an LLM proposes a quote that is not found, it is discarded as evidence; any `[S#]` marker in LLM prose that does not resolve to a verified span is stripped.

## Architecture

```
PDFs ─► parse (PyMuPDF | pypdf) ─► page-aware chunks ─► BM25 ┐
                                                      embeddings ─► FAISS/numpy ┴► RRF fusion ─► cross-encoder rerank
                                                                                                   │
Question ─► Research Director ─► Literature Agent ─► Methodology Agent ─► Evidence Agent ─► Research Gap Agent ─► Report Writer
            (plans sub-queries)  (retrieve + rank)   (objective, methods,   (audit claims,     (gap matrix +        (review, tables,
                                                      datasets, metrics,    flag conflicts)     proposals)           graph, sources)
                                                      results, limitations)
                                       every extracted fact ─► Verification layer (span matching) ─► VerifiedClaim
```

| Spec item | Implementation |
|---|---|
| PDF intelligence | `ingest/pdf_parser.py` – page-aware text, title/author/year inference, section tracking, sentence-aware chunks that keep their page number |
| Semantic retrieval | `retrieval/embeddings.py` – Sentence-Transformers (default `all-MiniLM-L6-v2`); offline hashing-n-gram fallback |
| Hybrid search | `retrieval/bm25.py` + dense, fused with Reciprocal Rank Fusion (`retrieval/hybrid.py`) |
| Reranking | `retrieval/reranker.py` – cross-encoder; lexical fallback |
| Vector store | FAISS if installed, exact numpy cosine otherwise (`retrieval/vector_store.py`) |
| Agent workflow | `agents/workflow.py` – LangGraph `StateGraph`; plain sequential runner if LangGraph is absent |
| Knowledge graph | `knowledge_graph.py` – NetworkX (Paper→Method/Dataset/Metric/Aspect); same triples can be written to Neo4j |
| Citation verification | `verification/span_match.py` – verbatim/fuzzy span matching, term coverage, cross-paper polarity conflicts |
| Evaluation | `evaluation/` – Recall@k, MRR, nDCG, retrieval ablation, citation validity & span faithfulness |
| Observability | optional Langfuse trace (`SYNAPSE_LANGFUSE=1`) + per-agent trace shown in the UI |
| Frontend | Next.js + TypeScript (`frontend/`) |

### Two modes
* **Heuristic mode (no API key):** extraction is lexicon/pattern based over each paper's own sentences. Every item is a real sentence, so it is always cited, but it is pattern matching, not understanding; recall is limited to the built-in method/metric lexicons.
* **LLM mode (`ANTHROPIC_API_KEY` set):** the model proposes `{claim, quote}` items and writes the review from numbered evidence only. Quotes are span-checked; unresolvable citations are stripped.

### What each claim carries
Paper title · authors · year · page number · verbatim source passage · extracted claim · supporting and contradicting spans · status (`supported` / `partially_supported` / `unverified` / `contradicted`) · confidence.

### Gap analysis semantics
A sentence that mentions an aspect **while saying it was missing** ("generalization to other locations was not evaluated") counts as evidence *of a gap*, not as coverage. "Not detected" means *not found in the uploaded corpus* – it is not proof the topic is unexplored in the literature, and every report says so.

## Quick start

```bash
# 1. Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt     # recommended: neural retrieval, FAISS, LLM mode
python scripts/make_sample_pdfs.py ../sample_papers   # already generated; re-run to regenerate
uvicorn app.main:app --reload --port 8000

# 2. Frontend (new terminal)
cd frontend && npm install && npm run dev      # http://localhost:3000
```
Open the UI → **Corpus → "Load synthetic demo papers"** → **Analyze → Run research team**.

No UI needed:
```bash
cd backend && python -m app "How can machine learning improve satellite communication link performance under heavy rainfall?" --pdf-dir ../sample_papers
```
Docker: `cp .env.example .env && docker compose up --build` (UI on :3000, API on :8000).

> The bundled demo PDFs are **synthetic and fictional** (each says so on its first page). They exist to exercise the software. Never cite them.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | active embedder / reranker / vector backend / mode |
| POST | `/api/papers` | upload PDFs (multipart `files`) |
| POST | `/api/demo/load` | load bundled synthetic papers |
| GET/DELETE | `/api/papers[/{id}]` | list / remove |
| POST | `/api/search` | hybrid retrieval `{question, top_k}` |
| POST | `/api/analyze` | run the five-agent workflow → `ResearchReport` |
| GET | `/api/reports/{id}[/markdown]` | stored report / downloadable Markdown |
| POST | `/api/verify` | check a free-text claim: supporting vs contradicting passages |

## Evaluation
```bash
cd backend
python -m pytest -q tests                       # 11 tests
python -m app.evaluation.run_eval               # retrieval ablation + citation quality
```
Ground truth lives in `eval/ground_truth.json` (question → paper + phrase that must appear in a retrieved chunk).

Result of the last run (**offline fallback embedder, 5-paper synthetic corpus, 5 queries – a smoke test, not a benchmark**):

| config | recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| BM25 only | 0.80 | 0.70 | 0.85 |
| dense only (hashing) | 0.67 | 0.54 | 0.56 |
| hybrid (RRF) | 0.87 | 0.61 | 0.66 |
| hybrid + rerank | 0.87 | 0.64 | 0.78 |

Citation quality on the demo question: 42/42 citations point to a real paper+page and 42/42 quotes are verbatim on that page. On this tiny corpus BM25 alone matches hybrid on ranking quality; the hybrid advantage should be measured with the neural embedder on a real corpus before claiming it. RAGAS is not wired in yet (it needs an LLM judge) – a natural next step.

## Layout
```
backend/app/{ingest,retrieval,verification,agents,evaluation}/  core pipeline
backend/app/{main,export,knowledge_graph,llm,config,schemas}.py
backend/tests/   eval/ground_truth.json   sample_papers/   frontend/
```

## Known limitations
* Scanned PDFs need OCR first (no OCR built in). Two-column layouts can interleave text; PyMuPDF handles this better than the pypdf fallback.
* Title/author inference is heuristic; check it on your own papers.
* Heuristic-mode contradiction detection is a polarity check – it flags candidates for human review, it does not adjudicate.
* Reports are held in memory (lost on restart); the corpus persists to `data/corpus.json`.

## Roadmap
Qdrant backend · Neo4j export · section-aware extraction of result tables · RAGAS faithfulness/answer-relevancy runs · OCR ingestion · streaming agent progress (SSE).
