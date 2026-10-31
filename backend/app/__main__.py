"""CLI: python -m app "question" [--pdf-dir DIR] -> prints Markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents.workflow import run_workflow
from .ingest.pdf_parser import parse_pdf
from .retrieval.hybrid import corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--pdf-dir", default="../sample_papers")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    for p in sorted(Path(a.pdf_dir).glob("*.pdf")):
        parsed = parse_pdf(p)
        corpus.add(parsed.paper, parsed.chunks)
    r = run_workflow(a.question)
    if a.json:
        print(r.model_dump_json(indent=2))
        return
    print(r.literature_review)
    print("\n## Sources")
    for sid, s in r.sources.items():
        print(f"[{sid}] {s.paper_title} ({', '.join(s.authors[:2]) or 'n/a'}, {s.year or 'n.d.'}), p.{s.page}: “{s.quote}”")
    print("\n## Gaps")
    for g in r.gaps:
        print(f"- {g.title}: {g.description}\n  -> {g.proposed_direction}")
    print("\nVerification:", r.verification_summary)
    print("Limitations:", *r.limitations, sep="\n - ")


if __name__ == "__main__":
    main()
