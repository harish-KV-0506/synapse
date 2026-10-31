"""python -m app.evaluation.run_eval [--pdf-dir ../sample_papers] [--gt ../eval/ground_truth.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..agents.workflow import run_workflow
from ..ingest.pdf_parser import parse_pdf
from ..retrieval.hybrid import corpus
from .metrics import ablation, citation_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="../sample_papers")
    ap.add_argument("--gt", default="../eval/ground_truth.json")
    a = ap.parse_args()
    for p in sorted(Path(a.pdf_dir).glob("*.pdf")):
        parsed = parse_pdf(p)
        corpus.add(parsed.paper, parsed.chunks)
    gt = json.loads(Path(a.gt).read_text())
    print("== Retrieval ablation ==")
    print(json.dumps(ablation(corpus, gt), indent=2))
    q = gt[0]["question"]
    print(f"\n== Citation quality for: {q}")
    print(json.dumps(citation_metrics(run_workflow(q), corpus), indent=2))


if __name__ == "__main__":
    main()
