import os
import sys
from pathlib import Path

# Deterministic, offline test configuration (set BEFORE importing the app).
os.environ.setdefault("SYNAPSE_USE_NEURAL", "0")
os.environ.setdefault("SYNAPSE_DATA_DIR", str(Path(__file__).parent / ".data"))
os.environ.pop("ANTHROPIC_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    from make_sample_pdfs import make
    d = tmp_path_factory.mktemp("sample_papers")
    make(d)
    return d


@pytest.fixture(scope="session")
def loaded_corpus(sample_dir):
    from app.ingest.pdf_parser import parse_pdf
    from app.retrieval.hybrid import corpus
    corpus.clear()
    for p in sorted(sample_dir.glob("*.pdf")):
        r = parse_pdf(p)
        corpus.add(r.paper, r.chunks)
    return corpus
