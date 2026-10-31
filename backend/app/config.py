"""Central configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("SYNAPSE_DATA_DIR", "./data")).resolve())
    # Chunking
    chunk_words: int = int(os.getenv("SYNAPSE_CHUNK_WORDS", "180"))
    chunk_overlap: int = int(os.getenv("SYNAPSE_CHUNK_OVERLAP", "40"))
    # Retrieval
    top_k_dense: int = int(os.getenv("SYNAPSE_TOP_K_DENSE", "30"))
    top_k_bm25: int = int(os.getenv("SYNAPSE_TOP_K_BM25", "30"))
    top_k_final: int = int(os.getenv("SYNAPSE_TOP_K_FINAL", "8"))
    max_papers: int = int(os.getenv("SYNAPSE_MAX_PAPERS", "15"))          # papers analysed per question
    paper_relevance_floor: float = float(os.getenv("SYNAPSE_PAPER_FLOOR", "0.10"))  # vs. best paper's score
    embedding_model: str = os.getenv("SYNAPSE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("SYNAPSE_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_neural: bool = _env_bool("SYNAPSE_USE_NEURAL", True)  # falls back automatically if libs are missing
    # LLM (optional)
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    llm_model: str = os.getenv("SYNAPSE_LLM_MODEL", "claude-sonnet-5-5")
    # Verification
    verify_threshold_supported: float = float(os.getenv("SYNAPSE_VERIFY_SUPPORTED", "0.80"))
    verify_threshold_partial: float = float(os.getenv("SYNAPSE_VERIFY_PARTIAL", "0.50"))
    # Observability (optional)
    langfuse_enabled: bool = _env_bool("SYNAPSE_LANGFUSE", False)

    @property
    def uploads_dir(self) -> Path:
        p = self.data_dir / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def index_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "corpus.json"


settings = Settings()
