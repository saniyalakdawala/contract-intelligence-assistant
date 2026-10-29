"""Configuration management and schema validation for ClauseTrace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ChunkingConfig(BaseModel):
    """Configuration for contract chunking while preserving exact original character offsets."""

    strategy: Literal["fixed", "clause_aware"] = "fixed"
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=100, ge=0, le=5000)
    min_chunk_size: int = Field(default=80, ge=10, le=2000)
    respect_section_boundaries: bool = False

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})."
            )
        if self.min_chunk_size > self.chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) cannot exceed chunk_size ({self.chunk_size})."
            )
        return self


class LegalContextConfig(BaseModel):
    """Configuration for legal context enrichment (document headers, query expansion, cross-references)."""

    include_doc_header: bool = False
    header_char_limit: int = Field(default=900, ge=100, le=4000)
    doc_routing_weight: float = Field(default=0.0, ge=0.0, le=2.0)
    enable_legal_query_expansion: bool = False
    resolve_cross_references: bool = False


class RetrievalConfig(BaseModel):
    """Configuration for lexical, dense, and hybrid retrieval."""

    strategy: Literal["tfidf", "bm25", "dense", "hybrid"] = "tfidf"
    top_k: int = Field(default=10, ge=1, le=64)
    candidate_pool_size: int = Field(default=30, ge=5, le=200)

    # TF-IDF parameters
    tfidf_ngram_min: int = Field(default=1, ge=1, le=3)
    tfidf_ngram_max: int = Field(default=1, ge=1, le=3)
    tfidf_sublinear_tf: bool = True
    tfidf_max_features: int | None = Field(default=25000, ge=500)

    # BM25 parameters
    bm25_k1: float = Field(default=1.5, gt=0.0, le=5.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    # Dense parameters
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_batch_size: int = Field(default=64, ge=1, le=512)

    # Hybrid fusion parameters
    hybrid_lexical_type: Literal["tfidf", "bm25"] = "bm25"
    fusion_method: Literal["rrf", "linear"] = "linear"
    lexical_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=200)

    @model_validator(mode="after")
    def validate_params(self) -> "RetrievalConfig":
        if self.tfidf_ngram_max < self.tfidf_ngram_min:
            raise ValueError("tfidf_ngram_max must be >= tfidf_ngram_min.")
        if self.candidate_pool_size < self.top_k:
            raise ValueError("candidate_pool_size must be >= top_k.")
        if self.strategy == "hybrid" and (self.lexical_weight + self.dense_weight) <= 0:
            raise ValueError("At least one hybrid weight (lexical_weight or dense_weight) must be > 0.")
        return self


class RerankConfig(BaseModel):
    """Configuration for optional second-stage cross-encoder or legal feature reranking."""

    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = Field(default=20, ge=1, le=100)
    alpha: float = Field(default=0.35, ge=0.0, le=1.0)


class AbstentionConfig(BaseModel):
    """Configuration for honest abstention when evidence is insufficient."""

    enabled: bool = True
    mode: Literal["single_threshold", "dual_gate"] = "single_threshold"
    min_top_score: float = Field(default=0.08, ge=0.0, le=1.0)
    min_clause_score: float = Field(default=0.0, ge=0.0, le=1.0)
    min_doc_match_score: float = Field(default=0.0, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    """Complete ClauseTrace retrieval pipeline configuration."""

    name: str = "baseline_tfidf"
    description: str = "Baseline fixed-window TF-IDF retrieval pipeline."
    seed: int = 42
    data_dir: str = "data/legalbench_rag"
    unanswerable_path: str = "data/unanswerable_queries.json"
    cache_dir: str = "outputs/cache"

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    legal_context: LegalContextConfig = Field(default_factory=LegalContextConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)

    def index_fingerprint(self) -> str:
        """Compute a deterministic SHA-256 fingerprint for caching index artifacts."""
        relevant = {
            "seed": self.seed,
            "chunking": self.chunking.model_dump(),
            "legal_context": self.legal_context.model_dump(),
            "retrieval": {
                "strategy": self.retrieval.strategy,
                "tfidf_ngram_min": self.retrieval.tfidf_ngram_min,
                "tfidf_ngram_max": self.retrieval.tfidf_ngram_max,
                "tfidf_sublinear_tf": self.retrieval.tfidf_sublinear_tf,
                "tfidf_max_features": self.retrieval.tfidf_max_features,
                "bm25_k1": self.retrieval.bm25_k1,
                "bm25_b": self.retrieval.bm25_b,
                "dense_model_name": self.retrieval.dense_model_name,
                "hybrid_lexical_type": self.retrieval.hybrid_lexical_type,
            },
        }
        canonical = json.dumps(relevant, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def full_fingerprint(self) -> str:
        """Compute a deterministic SHA-256 fingerprint for the entire pipeline config."""
        canonical = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load and validate configuration from a YAML file."""
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {in_path}")
        with open(in_path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

