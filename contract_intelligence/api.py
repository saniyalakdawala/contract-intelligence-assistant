"""FastAPI backend for ClauseTrace | Contract Intelligence Assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from contract_intelligence import __version__
from contract_intelligence.config import PipelineConfig
from contract_intelligence.dataset import LegalBenchMiniDataset, load_dataset
from contract_intelligence.retrieval import ClauseTraceRetriever
from contract_intelligence.validation import validate_ground_truth_offsets

app = FastAPI(
    title="ClauseTrace | Contract Intelligence API",
    description="Evidence-first legal clause retrieval API with exact character offsets and honest abstention.",
    version=__version__,
)

_DATASET: LegalBenchMiniDataset | None = None
_RETRIEVERS: dict[str, ClauseTraceRetriever] = {}


class SearchRequest(BaseModel):
    """Query payload for the /search endpoint."""

    query: str = Field(..., description="Natural-language question about an indexed contract.")
    top_k: int = Field(default=5, ge=1, le=20)
    config_path: str = Field(default="configs/final.yaml")


def _get_dataset() -> LegalBenchMiniDataset:
    global _DATASET
    if _DATASET is None:
        _DATASET = load_dataset()
    return _DATASET


def _get_retriever(config_path: str) -> ClauseTraceRetriever:
    path = Path(config_path)
    if not path.exists():
        path = Path("configs/baseline.yaml")
    cfg = PipelineConfig.from_yaml(path)
    fp = cfg.full_fingerprint()
    if fp not in _RETRIEVERS:
        ds = _get_dataset()
        _RETRIEVERS[fp] = ClauseTraceRetriever(ds.contracts, cfg, use_cache=True)
    return _RETRIEVERS[fp]


@app.get("/health")
def health() -> dict[str, Any]:
    ds = _get_dataset()
    return {
        "status": "ok",
        "version": __version__,
        "summary_counts": ds.summary_counts(),
    }


@app.get("/contracts")
def list_contracts() -> list[dict[str, Any]]:
    ds = _get_dataset()
    return [
        {
            "doc_id": doc.doc_id,
            "file_path": doc.file_path,
            "char_length": doc.char_length,
            "title_header": doc.title_header,
        }
        for doc in ds.contracts.values()
    ]


@app.get("/validate")
def validate_benchmark() -> dict[str, Any]:
    ds = _get_dataset()
    rep = validate_ground_truth_offsets(ds)
    return rep.to_dict()


@app.post("/search")
def search_clauses(req: SearchRequest) -> dict[str, Any]:
    try:
        retriever = _get_retriever(req.config_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    resp = retriever.search(req.query, top_k=req.top_k)
    return resp.to_dict()

