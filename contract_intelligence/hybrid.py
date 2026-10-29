"""Score normalization, hybrid lexical+dense fusion (RRF & linear), and cross-encoder reranking."""

from __future__ import annotations

import math
import numpy as np

from contract_intelligence.chunking import ContractChunk
from contract_intelligence.config import RerankConfig
from contract_intelligence.indexing import get_cross_encoder_model


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize a 1D array of scores into [0, 1] while preserving zero arrays."""
    if scores.size == 0:
        return scores
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if s_max <= 1e-12 or (s_max - s_min) <= 1e-12:
        return np.clip(scores, 0.0, 1.0)
    return (scores - s_min) / (s_max - s_min)


def bm25_saturation_normalize(scores: np.ndarray, k_sat: float = 8.0) -> np.ndarray:
    """Map unbounded non-negative BM25 scores into [0, 1) via monotonic saturation s / (s + k_sat)."""
    pos = np.maximum(scores, 0.0)
    return pos / (pos + k_sat)


def reciprocal_rank_fusion(
    lexical_scores: np.ndarray,
    dense_scores: np.ndarray,
    lexical_weight: float = 0.65,
    dense_weight: float = 0.35,
    rrf_k: int = 60,
) -> np.ndarray:
    """Combine lexical and dense rankings using weighted Reciprocal Rank Fusion (RRF), scaled to [0, 1]."""
    n = lexical_scores.size
    if n == 0:
        return np.array([], dtype=np.float64)

    lex_order = np.argsort(-lexical_scores)
    dense_order = np.argsort(-dense_scores)

    lex_ranks = np.empty(n, dtype=np.int32)
    dense_ranks = np.empty(n, dtype=np.int32)
    lex_ranks[lex_order] = np.arange(1, n + 1)
    dense_ranks[dense_order] = np.arange(1, n + 1)

    total_w = max(1e-9, lexical_weight + dense_weight)
    lw = lexical_weight / total_w
    dw = dense_weight / total_w

    rrf_raw = lw * (1.0 / (rrf_k + lex_ranks)) + dw * (1.0 / (rrf_k + dense_ranks))
    # Scale by (rrf_k + 1) so the theoretical maximum rank-1 agreement is 1.0
    rrf_scaled = rrf_raw * float(rrf_k + 1)
    # Modulate slightly by actual score magnitude so unanswerable queries don't all get 1.0
    lex_cal = bm25_saturation_normalize(lexical_scores) if np.max(lexical_scores) > 1.5 else np.clip(lexical_scores, 0.0, 1.0)
    dense_cal = np.clip(dense_scores, 0.0, 1.0)
    mag = lw * lex_cal + dw * dense_cal
    return 0.5 * rrf_scaled + 0.5 * mag


def linear_score_fusion(
    lexical_scores: np.ndarray,
    dense_scores: np.ndarray,
    lexical_type: str = "bm25",
    lexical_weight: float = 0.65,
    dense_weight: float = 0.35,
) -> np.ndarray:
    """Combine calibrated lexical and dense scores via convex linear combination."""
    if lexical_type == "bm25":
        lex_cal = bm25_saturation_normalize(lexical_scores, k_sat=8.0)
    else:
        lex_cal = np.clip(lexical_scores, 0.0, 1.0)
    dense_cal = np.clip(dense_scores, 0.0, 1.0)
    total_w = max(1e-9, lexical_weight + dense_weight)
    lw = lexical_weight / total_w
    dw = dense_weight / total_w
    return lw * lex_cal + dw * dense_cal


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


import pickle
from pathlib import Path

_CE_PAIR_CACHE: dict[tuple[str, str, str], float] = {}
_CE_CACHE_LOADED = False


def apply_cross_encoder_rerank(
    query_for_rerank: str,
    chunks: list[ContractChunk],
    candidate_indices: list[int],
    base_scores: np.ndarray,
    rerank_cfg: RerankConfig,
) -> tuple[list[int], dict[int, float]]:
    """Rerank top candidate chunks using a local CrossEncoder and return re-ordered indices + reranker scores."""
    global _CE_CACHE_LOADED
    if not rerank_cfg.enabled or not candidate_indices:
        return candidate_indices, {}

    cache_file = Path("outputs/cache/ce_pair_cache.pkl")
    if not _CE_CACHE_LOADED and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                _CE_PAIR_CACHE.update(pickle.load(f))
        except Exception:
            pass
        _CE_CACHE_LOADED = True

    top_n = min(rerank_cfg.top_n, len(candidate_indices))
    head_indices = candidate_indices[:top_n]
    tail_indices = candidate_indices[top_n:]

    model_name = rerank_cfg.model_name
    missing_indices = [
        idx
        for idx in head_indices
        if (model_name, query_for_rerank, chunks[idx].chunk_id) not in _CE_PAIR_CACHE
    ]
    if missing_indices:
        model = get_cross_encoder_model(model_name)
        pairs = [(query_for_rerank[:250], chunks[idx].raw_text[:650]) for idx in missing_indices]
        preds = model.predict(pairs, batch_size=64, show_progress_bar=False)
        for idx, logit in zip(missing_indices, preds):
            _CE_PAIR_CACHE[(model_name, query_for_rerank, chunks[idx].chunk_id)] = float(logit)
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(_CE_PAIR_CACHE, f)
        except Exception:
            pass

    rerank_scores_map: dict[int, float] = {}
    blended_head: list[tuple[int, float]] = []
    alpha = rerank_cfg.alpha
    for idx in head_indices:
        logit = _CE_PAIR_CACHE[(model_name, query_for_rerank, chunks[idx].chunk_id)]
        ce_prob = sigmoid(float(logit) / 3.0)
        rerank_scores_map[idx] = ce_prob
        blended = (1.0 - alpha) * float(base_scores[idx]) + alpha * ce_prob
        base_scores[idx] = blended
        blended_head.append((idx, blended))

    blended_head.sort(key=lambda x: x[1], reverse=True)
    reordered = [idx for idx, _ in blended_head] + tail_indices
    return reordered, rerank_scores_map
