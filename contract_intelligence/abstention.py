"""Honest abstention decision layer and train-only threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from contract_intelligence.config import AbstentionConfig


@dataclass
class AbstentionDecision:
    """Structured abstention decision with transparent diagnostics."""

    abstained: bool
    status: str
    reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def evaluate_abstention(
    query: str,
    top_score: float,
    top_clause_score: float,
    top_doc_match_score: float,
    cfg: AbstentionConfig,
) -> AbstentionDecision:
    """Evaluate whether the retrieval system should abstain from returning passages."""
    q_stripped = (query or "").strip()
    if len(q_stripped) < 3 or not any(ch.isalnum() for ch in q_stripped):
        return AbstentionDecision(
            abstained=True,
            status="abstained_invalid_query",
            reason="No sufficiently relevant supporting passage was retrieved because the query is empty or malformed.",
            diagnostics={
                "top_score": 0.0,
                "top_clause_score": 0.0,
                "top_doc_match_score": 0.0,
                "mode": cfg.mode,
            },
        )

    diag = {
        "mode": cfg.mode,
        "top_score": round(float(top_score), 4),
        "top_clause_score": round(float(top_clause_score), 4),
        "top_doc_match_score": round(float(top_doc_match_score), 4),
        "min_top_score": cfg.min_top_score,
        "min_clause_score": cfg.min_clause_score,
        "min_doc_match_score": cfg.min_doc_match_score,
    }

    if not cfg.enabled:
        return AbstentionDecision(
            abstained=False,
            status="answered_with_evidence",
            reason="Abstention layer disabled; returning top ranked candidate passages.",
            diagnostics=diag,
        )

    if top_score < cfg.min_top_score:
        return AbstentionDecision(
            abstained=True,
            status="abstained_low_overall_score",
            reason=(
                f"No sufficiently relevant supporting passage was found in the indexed corpus "
                f"(top retrieval score {top_score:.3f} < threshold {cfg.min_top_score:.3f}). "
                "Note: this indicates insufficient retrieved evidence, not a legal guarantee of absence."
            ),
            diagnostics=diag,
        )

    if cfg.mode == "dual_gate":
        if top_doc_match_score < cfg.min_doc_match_score:
            return AbstentionDecision(
                abstained=True,
                status="abstained_unmatched_contract",
                reason=(
                    f"No indexed contract matched the target agreement or parties specified in the query "
                    f"(document alignment score {top_doc_match_score:.3f} < threshold {cfg.min_doc_match_score:.3f})."
                ),
                diagnostics=diag,
            )
        if top_clause_score < cfg.min_clause_score:
            return AbstentionDecision(
                abstained=True,
                status="abstained_missing_clause_evidence",
                reason=(
                    f"While the target agreement may be present, no passage within it sufficiently matched the requested legal provision "
                    f"(clause relevance score {top_clause_score:.3f} < threshold {cfg.min_clause_score:.3f}). "
                    "Note: this indicates insufficient retrieved evidence rather than definitive legal absence."
                ),
                diagnostics=diag,
            )

    return AbstentionDecision(
        abstained=False,
        status="answered_with_evidence",
        reason="Retrieved passage meets all configured evidence relevance thresholds.",
        diagnostics=diag,
    )


def calibrate_abstention_on_train(
    train_records: list[dict[str, Any]],
    mode: str = "dual_gate",
    max_false_abstention_rate: float = 0.06,
) -> dict[str, float]:
    """Grid-search abstention thresholds exclusively on train-split query diagnostics."""
    ans = [r for r in train_records if r["is_answerable"]]
    unans = [r for r in train_records if not r["is_answerable"]]
    if not ans or not unans:
        return {"min_top_score": 0.08, "min_clause_score": 0.0, "min_doc_match_score": 0.0}

    ans_top = np.array([r["top_score"] for r in ans], dtype=np.float64)
    ans_clause = np.array([r["top_clause_score"] for r in ans], dtype=np.float64)
    ans_doc = np.array([r["top_doc_match_score"] for r in ans], dtype=np.float64)

    unans_top = np.array([r["top_score"] for r in unans], dtype=np.float64)
    unans_clause = np.array([r["top_clause_score"] for r in unans], dtype=np.float64)
    unans_doc = np.array([r["top_doc_match_score"] for r in unans], dtype=np.float64)

    if mode == "single_threshold":
        best_t = 0.08
        best_obj = -1.0
        for t in np.linspace(0.02, 0.70, 140):
            false_abs = float(np.mean(ans_top < t))
            corr_abs = float(np.mean(unans_top < t))
            if false_abs <= max_false_abstention_rate:
                obj = corr_abs - 1.5 * false_abs
                if obj > best_obj:
                    best_obj = obj
                    best_t = float(t)
        return {
            "min_top_score": round(best_t, 4),
            "min_clause_score": 0.0,
            "min_doc_match_score": 0.0,
        }

    # Dual-gate grid search on train split only
    best_params = {
        "min_top_score": 0.12,
        "min_clause_score": 0.08,
        "min_doc_match_score": 0.25,
    }
    best_obj = -1e9

    top_grid = np.unique(np.round(np.quantile(ans_top, np.linspace(0.0, 0.12, 10)), 4))
    clause_grid = np.unique(np.round(np.quantile(ans_clause, np.linspace(0.0, 0.15, 12)), 4))
    doc_grid = np.unique(np.round(np.quantile(ans_doc, np.linspace(0.0, 0.15, 12)), 4))

    for t_top in top_grid:
        for t_clause in clause_grid:
            for t_doc in doc_grid:
                ans_abstain = (ans_top < t_top) | (ans_clause < t_clause) | (ans_doc < t_doc)
                unans_abstain = (unans_top < t_top) | (unans_clause < t_clause) | (unans_doc < t_doc)
                false_abs_rate = float(np.mean(ans_abstain))
                corr_abs_rate = float(np.mean(unans_abstain))
                if false_abs_rate <= max_false_abstention_rate:
                    obj = corr_abs_rate - 2.0 * false_abs_rate
                    if obj > best_obj:
                        best_obj = obj
                        best_params = {
                            "min_top_score": round(float(t_top), 4),
                            "min_clause_score": round(float(t_clause), 4),
                            "min_doc_match_score": round(float(t_doc), 4),
                        }

    return best_params

