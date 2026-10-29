"""Automated tests for single-threshold and dual-gate abstention behavior and train-only calibration."""

from __future__ import annotations

from contract_intelligence.abstention import calibrate_abstention_on_train, evaluate_abstention
from contract_intelligence.config import AbstentionConfig


def test_single_threshold_and_dual_gate_abstention_decisions() -> None:
    """Verify abstention triggers accurately for low overall, low document match, or low clause scores."""
    cfg_single = AbstentionConfig(enabled=True, mode="single_threshold", min_top_score=0.25)

    # Below threshold -> abstain
    d1 = evaluate_abstention("What is the cap on liability?", 0.12, 0.10, 0.80, cfg_single)
    assert d1.abstained is True
    assert d1.status == "abstained_low_overall_score"

    # Above threshold -> answer
    d2 = evaluate_abstention("What is the cap on liability?", 0.42, 0.35, 0.80, cfg_single)
    assert d2.abstained is False
    assert d2.status == "answered_with_evidence"

    # Dual gate: high doc match (indexed contract) but missing clause (low clause score) -> abstain!
    cfg_dual = AbstentionConfig(
        enabled=True,
        mode="dual_gate",
        min_top_score=0.20,
        min_clause_score=0.15,
        min_doc_match_score=0.40,
    )
    d_missing_clause = evaluate_abstention(
        "Consider the Fox Mobile Agreement; What is the cap on liability?",
        top_score=0.38,
        top_clause_score=0.04,
        top_doc_match_score=0.85,
        cfg=cfg_dual,
    )
    assert d_missing_clause.abstained is True
    assert d_missing_clause.status == "abstained_missing_clause_evidence"

    # Dual gate: high clause match (generic clause) but unindexed contract (low doc match) -> abstain!
    d_unindexed_doc = evaluate_abstention(
        "Consider the Netflix / Sony Agreement; What is the governing law?",
        top_score=0.35,
        top_clause_score=0.45,
        top_doc_match_score=0.12,
        cfg=cfg_dual,
    )
    assert d_unindexed_doc.abstained is True
    assert d_unindexed_doc.status == "abstained_unmatched_contract"


def test_train_only_abstention_calibration() -> None:
    """Verify calibrate_abstention_on_train selects thresholds respecting false abstention ceiling."""
    synthetic_train = [
        {"is_answerable": True, "top_score": 0.55, "top_clause_score": 0.38, "top_doc_match_score": 0.78}
        for _ in range(25)
    ] + [
        {"is_answerable": False, "top_score": 0.30, "top_clause_score": 0.03, "top_doc_match_score": 0.75}
        for _ in range(6)
    ] + [
        {"is_answerable": False, "top_score": 0.22, "top_clause_score": 0.35, "top_doc_match_score": 0.10}
        for _ in range(6)
    ]

    params = calibrate_abstention_on_train(synthetic_train, mode="dual_gate", max_false_abstention_rate=0.05)
    assert 0.0 <= params["min_top_score"] <= 0.55
    assert 0.0 <= params["min_clause_score"] <= 0.38
    assert 0.0 <= params["min_doc_match_score"] <= 0.78

