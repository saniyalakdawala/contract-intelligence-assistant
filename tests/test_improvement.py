"""Automated tests for train/test isolation, guardrail enforcement, and improvement loop acceptance/rejection."""

from __future__ import annotations

import pytest

from contract_intelligence.improve import check_guardrails, run_improvement_loop


def test_improvement_loop_enforces_strict_train_isolation() -> None:
    """Verify passing split='test' or 'all' to run_improvement_loop raises RuntimeError immediately."""
    with pytest.raises(RuntimeError, match="Train/test isolation violation"):
        run_improvement_loop(split="test")

    with pytest.raises(RuntimeError, match="Train/test isolation violation"):
        run_improvement_loop(split="all")


def test_guardrail_acceptance_and_rejection_logic() -> None:
    """Verify check_guardrails accepts genuine improvements and rejects precision-bloated or slow candidates."""
    parent = {
        "char_recall@5": 0.7200,
        "char_precision@5": 0.1100,
        "false_abstention_rate": 0.0200,
        "correct_abstention_rate": 0.2000,
        "avg_latency_ms": 15.0,
    }

    # Good candidate: improves recall & abstention while preserving precision and latency
    good_cand = {
        "char_recall@5": 0.7550,
        "char_precision@5": 0.1050,
        "false_abstention_rate": 0.0350,
        "correct_abstention_rate": 0.8500,
        "avg_latency_ms": 22.0,
    }
    passed, details = check_guardrails(parent, good_cand)
    assert passed is True
    assert all(v["passed"] for v in details.values())

    # Bad candidate 1: oversized chunks bloat context and slash precision@5 to 0.040
    bloated_cand = {
        "char_recall@5": 0.8100,
        "char_precision@5": 0.0400,
        "false_abstention_rate": 0.0200,
        "correct_abstention_rate": 0.2000,
        "avg_latency_ms": 18.0,
    }
    passed_bloat, details_bloat = check_guardrails(parent, bloated_cand)
    assert passed_bloat is False
    assert details_bloat["precision_at_5_anti_bloat"]["passed"] is False

    # Bad candidate 2: excessive false abstention rate (> 6%)
    over_abstain_cand = {
        "char_recall@5": 0.7300,
        "char_precision@5": 0.1100,
        "false_abstention_rate": 0.1200,
        "correct_abstention_rate": 0.9500,
        "avg_latency_ms": 15.0,
    }
    passed_abs, details_abs = check_guardrails(parent, over_abstain_cand)
    assert passed_abs is False
    assert details_abs["false_abstention_ceiling"]["passed"] is False

