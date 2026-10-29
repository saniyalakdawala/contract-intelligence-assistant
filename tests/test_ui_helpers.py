"""Automated tests for the Streamlit sidebar configuration controls, backend integration, and UI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import (
    SAMPLE_QUERIES,
    SUPPORTED_STRATEGIES,
    build_abstention_dataframe,
    build_precision_recall_dataframe,
    build_ui_pipeline_config,
    highlight_query_terms,
    load_default_pipeline_config,
    load_experiment_jsonl,
    load_json_artifact,
)
from contract_intelligence.dataset import load_dataset
from contract_intelligence.retrieval import ClauseTraceRetriever
from contract_intelligence.validation import verify_evidence_offset


def test_ui_loads_real_evaluation_and_experiment_artifacts() -> None:
    """Verify the Streamlit app reads real evaluation and experiment logs from disk."""
    base_train = load_json_artifact("outputs/evaluation/baseline_train_metrics.json")
    final_train = load_json_artifact("outputs/evaluation/final_train_metrics.json")
    base_test = load_json_artifact("outputs/evaluation/baseline_test_metrics.json")
    final_test = load_json_artifact("outputs/evaluation/final_test_metrics.json")

    assert base_train is not None and base_train["split"] == "train"
    assert final_train is not None and final_train["split"] == "train"
    assert base_test is not None and base_test["split"] == "test"
    assert final_test is not None and final_test["split"] == "test"

    pr_df = build_precision_recall_dataframe(base_train["metrics"], final_train["metrics"])
    assert list(pr_df["Cutoff (K)"]) == ["K = 1", "K = 3", "K = 5", "K = 10"]
    assert float(pr_df.loc[pr_df["Cutoff (K)"] == "K = 5", "Final Recall@K"].iloc[0]) > 0.75

    abs_df = build_abstention_dataframe(base_test["metrics"], final_test["metrics"])
    assert len(abs_df) == 7

    exp_records = load_experiment_jsonl("experiments/experiment_log.jsonl")
    assert len(exp_records) == 8  # Baseline (0) + 7 iterations
    assert exp_records[0]["iteration"] == 0
    assert exp_records[-1]["iteration"] == 7
    assert exp_records[-1]["config_name"] == "iter7_final_calibrated_hybrid"


def test_ui_highlight_query_terms_escapes_html_and_highlights_keywords() -> None:
    """Verify highlight_query_terms safely escapes raw HTML and wraps query terms in <mark> tags."""
    raw_clause = "1.2 <script>alert(1)</script> Governing law shall be the laws of the State of Delaware."
    query = "What is the governing law for this contract?"
    rendered = highlight_query_terms(raw_clause, query)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert '<mark class="clause-hl">Governing</mark>' in rendered


def test_build_ui_pipeline_config_controls_and_immutability() -> None:
    """Verify every sidebar control updates PipelineConfig without mutating saved YAML files."""
    final_before = Path("configs/final.yaml").read_text(encoding="utf-8")
    baseline_before = Path("configs/baseline.yaml").read_text(encoding="utf-8")

    # Default UI state (top_k=5) has no temporary overrides
    cfg_exact, meta_exact = build_ui_pipeline_config(base_preset="final", top_k=5)
    assert meta_exact["is_temporary_ui_override"] is False
    assert cfg_exact.retrieval.top_k == 5
    assert len(meta_exact["overridden_fields"]) == 0

    # Changing strategy, top_k, abstention, fusion, and rerank updates config and diff metadata
    cfg_custom, meta_custom = build_ui_pipeline_config(
        base_preset="final",
        strategy="bm25",
        top_k=3,
        abstention_enabled=False,
        abstention_mode="single_threshold",
        min_top_score=0.25,
        fusion_method="rrf",
        rerank_enabled=True,
    )
    assert cfg_custom.retrieval.strategy == "bm25"
    assert cfg_custom.retrieval.top_k == 3
    assert cfg_custom.abstention.enabled is False
    assert cfg_custom.abstention.mode == "single_threshold"
    assert cfg_custom.abstention.min_top_score == 0.25
    assert cfg_custom.retrieval.fusion_method == "rrf"
    assert cfg_custom.rerank.enabled is True
    assert meta_custom["is_temporary_ui_override"] is True
    assert meta_custom["saved_fingerprint"] != meta_custom["active_fingerprint"]
    assert "retrieval.strategy" in meta_custom["overridden_fields"]
    assert "abstention.enabled" in meta_custom["overridden_fields"]

    # Invalid inputs raise ValueError
    with pytest.raises(ValueError):
        build_ui_pipeline_config(top_k=0)

    # Saved files on disk must be untouched
    assert Path("configs/final.yaml").read_text(encoding="utf-8") == final_before
    assert Path("configs/baseline.yaml").read_text(encoding="utf-8") == baseline_before


def test_all_four_strategies_and_top_k_values_on_real_corpus() -> None:
    """Test all 4 supported strategies (tfidf, bm25, dense, hybrid) and Top-K in {1, 3, 5} on the real corpus."""
    ds = load_dataset()
    sample_query = SAMPLE_QUERIES[1]["query"]  # BicycleTX governing law query

    for strat in SUPPORTED_STRATEGIES:
        cfg, _ = build_ui_pipeline_config(
            base_preset="final",
            strategy=strat,  # type: ignore[arg-type]
            top_k=5,
            abstention_enabled=True,
        )
        retriever = ClauseTraceRetriever(ds.contracts, cfg, use_cache=True)
        for k in (1, 3, 5):
            resp = retriever.search(sample_query, top_k=k)
            assert resp.retrieval_method == strat
            assert not resp.abstained
            assert len(resp.passages) == k
            assert [p.rank for p in resp.passages] == list(range(1, k + 1))
            for p in resp.passages:
                assert verify_evidence_offset(
                    ds.contracts, p.doc_id, p.char_start, p.char_end, p.passage_text
                )


def test_abstention_toggle_on_answerable_and_unanswerable_queries() -> None:
    """Verify enabling vs disabling the Abstention Gate changes backend abstention behavior."""
    ds = load_dataset()
    answerable_q = SAMPLE_QUERIES[1]["query"]
    unanswerable_q = SAMPLE_QUERIES[5]["query"]  # Missing cloud gaming sublicensing clause

    cfg_on, _ = build_ui_pipeline_config(base_preset="final", top_k=3, abstention_enabled=True)
    ret_on = ClauseTraceRetriever(ds.contracts, cfg_on, use_cache=True)

    # Answerable query returns passages when gate is ON
    resp_ans = ret_on.search(answerable_q, top_k=3)
    assert resp_ans.abstained is False
    assert len(resp_ans.passages) == 3

    # Unanswerable query abstains when gate is ON
    resp_unans_on = ret_on.search(unanswerable_q, top_k=3)
    assert resp_unans_on.abstained is True
    assert len(resp_unans_on.passages) == 0

    # When gate is OFF, the same unanswerable query returns top-K candidate passages
    cfg_off, _ = build_ui_pipeline_config(base_preset="final", top_k=3, abstention_enabled=False)
    ret_off = ClauseTraceRetriever(ds.contracts, cfg_off, use_cache=True)
    resp_unans_off = ret_off.search(unanswerable_q, top_k=3)
    assert resp_unans_off.abstained is False
    assert resp_unans_off.abstention.status == "answered_with_evidence"
    assert len(resp_unans_off.passages) == 3

    # Input validation safety check still rejects empty/punctuation-only queries even when gate is OFF
    resp_empty = ret_off.search("   ", top_k=3)
    assert resp_empty.abstained is True
    assert resp_empty.abstention.status == "abstained_invalid_query"
