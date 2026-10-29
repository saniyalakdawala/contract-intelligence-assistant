"""Automated tests for character-level precision/recall calculation and span merging."""

from __future__ import annotations

import pytest

from contract_intelligence.dataset import GroundTruthSnippet
from contract_intelligence.evaluation import compute_char_metrics_at_k, sort_and_merge_spans
from contract_intelligence.retrieval import EvidencePassage


def test_sort_and_merge_spans_deduplicates_overlaps() -> None:
    """Verify overlapping and adjacent character spans merge cleanly without double counting."""
    spans = [(100, 300), (200, 450), (450, 500), (600, 700)]
    merged = sort_and_merge_spans(spans)
    assert merged == [(100, 500), (600, 700)]


def test_character_precision_and_recall_exact_math() -> None:
    """Verify character-level precision@k and recall@k follow LegalBench-RAG exact span overlap definitions."""
    gt = [
        GroundTruthSnippet(file_path="cuad/docA.txt", char_start=100, char_end=300, answer="x" * 200),
    ]
    # Retrieved chunk 1 covers [50, 250] in docA (200 chars retrieved, 150 chars overlap with [100, 300])
    # Retrieved chunk 2 covers [200, 350] in docA (overlaps chunk 1! Merged retrieved span is [50, 350] = 300 chars, overlap with [100, 300] is 200 chars!)
    # Retrieved chunk 3 is in wrong document docB [0, 200] (200 chars retrieved, 0 overlap)
    passages = [
        EvidencePassage(
            doc_id="cuad/docA.txt",
            file_path="cuad/docA.txt",
            char_start=50,
            char_end=250,
            passage_text="a" * 200,
            retrieval_method="test",
            rank=1,
            retrieval_score=0.9,
        ),
        EvidencePassage(
            doc_id="cuad/docA.txt",
            file_path="cuad/docA.txt",
            char_start=200,
            char_end=350,
            passage_text="b" * 150,
            retrieval_method="test",
            rank=2,
            retrieval_score=0.8,
        ),
        EvidencePassage(
            doc_id="cuad/docB.txt",
            file_path="cuad/docB.txt",
            char_start=0,
            char_end=200,
            passage_text="c" * 200,
            retrieval_method="test",
            rank=3,
            retrieval_score=0.7,
        ),
    ]

    # At k=1: retrieved=[50, 250] (200 chars), overlap=[100, 250] (150 chars), gt=[100, 300] (200 chars)
    m1 = compute_char_metrics_at_k(gt, passages, k=1)
    assert m1["precision"] == pytest.approx(150.0 / 200.0)
    assert m1["recall"] == pytest.approx(150.0 / 200.0)
    assert m1["doc_hit"] == 1.0

    # At k=2: merged retrieved in docA is [50, 350] (300 chars), overlap is [100, 300] (200 chars)
    # Without span deduplication, recall would erroneously exceed 1.0 (150+100)/200 = 1.25!
    m2 = compute_char_metrics_at_k(gt, passages, k=2)
    assert m2["recall"] == pytest.approx(1.0)
    assert m2["precision"] == pytest.approx(200.0 / 300.0)

    # At k=3: total retrieved is 300 (docA) + 200 (docB) = 500 chars; overlap is 200 chars
    m3 = compute_char_metrics_at_k(gt, passages, k=3)
    assert m3["recall"] == pytest.approx(1.0)
    assert m3["precision"] == pytest.approx(200.0 / 500.0)

    # Empty retrieved list (abstained) -> precision=0, recall=0
    m_empty = compute_char_metrics_at_k(gt, [], k=5)
    assert m_empty["precision"] == 0.0
    assert m_empty["recall"] == 0.0

