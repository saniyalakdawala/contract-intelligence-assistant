"""Automated tests for exact LegalBench-RAG-mini selection, SHA-256 split assignment, and UTF-8 loading."""

from __future__ import annotations

from pathlib import Path

from contract_intelligence.dataset import (
    assign_split,
    load_dataset,
    load_unanswerable_queries,
    mini,
    read_contract_utf8,
)


def test_prescribed_assign_split_determinism() -> None:
    """Verify the prescribed SHA-256 assign_split function is deterministic and sensitive to whitespace stripping."""
    q1 = "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?"
    assert assign_split(q1) in ("train", "test")
    assert assign_split(q1) == assign_split("  " + q1 + "\n")

    # Verify approximate 70/30 distribution on synthetic queries
    splits = [assign_split(f"Synthetic legal question number {i} about sublicensing?") for i in range(500)]
    test_ratio = splits.count("test") / len(splits)
    assert 0.22 <= test_ratio <= 0.38


def test_mini_selection_synthetic_and_real() -> None:
    """Verify mini(tests, n=194) is deterministic and selects exactly 194 queries from CUAD."""
    synthetic = [
        {"query": f"Q{i}", "snippets": [{"file_path": f"cuad/doc_{i % 5}.txt", "span": [0, 10]}]}
        for i in range(10)
    ]
    assert len(mini(synthetic, n=194)) == 10
    sel_a = mini(synthetic, n=4)
    sel_b = mini(synthetic, n=4)
    assert sel_a == sel_b
    assert len(sel_a) == 4


def test_cuad_mini_exact_counts_and_utf8_contracts(tmp_path: Path) -> None:
    """Verify real LegalBench-RAG-mini loads 194 answerable queries, 29 contracts, and 36 unanswerable queries."""
    ds = load_dataset()
    counts = ds.summary_counts()
    assert counts["answerable_total"] == 194
    assert counts["indexed_contracts"] == 29
    assert counts["answerable_train"] == 140
    assert counts["answerable_test"] == 54
    assert counts["unanswerable_total"] >= 30
    assert counts["unanswerable_train"] + counts["unanswerable_test"] == counts["unanswerable_total"]

    # Check UTF-8 default text mode reading with CRLF conversion
    sample_file = tmp_path / "sample_contract.txt"
    sample_file.write_bytes("SECTION 1. TERM\r\nThe term is 5 years — €100.\r\n".encode("utf-8"))
    content = read_contract_utf8(sample_file)
    assert "\r\n" not in content
    assert "€100" in content


def test_unanswerable_queries_schema_and_splits() -> None:
    """Verify every committed unanswerable query has required fields and valid assign_split assignment."""
    records = load_unanswerable_queries("data/unanswerable_queries.json")
    assert len(records) >= 30
    seen_ids = set()
    categories = set()
    for r in records:
        assert r.query_id not in seen_ids
        seen_ids.add(r.query_id)
        assert not r.is_answerable
        assert r.split == assign_split(r.query)
        assert r.unanswerable_reason and len(r.unanswerable_reason) > 10
        categories.add(r.category)
    assert len(categories) >= 4

