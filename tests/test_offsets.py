"""Automated tests for ground-truth span validation and chunk-to-original character offset invariants."""

from __future__ import annotations

from contract_intelligence.chunking import chunk_document
from contract_intelligence.config import ChunkingConfig, LegalContextConfig
from contract_intelligence.dataset import ContractDocument, load_dataset
from contract_intelligence.validation import validate_ground_truth_offsets, verify_evidence_offset


SYNTHETIC_CONTRACT_TEXT = (
    "COMMERCIAL LICENSE AGREEMENT\n\n"
    "This Commercial License Agreement ('Agreement') is entered into as of January 15, 2024 "
    "by and between Acme Media Corp. ('Licensor') and Star Stream Ltd. ('Licensee').\n\n"
    "ARTICLE I\nDEFINITIONS\n\n"
    "1.1 'Licensed Territory' means India, Singapore, and the United Kingdom.\n\n"
    "SECTION 2. GRANT OF RIGHTS\n\n"
    "2.1 Subject to Section 4.2, Licensor hereby grants to Licensee an exclusive license to stream "
    "the Programs during the Term. Licensee shall not sublicense the Programs without prior written consent.\n\n"
    "SECTION 4. LIMITATION OF LIABILITY\n\n"
    "4.2 Neither party's aggregate liability shall exceed $500,000 under any theory of law.\n"
)


def test_ground_truth_span_validation_on_cuad_mini() -> None:
    """Verify all 272 ground-truth snippets in LegalBench-RAG-mini match document[start:end] exactly."""
    ds = load_dataset()
    report = validate_ground_truth_offsets(ds)
    assert report.is_valid
    assert report.total_queries_checked == 194
    assert report.total_snippets_checked == 272
    assert report.exact_matches == 272
    assert len(report.mismatches) == 0


def test_fixed_and_clause_aware_chunk_offsets_match_original_document() -> None:
    """Verify every chunk produced by fixed and clause_aware strategies satisfies doc.text[start:end] == raw_text."""
    doc = ContractDocument(
        doc_id="synthetic/acme_star_license.txt",
        file_path="synthetic/acme_star_license.txt",
        text=SYNTHETIC_CONTRACT_TEXT,
        char_length=len(SYNTHETIC_CONTRACT_TEXT),
        title_header="[Document: acme star license]",
    )

    for strategy in ("fixed", "clause_aware"):
        cfg = ChunkingConfig(
            strategy=strategy,  # type: ignore[arg-type]
            chunk_size=180,
            chunk_overlap=40,
            min_chunk_size=30,
            respect_section_boundaries=True,
        )
        legal_cfg = LegalContextConfig(
            include_doc_header=True,
            resolve_cross_references=True,
        )
        chunks = chunk_document(doc, cfg, legal_cfg)
        assert len(chunks) >= 3
        for ch in chunks:
            assert 0 <= ch.char_start < ch.char_end <= len(doc.text)
            assert doc.text[ch.char_start : ch.char_end] == ch.raw_text
            assert verify_evidence_offset(
                {doc.doc_id: doc}, doc.doc_id, ch.char_start, ch.char_end, ch.raw_text
            )
            # Even when index_text has header/cross-ref enrichment, raw_text is untouched
            assert ch.raw_text in ch.index_text

    # Verify cross-reference detection found 'Section 4.2'
    clause_chunks = chunk_document(
        doc,
        ChunkingConfig(strategy="clause_aware", chunk_size=260, chunk_overlap=40),
        LegalContextConfig(include_doc_header=True, resolve_cross_references=True),
    )
    all_xrefs = [ref for ch in clause_chunks for ref in ch.cross_references]
    assert any("4.2" in ref for ref in all_xrefs)

