"""Ground-truth character offset validation and runtime evidence span verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contract_intelligence.dataset import ContractDocument, LegalBenchMiniDataset, QueryRecord


@dataclass
class SnippetMismatch:
    """Record of any discrepancy between ground-truth span and supplied snippet text."""

    query_id: str
    file_path: str
    span: tuple[int, int]
    expected_answer: str
    extracted_text: str
    reason: str


@dataclass
class ValidationReport:
    """Summary of benchmark ground-truth validation."""

    total_queries_checked: int
    total_snippets_checked: int
    exact_matches: int
    mismatches: list[SnippetMismatch] = field(default_factory=list)
    overlapping_gt_spans: int = 0
    unanswerable_queries_checked: int = 0
    crlf_simulation_mismatches: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.mismatches) == 0 and self.overlapping_gt_spans == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_queries_checked": self.total_queries_checked,
            "total_snippets_checked": self.total_snippets_checked,
            "exact_matches": self.exact_matches,
            "mismatch_count": len(self.mismatches),
            "overlapping_gt_spans": self.overlapping_gt_spans,
            "unanswerable_queries_checked": self.unanswerable_queries_checked,
            "crlf_simulation_mismatches": self.crlf_simulation_mismatches,
            "mismatches": [
                {
                    "query_id": m.query_id,
                    "file_path": m.file_path,
                    "span": list(m.span),
                    "reason": m.reason,
                    "expected_preview": m.expected_answer[:120],
                    "extracted_preview": m.extracted_text[:120],
                }
                for m in self.mismatches
            ],
        }


def validate_ground_truth_offsets(dataset: LegalBenchMiniDataset) -> ValidationReport:
    """Validate ground-truth character offsets against supplied answer snippets without modifying the benchmark."""
    total_snippets = 0
    exact_matches = 0
    mismatches: list[SnippetMismatch] = []
    overlapping_gt = 0
    crlf_mismatches = 0

    for q in dataset.answerable_queries:
        by_file: dict[str, list[tuple[int, int]]] = {}
        for s in q.snippets:
            total_snippets += 1
            by_file.setdefault(s.file_path, []).append((s.char_start, s.char_end))

            if s.file_path not in dataset.contracts:
                mismatches.append(
                    SnippetMismatch(
                        query_id=q.query_id,
                        file_path=s.file_path,
                        span=(s.char_start, s.char_end),
                        expected_answer=s.answer,
                        extracted_text="",
                        reason="Referenced contract missing from indexed corpus.",
                    )
                )
                continue

            doc = dataset.contracts[s.file_path]
            if s.char_start < 0 or s.char_end > doc.char_length or s.char_start >= s.char_end:
                mismatches.append(
                    SnippetMismatch(
                        query_id=q.query_id,
                        file_path=s.file_path,
                        span=(s.char_start, s.char_end),
                        expected_answer=s.answer,
                        extracted_text="",
                        reason=f"Invalid span bounds [{s.char_start}, {s.char_end}] for document length {doc.char_length}.",
                    )
                )
                continue

            extracted = doc.text[s.char_start : s.char_end]
            if extracted == s.answer:
                exact_matches += 1
            else:
                mismatches.append(
                    SnippetMismatch(
                        query_id=q.query_id,
                        file_path=s.file_path,
                        span=(s.char_start, s.char_end),
                        expected_answer=s.answer,
                        extracted_text=extracted,
                        reason="Extracted substring does not match snippet['answer'].",
                    )
                )

            # Check how CRLF (binary/untranslated Windows newlines) would corrupt offsets
            crlf_doc = doc.text.replace("\n", "\r\n")
            if crlf_doc[s.char_start : s.char_end] != s.answer:
                crlf_mismatches += 1

        # Verify ground-truth spans per document are disjoint
        for fp, spans in by_file.items():
            spans_sorted = sorted(spans, key=lambda x: x[0])
            for i in range(1, len(spans_sorted)):
                if spans_sorted[i - 1][1] > spans_sorted[i][0]:
                    overlapping_gt += 1

    return ValidationReport(
        total_queries_checked=len(dataset.answerable_queries),
        total_snippets_checked=total_snippets,
        exact_matches=exact_matches,
        mismatches=mismatches,
        overlapping_gt_spans=overlapping_gt,
        unanswerable_queries_checked=len(dataset.unanswerable_queries),
        crlf_simulation_mismatches=crlf_mismatches,
    )


def verify_evidence_offset(
    contracts: dict[str, ContractDocument] | dict[str, str],
    doc_id: str,
    char_start: int,
    char_end: int,
    passage_text: str,
) -> bool:
    """Verify original_document[start:end] == returned_passage for any retrieved evidence span."""
    if doc_id not in contracts:
        raise KeyError(f"Document '{doc_id}' not found in indexed contracts.")
    doc_obj = contracts[doc_id]
    raw_doc = doc_obj.text if isinstance(doc_obj, ContractDocument) else doc_obj
    if char_start < 0 or char_end > len(raw_doc) or char_start >= char_end:
        return False
    return raw_doc[char_start:char_end] == passage_text

