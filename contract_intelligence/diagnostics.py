"""Failure classification and error analysis diagnostics on evaluation predictions."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from contract_intelligence.dataset import QueryRecord
from contract_intelligence.retrieval import RetrievalResponse

CROSS_REF_CUE_RE = re.compile(
    r"\b(?:as defined in|pursuant to|subject to|in accordance with|section\s+\d+|article\s+[ivxlcdm0-9]+|exhibit\s+[a-z0-9])\b",
    re.IGNORECASE,
)


def classify_query_Result(
    query: QueryRecord,
    response: RetrievalResponse,
    metrics_row: dict[str, Any],
) -> tuple[str, str]:
    """Classify a single query evaluation result into a specific failure taxonomy category."""
    if not query.is_answerable:
        if response.abstained:
            return "none", "Correctly abstained on unanswerable query."
        top_doc = response.passages[0].doc_id if response.passages else "unknown"
        return (
            "failed_to_abstain",
            f"Returned passage from '{top_doc}' (score={response.top_score:.3f}, "
            f"clause={response.top_clause_score:.3f}, doc={response.top_doc_match_score:.3f}) "
            f"instead of abstaining ({query.unanswerable_reason}).",
        )

    # Answerable benchmark query
    if response.abstained:
        return (
            "false_abstention",
            f"Falsely abstained on answerable query ({response.abstention.status}: "
            f"score={response.top_score:.3f}, clause={response.top_clause_score:.3f}, "
            f"doc={response.top_doc_match_score:.3f}).",
        )

    rec1 = float(metrics_row.get("char_recall@1", 0.0) or 0.0)
    rec5 = float(metrics_row.get("char_recall@5", 0.0) or 0.0)
    rec10 = float(metrics_row.get("char_recall@10", 0.0) or 0.0)
    doc5 = float(metrics_row.get("doc_hit@5", 0.0) or 0.0)

    # Consider >= 0.75 recall@5 with reasonable rank as non-error
    if rec5 >= 0.75 and rec1 >= 0.35:
        return "none", f"Strong evidence coverage (Recall@1={rec1:.2f}, Recall@5={rec5:.2f})."

    gt_text_joined = " ".join(s.answer for s in query.snippets)
    has_xref_cue = bool(CROSS_REF_CUE_RE.search(gt_text_joined))

    # 1. Wrong document retrieved in top-5
    if doc5 == 0.0:
        top_doc = response.passages[0].doc_id if response.passages else "none"
        gt_doc = query.snippets[0].file_path if query.snippets else "none"
        return (
            "wrong_document",
            f"Top-5 passages missed ground-truth contract '{gt_doc}' (top-1 was '{top_doc}').",
        )

    # 2. Correct passage ranked too low (missed at top-1/top-3 or only appears at rank 4-10)
    if rec1 == 0.0 and (rec5 >= 0.50 or rec10 >= 0.50):
        return (
            "ranked_too_low",
            f"Relevant passage missed at rank 1 (Recall@1=0.00) but recovered deeper in ranking (Recall@5={rec5:.2f}, Recall@10={rec10:.2f}).",
        )

    # 3. Right document retrieved in top-5, but zero overlap with ground-truth clause
    if rec5 == 0.0 and doc5 > 0.0:
        if has_xref_cue:
            return (
                "cross_reference_or_defined_term",
                "Retrieved correct contract but missed target clause containing cross-references or defined terms.",
            )
        return (
            "right_document_wrong_clause",
            "Retrieved target contract in top-5 but selected wrong sections/clauses (Recall@5=0.00).",
        )

    # 4. Partial span coverage or chunk boundary truncation (0 < Recall@5 < 0.75)
    if 0.0 < rec5 < 0.75:
        # Check if a retrieved chunk boundary sliced right through a ground-truth span
        boundary_cut = False
        for s in query.snippets:
            for p in response.passages[:5]:
                if p.doc_id == s.file_path:
                    if (s.char_start < p.char_start < s.char_end) or (
                        s.char_start < p.char_end < s.char_end
                    ):
                        boundary_cut = True
                        break
        if boundary_cut:
            return (
                "chunk_boundary_truncation",
                f"Retrieved relevant clause but chunk boundary sliced through ground-truth span (Recall@5={rec5:.2f}).",
            )
        if has_xref_cue:
            return (
                "cross_reference_or_defined_term",
                f"Multi-part or cross-referenced answer span only partially retrieved (Recall@5={rec5:.2f}).",
            )
        return (
            "partial_span_coverage",
            f"Relevant section retrieved but multi-snippet or long answer span incompletely covered (Recall@5={rec5:.2f}).",
        )

    return "ranked_too_low", f"Top-1 coverage low (Recall@1={rec1:.2f}, Recall@5={rec5:.2f})."


def diagnose_evaluation_failures(eval_output: Any) -> dict[str, Any]:
    """Aggregate failure categories and extract concrete failure examples from an EvaluationOutput."""
    counts: Counter[str] = Counter()
    examples_by_category: dict[str, list[dict[str, Any]]] = {}

    for row in eval_output.per_query_results:
        cat = row.get("error_category", "none")
        if cat == "none":
            continue
        counts[cat] += 1
        examples_by_category.setdefault(cat, [])
        if len(examples_by_category[cat]) < 5:
            examples_by_category[cat].append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "is_answerable": row["is_answerable"],
                    "error_category": cat,
                    "error_detail": row.get("error_detail", ""),
                    "abstained": row["abstained"],
                    "top_score": row["top_score"],
                    "top_clause_score": row["top_clause_score"],
                    "top_doc_match_score": row["top_doc_match_score"],
                    "char_precision@5": row.get("char_precision@5"),
                    "char_recall@5": row.get("char_recall@5"),
                    "gt_doc_id": row.get("gt_doc_id", ""),
                    "top1_doc_id": row.get("top1_doc_id", ""),
                }
            )

    ordered_counts = dict(counts.most_common())
    dominant_failure = next(iter(ordered_counts.keys()), "none")
    return {
        "total_failures": sum(counts.values()),
        "dominant_failure_category": dominant_failure,
        "category_counts": ordered_counts,
        "examples_by_category": examples_by_category,
    }

