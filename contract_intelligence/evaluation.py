"""Character-level precision/recall evaluation (LegalBench-RAG definition) and abstention metrics."""

from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from contract_intelligence.config import PipelineConfig
from contract_intelligence.dataset import GroundTruthSnippet, LegalBenchMiniDataset, QueryRecord
from contract_intelligence.retrieval import ClauseTraceRetriever, EvidencePassage, RetrievalResponse

EVAL_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


def sort_and_merge_spans(
    spans: list[tuple[int, int]], *, max_bridge_gap_len: int = 0
) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent character spans (exact algorithm from LegalBench-RAG)."""
    if len(spans) == 0:
        return []
    sorted_spans = sorted(spans, key=lambda x: x[0])
    merged: list[tuple[int, int]] = [sorted_spans[0]]
    for span in sorted_spans[1:]:
        if span[0] <= merged[-1][1] + max_bridge_gap_len:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged.append(span)
    return merged


def compute_char_metrics_at_k(
    gt_snippets: list[GroundTruthSnippet],
    retrieved_passages: list[EvidencePassage],
    k: int,
) -> dict[str, float]:
    """Compute character-level precision@k, recall@k, f1@k, and doc_hit@k for one answerable query."""
    top_k = retrieved_passages[:k]

    # Group and merge ground-truth spans by file_path
    gt_by_file: dict[str, list[tuple[int, int]]] = {}
    for g in gt_snippets:
        gt_by_file.setdefault(g.file_path, []).append((g.char_start, g.char_end))
    merged_gt_by_file = {fp: sort_and_merge_spans(sp) for fp, sp in gt_by_file.items()}

    # Group and merge retrieved spans by doc_id so overlapping chunks are never double-counted
    ret_by_file: dict[str, list[tuple[int, int]]] = {}
    for p in top_k:
        ret_by_file.setdefault(p.doc_id, []).append((p.char_start, p.char_end))
    merged_ret_by_file = {fp: sort_and_merge_spans(sp) for fp, sp in ret_by_file.items()}

    total_gt_len = sum(
        (e - s) for spans in merged_gt_by_file.values() for (s, e) in spans
    )
    total_ret_len = sum(
        (e - s) for spans in merged_ret_by_file.values() for (s, e) in spans
    )

    overlap_len = 0
    for fp, gt_spans in merged_gt_by_file.items():
        ret_spans = merged_ret_by_file.get(fp, [])
        for gs, ge in gt_spans:
            for rs, re_pos in ret_spans:
                common_min = max(gs, rs)
                common_max = min(ge, re_pos)
                if common_max > common_min:
                    overlap_len += common_max - common_min

    precision = (overlap_len / float(total_ret_len)) if total_ret_len > 0 else 0.0
    recall = (overlap_len / float(total_gt_len)) if total_gt_len > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    doc_hit = 1.0 if any(fp in merged_gt_by_file for fp in merged_ret_by_file) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "doc_hit": doc_hit,
        "overlap_chars": float(overlap_len),
        "retrieved_chars": float(total_ret_len),
        "gt_chars": float(total_gt_len),
    }


def compute_primary_objective(metrics: dict[str, float]) -> float:
    """Compute the composite Evidence Quality Objective score from split metrics."""
    r5 = metrics.get("char_recall@5", 0.0)
    f1_5 = metrics.get("char_f1@5", 0.0)
    r1 = metrics.get("char_recall@1", 0.0)
    corr_abs = metrics.get("correct_abstention_rate", 0.0)
    false_abs = metrics.get("false_abstention_rate", 0.0)
    return float(
        0.50 * r5
        + 0.25 * f1_5
        + 0.15 * r1
        + 0.10 * corr_abs
        - 0.25 * false_abs
    )


@dataclass
class EvaluationOutput:
    """Complete evaluation results for a split, including summary metrics and per-query records."""

    config_name: str
    config_fingerprint: str
    split: str
    timestamp: str
    summary: dict[str, Any]
    per_query_results: list[dict[str, Any]] = field(default_factory=list)

    def save_artifacts(self, out_dir: str | Path, prefix: str | None = None) -> dict[str, str]:
        """Write machine-readable JSON summaries, per-query predictions, and error CSV."""
        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)
        tag = prefix or f"{self.config_name}_{self.split}"

        summary_path = base / f"{tag}_metrics.json"
        preds_path = base / f"{tag}_predictions.json"
        errors_csv_path = base / f"{tag}_errors.csv"

        summary_path.write_text(
            json.dumps(
                {
                    "config_name": self.config_name,
                    "config_fingerprint": self.config_fingerprint,
                    "split": self.split,
                    "timestamp": self.timestamp,
                    "metrics": self.summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        preds_path.write_text(
            json.dumps(self.per_query_results, indent=2) + "\n",
            encoding="utf-8",
        )

        # Write query-level error analysis CSV
        error_rows = [
            r for r in self.per_query_results if r.get("error_category", "none") != "none"
        ]
        fieldnames = [
            "query_id",
            "split",
            "is_answerable",
            "error_category",
            "abstained",
            "top_score",
            "top_clause_score",
            "top_doc_match_score",
            "char_precision@5",
            "char_recall@5",
            "gt_doc_id",
            "top1_doc_id",
            "query",
        ]
        with open(errors_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in error_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        return {
            "summary_json": str(summary_path),
            "predictions_json": str(preds_path),
            "errors_csv": str(errors_csv_path),
        }


def _record_test_run(
    config: PipelineConfig,
    summary: dict[str, Any],
    ledger_path: Path = Path("outputs/evaluation/test_run_ledger.json"),
) -> int:
    """Record every test-split evaluation in a persistent ledger to audit test set usage."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        data = {"test_evaluation_runs": 0, "runs": []}

    data["test_evaluation_runs"] = int(data.get("test_evaluation_runs", 0)) + 1
    data["runs"].append(
        {
            "run_number": data["test_evaluation_runs"],
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_name": config.name,
            "config_fingerprint": config.full_fingerprint(),
            "char_recall@5": summary.get("char_recall@5"),
            "char_precision@5": summary.get("char_precision@5"),
            "correct_abstention_rate": summary.get("correct_abstention_rate"),
            "false_abstention_rate": summary.get("false_abstention_rate"),
        }
    )
    ledger_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return int(data["test_evaluation_runs"])


def evaluate_pipeline(
    dataset: LegalBenchMiniDataset,
    config: PipelineConfig,
    split: Literal["train", "test"] = "train",
    retriever: ClauseTraceRetriever | None = None,
    k_values: tuple[int, ...] = EVAL_K_VALUES,
    record_test_ledger: bool = True,
) -> EvaluationOutput:
    """Evaluate a PipelineConfig on the specified split ('train' or 'test')."""
    from contract_intelligence.diagnostics import classify_query_Result

    max_k = max(k_values)
    active_retriever = retriever or ClauseTraceRetriever(
        contracts=dataset.contracts, config=config, use_cache=True
    )

    ans_queries = dataset.get_queries(split=split, include_answerable=True, include_unanswerable=False)
    unans_queries = dataset.get_queries(split=split, include_answerable=False, include_unanswerable=True)
    all_queries = ans_queries + unans_queries

    # Pre-warm batch query embeddings if strategy uses dense embeddings
    if config.retrieval.strategy in ("dense", "hybrid"):
        from contract_intelligence.indexing import decompose_query, encode_query_texts

        batch_texts: list[str] = []
        for q in all_queries:
            q_c = q.query.strip()
            _, _, exp_q = decompose_query(q_c, expand_legal=config.legal_context.enable_legal_query_expansion)
            batch_texts.append(q_c)
            batch_texts.append(exp_q)
        encode_query_texts(config.retrieval.dense_model_name, batch_texts, cache_dir=config.cache_dir)

    per_query_records: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    prec_by_k: dict[int, list[float]] = {k: [] for k in k_values}
    rec_by_k: dict[int, list[float]] = {k: [] for k in k_values}
    f1_by_k: dict[int, list[float]] = {k: [] for k in k_values}
    dochit_by_k: dict[int, list[float]] = {k: [] for k in k_values}

    false_abstentions = 0
    correct_abstentions = 0

    for q in all_queries:
        resp: RetrievalResponse = active_retriever.search(q.query, top_k=max_k)
        latencies_ms.append(resp.latency_ms)

        gt_docs = sorted({s.file_path for s in q.snippets})
        top1_doc = resp.unfiltered_top_passages[0].doc_id if resp.unfiltered_top_passages else ""

        q_record: dict[str, Any] = {
            "query_id": q.query_id,
            "query": q.query,
            "split": q.split,
            "is_answerable": q.is_answerable,
            "category": q.category,
            "abstained": resp.abstained,
            "abstention_status": resp.abstention.status,
            "abstention_reason": resp.abstention.reason,
            "latency_ms": round(resp.latency_ms, 3),
            "top_score": round(resp.top_score, 4),
            "top_clause_score": round(resp.top_clause_score, 4),
            "top_doc_match_score": round(resp.top_doc_match_score, 4),
            "gt_doc_id": gt_docs[0] if gt_docs else "",
            "top1_doc_id": top1_doc,
            "gt_spans": [
                {"file_path": s.file_path, "span": [s.char_start, s.char_end], "answer": s.answer}
                for s in q.snippets
            ],
            "retrieved_passages": [p.to_dict() for p in resp.passages[:max_k]],
            "unfiltered_top3": [p.to_dict() for p in resp.unfiltered_top_passages[:3]],
        }

        if q.is_answerable:
            if resp.abstained:
                false_abstentions += 1
            for k in k_values:
                m_k = compute_char_metrics_at_k(q.snippets, resp.passages, k=k)
                prec_by_k[k].append(m_k["precision"])
                rec_by_k[k].append(m_k["recall"])
                f1_by_k[k].append(m_k["f1"])
                dochit_by_k[k].append(m_k["doc_hit"])
                q_record[f"char_precision@{k}"] = round(m_k["precision"], 4)
                q_record[f"char_recall@{k}"] = round(m_k["recall"], 4)
                q_record[f"char_f1@{k}"] = round(m_k["f1"], 4)
                q_record[f"doc_hit@{k}"] = m_k["doc_hit"]
        else:
            if resp.abstained:
                correct_abstentions += 1
            for k in k_values:
                q_record[f"char_precision@{k}"] = None
                q_record[f"char_recall@{k}"] = None
                q_record[f"char_f1@{k}"] = None
                q_record[f"doc_hit@{k}"] = None

        err_cat, err_detail = classify_query_Result(q, resp, q_record)
        q_record["error_category"] = err_cat
        q_record["error_detail"] = err_detail
        per_query_records.append(q_record)

    n_ans = len(ans_queries)
    n_unans = len(unans_queries)

    summary: dict[str, Any] = {
        "split": split,
        "answerable_query_count": n_ans,
        "unanswerable_query_count": n_unans,
        "total_query_count": n_ans + n_unans,
        "indexed_contract_count": len(dataset.contracts),
        "num_chunks": len(active_retriever.index.chunks),
        "index_build_time_s": round(active_retriever.index.build_time_seconds, 4),
        "index_size_bytes": active_retriever.index.index_size_bytes,
    }

    for k in k_values:
        p_mean = float(np.mean(prec_by_k[k])) if prec_by_k[k] else 0.0
        r_mean = float(np.mean(rec_by_k[k])) if rec_by_k[k] else 0.0
        f1_mean = float(np.mean(f1_by_k[k])) if f1_by_k[k] else 0.0
        dh_mean = float(np.mean(dochit_by_k[k])) if dochit_by_k[k] else 0.0
        summary[f"char_precision@{k}"] = round(p_mean, 4)
        summary[f"char_recall@{k}"] = round(r_mean, 4)
        summary[f"char_f1@{k}"] = round(f1_mean, 4)
        summary[f"doc_hit@{k}"] = round(dh_mean, 4)

    false_abs_rate = (false_abstentions / float(n_ans)) if n_ans > 0 else 0.0
    corr_abs_rate = (correct_abstentions / float(n_unans)) if n_unans > 0 else 0.0
    inc_ret_rate = 1.0 - corr_abs_rate
    ans_coverage = 1.0 - false_abs_rate

    summary["correct_abstention_count"] = correct_abstentions
    summary["correct_abstention_rate"] = round(corr_abs_rate, 4)
    summary["incorrect_retrieval_rate_unanswerable"] = round(inc_ret_rate, 4)
    summary["false_abstention_count"] = false_abstentions
    summary["false_abstention_rate"] = round(false_abs_rate, 4)
    summary["answer_coverage"] = round(ans_coverage, 4)
    summary["abstention_balanced_accuracy"] = round(0.5 * (corr_abs_rate + ans_coverage), 4)

    summary["avg_latency_ms"] = round(float(np.mean(latencies_ms)) if latencies_ms else 0.0, 3)
    summary["p95_latency_ms"] = round(float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0, 3)
    summary["primary_objective"] = round(compute_primary_objective(summary), 4)

    if split == "test" and record_test_ledger:
        run_num = _record_test_run(config, summary)
        summary["test_evaluation_run_number"] = run_num

    return EvaluationOutput(
        config_name=config.name,
        config_fingerprint=config.full_fingerprint(),
        split=split,
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        summary=summary,
        per_query_results=per_query_records,
    )

