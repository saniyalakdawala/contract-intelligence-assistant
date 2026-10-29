"""Evaluation tables and Matplotlib chart generation for ClauseTrace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#0F1E36"
TEAL = "#1F7A8C"
GOLD = "#C59B27"
CORAL = "#C85A32"
SLATE = "#5C6B73"
IVORY = "#FAF8F5"


def generate_evaluation_charts(
    eval_dir: str | Path = "outputs/evaluation",
    exp_dir: str | Path = "experiments",
    fig_dir: str | Path = "outputs/figures",
) -> dict[str, str]:
    """Generate comparison charts from real saved evaluation and experiment artifacts."""
    e_dir = Path(eval_dir)
    x_dir = Path(exp_dir)
    f_dir = Path(fig_dir)
    f_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, str] = {}

    base_train_file = e_dir / "baseline_train_metrics.json"
    final_train_file = e_dir / "final_train_metrics.json"
    base_test_file = e_dir / "baseline_test_metrics.json"
    final_test_file = e_dir / "final_test_metrics.json"

    if base_train_file.exists() and final_train_file.exists():
        bt = json.loads(base_train_file.read_text(encoding="utf-8"))["metrics"]
        ft = json.loads(final_train_file.read_text(encoding="utf-8"))["metrics"]
        b_test = (
            json.loads(base_test_file.read_text(encoding="utf-8"))["metrics"]
            if base_test_file.exists()
            else None
        )
        f_test = (
            json.loads(final_test_file.read_text(encoding="utf-8"))["metrics"]
            if final_test_file.exists()
            else None
        )

        # 1. Precision@k and Recall@k comparison chart
        ks = [1, 3, 5, 10]
        x = np.arange(len(ks))
        width = 0.20 if f_test else 0.32

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)
        fig.patch.set_facecolor(IVORY)

        for ax in axes:
            ax.set_facecolor("#FFFFFF")
            ax.grid(axis="y", linestyle="--", alpha=0.35)

        # Panel A: Recall@k
        ax_r = axes[0]
        bt_r = [bt.get(f"char_recall@{k}", 0.0) for k in ks]
        ft_r = [ft.get(f"char_recall@{k}", 0.0) for k in ks]
        if b_test and f_test:
            btest_r = [b_test.get(f"char_recall@{k}", 0.0) for k in ks]
            ftest_r = [f_test.get(f"char_recall@{k}", 0.0) for k in ks]
            ax_r.bar(x - 1.5 * width, bt_r, width, label="Baseline (Train)", color=SLATE, alpha=0.75)
            ax_r.bar(x - 0.5 * width, ft_r, width, label="Final (Train)", color=NAVY)
            ax_r.bar(x + 0.5 * width, btest_r, width, label="Baseline (Test)", color=GOLD, alpha=0.75)
            ax_r.bar(x + 1.5 * width, ftest_r, width, label="Final (Test)", color=TEAL)
        else:
            ax_r.bar(x - width / 2, bt_r, width, label="Baseline (Train)", color=SLATE)
            ax_r.bar(x + width / 2, ft_r, width, label="Final (Train)", color=NAVY)

        ax_r.set_title("Character-Level Recall@k", fontsize=12, fontweight="bold", color=NAVY)
        ax_r.set_xticks(x)
        ax_r.set_xticklabels([f"k={k}" for k in ks])
        ax_r.set_ylim(0.0, 1.0)
        ax_r.set_ylabel("Character Recall")
        ax_r.legend(frameon=True, fontsize=8.5)

        # Panel B: Precision@k
        ax_p = axes[1]
        bt_p = [bt.get(f"char_precision@{k}", 0.0) for k in ks]
        ft_p = [ft.get(f"char_precision@{k}", 0.0) for k in ks]
        if b_test and f_test:
            btest_p = [b_test.get(f"char_precision@{k}", 0.0) for k in ks]
            ftest_p = [f_test.get(f"char_precision@{k}", 0.0) for k in ks]
            ax_p.bar(x - 1.5 * width, bt_p, width, label="Baseline (Train)", color=SLATE, alpha=0.75)
            ax_p.bar(x - 0.5 * width, ft_p, width, label="Final (Train)", color=NAVY)
            ax_p.bar(x + 0.5 * width, btest_p, width, label="Baseline (Test)", color=GOLD, alpha=0.75)
            ax_p.bar(x + 1.5 * width, ftest_p, width, label="Final (Test)", color=TEAL)
        else:
            ax_p.bar(x - width / 2, bt_p, width, label="Baseline (Train)", color=SLATE)
            ax_p.bar(x + width / 2, ft_p, width, label="Final (Train)", color=NAVY)

        ax_p.set_title("Character-Level Precision@k", fontsize=12, fontweight="bold", color=NAVY)
        ax_p.set_xticks(x)
        ax_p.set_xticklabels([f"k={k}" for k in ks])
        ax_p.set_ylabel("Character Precision")
        ax_p.legend(frameon=True, fontsize=8.5)

        fig.suptitle(
            "ClauseTrace: Baseline TF-IDF vs. Final Hybrid Pipeline on LegalBench-RAG-mini (CUAD)",
            fontsize=13,
            fontweight="bold",
            color=NAVY,
        )
        fig.tight_layout()
        pr_chart_path = f_dir / "precision_recall_by_k.png"
        fig.savefig(pr_chart_path)
        plt.close(fig)
        generated["precision_recall_by_k"] = str(pr_chart_path)

        # 2. Abstention comparison chart
        fig2, ax2 = plt.subplots(figsize=(8.5, 4.8), dpi=160)
        fig2.patch.set_facecolor(IVORY)
        ax2.set_facecolor("#FFFFFF")
        ax2.grid(axis="y", linestyle="--", alpha=0.35)

        metrics_keys = [
            ("correct_abstention_rate", "Correct Abstention\n(Unanswerable)"),
            ("answer_coverage", "Answer Coverage\n(Answerable)"),
            ("false_abstention_rate", "False Abstention\n(Lower is Better)"),
            ("abstention_balanced_accuracy", "Balanced Abstention\nAccuracy"),
        ]
        labels = [m[1] for m in metrics_keys]
        xm = np.arange(len(labels))
        w2 = 0.20 if f_test else 0.32

        bt_a = [bt.get(m[0], 0.0) for m in metrics_keys]
        ft_a = [ft.get(m[0], 0.0) for m in metrics_keys]
        if b_test and f_test:
            btest_a = [b_test.get(m[0], 0.0) for m in metrics_keys]
            ftest_a = [f_test.get(m[0], 0.0) for m in metrics_keys]
            ax2.bar(xm - 1.5 * w2, bt_a, w2, label="Baseline (Train)", color=SLATE, alpha=0.75)
            ax2.bar(xm - 0.5 * w2, ft_a, w2, label="Final (Train)", color=NAVY)
            ax2.bar(xm + 0.5 * w2, btest_a, w2, label="Baseline (Test)", color=GOLD, alpha=0.75)
            ax2.bar(xm + 1.5 * w2, ftest_a, w2, label="Final (Test)", color=TEAL)
        else:
            ax2.bar(xm - w2 / 2, bt_a, w2, label="Baseline (Train)", color=SLATE)
            ax2.bar(xm + w2 / 2, ft_a, w2, label="Final (Train)", color=NAVY)

        ax2.set_xticks(xm)
        ax2.set_xticklabels(labels)
        ax2.set_ylim(0.0, 1.08)
        ax2.set_ylabel("Rate")
        ax2.set_title("Bidirectional Abstention Performance", fontsize=12, fontweight="bold", color=NAVY)
        ax2.legend(frameon=True, fontsize=9)
        fig2.tight_layout()
        abs_chart_path = f_dir / "abstention_comparison.png"
        fig2.savefig(abs_chart_path)
        plt.close(fig2)
        generated["abstention_comparison"] = str(abs_chart_path)

    # 3. Improvement loop trajectory chart
    log_jsonl = x_dir / "experiment_log.jsonl"
    if log_jsonl.exists():
        rows = [
            json.loads(line)
            for line in log_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rows:
            iters = [r["iteration"] for r in rows]
            objs = [r["metrics_after"]["primary_objective"] for r in rows]
            r5s = [r["metrics_after"]["char_recall@5"] for r in rows]
            c_abs = [r["metrics_after"]["correct_abstention_rate"] for r in rows]
            accepted_flags = [r["accepted"] for r in rows]

            fig3, ax3 = plt.subplots(figsize=(9.5, 4.8), dpi=160)
            fig3.patch.set_facecolor(IVORY)
            ax3.set_facecolor("#FFFFFF")
            ax3.grid(True, linestyle="--", alpha=0.35)

            ax3.plot(iters, objs, marker="o", linewidth=2.2, color=NAVY, label="Primary Objective (Train)")
            ax3.plot(iters, r5s, marker="s", linewidth=1.8, linestyle="--", color=TEAL, label="Char Recall@5 (Train)")
            ax3.plot(iters, c_abs, marker="^", linewidth=1.8, linestyle=":", color=GOLD, label="Correct Abstention Rate (Train)")

            for it, obj, acc in zip(iters, objs, accepted_flags):
                color = "#1B8A5A" if acc else CORAL
                tag = "Accepted" if acc else "Rejected"
                if it == 0:
                    tag = "Baseline"
                ax3.scatter([it], [obj], s=95, color=color, zorder=5)
                ax3.annotate(
                    f"Iter {it}\n({tag})",
                    (it, obj),
                    textcoords="offset points",
                    xytext=(0, 10 if acc else -22),
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )

            ax3.set_xticks(iters)
            ax3.set_xlabel("Improvement Loop Iteration (Train Split Only)")
            ax3.set_ylabel("Score / Rate")
            ax3.set_ylim(0.0, 1.05)
            ax3.set_title(
                "Self-Improvement Loop Trajectory on Train Split (Accepted vs. Rejected Experiments)",
                fontsize=11.5,
                fontweight="bold",
                color=NAVY,
            )
            ax3.legend(loc="lower right", frameon=True, fontsize=9)
            fig3.tight_layout()
            traj_path = f_dir / "improvement_trajectory.png"
            fig3.savefig(traj_path)
            plt.close(fig3)
            generated["improvement_trajectory"] = str(traj_path)

    return generated


def build_markdown_comparison_table(
    eval_dir: str | Path = "outputs/evaluation",
) -> str:
    """Create a formatted Markdown table comparing Baseline and Final metrics across Train and Test splits."""
    e_dir = Path(eval_dir)
    files = {
        "Baseline (Train)": e_dir / "baseline_train_metrics.json",
        "Final (Train)": e_dir / "final_train_metrics.json",
        "Baseline (Test)": e_dir / "baseline_test_metrics.json",
        "Final (Test)": e_dir / "final_test_metrics.json",
    }
    loaded: dict[str, dict[str, Any]] = {}
    for label, path in files.items():
        if path.exists():
            loaded[label] = json.loads(path.read_text(encoding="utf-8"))["metrics"]

    if not loaded:
        return "_No evaluation artifacts found._"

    metric_rows = [
        ("Answerable Queries", "answerable_query_count"),
        ("Unanswerable Queries", "unanswerable_query_count"),
        ("Primary Objective Score", "primary_objective"),
        ("Char Precision@1", "char_precision@1"),
        ("Char Recall@1", "char_recall@1"),
        ("Char Precision@3", "char_precision@3"),
        ("Char Recall@3", "char_recall@3"),
        ("Char Precision@5", "char_precision@5"),
        ("Char Recall@5", "char_recall@5"),
        ("Char F1@5", "char_f1@5"),
        ("Char Precision@10", "char_precision@10"),
        ("Char Recall@10", "char_recall@10"),
        ("Doc Hit@5", "doc_hit@5"),
        ("Correct Abstention Rate (Unanswerable)", "correct_abstention_rate"),
        ("Incorrect Retrieval Rate (Unanswerable)", "incorrect_retrieval_rate_unanswerable"),
        ("False Abstention Rate (Answerable)", "false_abstention_rate"),
        ("Answer Coverage (Answerable)", "answer_coverage"),
        ("Balanced Abstention Accuracy", "abstention_balanced_accuracy"),
        ("Avg Latency (ms/query)", "avg_latency_ms"),
        ("P95 Latency (ms/query)", "p95_latency_ms"),
    ]

    cols = list(loaded.keys())
    header = "| Metric | " + " | ".join(cols) + " |"
    sep = "| :--- | " + " | ".join(["---:" for _ in cols]) + " |"
    lines = [header, sep]

    for label, key in metric_rows:
        vals = [str(loaded[c].get(key, "N/A")) for c in cols]
        lines.append(f"| **{label}** | " + " | ".join(vals) + " |")

    return "\n".join(lines)

