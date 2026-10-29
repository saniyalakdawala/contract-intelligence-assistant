"""Typer CLI for ClauseTrace (`python -m contract_intelligence <command>`)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from contract_intelligence.config import PipelineConfig
from contract_intelligence.dataset import download_legalbench_cuad, load_dataset
from contract_intelligence.evaluation import evaluate_pipeline
from contract_intelligence.improve import run_improvement_loop
from contract_intelligence.indexing import build_index
from contract_intelligence.reporting import (
    build_markdown_comparison_table,
    generate_evaluation_charts,
)
from contract_intelligence.retrieval import ClauseTraceRetriever
from contract_intelligence.validation import validate_ground_truth_offsets

app = typer.Typer(
    name="contract_intelligence",
    help="ClauseTrace: Evaluation-driven Contract Intelligence Assistant for legal clause retrieval.",
    add_completion=False,
)
console = Console()


@app.command("download-data")
def cmd_download_data(
    data_dir: str = typer.Option("data/legalbench_rag", "--data-dir", help="Target directory for LegalBench-RAG CUAD files."),
) -> None:
    """Download benchmarks/cuad.json and the 29 referenced CUAD-mini contract files."""
    info = download_legalbench_cuad(data_dir=data_dir)
    console.print(Panel(json.dumps(info, indent=2), title="Dataset Download Complete", border_style="green"))


@app.command("validate")
def cmd_validate(
    data_dir: str = typer.Option("data/legalbench_rag", "--data-dir"),
    unanswerable_path: str = typer.Option("data/unanswerable_queries.json", "--unanswerable"),
) -> None:
    """Validate ground-truth character offsets against supplied answer snippets and check split counts."""
    ds = load_dataset(data_dir=data_dir, unanswerable_path=unanswerable_path)
    counts = ds.summary_counts()
    rep = validate_ground_truth_offsets(ds)
    payload = {
        "dataset_counts": counts,
        "validation": rep.to_dict(),
    }
    console.print(Panel(json.dumps(payload, indent=2), title="Ground-Truth Offset & Split Validation", border_style="cyan"))
    if not rep.is_valid:
        raise typer.Exit(code=1)


@app.command("prepare-mini")
def cmd_prepare_mini(
    data_dir: str = typer.Option("data/legalbench_rag", "--data-dir"),
    out_file: str = typer.Option("outputs/evaluation/cuad_mini_manifest.json", "--out"),
) -> None:
    """Export the exact 194-query LegalBench-RAG-mini selection and split manifest."""
    ds = load_dataset(data_dir=data_dir)
    manifest = {
        "counts": ds.summary_counts(),
        "indexed_contracts": sorted(ds.contracts.keys()),
        "answerable_queries": [
            {
                "query_id": q.query_id,
                "split": q.split,
                "query": q.query,
                "num_snippets": len(q.snippets),
                "file_path": q.snippets[0].file_path if q.snippets else "",
            }
            for q in ds.answerable_queries
        ],
    }
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Saved 194-query LegalBench-RAG-mini manifest to {out_p}[/green]")


@app.command("build-index")
def cmd_build_index(
    config: str = typer.Option("configs/final.yaml", "--config", "-c", help="Path to YAML configuration file."),
) -> None:
    """Build and cache retrieval index for the specified configuration."""
    cfg_path = Path(config) if Path(config).exists() else Path("configs/baseline.yaml")
    cfg = PipelineConfig.from_yaml(cfg_path)
    ds = load_dataset(data_dir=cfg.data_dir, unanswerable_path=cfg.unanswerable_path)
    idx = build_index(ds.contracts, cfg, use_cache=True)
    console.print(
        Panel(
            json.dumps(
                {
                    "config_name": cfg.name,
                    "config_fingerprint": idx.config_fingerprint,
                    "contracts_indexed": len(idx.doc_ids),
                    "total_chunks": len(idx.chunks),
                    "build_time_seconds": round(idx.build_time_seconds, 4),
                    "index_size_bytes": idx.index_size_bytes,
                },
                indent=2,
            ),
            title="Index Built Successfully",
            border_style="green",
        )
    )


@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Natural-language contract question."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top evidence passages to return."),
    config: str = typer.Option("configs/final.yaml", "--config", "-c", help="Path to YAML configuration."),
    strategy: str | None = typer.Option(None, "--strategy", "-s", help="Optional strategy override (tfidf, bm25, dense, hybrid)."),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON output."),
) -> None:
    """Search indexed contracts for exact supporting clauses and character offsets."""
    cfg_path = Path(config) if Path(config).exists() else Path("configs/baseline.yaml")
    cfg = PipelineConfig.from_yaml(cfg_path)
    if strategy is not None:
        cfg.retrieval.strategy = strategy  # type: ignore[assignment]
    ds = load_dataset(data_dir=cfg.data_dir, unanswerable_path=cfg.unanswerable_path)
    retriever = ClauseTraceRetriever(ds.contracts, cfg, use_cache=True)
    resp = retriever.search(query, top_k=top_k)

    if json_output:
        print(json.dumps(resp.to_dict(), indent=2))
        return

    console.print(
        Panel(
            f"[bold]Query:[/bold] {resp.query}\n"
            f"[bold]Method:[/bold] {resp.retrieval_method} | "
            f"[bold]Latency:[/bold] {resp.latency_ms:.2f} ms | "
            f"[bold]Status:[/bold] {resp.abstention.status}",
            title="ClauseTrace Search Result",
            border_style="yellow" if resp.abstained else "cyan",
        )
    )

    if resp.abstained:
        console.print(f"[bold yellow]ABSTAINED:[/bold yellow] {resp.abstention.reason}")
        console.print(f"Diagnostics: {json.dumps(resp.abstention.diagnostics)}")
        return

    for p in resp.passages:
        header = (
            f"Rank #{p.rank} | {p.doc_id} | Offsets [{p.char_start}:{p.char_end}] "
            f"| Score={p.retrieval_score:.4f} (Clause={p.scores.get('clause_score', 0.0):.3f}, Doc={p.scores.get('doc_match_score', 0.0):.3f})"
        )
        meta_lines: list[str] = []
        if p.section_heading:
            meta_lines.append(f"[dim]Section: {p.section_heading}[/dim]")
        if p.cross_references:
            meta_lines.append(f"[dim]Cross-References: {', '.join(p.cross_references)}[/dim]")
        body = ("\n".join(meta_lines) + "\n\n" if meta_lines else "") + p.passage_text
        if p.referenced_context:
            body += f"\n\n[dim]Heuristic Cross-Reference Context (separate from exact evidence): {p.referenced_context}[/dim]"
        console.print(Panel(body, title=header, border_style="blue"))


@app.command("evaluate")
def cmd_evaluate(
    split: str = typer.Option("train", "--split", help="Evaluation split: 'train' or 'test'."),
    config: str = typer.Option("configs/final.yaml", "--config", "-c", help="Configuration YAML path."),
    out_dir: str = typer.Option("outputs/evaluation", "--out-dir", help="Directory for evaluation artifacts."),
    prefix: str | None = typer.Option(None, "--prefix", help="Optional filename prefix for saved artifacts."),
) -> None:
    """Evaluate a pipeline configuration on the train or test split."""
    if split not in ("train", "test"):
        raise typer.BadParameter("split must be 'train' or 'test'")
    cfg_path = Path(config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = PipelineConfig.from_yaml(cfg_path)
    ds = load_dataset(data_dir=cfg.data_dir, unanswerable_path=cfg.unanswerable_path)
    out = evaluate_pipeline(ds, cfg, split=split, record_test_ledger=(split == "test"))  # type: ignore[arg-type]

    default_tag = prefix or (
        f"baseline_{split}" if "baseline" in cfg.name else f"final_{split}"
    )
    paths = out.save_artifacts(out_dir=out_dir, prefix=default_tag)

    table = Table(title=f"ClauseTrace Evaluation ({cfg.name} on '{split}' split)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")
    for k, v in out.summary.items():
        table.add_row(str(k), str(v))
    console.print(table)
    console.print(f"[green]Saved artifacts:[/green] {json.dumps(paths)}")


@app.command("improve")
def cmd_improve(
    baseline_config: str = typer.Option("configs/baseline.yaml", "--baseline"),
    final_config: str = typer.Option("configs/final.yaml", "--final"),
    iterations: int = typer.Option(7, "--iterations", "-n", min=5, max=7),
) -> None:
    """Run the automated train-only self-improvement loop (at least 5 iterations beyond baseline)."""
    res = run_improvement_loop(
        baseline_config_path=baseline_config,
        final_config_path=final_config,
        num_iterations=iterations,
        split="train",
    )
    console.print(
        Panel(
            json.dumps(
                {
                    "baseline_config": res["baseline_config"],
                    "final_selected_config": res["final_selected_config"],
                    "accepted_iterations": res["accepted_iterations"],
                    "rejected_iterations": res["rejected_iterations"],
                    "baseline_primary_objective": res["baseline_train_metrics"]["primary_objective"],
                    "final_primary_objective": res["final_train_metrics"]["primary_objective"],
                    "final_config_path": res["final_config_path"],
                },
                indent=2,
            ),
            title="Self-Improvement Loop Finished (Train Only)",
            border_style="green",
        )
    )


@app.command("export-report")
def cmd_export_report(
    eval_dir: str = typer.Option("outputs/evaluation", "--eval-dir"),
    exp_dir: str = typer.Option("experiments", "--exp-dir"),
    fig_dir: str = typer.Option("outputs/figures", "--fig-dir"),
) -> None:
    """Generate evaluation comparison tables and Matplotlib charts."""
    charts = generate_evaluation_charts(eval_dir=eval_dir, exp_dir=exp_dir, fig_dir=fig_dir)
    md_table = build_markdown_comparison_table(eval_dir=eval_dir)
    table_file = Path(eval_dir) / "comparison_table.md"
    table_file.parent.mkdir(parents=True, exist_ok=True)
    table_file.write_text(md_table + "\n", encoding="utf-8")
    console.print(f"[green]Generated charts:[/green] {json.dumps(charts, indent=2)}")
    console.print(f"[green]Saved Markdown table to:[/green] {table_file}")


@app.command("test")
def cmd_test() -> None:
    """Run the automated pytest suite (never triggers a test-split evaluation)."""
    ret = subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)
    raise typer.Exit(code=ret.returncode)


if __name__ == "__main__":
    app()

