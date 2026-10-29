"""Streamlit web application for ClauseTrace | Contract Intelligence Assistant.

Provides a clean, accessible legal-tech interface with a functional Pipeline Configuration
Sidebar connected directly to the ClauseTrace retrieval, abstention, evaluation, and
experiment-logging backend.
"""

from __future__ import annotations

import copy
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("USE_TF", "NO")
os.environ.setdefault("USE_TORCH", "YES")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import streamlit as st
import yaml

from contract_intelligence.config import PipelineConfig
from contract_intelligence.dataset import LegalBenchMiniDataset, load_dataset
from contract_intelligence.indexing import decompose_query, legal_tokenize
from contract_intelligence.retrieval import ClauseTraceRetriever
from contract_intelligence.validation import validate_ground_truth_offsets, verify_evidence_offset

DEFAULT_FINAL_CONFIG_PATH = Path("configs/final.yaml")
DEFAULT_BASELINE_CONFIG_PATH = Path("configs/baseline.yaml")

SUPPORTED_STRATEGIES: dict[str, str] = {
    "hybrid": "Hybrid (BM25 Lexical + Dense MiniLM-L6-v2)",
    "bm25": "BM25 Lexical Retrieval (Okapi BM25)",
    "dense": "Dense Semantic Retrieval (all-MiniLM-L6-v2 + FAISS)",
    "tfidf": "TF-IDF Cosine Similarity (Baseline Lexical)",
}

SAMPLE_QUERIES: list[dict[str, str]] = [
    {
        "label": "Custom question (type below)",
        "query": "",
    },
    {
        "label": "[Answerable · Governing Law] BicycleTX Ltd & Nigel Crockett Service Agreement",
        "query": "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; What is the governing law for this contract?",
    },
    {
        "label": "[Answerable · Expiration Date] Fox Mobile & Glu Mobile Amendment No. 3",
        "query": "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?",
    },
    {
        "label": "[Answerable · Liability Cap] HPIL ENERGYTECH & GINARES GROUP Cooperation Agreement",
        "query": "Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is there a cap on liability under this contract?",
    },
    {
        "label": "[Answerable · Exclusivity] ABG-Shaq, LLC & Papa John's Endorsement Agreement",
        "query": "Consider the Endorsement Agreement between ABG-Shaq, LLC and Papa John's International, Inc.; Does this contract include an exclusivity agreement?",
    },
    {
        "label": "[Unanswerable · Missing Clause] Cloud gaming sublicensing in Fox Mobile & Glu Mobile",
        "query": "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; Does this agreement allow sublicensing of the mobile games to third-party cloud gaming platforms?",
    },
    {
        "label": "[Unanswerable · Out-of-Corpus Contract] Netflix & Sony Pictures Distribution Agreement",
        "query": "Consider the Content Distribution Agreement between Netflix, Inc. and Sony Pictures Television Inc.; Does this agreement allow sublicensing to third-party streaming platforms?",
    },
    {
        "label": "[Unanswerable · Missing Obligation] Source code escrow in Town of Vinton Gas Franchise",
        "query": "Consider the Gas Franchise Agreement between the Town of Vinton and Roanoke Gas Company; Is there a source code escrow or software maintenance obligation in this contract?",
    },
]


# ---------------------------------------------------------------------------
# Pure helper functions (importable and unit-tested independently of browser)
# ---------------------------------------------------------------------------


def load_json_artifact(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON file from disk if it exists, returning None otherwise."""
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_experiment_jsonl(path: str | Path = "experiments/experiment_log.jsonl") -> list[dict[str, Any]]:
    """Read all logged iterations from `experiments/experiment_log.jsonl`."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_default_pipeline_config(preset: str = "final") -> PipelineConfig:
    """Load a fresh `PipelineConfig` from `configs/final.yaml` or `configs/baseline.yaml`."""
    if preset == "baseline" and DEFAULT_BASELINE_CONFIG_PATH.exists():
        return PipelineConfig.from_yaml(DEFAULT_BASELINE_CONFIG_PATH)
    if DEFAULT_FINAL_CONFIG_PATH.exists():
        return PipelineConfig.from_yaml(DEFAULT_FINAL_CONFIG_PATH)
    if DEFAULT_BASELINE_CONFIG_PATH.exists():
        return PipelineConfig.from_yaml(DEFAULT_BASELINE_CONFIG_PATH)
    return PipelineConfig()


def build_ui_pipeline_config(
    base_preset: Literal["final", "baseline"] = "final",
    strategy: Literal["hybrid", "bm25", "dense", "tfidf"] | None = None,
    top_k: int = 5,
    abstention_enabled: bool | None = None,
    abstention_mode: Literal["dual_gate", "single_threshold"] | None = None,
    min_top_score: float | None = None,
    min_clause_score: float | None = None,
    min_doc_match_score: float | None = None,
    fusion_method: Literal["linear", "rrf"] | None = None,
    rerank_enabled: bool | None = None,
) -> tuple[PipelineConfig, dict[str, Any]]:
    """Construct an immutable validated `PipelineConfig` from saved YAML + temporary UI selections.

    Never mutates shared configuration objects or saved YAML files on disk.
    Returns `(active_config, metadata_diff)` detailing which settings come from the saved
    configuration file and which are temporary UI overrides.
    """
    if top_k < 1 or top_k > 64:
        raise ValueError(f"top_k must be between 1 and 64 (got {top_k}).")

    saved_cfg = load_default_pipeline_config(base_preset)
    saved_dict = copy.deepcopy(saved_cfg.model_dump())
    active_dict = copy.deepcopy(saved_dict)

    if strategy is not None:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")
        active_dict["retrieval"]["strategy"] = strategy

    active_dict["retrieval"]["top_k"] = int(top_k)
    active_dict["retrieval"]["candidate_pool_size"] = max(
        int(active_dict["retrieval"]["candidate_pool_size"]), int(top_k)
    )

    if abstention_enabled is not None:
        active_dict["abstention"]["enabled"] = bool(abstention_enabled)
    if abstention_mode is not None:
        active_dict["abstention"]["mode"] = abstention_mode
    if min_top_score is not None:
        active_dict["abstention"]["min_top_score"] = round(float(min_top_score), 4)
    if min_clause_score is not None:
        active_dict["abstention"]["min_clause_score"] = round(float(min_clause_score), 4)
    if min_doc_match_score is not None:
        active_dict["abstention"]["min_doc_match_score"] = round(float(min_doc_match_score), 4)
    if fusion_method is not None:
        active_dict["retrieval"]["fusion_method"] = fusion_method
    if rerank_enabled is not None:
        active_dict["rerank"]["enabled"] = bool(rerank_enabled)

    # Compare against saved preset defaults to identify temporary UI overrides
    tracked_paths = [
        ("retrieval", "strategy"),
        ("retrieval", "fusion_method"),
        ("rerank", "enabled"),
        ("abstention", "enabled"),
        ("abstention", "mode"),
        ("abstention", "min_top_score"),
        ("abstention", "min_clause_score"),
        ("abstention", "min_doc_match_score"),
    ]
    overrides: dict[str, dict[str, Any]] = {}
    if int(top_k) != 5:
        overrides["retrieval.top_k"] = {"saved": 5, "active": int(top_k)}
    for section, key in tracked_paths:
        saved_val = saved_dict[section][key]
        active_val = active_dict[section][key]
        if saved_val != active_val:
            overrides[f"{section}.{key}"] = {"saved": saved_val, "active": active_val}

    if overrides:
        active_dict["name"] = f"{saved_cfg.name}_ui_custom"

    active_cfg = PipelineConfig.model_validate(active_dict)
    preset_file = "configs/baseline.yaml" if base_preset == "baseline" else "configs/final.yaml"
    meta = {
        "base_preset": base_preset,
        "base_preset_file": preset_file,
        "saved_config_name": saved_cfg.name,
        "active_config_name": active_cfg.name,
        "saved_fingerprint": saved_cfg.full_fingerprint(),
        "active_fingerprint": active_cfg.full_fingerprint(),
        "index_fingerprint": active_cfg.index_fingerprint(),
        "is_temporary_ui_override": bool(overrides),
        "overridden_fields": overrides,
        "top_k": int(top_k),
    }
    return active_cfg, meta


def highlight_query_terms(passage_text: str, query: str) -> str:
    """HTML-escape passage text and highlight informative query terms safely."""
    escaped = html.escape(passage_text)
    if not query or not query.strip():
        return escaped
    _, _q_part, exp_part = decompose_query(query, expand_legal=True)
    tokens = sorted(
        {t for t in legal_tokenize(exp_part, remove_stopwords=True) if len(t) >= 4},
        key=len,
        reverse=True,
    )[:14]
    if not tokens:
        return escaped
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b", re.IGNORECASE)
    return pattern.sub(r'<mark class="clause-hl">\1</mark>', escaped)


def build_precision_recall_dataframe(
    baseline_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Build a clean table of character-level Precision@K, Recall@K, and F1@K for K in {1, 3, 5, 10}."""
    rows: list[dict[str, Any]] = []
    for k in (1, 3, 5, 10):
        bp = float(baseline_metrics.get(f"char_precision@{k}", 0.0))
        fp = float(final_metrics.get(f"char_precision@{k}", 0.0))
        br = float(baseline_metrics.get(f"char_recall@{k}", 0.0))
        fr = float(final_metrics.get(f"char_recall@{k}", 0.0))
        bf1 = float(baseline_metrics.get(f"char_f1@{k}", 0.0))
        ff1 = float(final_metrics.get(f"char_f1@{k}", 0.0))
        rows.append(
            {
                "Cutoff (K)": f"K = {k}",
                "Baseline Precision@K": f"{bp:.4f}",
                "Final Precision@K": f"{fp:.4f}",
                "Δ Precision@K": f"{fp - bp:+.4f}",
                "Baseline Recall@K": f"{br:.4f}",
                "Final Recall@K": f"{fr:.4f}",
                "Δ Recall@K": f"{fr - br:+.4f}",
                "Baseline F1@K": f"{bf1:.4f}",
                "Final F1@K": f"{ff1:.4f}",
            }
        )
    return pd.DataFrame(rows)


def build_abstention_dataframe(
    baseline_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Build a structured table of abstention and operational metrics."""
    ans_n = int(final_metrics.get("answerable_query_count", 0))
    unans_n = int(final_metrics.get("unanswerable_query_count", 0))
    b_ca = int(baseline_metrics.get("correct_abstention_count", 0))
    f_ca = int(final_metrics.get("correct_abstention_count", 0))
    b_fa = int(baseline_metrics.get("false_abstention_count", 0))
    f_fa = int(final_metrics.get("false_abstention_count", 0))

    rows = [
        {
            "Metric": f"Correct Abstention Rate (Unanswerable, n={unans_n})",
            "Baseline (TF-IDF)": f"{float(baseline_metrics.get('correct_abstention_rate', 0.0)):.4f} ({b_ca}/{unans_n})",
            "Final (Calibrated Hybrid)": f"{float(final_metrics.get('correct_abstention_rate', 0.0)):.4f} ({f_ca}/{unans_n})",
            "Change (Final − Baseline)": f"{float(final_metrics.get('correct_abstention_rate', 0.0)) - float(baseline_metrics.get('correct_abstention_rate', 0.0)):+.4f}",
        },
        {
            "Metric": f"Incorrect Retrieval Rate on Unanswerable (n={unans_n})",
            "Baseline (TF-IDF)": f"{float(baseline_metrics.get('incorrect_retrieval_rate_unanswerable', 0.0)):.4f}",
            "Final (Calibrated Hybrid)": f"{float(final_metrics.get('incorrect_retrieval_rate_unanswerable', 0.0)):.4f}",
            "Change (Final − Baseline)": f"{float(final_metrics.get('incorrect_retrieval_rate_unanswerable', 0.0)) - float(baseline_metrics.get('incorrect_retrieval_rate_unanswerable', 0.0)):+.4f}",
        },
        {
            "Metric": f"False Abstention Rate (Answerable, n={ans_n})",
            "Baseline (TF-IDF)": f"{float(baseline_metrics.get('false_abstention_rate', 0.0)):.4f} ({b_fa}/{ans_n})",
            "Final (Calibrated Hybrid)": f"{float(final_metrics.get('false_abstention_rate', 0.0)):.4f} ({f_fa}/{ans_n})",
            "Change (Final − Baseline)": f"{float(final_metrics.get('false_abstention_rate', 0.0)) - float(baseline_metrics.get('false_abstention_rate', 0.0)):+.4f}",
        },
        {
            "Metric": f"Answerable Coverage Rate (n={ans_n})",
            "Baseline (TF-IDF)": f"{float(baseline_metrics.get('answer_coverage', 0.0)):.4f}",
            "Final (Calibrated Hybrid)": f"{float(final_metrics.get('answer_coverage', 0.0)):.4f}",
            "Change (Final − Baseline)": f"{float(final_metrics.get('answer_coverage', 0.0)) - float(baseline_metrics.get('answer_coverage', 0.0)):+.4f}",
        },
        {
            "Metric": "Abstention Balanced Accuracy",
            "Baseline (TF-IDF)": f"{float(baseline_metrics.get('abstention_balanced_accuracy', 0.0)):.4f}",
            "Final (Calibrated Hybrid)": f"{float(final_metrics.get('abstention_balanced_accuracy', 0.0)):.4f}",
            "Change (Final − Baseline)": f"{float(final_metrics.get('abstention_balanced_accuracy', 0.0)) - float(baseline_metrics.get('abstention_balanced_accuracy', 0.0)):+.4f}",
        },
        {
            "Metric": "Primary Composite Objective",
            "Baseline (TF-IDF)": f"{float(baseline_metrics.get('primary_objective', 0.0)):.4f}",
            "Final (Calibrated Hybrid)": f"{float(final_metrics.get('primary_objective', 0.0)):.4f}",
            "Change (Final − Baseline)": f"{float(final_metrics.get('primary_objective', 0.0)) - float(baseline_metrics.get('primary_objective', 0.0)):+.4f}",
        },
        {
            "Metric": "Mean Query Latency (ms)",
            "Baseline (TF-IDF)": f"{float(baseline_metrics.get('avg_latency_ms', 0.0)):.2f} ms",
            "Final (Calibrated Hybrid)": f"{float(final_metrics.get('avg_latency_ms', 0.0)):.2f} ms",
            "Change (Final − Baseline)": f"{float(final_metrics.get('avg_latency_ms', 0.0)) - float(baseline_metrics.get('avg_latency_ms', 0.0)):+.2f} ms",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cached Dataset & Retriever Resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading LegalBench-RAG-mini corpus (29 contracts, 194 benchmark queries, 36 unanswerable queries)...")
def get_cached_dataset() -> LegalBenchMiniDataset:
    return load_dataset()


@st.cache_resource(show_spinner="Building or loading cached retrieval index...")
def get_cached_retriever(config_json: str) -> ClauseTraceRetriever:
    ds = get_cached_dataset()
    cfg = PipelineConfig.model_validate(json.loads(config_json))
    return ClauseTraceRetriever(ds.contracts, cfg, use_cache=True)


# ---------------------------------------------------------------------------
# Custom CSS Styling (Accessible Legal-Tech Theme)
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    """Apply clean, high-contrast legal-tech styling (off-white, deep navy, subtle gold)."""
    st.markdown(
        """
        <style>
        /* Force readable light legal-tech palette regardless of OS dark-mode setting */
        :root {
            --bg-offwhite: #FAF8F5;
            --bg-surface: #FFFFFF;
            --bg-subtle: #F3F5F8;
            --navy-primary: #0F1E36;
            --navy-secondary: #1B3358;
            --text-dark: #14243B;
            --text-muted: #475569;
            --gold-accent: #C59B27;
            --gold-soft: #FBF6E8;
            --teal-accent: #1F7A8C;
            --border-clean: #D8E0E8;
        }

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: var(--bg-offwhite) !important;
            color: var(--text-dark) !important;
        }

        [data-testid="stSidebar"] {
            background-color: #F2EFE9 !important;
            border-right: 1px solid var(--border-clean) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text-dark) !important;
        }

        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1220px !important;
        }

        /* Typography & Headings */
        h1, h2, h3, h4, h5, h6 {
            color: var(--navy-primary) !important;
            font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
            letter-spacing: -0.01em;
        }

        p, li, label, span, div {
            color: var(--text-dark);
        }

        /* Widget labels & inputs — guaranteed high contrast */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        .stTextInput label,
        .stSelectbox label,
        .stSlider label,
        .stNumberInput label,
        .stRadio label {
            color: var(--navy-primary) !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        [data-baseweb="select"] > div {
            background-color: var(--bg-surface) !important;
            color: var(--text-dark) !important;
            border: 1px solid #BCCCDC !important;
            border-radius: 6px !important;
        }

        /* Dropdown popover list contrast */
        [data-baseweb="popover"] li,
        [data-baseweb="menu"] li,
        [role="listbox"] li {
            background-color: var(--bg-surface) !important;
            color: var(--text-dark) !important;
        }

        /* Primary & Secondary Buttons */
        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"] {
            background-color: var(--navy-primary) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--navy-primary) !important;
            border-bottom: 2px solid var(--gold-accent) !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            padding: 0.45rem 1.05rem !important;
        }
        .stButton > button[kind="primary"] *,
        .stFormSubmitButton > button[kind="primary"] * {
            color: #FFFFFF !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[kind="primary"]:hover {
            background-color: var(--navy-secondary) !important;
        }
        .stButton > button[kind="secondary"] {
            background-color: var(--bg-surface) !important;
            color: var(--navy-primary) !important;
            border: 1px solid #BCCCDC !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
        }

        /* Navigation Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            background-color: #EFECE6;
            padding: 0.35rem 0.45rem;
            border-radius: 8px;
            border: 1px solid var(--border-clean);
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.45rem;
            padding: 0 1.1rem;
            border-radius: 6px;
            color: var(--navy-primary) !important;
            font-weight: 600;
            background-color: transparent;
            border: none !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--navy-primary) !important;
            color: #FFFFFF !important;
            border-bottom: 2px solid var(--gold-accent) !important;
        }
        .stTabs [aria-selected="true"] p,
        .stTabs [aria-selected="true"] span {
            color: #FFFFFF !important;
        }

        /* Compact Metric Cards */
        [data-testid="stMetric"] {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-clean);
            border-top: 3px solid var(--navy-primary);
            border-radius: 6px;
            padding: 0.6rem 0.8rem;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--text-muted) !important;
            font-size: 0.79rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] div {
            color: var(--navy-primary) !important;
            font-size: 1.2rem !important;
            font-weight: 700 !important;
        }

        /* App Header Banner */
        .legal-banner {
            background-color: var(--navy-primary);
            color: #FFFFFF;
            padding: 1rem 1.35rem;
            border-radius: 8px;
            border-left: 5px solid var(--gold-accent);
            margin-bottom: 0.9rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        .legal-banner h1 {
            color: #FFFFFF !important;
            font-size: 1.35rem !important;
            margin: 0 !important;
            font-weight: 700 !important;
        }
        .legal-banner p {
            color: #E2E8F0 !important;
            font-size: 0.86rem !important;
            margin: 0.2rem 0 0 0 !important;
        }
        .banner-pill {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(197, 155, 39, 0.65);
            color: #F8FAFC !important;
            padding: 0.28rem 0.6rem;
            border-radius: 4px;
            font-size: 0.76rem;
            font-family: monospace;
        }

        /* Sidebar Active Summary Box */
        .sidebar-config-box {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-clean);
            border-left: 4px solid var(--gold-accent);
            border-radius: 6px;
            padding: 0.65rem 0.8rem;
            font-size: 0.81rem;
            line-height: 1.5;
            margin-top: 0.4rem;
        }

        /* Evidence Result Cards */
        .clause-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-clean);
            border-left: 4px solid var(--navy-primary);
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.75rem;
        }
        .clause-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 0.35rem;
        }
        .clause-card-title {
            font-weight: 700;
            font-size: 0.95rem;
            color: var(--navy-primary) !important;
        }
        .clause-meta-bar {
            font-size: 0.8rem;
            color: var(--text-muted) !important;
            font-family: "Consolas", "SFMono-Regular", monospace;
            background-color: var(--bg-subtle);
            padding: 0.35rem 0.6rem;
            border-radius: 4px;
            margin-bottom: 0.55rem;
            border: 1px solid #E2E8F0;
        }
        .clause-body {
            background-color: #FCFBF9;
            border: 1px solid #E5E7EB;
            border-radius: 5px;
            padding: 0.8rem 0.95rem;
            font-family: "Georgia", "Times New Roman", serif;
            font-size: 0.94rem;
            line-height: 1.58;
            color: #111827 !important;
            white-space: pre-wrap;
        }
        mark.clause-hl {
            background-color: #FDE68A !important;
            color: #0F1E36 !important;
            padding: 0.05rem 0.22rem;
            border-radius: 3px;
            font-weight: 600;
        }

        /* Badges */
        .badge-offset-ok {
            background-color: #ECFDF5;
            color: #065F46 !important;
            border: 1px solid #A7F3D0;
            padding: 0.14rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-accepted {
            background-color: #ECFDF5;
            color: #065F46 !important;
            border: 1px solid #6EE7B7;
            padding: 0.14rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
        }
        .badge-rejected {
            background-color: #FEF2F2;
            color: #991B1B !important;
            border: 1px solid #FCA5A5;
            padding: 0.14rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
        }
        .badge-split-train {
            background-color: #EFF6FF;
            color: #1E40AF !important;
            border: 1px solid #BFDBFE;
            padding: 0.18rem 0.6rem;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .badge-split-test {
            background-color: var(--gold-soft);
            color: #78350F !important;
            border: 1px solid #FCD34D;
            padding: 0.18rem 0.6rem;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        /* Abstention Notice Panel */
        .abstention-panel {
            background-color: var(--gold-soft);
            border: 1px solid #EAB308;
            border-left: 5px solid var(--gold-accent);
            border-radius: 6px;
            padding: 0.95rem 1.15rem;
            margin: 0.7rem 0;
            color: #422006 !important;
        }
        .abstention-panel h4 {
            color: #713F12 !important;
            margin: 0 0 0.35rem 0 !important;
            font-size: 0.98rem !important;
        }
        .abstention-panel p {
            color: #422006 !important;
            font-size: 0.88rem !important;
            margin: 0.22rem 0 !important;
        }

        /* Probability & Gate Notice */
        .score-disclaimer {
            background-color: var(--bg-subtle);
            border-left: 3px solid var(--gold-accent);
            padding: 0.42rem 0.72rem;
            border-radius: 4px;
            font-size: 0.81rem;
            color: var(--text-muted) !important;
            margin: 0.4rem 0 0.75rem 0;
        }

        /* Section Sub-box */
        .section-box {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-clean);
            border-radius: 6px;
            padding: 0.85rem 1.05rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: Functional Pipeline Configuration
# ---------------------------------------------------------------------------


def _sync_sidebar_to_preset(preset_key: Literal["final", "baseline"]) -> None:
    """Reset sidebar widget state to match the selected saved YAML preset (`final` or `baseline`)."""
    cfg = load_default_pipeline_config(preset_key)
    st.session_state["sb_strategy"] = cfg.retrieval.strategy
    st.session_state["sb_top_k"] = 5
    st.session_state["sb_abstention_enabled"] = bool(cfg.abstention.enabled)
    st.session_state["sb_abstention_mode"] = cfg.abstention.mode
    st.session_state["sb_min_top_score"] = float(cfg.abstention.min_top_score)
    st.session_state["sb_min_clause_score"] = float(cfg.abstention.min_clause_score)
    st.session_state["sb_min_doc_match_score"] = float(cfg.abstention.min_doc_match_score)
    st.session_state["sb_fusion_method"] = cfg.retrieval.fusion_method
    st.session_state["sb_rerank_enabled"] = bool(cfg.rerank.enabled)


def render_pipeline_sidebar(counts: dict[str, int]) -> tuple[PipelineConfig, dict[str, Any]]:
    """Render the Pipeline Configuration Sidebar and return `(active_config, config_meta)`."""
    if "sb_initialized" not in st.session_state:
        st.session_state["sb_base_preset"] = "final"
        _sync_sidebar_to_preset("final")
        st.session_state["sb_initialized"] = True

    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Configuration")
        st.caption(
            "Configure live retrieval settings. Saved YAML files (`configs/final.yaml`, `configs/baseline.yaml`) are never modified."
        )

        preset_labels = {
            "final": "Frozen Final Config (configs/final.yaml)",
            "baseline": "Baseline TF-IDF (configs/baseline.yaml)",
        }
        selected_preset: Literal["final", "baseline"] = st.selectbox(
            "Base Configuration Preset",
            options=["final", "baseline"],
            format_func=lambda k: preset_labels[k],
            key="sb_base_preset",
            on_change=lambda: _sync_sidebar_to_preset(st.session_state["sb_base_preset"]),
            help="Load defaults from the frozen final configuration or the initial TF-IDF baseline.",
        )

        st.button(
            "Reset to Saved Preset Defaults",
            type="secondary",
            width="stretch",
            on_click=lambda: _sync_sidebar_to_preset(st.session_state.get("sb_base_preset", "final")),
        )

        st.markdown("#### 1. Retrieval Strategy")
        strategy_keys = list(SUPPORTED_STRATEGIES.keys())
        selected_strategy: Literal["hybrid", "bm25", "dense", "tfidf"] = st.selectbox(
            "Retrieval Approach",
            options=strategy_keys,
            format_func=lambda k: SUPPORTED_STRATEGIES[k],
            key="sb_strategy",
            help="Changes `retrieval.strategy` in the backend `ClauseTraceRetriever`.",
        )

        st.markdown("#### 2. Top-K Passages")
        selected_top_k: int = int(
            st.slider(
                "Maximum Passages to Return (Top-K)",
                min_value=1,
                max_value=10,
                step=1,
                key="sb_top_k",
                help="Passed directly to `ClauseTraceRetriever.search(query, top_k=Top-K)`. At most Top-K passages are returned.",
            )
        )

        st.markdown("#### 3. Abstention Gate")
        abstention_enabled: bool = bool(
            st.toggle(
                "Enable Abstention Gate",
                key="sb_abstention_enabled",
                help=(
                    "When enabled, queries failing the configured relevance criteria return 0 passages with an explicit abstention reason. "
                    "When disabled, relevance thresholds are bypassed (only empty/non-alphanumeric input validation remains active)."
                ),
            )
        )
        if not abstention_enabled:
            st.caption(
                "ℹ️ **Gate Disabled:** Score thresholds are bypassed so top-K candidate clauses are always returned. "
                "*(Safety check active: empty or punctuation-only queries `< 3` chars still abstain cleanly as `abstained_invalid_query`.)*"
            )

        with st.expander("4. Advanced Settings (Thresholds & Fusion)", expanded=False):
            st.caption(
                "Chunk size (`750`/`1000`), overlap (`120`/`100`), and routing weights remain fixed in the YAML config files for index reproducibility."
            )
            abstention_mode: Literal["dual_gate", "single_threshold"] = st.selectbox(
                "Abstention Mode (`abstention.mode`)",
                options=["dual_gate", "single_threshold"],
                format_func=lambda m: "Dual-Gate (Score + Contract + Clause)" if m == "dual_gate" else "Single Combined Score Threshold",
                key="sb_abstention_mode",
                disabled=not abstention_enabled,
                help="Dual-Gate checks overall score, target contract alignment, and raw clause relevance.",
            )
            min_top_score: float = float(
                st.slider(
                    "Min Overall Score (`min_top_score`)",
                    min_value=0.0,
                    max_value=0.50,
                    step=0.005,
                    key="sb_min_top_score",
                    disabled=not abstention_enabled,
                )
            )
            min_clause_score: float = float(
                st.slider(
                    "Min Clause Score (`min_clause_score`)",
                    min_value=0.0,
                    max_value=0.50,
                    step=0.005,
                    key="sb_min_clause_score",
                    disabled=(not abstention_enabled) or (abstention_mode != "dual_gate"),
                )
            )
            min_doc_match_score: float = float(
                st.slider(
                    "Min Contract Match (`min_doc_match_score`)",
                    min_value=0.0,
                    max_value=0.75,
                    step=0.005,
                    key="sb_min_doc_match_score",
                    disabled=(not abstention_enabled) or (abstention_mode != "dual_gate"),
                )
            )
            fusion_method: Literal["linear", "rrf"] = st.selectbox(
                "Hybrid Fusion (`retrieval.fusion_method`)",
                options=["linear", "rrf"],
                format_func=lambda f: "Linear Weighted Fusion" if f == "linear" else "Reciprocal Rank Fusion (RRF)",
                key="sb_fusion_method",
                disabled=(selected_strategy != "hybrid"),
            )
            rerank_enabled: bool = bool(
                st.toggle(
                    "Cross-Encoder Reranker (`rerank.enabled`)",
                    key="sb_rerank_enabled",
                    help="Optional second-stage MS-MARCO Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).",
                )
            )

        active_cfg, cfg_meta = build_ui_pipeline_config(
            base_preset=selected_preset,
            strategy=selected_strategy,
            top_k=selected_top_k,
            abstention_enabled=abstention_enabled,
            abstention_mode=abstention_mode,
            min_top_score=min_top_score,
            min_clause_score=min_clause_score,
            min_doc_match_score=min_doc_match_score,
            fusion_method=fusion_method,
            rerank_enabled=rerank_enabled,
        )

        st.markdown("#### 5. Active Configuration")
        source_status = (
            '<span class="badge-split-test">Temporary UI Selection</span>'
            if cfg_meta["is_temporary_ui_override"]
            else f'<span class="badge-accepted">Matches Saved {html.escape(cfg_meta["base_preset_file"])}</span>'
        )
        gate_label = (
            f"Enabled ({active_cfg.abstention.mode})"
            if active_cfg.abstention.enabled
            else "Disabled (Top-K Always Returned)"
        )
        override_keys = ", ".join(cfg_meta["overridden_fields"].keys()) if cfg_meta["overridden_fields"] else "None"

        st.markdown(
            f"""
            <div class="sidebar-config-box">
                <div style="margin-bottom:0.3rem;">{source_status}</div>
                <strong>Strategy:</strong> <code>{html.escape(active_cfg.retrieval.strategy + ('+rerank' if active_cfg.rerank.enabled else ''))}</code><br/>
                <strong>Top-K Limit:</strong> <code>{selected_top_k}</code> passages<br/>
                <strong>Abstention Gate:</strong> <code>{html.escape(gate_label)}</code><br/>
                <strong>Config Fingerprint:</strong> <code>{active_cfg.full_fingerprint()}</code><br/>
                <strong>Index Fingerprint:</strong> <code>{active_cfg.index_fingerprint()}</code><br/>
                <strong>UI Overrides vs File:</strong> <code>{html.escape(override_keys)}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"**Corpus:** {counts['indexed_contracts']} CUAD-mini contracts · "
            f"**Split (Frozen SHA-256):** {counts['answerable_train']}+{counts['unanswerable_train']} Train / "
            f"{counts['answerable_test']}+{counts['unanswerable_test']} Test"
        )

    return active_cfg, cfg_meta


# ---------------------------------------------------------------------------
# Tab 1: Clause Search Workspace (Primary Workflow)
# ---------------------------------------------------------------------------


def render_clause_search_page(
    ds: LegalBenchMiniDataset,
    active_cfg: PipelineConfig,
    cfg_meta: dict[str, Any],
) -> None:
    """Render Tab 1: Natural-language clause search, ranked evidence, and honest abstention."""
    st.markdown("### Clause Search Workspace")
    st.caption(
        "Retrieve exact supporting contract clauses from the 29 indexed CUAD-mini agreements. "
        "Every control in the left sidebar is connected directly to `ClauseTraceRetriever`."
    )

    if "query_input" not in st.session_state:
        st.session_state["query_input"] = SAMPLE_QUERIES[1]["query"]

    sample_labels = [item["label"] for item in SAMPLE_QUERIES]
    selected_sample_label = st.selectbox(
        "Quick-Load Sample Benchmark or Unanswerable Query (optional)",
        options=sample_labels,
        index=1,
        key="sample_selector",
    )

    sample_map = {item["label"]: item["query"] for item in SAMPLE_QUERIES}
    chosen_sample_query = sample_map.get(selected_sample_label, "")
    if st.session_state.get("_last_sample_label") != selected_sample_label:
        st.session_state["_last_sample_label"] = selected_sample_label
        if chosen_sample_query:
            st.session_state["query_input"] = chosen_sample_query

    with st.form(key="clause_search_form", clear_on_submit=False):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            query_text = st.text_input(
                "Natural-Language Contract Question",
                value=st.session_state.get("query_input", SAMPLE_QUERIES[1]["query"]),
                placeholder="e.g., Consider the <Agreement>; Does this agreement allow sublicensing to third-party platforms?",
            )
        with btn_col:
            st.write("")
            submitted = st.form_submit_button("Search", type="primary", width="stretch")

    if submitted:
        st.session_state["query_input"] = query_text

    active_query = st.session_state.get("query_input", "").strip()
    if not active_query:
        st.info("Enter a contract question above and click **Search** to retrieve supporting clauses.")
        return

    top_k = int(cfg_meta["top_k"])
    retriever = get_cached_retriever(json.dumps(active_cfg.model_dump(), sort_keys=True))
    resp = retriever.search(active_query, top_k=top_k)

    # Compact summary row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Strategy Used", resp.retrieval_method.upper())
    m2.metric(
        "Decision / Count",
        "Abstained (0)" if resp.abstained else f"{len(resp.passages)} / {top_k} Passages",
    )
    m3.metric("Top Combined Score", f"{resp.top_score:.4f}")
    m4.metric("Clause Relevance", f"{resp.top_clause_score:.4f}")
    m5.metric("Latency", f"{resp.latency_ms:.1f} ms")

    st.markdown(
        """
        <div class="score-disclaimer">
            <strong>Score Interpretation Note:</strong> Retrieval, lexical, semantic, and document-alignment scores are uncalibrated similarity measures in <code>[0, 1]</code> used to rank candidate passages and gate abstention. They are <strong>not</strong> calibrated legal probabilities that a contractual right or obligation exists.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if resp.abstained:
        st.markdown(
            f"""
            <div class="abstention-panel">
                <h4>Abstention Triggered — No Passage Met Configured Evidence Thresholds</h4>
                <p><strong>Decision Status:</strong> <code>{html.escape(resp.abstention.status)}</code></p>
                <p><strong>Explanation:</strong> {html.escape(resp.abstention.reason)}</p>
                <p style="font-size:0.82rem; margin-top:0.35rem;">
                    <strong>Scope Distinction:</strong> Abstaining indicates that no passage in the 29 indexed contracts satisfied the active relevance criteria (<code>mode={html.escape(active_cfg.abstention.mode)}</code>, <code>min_top_score={active_cfg.abstention.min_top_score:.3f}</code>). Toggle <em>Enable Abstention Gate</em> off in the sidebar to inspect the unfiltered top-{top_k} candidates directly.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("View Abstention Diagnostics & Highest Sub-Threshold Candidate"):
            st.json(resp.abstention.diagnostics)
            if resp.unfiltered_top_passages:
                sub = resp.unfiltered_top_passages[0]
                st.caption(
                    f"Highest sub-threshold candidate filtered out: `{sub.doc_id}` "
                    f"offsets `[{sub.char_start}:{sub.char_end}]` (combined={sub.retrieval_score:.4f}, "
                    f"clause={sub.scores.get('clause_score', 0.0):.4f}, doc={sub.scores.get('doc_match_score', 0.0):.4f})"
                )
                st.code(sub.passage_text, language="text")
        return

    # Render ranked passage cards (guaranteed <= top_k)
    for p in resp.passages:
        offset_verified = verify_evidence_offset(
            ds.contracts, p.doc_id, p.char_start, p.char_end, p.passage_text
        )
        badge_html = (
            '<span class="badge-offset-ok">✓ Offset Verified: doc[start:end] == passage</span>'
            if offset_verified
            else '<span class="badge-rejected">Offset Mismatch</span>'
        )
        sec_str = f" · Section: {html.escape(p.section_heading)}" if p.section_heading else ""
        clause_str = f" · Clause: {html.escape(p.clause_number)}" if p.clause_number else ""
        rerank_str = (
            f" · Rerank: <strong>{p.scores['rerank_score']:.4f}</strong>"
            if "rerank_score" in p.scores
            else ""
        )
        highlighted_clause = highlight_query_terms(p.passage_text, active_query)

        st.markdown(
            f"""
            <div class="clause-card">
                <div class="clause-card-header">
                    <span class="clause-card-title">Rank #{p.rank} · {html.escape(p.doc_id)}</span>
                    {badge_html}
                </div>
                <div class="clause-meta-bar">
                    Source Path: {html.escape(p.file_path)} · Offsets: [{p.char_start}:{p.char_end}] ({p.char_end - p.char_start} chars){sec_str}{clause_str}<br/>
                    Combined Score: <strong>{p.retrieval_score:.4f}</strong> · Clause Score: <strong>{p.scores.get('clause_score', 0.0):.4f}</strong> · Contract Match: <strong>{p.scores.get('doc_match_score', 0.0):.4f}</strong> · Lexical: {p.scores.get('lexical_score', 0.0):.4f} · Dense: {p.scores.get('dense_score', 0.0):.4f}{rerank_str}
                </div>
                <div class="clause-body">{highlighted_clause}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            with st.expander(f"Inspect Surrounding Document Text (±350 chars around [{p.char_start}:{p.char_end}])"):
                doc_text = ds.contracts[p.doc_id].text
                w_start = max(0, p.char_start - 350)
                w_end = min(len(doc_text), p.char_end + 350)
                prefix_txt = doc_text[w_start : p.char_start]
                suffix_txt = doc_text[p.char_end : w_end]
                st.caption(f"Document slice `[{w_start}:{w_end}]` from `{p.file_path}`:")
                st.code(f"...{prefix_txt}[[[{p.passage_text}]]]{suffix_txt}...", language="text")
        with exp_col2:
            if p.cross_references or p.referenced_context:
                with st.expander(f"Cross-References in Clause ({len(p.cross_references)})"):
                    st.write(
                        "**Detected Intra-Contract References:**",
                        ", ".join(p.cross_references) if p.cross_references else "None",
                    )
                    if p.referenced_context:
                        st.caption(
                            "Heuristic linked section preview (kept separate from the exact retrieved passage):"
                        )
                        st.code(p.referenced_context, language="text")


# ---------------------------------------------------------------------------
# Tab 2: Evaluation
# ---------------------------------------------------------------------------


def render_evaluation_page(ds: LegalBenchMiniDataset) -> None:
    """Render Tab 2: Actual character-level precision/recall at K=1,3,5,10 and abstention metrics."""
    st.markdown("### Evaluation Results")
    st.caption(
        "Character-level Precision@K and Recall@K (following the LegalBench-RAG span-union formulation) "
        "and bidirectional abstention accuracy on the prescribed SHA-256 train and held-out test splits. "
        "All metrics are read directly from `outputs/evaluation/*_metrics.json`."
    )

    base_train = load_json_artifact("outputs/evaluation/baseline_train_metrics.json")
    final_train = load_json_artifact("outputs/evaluation/final_train_metrics.json")
    base_test = load_json_artifact("outputs/evaluation/baseline_test_metrics.json")
    final_test = load_json_artifact("outputs/evaluation/final_test_metrics.json")
    ledger = load_json_artifact("outputs/evaluation/test_run_ledger.json")

    if not (base_train and final_train and base_test and final_test):
        st.warning("Evaluation artifacts not found in `outputs/evaluation/`. Run `python -m contract_intelligence evaluate`.")
        return

    bt_m = base_train["metrics"]
    ft_m = final_train["metrics"]
    btest_m = base_test["metrics"]
    ftest_m = final_test["metrics"]

    split_choice = st.radio(
        "Evaluation Split View (Read-Only Benchmark Artifacts)",
        options=[
            "Train Split (140 Answerable + 26 Unanswerable — Used for Tuning & Calibration)",
            "Held-Out Test Split (54 Answerable + 10 Unanswerable — Final Reporting Only)",
            "Side-by-Side Train vs. Held-Out Test Summary",
        ],
        horizontal=True,
    )

    if split_choice.startswith("Train"):
        st.markdown(
            '<span class="badge-split-train">TRAIN SPLIT · 140 Answerable Queries + 26 Unanswerable Queries (166 Total)</span>',
            unsafe_allow_html=True,
        )
        active_base, active_final = bt_m, ft_m
        err_csv_path = Path("outputs/evaluation/final_train_errors.csv")
    elif split_choice.startswith("Held-Out Test"):
        test_runs = ledger.get("test_evaluation_runs", len(ledger.get("runs", []))) if ledger else 0
        st.markdown(
            f'<span class="badge-split-test">HELD-OUT TEST SPLIT · 54 Answerable Queries + 10 Unanswerable Queries (64 Total) · Audit Ledger Count: {test_runs}</span>',
            unsafe_allow_html=True,
        )
        active_base, active_final = btest_m, ftest_m
        err_csv_path = Path("outputs/evaluation/final_test_errors.csv")
    else:
        active_base, active_final = None, None
        err_csv_path = None

    if active_base and active_final:
        st.write("")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(
            "Char Recall@5",
            f"{active_final['char_recall@5']:.4f}",
            delta=f"{active_final['char_recall@5'] - active_base['char_recall@5']:+.4f} vs baseline",
        )
        k2.metric(
            "Char Precision@5",
            f"{active_final['char_precision@5']:.4f}",
            delta=f"{active_final['char_precision@5'] - active_base['char_precision@5']:+.4f} vs baseline",
        )
        k3.metric(
            "Char Recall@1",
            f"{active_final['char_recall@1']:.4f}",
            delta=f"{active_final['char_recall@1'] - active_base['char_recall@1']:+.4f} vs baseline",
        )
        k4.metric(
            "Correct Abstention",
            f"{active_final['correct_abstention_rate']:.4f}",
            delta=f"{active_final['correct_abstention_rate'] - active_base['correct_abstention_rate']:+.4f} vs baseline",
        )
        k5.metric(
            "False Abstention",
            f"{active_final['false_abstention_rate']:.4f}",
            delta=f"{active_final['false_abstention_rate'] - active_base['false_abstention_rate']:+.4f} vs baseline",
            delta_color="inverse",
        )

        st.markdown("#### 1. Character-Level Precision@K, Recall@K, and F1@K (K = 1, 3, 5, 10)")
        pr_df = build_precision_recall_dataframe(active_base, active_final)
        st.dataframe(pr_df, width="stretch", hide_index=True)

        st.markdown("#### 2. Abstention Accuracy & Coverage (Answerable vs. Unanswerable Queries)")
        abs_df = build_abstention_dataframe(active_base, active_final)
        st.dataframe(abs_df, width="stretch", hide_index=True)

        if err_csv_path and err_csv_path.exists():
            with st.expander(f"Inspect Query-Level Error Diagnostics ({err_csv_path.name})"):
                err_df = pd.read_csv(err_csv_path)
                st.caption(f"Logged failure cases in final configuration: {len(err_df)} queries")
                st.dataframe(err_df, width="stretch", hide_index=True)
    else:
        st.markdown("#### Side-by-Side Train Split vs. Held-Out Test Split Comparison")
        side_rows = [
            ("Answerable Queries", bt_m["answerable_query_count"], ft_m["answerable_query_count"], btest_m["answerable_query_count"], ftest_m["answerable_query_count"]),
            ("Unanswerable Queries", bt_m["unanswerable_query_count"], ft_m["unanswerable_query_count"], btest_m["unanswerable_query_count"], ftest_m["unanswerable_query_count"]),
            ("Primary Objective", f"{bt_m['primary_objective']:.4f}", f"{ft_m['primary_objective']:.4f}", f"{btest_m['primary_objective']:.4f}", f"{ftest_m['primary_objective']:.4f}"),
            ("Char Precision@1", f"{bt_m['char_precision@1']:.4f}", f"{ft_m['char_precision@1']:.4f}", f"{btest_m['char_precision@1']:.4f}", f"{ftest_m['char_precision@1']:.4f}"),
            ("Char Recall@1", f"{bt_m['char_recall@1']:.4f}", f"{ft_m['char_recall@1']:.4f}", f"{btest_m['char_recall@1']:.4f}", f"{ftest_m['char_recall@1']:.4f}"),
            ("Char Precision@3", f"{bt_m['char_precision@3']:.4f}", f"{ft_m['char_precision@3']:.4f}", f"{btest_m['char_precision@3']:.4f}", f"{ftest_m['char_precision@3']:.4f}"),
            ("Char Recall@3", f"{bt_m['char_recall@3']:.4f}", f"{ft_m['char_recall@3']:.4f}", f"{btest_m['char_recall@3']:.4f}", f"{ftest_m['char_recall@3']:.4f}"),
            ("Char Precision@5", f"{bt_m['char_precision@5']:.4f}", f"{ft_m['char_precision@5']:.4f}", f"{btest_m['char_precision@5']:.4f}", f"{ftest_m['char_precision@5']:.4f}"),
            ("Char Recall@5", f"{bt_m['char_recall@5']:.4f}", f"{ft_m['char_recall@5']:.4f}", f"{btest_m['char_recall@5']:.4f}", f"{ftest_m['char_recall@5']:.4f}"),
            ("Char Precision@10", f"{bt_m['char_precision@10']:.4f}", f"{ft_m['char_precision@10']:.4f}", f"{btest_m['char_precision@10']:.4f}", f"{ftest_m['char_precision@10']:.4f}"),
            ("Char Recall@10", f"{bt_m['char_recall@10']:.4f}", f"{ft_m['char_recall@10']:.4f}", f"{btest_m['char_recall@10']:.4f}", f"{ftest_m['char_recall@10']:.4f}"),
            ("Correct Abstention Rate", f"{bt_m['correct_abstention_rate']:.4f} ({bt_m['correct_abstention_count']}/26)", f"{ft_m['correct_abstention_rate']:.4f} ({ft_m['correct_abstention_count']}/26)", f"{btest_m['correct_abstention_rate']:.4f} ({btest_m['correct_abstention_count']}/10)", f"{ftest_m['correct_abstention_rate']:.4f} ({ftest_m['correct_abstention_count']}/10)"),
            ("False Abstention Rate", f"{bt_m['false_abstention_rate']:.4f} ({bt_m['false_abstention_count']}/140)", f"{ft_m['false_abstention_rate']:.4f} ({ft_m['false_abstention_count']}/140)", f"{btest_m['false_abstention_rate']:.4f} ({btest_m['false_abstention_count']}/54)", f"{ftest_m['false_abstention_rate']:.4f} ({ftest_m['false_abstention_count']}/54)"),
            ("Answer Coverage", f"{bt_m['answer_coverage']:.4f}", f"{ft_m['answer_coverage']:.4f}", f"{btest_m['answer_coverage']:.4f}", f"{ftest_m['answer_coverage']:.4f}"),
            ("Mean Latency (ms)", f"{bt_m['avg_latency_ms']:.2f}", f"{ft_m['avg_latency_ms']:.2f}", f"{btest_m['avg_latency_ms']:.2f}", f"{ftest_m['avg_latency_ms']:.2f}"),
        ]
        side_df = pd.DataFrame(
            side_rows,
            columns=[
                "Metric",
                "Train Baseline (TF-IDF)",
                "Train Final (Hybrid)",
                "Test Baseline (TF-IDF)",
                "Test Final (Hybrid)",
            ],
        )
        st.dataframe(side_df, width="stretch", hide_index=True)

    with st.expander("Dataset Split & Ground-Truth Offset Validation Summary"):
        val_rep = validate_ground_truth_offsets(ds)
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Prescribed SHA-256 Split Counts**")
            st.json(ds.summary_counts())
        with c_right:
            st.markdown("**Ground-Truth Character Offset Verification (`272/272` snippets)**")
            st.json(val_rep.to_dict())


# ---------------------------------------------------------------------------
# Tab 3: Experiments
# ---------------------------------------------------------------------------


def render_experiments_page() -> None:
    """Render Tab 3: Baseline and all 7 logged self-improvement iterations from disk."""
    st.markdown("### Self-Improvement Experiments (Train Split Only)")
    st.caption(
        "All experiment iterations are read directly from `experiments/experiment_log.jsonl` and `experiments/improvement_summary.json`. "
        "Every candidate change was evaluated strictly on the 166 train queries (`test_set_touched = False`)."
    )

    summary = load_json_artifact("experiments/improvement_summary.json")
    records = load_experiment_jsonl("experiments/experiment_log.jsonl")
    exp_csv = Path("experiments/experiment_log.csv")

    if not records:
        st.warning("No experiment logs found in `experiments/experiment_log.jsonl`. Run `python -m contract_intelligence.improve`.")
        return

    if summary:
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Evaluation Split", str(summary.get("split_used", "train")).upper())
        s2.metric("Post-Baseline Iterations", str(summary.get("iterations_run", 7)))
        s3.metric(
            "Accepted / Rejected",
            f"{len(summary.get('accepted_iterations', []))} Acc / {len(summary.get('rejected_iterations', []))} Rej",
        )
        s4.metric(
            "Train Objective Gain",
            f"{summary['final_train_metrics']['primary_objective']:.4f}",
            delta=f"{summary['final_train_metrics']['primary_objective'] - summary['baseline_train_metrics']['primary_objective']:+.4f} vs baseline",
        )
        s5.metric("Selected Final Config", str(summary.get("final_selected_config", "")))

        obj_decl = summary.get("objective_declaration", {})
        with st.expander("Primary Optimization Metric & Guardrail Definitions", expanded=False):
            st.markdown(f"**Formula:** `{obj_decl.get('formula', '')}`")
            st.write(obj_decl.get("rationale", ""))
            st.markdown("**Enforced Guardrail Conditions:**")
            st.json(obj_decl.get("guardrails", {}))

    st.markdown("#### Summary Table of Logged Iterations (`experiments/experiment_log.csv`)")
    if exp_csv.exists():
        df_log = pd.read_csv(exp_csv)
        st.dataframe(df_log, width="stretch", hide_index=True)

    st.markdown("#### Detailed Iteration Log (Baseline + 7 Improvement Iterations)")
    for rec in records:
        it_num = int(rec.get("iteration", 0))
        cfg_name = str(rec.get("config_name", ""))
        accepted = bool(rec.get("accepted", False))
        m_before = rec.get("metrics_before", {})
        m_after = rec.get("metrics_after", {})

        if it_num == 0:
            status_badge = '<span class="badge-split-train">BASELINE ANCHOR (Iteration 0)</span>'
        elif accepted:
            status_badge = '<span class="badge-accepted">✓ ACCEPTED</span>'
        else:
            status_badge = '<span class="badge-rejected">✗ REJECTED (Reverted to Parent)</span>'

        obj_before = float(m_before.get("primary_objective", 0.0))
        obj_after = float(m_after.get("primary_objective", 0.0))
        r5_before = float(m_before.get("char_recall@5", 0.0))
        r5_after = float(m_after.get("char_recall@5", 0.0))
        p5_before = float(m_before.get("char_precision@5", 0.0))
        p5_after = float(m_after.get("char_precision@5", 0.0))
        ca_after = float(m_after.get("correct_abstention_rate", 0.0))
        fa_after = float(m_after.get("false_abstention_rate", 0.0))

        st.markdown(
            f"""
            <div class="section-box">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.35rem;">
                    <strong style="font-size:0.96rem; color:#0F1E36;">Iteration {it_num}: <code>{html.escape(cfg_name)}</code> (Parent: <code>{html.escape(str(rec.get('parent_config_name', 'none')))}</code>)</strong>
                    {status_badge}
                </div>
                <p style="margin:0.2rem 0; font-size:0.88rem;"><strong>Proposed Change:</strong> {html.escape(str(rec.get('proposed_change', '')))}</p>
                <p style="margin:0.2rem 0; font-size:0.87rem; color:#334155;"><strong>Diagnostic Rationale:</strong> {html.escape(str(rec.get('reason', '')))}</p>
                <p style="margin:0.2rem 0; font-size:0.87rem;"><strong>Decision:</strong> {html.escape(str(rec.get('decision_reason', '')))}</p>
                <div class="clause-meta-bar" style="margin-top:0.45rem; margin-bottom:0;">
                    Objective: {obj_before:.4f} → <strong>{obj_after:.4f}</strong> ({obj_after - obj_before:+.4f}) ·
                    Recall@5: {r5_before:.4f} → <strong>{r5_after:.4f}</strong> ·
                    Precision@5: {p5_before:.4f} → <strong>{p5_after:.4f}</strong> ·
                    Correct Abstention: <strong>{ca_after:.4f}</strong> ·
                    False Abstention: <strong>{fa_after:.4f}</strong> ·
                    Latency: {float(m_after.get('avg_latency_ms', 0.0)):.1f} ms
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        guardrails = rec.get("guardrail_checks", {})
        diag = rec.get("failure_diagnosis", {})
        if guardrails or diag:
            with st.expander(f"Inspect Iteration {it_num} Guardrail Checks & Failure Taxonomy"):
                g_col, d_col = st.columns(2)
                with g_col:
                    st.markdown("**Guardrail Verification**")
                    st.json(guardrails if guardrails else {"status": "Baseline anchor (no parent comparison)"})
                with d_col:
                    st.markdown("**Train Failure Taxonomy Counts**")
                    st.json(diag)


# ---------------------------------------------------------------------------
# Tab 4: Active Configuration & Saved Presets Inspector
# ---------------------------------------------------------------------------


def render_active_configuration_page(
    active_cfg: PipelineConfig,
    cfg_meta: dict[str, Any],
) -> None:
    """Render Tab 4: Inspect active configuration vs saved baseline/final YAML files."""
    st.markdown("### Active Configuration & Reproducibility Inspector")
    st.caption(
        "Compare the active sidebar configuration against the frozen YAML files (`configs/final.yaml` and `configs/baseline.yaml`). "
        "Interactive UI changes are held in memory for live exploration and never overwrite the committed experiment configurations."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base YAML Preset", cfg_meta["base_preset_file"])
    c2.metric("Saved SHA-256", cfg_meta["saved_fingerprint"][:12])
    c3.metric("Active SHA-256", cfg_meta["active_fingerprint"][:12])
    c4.metric(
        "Config State",
        "Temporary UI Override" if cfg_meta["is_temporary_ui_override"] else "Exact Saved Preset",
    )

    if cfg_meta["is_temporary_ui_override"]:
        st.markdown("#### Temporary UI Overrides Applied Over Saved Preset")
        override_rows = [
            {
                "Parameter Path": k,
                f"Saved ({cfg_meta['base_preset_file']})": str(v["saved"]),
                "Active (Sidebar Selection)": str(v["active"]),
            }
            for k, v in cfg_meta["overridden_fields"].items()
        ]
        st.dataframe(pd.DataFrame(override_rows), width="stretch", hide_index=True)
    else:
        st.success(
            f"The active sidebar configuration is identical to the saved `{cfg_meta['base_preset_file']}` "
            f"(Fingerprint `{cfg_meta['active_fingerprint']}`)."
        )

    col_active, col_saved = st.columns(2)
    with col_active:
        st.markdown(f"#### Active In-Memory Configuration (`{active_cfg.full_fingerprint()}`)")
        st.code(yaml.safe_dump(active_cfg.model_dump(), sort_keys=False), language="yaml")
    with col_saved:
        preset_path = Path(cfg_meta["base_preset_file"])
        st.markdown(f"#### Saved File on Disk (`{preset_path.as_posix()}`)")
        if preset_path.exists():
            st.code(preset_path.read_text(encoding="utf-8"), language="yaml")


# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="ClauseTrace | Contract Intelligence Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    ds = get_cached_dataset()
    counts = ds.summary_counts()
    active_cfg, cfg_meta = render_pipeline_sidebar(counts)

    gate_short = "ON" if active_cfg.abstention.enabled else "OFF"
    state_short = "UI-CUSTOM" if cfg_meta["is_temporary_ui_override"] else cfg_meta["base_preset"].upper()

    st.markdown(
        f"""
        <div class="legal-banner">
            <div>
                <h1>⚖️ ClauseTrace · Contract Intelligence Assistant</h1>
                <p>Evidence-first legal clause retrieval over LegalBench-RAG-mini (CUAD) with exact character offsets and calibrated abstention.</p>
            </div>
            <div style="display:flex; gap:0.45rem; flex-wrap:wrap;">
                <span class="banner-pill">Strategy: {html.escape(active_cfg.retrieval.strategy.upper())}</span>
                <span class="banner-pill">Top-K: {cfg_meta['top_k']}</span>
                <span class="banner-pill">Abstention: {gate_short}</span>
                <span class="banner-pill">Preset: {state_short} ({active_cfg.full_fingerprint()[:8]})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_search, tab_eval, tab_exp, tab_cfg = st.tabs(
        [
            "1. Clause Search",
            "2. Evaluation",
            "3. Experiments",
            "4. Active Configuration",
        ]
    )

    with tab_search:
        render_clause_search_page(ds, active_cfg, cfg_meta)

    with tab_eval:
        render_evaluation_page(ds)

    with tab_exp:
        render_experiments_page()

    with tab_cfg:
        render_active_configuration_page(active_cfg, cfg_meta)


if __name__ == "__main__":
    main()
