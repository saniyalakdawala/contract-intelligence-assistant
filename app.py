"""Streamlit web interface for ClauseTrace | Contract Intelligence Assistant."""

from __future__ import annotations

import copy
import html
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import pandas as pd
import streamlit as st

from contract_intelligence import __version__
from contract_intelligence.config import PipelineConfig
from contract_intelligence.dataset import LegalBenchMiniDataset, load_dataset
from contract_intelligence.indexing import decompose_query, legal_tokenize
from contract_intelligence.retrieval import ClauseTraceRetriever
from contract_intelligence.validation import validate_ground_truth_offsets, verify_evidence_offset

st.set_page_config(
    page_title="ClauseTrace | Contract Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Deep Navy (#0F1E36), Warm Ivory (#FAF8F5), Muted Teal (#1F7A8C), and Subtle Gold (#C59B27)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FAF8F5;
        color: #0F1E36;
    }
    .clausetrace-header {
        background: linear-gradient(135deg, #0F1E36 0%, #182F52 100%);
        color: #FAF8F5;
        padding: 1.4rem 1.8rem;
        border-radius: 10px;
        border-left: 6px solid #C59B27;
        margin-bottom: 1.2rem;
    }
    .clausetrace-header h1 {
        color: #FAF8F5 !important;
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .clausetrace-header p {
        color: #D9E2EC;
        margin: 0.35rem 0 0 0;
        font-size: 0.96rem;
    }
    .evidence-card {
        background-color: #FFFFFF;
        border: 1px solid #D8E0E8;
        border-left: 5px solid #1F7A8C;
        border-radius: 8px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1.0rem;
        box-shadow: 0 2px 5px rgba(15, 30, 54, 0.04);
    }
    .evidence-meta {
        font-size: 0.84rem;
        color: #486581;
        margin-bottom: 0.6rem;
        font-family: monospace;
    }
    .evidence-passage {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.95rem 1.1rem;
        font-family: "Georgia", serif;
        font-size: 0.97rem;
        line-height: 1.62;
        color: #102A43;
        white-space: pre-wrap;
    }
    .abstention-box {
        background-color: #FFFBEB;
        border: 1px solid #F59E0B;
        border-left: 6px solid #C59B27;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        color: #78350F;
    }
    .badge-verified {
        background-color: #E6FFFA;
        color: #047857;
        border: 1px solid #A7F3D0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-split {
        background-color: #0F1E36;
        color: #FAF8F5;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.80rem;
        font-weight: 600;
    }
    mark.clause-hl {
        background-color: #FEF08A;
        color: #0F1E36;
        padding: 0 2px;
        border-radius: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading LegalBench-RAG-mini (194 queries, 29 contracts, 36 unanswerable queries)...")
def get_cached_dataset() -> LegalBenchMiniDataset:
    return load_dataset()


@st.cache_resource(show_spinner="Building & caching retrieval index...")
def get_cached_retriever(config_json: str) -> ClauseTraceRetriever:
    ds = get_cached_dataset()
    cfg = PipelineConfig.model_validate(json.loads(config_json))
    return ClauseTraceRetriever(ds.contracts, cfg, use_cache=True)


def highlight_query_terms(passage_text: str, query: str) -> str:
    """Safely HTML-escape the passage text and highlight informative query terms."""
    escaped = html.escape(passage_text)
    _, q_part, exp_part = decompose_query(query, expand_legal=True)
    tokens = sorted(
        {t for t in legal_tokenize(exp_part, remove_stopwords=True) if len(t) >= 4},
        key=len,
        reverse=True,
    )[:14]
    if not tokens:
        return escaped
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b", re.IGNORECASE)
    return pattern.sub(r'<mark class="clause-hl">\1</mark>', escaped)


def load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def main() -> None:
    st.markdown(
        """
        <div class="clausetrace-header">
            <h1>⚖️ ClauseTrace | Contract Intelligence Assistant</h1>
            <p>Evidence-first legal clause retrieval over LegalBench-RAG-mini (CUAD) with exact character offsets, calibrated abstention, and reproducible train-only self-improvement.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ds = get_cached_dataset()
    counts = ds.summary_counts()

    # Sidebar controls
    with st.sidebar:
        st.subheader("Pipeline Configuration")
        preset_options = {
            "Final Calibrated Hybrid (configs/final.yaml)": "final",
            "Baseline TF-IDF (configs/baseline.yaml)": "baseline",
            "BM25 Lexical + Clause Chunking": "bm25",
            "Dense Semantic (all-MiniLM-L6-v2)": "dense",
            "Hybrid + Cross-Encoder Rerank": "hybrid_rerank",
        }
        selected_preset_label = st.selectbox(
            "Retrieval Strategy Preset",
            options=list(preset_options.keys()),
            index=0,
        )
        preset_key = preset_options[selected_preset_label]

        final_path = Path("configs/final.yaml")
        base_path = Path("configs/baseline.yaml")
        active_base_cfg = PipelineConfig.from_yaml(
            final_path if (preset_key != "baseline" and final_path.exists()) else base_path
        )
        cfg_dict = copy.deepcopy(active_base_cfg.model_dump())

        if preset_key == "baseline":
            cfg_dict = PipelineConfig.from_yaml(base_path).model_dump()
        elif preset_key == "bm25":
            cfg_dict["name"] = "interactive_bm25"
            cfg_dict["retrieval"]["strategy"] = "bm25"
            cfg_dict["rerank"]["enabled"] = False
        elif preset_key == "dense":
            cfg_dict["name"] = "interactive_dense"
            cfg_dict["retrieval"]["strategy"] = "dense"
            cfg_dict["rerank"]["enabled"] = False
        elif preset_key == "hybrid_rerank":
            cfg_dict["name"] = "interactive_hybrid_rerank"
            cfg_dict["retrieval"]["strategy"] = "hybrid"
            cfg_dict["rerank"]["enabled"] = True
            cfg_dict["rerank"]["alpha"] = 0.25
            cfg_dict["rerank"]["top_n"] = 10

        top_k = st.select_slider("Top-K Passages", options=[1, 3, 5, 10], value=5)
        enable_abstention = st.toggle("Enable Abstention Gate", value=bool(cfg_dict["abstention"]["enabled"]))
        cfg_dict["abstention"]["enabled"] = enable_abstention

        active_config = PipelineConfig.model_validate(cfg_dict)

        st.divider()
        st.caption(
            f"**Corpus:** {counts['indexed_contracts']} CUAD-mini contracts\n\n"
            f"**Answerable Queries:** {counts['answerable_total']} ({counts['answerable_train']} train / {counts['answerable_test']} test)\n\n"
            f"**Unanswerable Queries:** {counts['unanswerable_total']} ({counts['unanswerable_train']} train / {counts['unanswerable_test']} test)\n\n"
            f"**Config Fingerprint:** `{active_config.full_fingerprint()}`"
        )

    tab_search, tab_eval, tab_repro = st.tabs(
        [
            "🔍 A–C. Clause Search & Evidence Workspace",
            "📊 D. Evaluation & Self-Improvement Dashboard",
            "🧬 E. Reproducibility & Audit Panel",
        ]
    )

    # =========================================================================
    # TAB 1: SEARCH WORKSPACE, EVIDENCE RESULTS, AND ABSTENTION STATE
    # =========================================================================
    with tab_search:
        sample_queries = [
            "(Custom Query — type below)",
            "[Answerable] Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; What is the governing law for this contract?",
            "[Answerable] Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?",
            "[Answerable] Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is there a cap on liability under this contract?",
            "[Answerable] Consider the Endorsement Agreement between ABG-Shaq, LLC and Papa John's International, Inc.; Does this contract include an exclusivity agreement?",
            "[Unanswerable - Missing Clause] Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; Does this agreement allow sublicensing of the mobile games to third-party cloud gaming platforms?",
            "[Unanswerable - Unindexed Contract] Consider the Content Distribution Agreement between Netflix, Inc. and Sony Pictures Television Inc.; Does this agreement allow sublicensing to third-party streaming platforms?",
            "[Unanswerable - Missing Term] Consider the Gas Franchise Agreement between the Town of Vinton and Roanoke Gas Company; Is there a source code escrow or software maintenance obligation in this contract?",
        ]
        chosen_sample = st.selectbox("Quick-Load Sample Query (or select Custom Query):", options=sample_queries, index=1)
        default_q = (
            chosen_sample.split("] ", 1)[1]
            if chosen_sample.startswith("[")
            else "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; What is the governing law for this contract?"
        )

        col_q, col_btn = st.columns([5, 1])
        with col_q:
            query_text = st.text_input(
                "Natural-Language Contract Question:",
                value=default_q,
                placeholder="e.g., Consider the <Contract>; Does this agreement allow sublicensing to third-party platforms?",
            )
        with col_btn:
            st.write("")
            st.write("")
            run_search = st.button("Search Clauses", type="primary", use_container_width=True)

        if run_search or query_text:
            if not query_text.strip():
                st.warning("Please enter a natural-language question about a contract.")
            else:
                retriever = get_cached_retriever(json.dumps(active_config.model_dump(), sort_keys=True))
                resp = retriever.search(query_text, top_k=top_k)

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Retrieval Strategy", resp.retrieval_method)
                m2.metric("Decision Status", "Abstained" if resp.abstained else f"{len(resp.passages)} Passages")
                m3.metric("Top Retrieval Score", f"{resp.top_score:.4f}")
                m4.metric("Clause Relevance Score", f"{resp.top_clause_score:.4f}")
                m5.metric("Latency", f"{resp.latency_ms:.1f} ms")

                st.caption(
                    "Note: Retrieval scores are uncalibrated lexical/semantic similarity measures in [0, 1], not calibrated legal probabilities."
                )

                if resp.abstained:
                    st.markdown(
                        f"""
                        <div class="abstention-box">
                            <h4 style="margin-top:0;">⚠️ Abstention Triggered — Insufficient Supporting Evidence</h4>
                            <p><strong>Status Code:</strong> <code>{html.escape(resp.abstention.status)}</code></p>
                            <p>{html.escape(resp.abstention.reason)}</p>
                            <p style="font-size:0.86rem; margin-bottom:0;">
                                <strong>Important Distinction:</strong> This response means no passage in the indexed 29-contract corpus met the configured evidence thresholds; it is not a legal guarantee that an external or unindexed agreement lacks such a provision.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("Inspect Abstention Gate Diagnostics & Sub-Threshold Candidates"):
                        st.json(resp.abstention.diagnostics)
                        if resp.unfiltered_top_passages:
                            st.caption("Top sub-threshold candidate that was filtered out by the abstention gate:")
                            sub_p = resp.unfiltered_top_passages[0]
                            st.code(
                                f"{sub_p.doc_id} [{sub_p.char_start}:{sub_p.char_end}] (score={sub_p.retrieval_score:.4f})\n\n{sub_p.passage_text[:500]}",
                                language="text",
                            )
                else:
                    for p in resp.passages:
                        offset_ok = verify_evidence_offset(
                            ds.contracts, p.doc_id, p.char_start, p.char_end, p.passage_text
                        )
                        badge_html = (
                            '<span class="badge-verified">✓ Exact Offset Verified: doc[start:end] == passage</span>'
                            if offset_ok
                            else '<span style="color:red;">Offset Mismatch</span>'
                        )
                        sec_info = f" | Section: {html.escape(p.section_heading)}" if p.section_heading else ""
                        clause_info = f" | Clause: {html.escape(p.clause_number)}" if p.clause_number else ""
                        highlighted_html = highlight_query_terms(p.passage_text, query_text)

                        st.markdown(
                            f"""
                            <div class="evidence-card">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                                    <strong style="font-size:1.05rem; color:#0F1E36;">Rank #{p.rank} — {html.escape(p.doc_id)}</strong>
                                    {badge_html}
                                </div>
                                <div class="evidence-meta">
                                    Path: {html.escape(p.file_path)} | Character Offsets: [{p.char_start} : {p.char_end}] ({p.char_end - p.char_start} chars){sec_info}{clause_info}<br/>
                                    Retrieval Score: <b>{p.retrieval_score:.4f}</b> | Clause Score: <b>{p.scores.get('clause_score', 0.0):.4f}</b> | Doc Alignment: <b>{p.scores.get('doc_match_score', 0.0):.4f}</b> | Lexical: {p.scores.get('lexical_score', 0.0):.4f} | Dense: {p.scores.get('dense_score', 0.0):.4f}
                                </div>
                                <div class="evidence-passage">{highlighted_html}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        col_exp1, col_exp2 = st.columns(2)
                        with col_exp1:
                            with st.expander(f"Expand Surrounding Contract Context (±400 chars around [{p.char_start}:{p.char_end}])"):
                                full_doc = ds.contracts[p.doc_id].text
                                ctx_start = max(0, p.char_start - 400)
                                ctx_end = min(len(full_doc), p.char_end + 400)
                                before_txt = full_doc[ctx_start : p.char_start]
                                after_txt = full_doc[p.char_end : ctx_end]
                                st.caption(f"Surrounding window `[{ctx_start}:{ctx_end}]` in `{p.doc_id}` (exact retrieved span in bold brackets):")
                                st.text(f"...{before_txt}【{p.passage_text}】{after_txt}...")
                        with col_exp2:
                            if p.cross_references or p.referenced_context:
                                with st.expander(f"Cross-References Detected ({len(p.cross_references)})"):
                                    st.write("**Detected References:**", ", ".join(p.cross_references) if p.cross_references else "None")
                                    if p.referenced_context:
                                        st.info(
                                            f"**Heuristic Linked Context (distinct from retrieved evidence):**\n\n{p.referenced_context}"
                                        )

    # =========================================================================
    # TAB 2: EVALUATION & SELF-IMPROVEMENT DASHBOARD
    # =========================================================================
    with tab_eval:
        st.subheader("Measured Evaluation Results (LegalBench-RAG Character-Level Precision & Recall)")
        base_train = load_json_if_exists("outputs/evaluation/baseline_train_metrics.json")
        final_train = load_json_if_exists("outputs/evaluation/final_train_metrics.json")
        base_test = load_json_if_exists("outputs/evaluation/baseline_test_metrics.json")
        final_test = load_json_if_exists("outputs/evaluation/final_test_metrics.json")

        split_view = st.radio(
            "Select Evaluation Split to Inspect:",
            options=["Train Split (140 Answerable + 26 Unanswerable)", "Test Split (54 Answerable + 10 Unanswerable)", "Side-by-Side Train & Test Table"],
            horizontal=True,
        )

        if split_view.startswith("Train"):
            b_data = base_train["metrics"] if base_train else None
            f_data = final_train["metrics"] if final_train else None
            split_badge = "TRAIN SPLIT (Tuning & Self-Improvement)"
        elif split_view.startswith("Test"):
            b_data = base_test["metrics"] if base_test else None
            f_data = final_test["metrics"] if final_test else None
            split_badge = "TEST SPLIT (Held-Out Final Reporting Only)"
        else:
            b_data, f_data = None, None
            split_badge = "TRAIN & TEST COMPARISON"

        st.markdown(f'<span class="badge-split">{split_badge}</span>', unsafe_allow_html=True)

        if b_data and f_data:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(
                "Char Recall@5",
                f"{f_data['char_recall@5']:.4f}",
                delta=f"{f_data['char_recall@5'] - b_data['char_recall@5']:+.4f} vs baseline",
            )
            c2.metric(
                "Char Precision@5",
                f"{f_data['char_precision@5']:.4f}",
                delta=f"{f_data['char_precision@5'] - b_data['char_precision@5']:+.4f} vs baseline",
            )
            c3.metric(
                "Char Recall@1",
                f"{f_data['char_recall@1']:.4f}",
                delta=f"{f_data['char_recall@1'] - b_data['char_recall@1']:+.4f} vs baseline",
            )
            c4.metric(
                "Correct Abstention Rate",
                f"{f_data['correct_abstention_rate']:.4f}",
                delta=f"{f_data['correct_abstention_rate'] - b_data['correct_abstention_rate']:+.4f} vs baseline",
            )
            c5.metric(
                "False Abstention Rate",
                f"{f_data['false_abstention_rate']:.4f}",
                delta=f"{f_data['false_abstention_rate'] - b_data['false_abstention_rate']:+.4f} vs baseline",
                delta_color="inverse",
            )

            rows = []
            for k in [1, 3, 5, 10]:
                rows.append(
                    {
                        "Cutoff k": f"k = {k}",
                        "Baseline Precision@k": b_data.get(f"char_precision@{k}"),
                        "Final Precision@k": f_data.get(f"char_precision@{k}"),
                        "Baseline Recall@k": b_data.get(f"char_recall@{k}"),
                        "Final Recall@k": f_data.get(f"char_recall@{k}"),
                        "Baseline F1@k": b_data.get(f"char_f1@{k}"),
                        "Final F1@k": f_data.get(f"char_f1@{k}"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        elif split_view == "Side-by-Side Train & Test Table":
            from contract_intelligence.reporting import build_markdown_comparison_table

            st.markdown(build_markdown_comparison_table("outputs/evaluation"))
        else:
            st.info("Run `python -m contract_intelligence evaluate` to generate split metrics.")

        st.divider()
        st.subheader("Visual Comparison Charts")
        fig_cols = st.columns(2)
        pr_fig = Path("outputs/figures/precision_recall_by_k.png")
        abs_fig = Path("outputs/figures/abstention_comparison.png")
        traj_fig = Path("outputs/figures/improvement_trajectory.png")
        if pr_fig.exists():
            fig_cols[0].image(str(pr_fig), caption="Character-Level Recall@k and Precision@k")
        if abs_fig.exists():
            fig_cols[1].image(str(abs_fig), caption="Bidirectional Abstention Performance")
        if traj_fig.exists():
            st.image(str(traj_fig), caption="Self-Improvement Loop Trajectory on Train Split")

        st.divider()
        st.subheader("Self-Improvement Loop Experiment History (Train Split Only)")
        exp_csv = Path("experiments/experiment_log.csv")
        if exp_csv.exists():
            df_exp = pd.read_csv(exp_csv)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)

        st.subheader("Error Taxonomy & Query-Level Diagnostics")
        err_file = Path("outputs/evaluation/final_train_errors.csv")
        if err_file.exists():
            df_err = pd.read_csv(err_file)
            st.write(f"Remaining Train Failure Cases in Final Configuration ({len(df_err)} total across 166 train queries):")
            st.dataframe(df_err, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: REPRODUCIBILITY & AUDIT PANEL
    # =========================================================================
    with tab_repro:
        st.subheader("Reproducibility, Dataset Integrity & Test-Isolation Audit")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### Dataset & Split Counts")
            st.json(counts)
            val_rep = validate_ground_truth_offsets(ds)
            st.markdown("#### Ground-Truth Character Offset Validation")
            st.json(val_rep.to_dict())

        with col_r2:
            st.markdown("#### Test-Set Evaluation Audit Ledger")
            ledger = load_json_if_exists("outputs/evaluation/test_run_ledger.json")
            if ledger:
                st.json(ledger)
            else:
                st.info("Test set has been evaluated 0 times so far.")

            st.markdown("#### Active Pipeline Configuration (`YAML`)")
            st.json(active_config.model_dump())


if __name__ == "__main__":
    main()

