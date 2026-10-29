# ClauseTrace – Contract Intelligence Assistant

**ClauseTrace** is an evaluation-driven legal clause retrieval and evidence verification system built on the **LegalBench-RAG** benchmark (specifically the **194-query CUAD-mini** subset and its **29 referenced commercial agreements**, augmented with a frozen **36-query unanswerable dataset**).

---

## 1. Project Overview and Objective

Given a natural-language question about a contract — such as:
- *"Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; What is the governing law for this contract?"*
- *"Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?"*
- *"Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is there a cap on liability under this contract?"*

ClauseTrace retrieves the exact supporting passage(s) from the indexed contract corpus and returns:
- **Ranked supporting passages** with retrieval rank (`1..K`).
- **Original source document identifiers (`doc_id`) and relative file paths (`file_path`)**.
- **Exact character start and end offsets (`[char_start:char_end]`)** verified at runtime against the unmodified UTF-8 contract text (`original_document[char_start:char_end] == passage_text`).
- **Uncalibrated relevance scores** (`retrieval_score`, `clause_score`, `doc_match_score`, `lexical_score`, `dense_score`) explicitly distinguished from legal probabilities.
- **Honest abstention decisions** (`abstained: true`, status code, and diagnostic reason) when no passage in the indexed corpus sufficiently supports an answer.

ClauseTrace is strictly an **evidence-retrieval system**, not a generative legal advice chatbot. It never fabricates clauses or generates unsupported answers.

> **Evaluation Report:** Full benchmark tables, self-improvement iteration logs, charts, and 12 concrete failure analyses are documented in [`REPORT.md`](./REPORT.md).

---

## 2. Main Features

- **Exact LegalBench-RAG-mini Reproduction:** Deterministic selection of the 194 CUAD-mini queries (`mini(tests, n=194)`) and indexing of only the 29 referenced contracts (`contract_intelligence/dataset.py`).
- **Prescribed SHA-256 Train/Test Split:** Exact implementation of `assign_split(query)` partitioning both the 194 answerable queries (`140 train`, `54 test`) and the 36 frozen unanswerable queries (`26 train`, `10 test`).
- **Strict Character-Offset Verification:** Validates all `272/272` ground-truth answer spans against UTF-8 contract files before evaluation and enforces `original_document[char_start:char_end] == passage_text` on every retrieved chunk (`contract_intelligence/validation.py`).
- **Four Configurable Retrieval Strategies:**
  1. `tfidf` — Unigram TF-IDF cosine similarity baseline (`scikit-learn`).
  2. `bm25` — Okapi BM25 (`rank-bm25`, `k1=1.5, b=0.75`) with legal terminology expansion.
  3. `dense` — Local semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`) indexed with `FAISS` (`IndexFlatIP`).
  4. `hybrid` — Configurable linear or Reciprocal Rank Fusion (`RRF`) combining lexical (`BM25`) and dense (`MiniLM`) scores with hierarchical contract-to-clause routing.
- **Clause-Aware Structural Chunking:** Preserves `ARTICLE`/`SECTION` headers, numbered clauses (`1.1`, `(a)`), paragraph boundaries, and heuristic intra-contract cross-reference links (`Section 4.2`, `Article VII`) without altering original character offsets (`contract_intelligence/chunking.py`).
- **Calibrated Dual-Gate Abstention:** Evaluates overall score (`min_top_score`), target contract alignment (`min_doc_match_score`), and raw clause relevance (`min_clause_score`), calibrated exclusively on the 166 training queries (`contract_intelligence/abstention.py`).
- **Single-Command Train-Only Self-Improvement Loop:** Runs 7 logged post-baseline iterations on the training split with 5 automated guardrails (`python -m contract_intelligence.improve`).
- **Streamlit Web UI, Typer CLI, and FastAPI Service:** Interactive 4-tab Streamlit application (`app.py`) with a live Pipeline Configuration Sidebar, full-featured CLI (`contract_intelligence/cli.py`), and REST API (`contract_intelligence/api.py`).

---

## 3. Technology Stack

| Component | Package / Version | Purpose |
| :--- | :--- | :--- |
| **Language & Runtime** | Python `3.11+` (tested on `3.13.5`) | Core implementation |
| **Web Frontend** | `streamlit >= 1.35.0` | Interactive 4-tab legal-tech web UI (`app.py`) |
| **CLI Framework** | `typer >= 0.12.0`, `rich >= 13.7.0` | Command-line interface (`python -m contract_intelligence`) |
| **REST Backend** | `fastapi >= 0.110.0`, `uvicorn >= 0.29.0` | Optional JSON HTTP API (`contract_intelligence/api.py`) |
| **Lexical Retrieval** | `scikit-learn >= 1.4.0`, `rank-bm25 >= 0.2.2` | TF-IDF baseline and BM25Okapi lexical scoring |
| **Dense Retrieval** | `sentence-transformers >= 2.7.0`, `faiss-cpu >= 1.8.0` | Local `all-MiniLM-L6-v2` embeddings and vector search |
| **Reranking (Optional)** | `sentence-transformers` (`CrossEncoder`) | Optional second-stage `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Config & Validation** | `pydantic >= 2.7.0`, `pyyaml >= 6.0.1` | Schema validation and SHA-256 config fingerprinting |
| **Reporting & Charts** | `pandas >= 2.2.0`, `matplotlib >= 3.8.0` | Evaluation tables, CSV logs, and PNG figures |
| **Testing** | `pytest >= 8.2.0` | 20 automated unit and integration tests |

---

## 4. Actual Project Folder Structure

```text
.
├── app.py                              # Streamlit web application entry point
├── README.md                           # Beginner-friendly setup, usage, and architecture guide
├── REPORT.md                           # Complete evaluation report, tables, charts, and error analysis
├── pyproject.toml                      # Python package metadata and dependencies
├── .gitignore                          # Excludes raw downloaded corpus, cache, and secrets
├── .env.example                        # Template for optional environment variables (no secrets required)
├── configs/
│   ├── baseline.yaml                   # Initial fixed-window TF-IDF baseline configuration
│   └── final.yaml                      # Frozen configuration selected by the train-only improvement loop
├── contract_intelligence/
│   ├── __init__.py                     # Package version (`0.1.0`)
│   ├── __main__.py                     # Entry point for `python -m contract_intelligence`
│   ├── api.py                          # FastAPI REST service (`/health`, `/search`, `/validate`)
│   ├── cli.py                          # Typer CLI commands
│   ├── config.py                       # Pydantic config schemas and SHA-256 fingerprinting
│   ├── dataset.py                      # Exact `mini(194)`, `assign_split`, and UTF-8 contract loader
│   ├── validation.py                   # Ground-truth span and runtime offset verification
│   ├── chunking.py                     # `fixed` and `clause_aware` chunking with offset preservation
│   ├── indexing.py                     # TF-IDF, BM25, and cached Dense FAISS indexers
│   ├── retrieval.py                    # `ClauseTraceRetriever`, contract routing, and evidence assembly
│   ├── hybrid.py                       # Linear fusion, RRF, and Cross-Encoder reranking
│   ├── abstention.py                   # `single_threshold` & `dual_gate` abstention + train calibration
│   ├── evaluation.py                   # LegalBench-RAG character-level P@k, R@k, and test run ledger
│   ├── diagnostics.py                  # Automated failure classification taxonomy
│   ├── improve.py                      # Single-command train-only self-improvement loop
│   └── reporting.py                    # Matplotlib chart and Markdown comparison table generators
├── data/
│   ├── README.md                       # Data directory documentation
│   ├── unanswerable_queries.json       # Frozen 36-query unanswerable dataset (26 train, 10 test)
│   └── legalbench_rag/                 # Created by `download-data` (gitignored):
│       ├── benchmarks/cuad.json        # Full CUAD benchmark JSON (4,042 queries -> 194 selected)
│       └── corpus/cuad/*.txt           # 29 referenced contract text files (UTF-8)
├── experiments/
│   ├── README.md                       # Experiment log schema documentation
│   ├── experiment_log.jsonl            # JSONL log of Baseline (Iter 0) + 7 improvement iterations
│   ├── experiment_log.csv              # Tabular CSV summary of all logged iterations
│   ├── improvement_summary.json        # Summary of objective declaration, guardrails, and train results
│   └── iterations/                     # Per-iteration config YAML, train metrics, predictions, and errors
├── outputs/
│   ├── evaluation/                     # Saved train/test metrics JSON, predictions, error CSVs, and test ledger
│   └── figures/                        # Generated comparison charts (`precision_recall_by_k.png`, etc.)
└── tests/
    ├── test_dataset.py                 # Tests `mini(194)`, `assign_split`, and 29-contract scope
    ├── test_offsets.py                 # Tests 272/272 GT span validation and chunk offset invariants
    ├── test_retrieval.py               # Tests retrieval strategies, malformed queries, and config validation
    ├── test_abstention.py              # Tests single/dual-gate abstention and train-only calibration
    ├── test_evaluation.py              # Tests character-level overlap math and span deduplication
    ├── test_improvement.py             # Tests strict train/test isolation and guardrail enforcement
    └── test_ui_helpers.py              # Tests Streamlit sidebar config builder, strategies, Top-K, and UI helpers
```

---

## 5. Dataset Description and LegalBench-RAG CUAD-Mini Usage

ClauseTrace uses the CUAD benchmark from **LegalBench-RAG** (`benchmarks/cuad.json` and `corpus/cuad/*.txt`):

1. **Exact 194-Query Selection (`contract_intelligence/dataset.py`):**
   Applies the benchmark authors' exact deterministic stride selector (`mini(tests, n=194)` with `random.seed(42)`), selecting **194 answerable queries** (`272` total ground-truth answer spans) from the `4,042` total CUAD queries.
2. **Scoped 29-Contract Corpus:**
   Indexes **only the 29 unique contract `.txt` files** referenced by those 194 queries (`corpus/cuad/*.txt`), read using Python's default text mode (`open(path, "r", encoding="utf-8")`).
3. **Prescribed SHA-256 Train/Test Split:**
   Every query is assigned to `train` or `test` using the exact SHA-256 hash function:
   ```python
   def assign_split(query: str) -> str:
       h = int(hashlib.sha256(query.encode("utf-8")).hexdigest(), 16)
       return "test" if h % 10 < 3 else "train"
   ```
4. **Frozen Unanswerable Dataset (`data/unanswerable_queries.json`):**
   Contains **36 manually verified unanswerable queries** across 5 categories (`unindexed_contract`, `missing_clause_or_term`, `unsupported_jurisdiction`, `missing_date_or_renewal`, `unestablished_ip_or_sublicensing`), committed in Git commit `bfc37b6` before running the improvement loop and never edited afterward.

| Dataset Subset | Total Queries | Train Split | Test Split | Indexed Contracts |
| :--- | ---: | ---: | ---: | ---: |
| **Answerable (CUAD-mini)** | `194` | `140` (`199` GT spans) | `54` (`73` GT spans) | `29` |
| **Unanswerable (`data/unanswerable_queries.json`)** | `36` | `26` | `10` | `29` |
| **Combined Evaluation Benchmark** | **`230`** | **`166`** | **`64`** | **`29`** |

---

## 6. Installation and Environment Setup

### Prerequisites
- **Python:** `3.11`, `3.12`, or `3.13`
- **OS:** Windows, macOS, or Linux (runs 100% on CPU; no GPU or paid API key required)

### Step-by-Step Setup

```powershell
# 1. Clone the repository and navigate into the project directory
git clone https://github.com/saniyalakdawala/contract-intelligence-assistant.git
cd contract-intelligence-assistant

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS / Linux:
# source .venv/bin/activate

# 3. Install the package and all dependencies in editable mode
pip install -e .

# 4. Download the LegalBench-RAG CUAD benchmark and 29 referenced contract files (~1.2 MB)
python -m contract_intelligence download-data

# 5. Validate ground-truth character offsets (272/272 exact matches) and split counts
python -m contract_intelligence validate
```

---

## 7. How to Run the Web Application

 The Streamlit entry-point file is [`app.py`](./app.py). Start it from the project root:

```powershell
python -m streamlit run app.py
```

Or run headless on port `8501`:

```powershell
python -m streamlit run app.py --server.headless true --server.port 8501
```

Then open **http://localhost:8501** in your browser.

### What You Can Do in the Web UI
- **Pipeline Configuration Sidebar (Left Panel):**
  1. **Base Configuration Preset:** Load defaults from `Frozen Final Config (configs/final.yaml)` or `Baseline TF-IDF (configs/baseline.yaml)`, or click **Reset to Saved Preset Defaults**.
  2. **Retrieval Strategy:** Switch live between `hybrid`, `bm25`, `dense`, and `tfidf`.
  3. **Top-K Passages:** Slider (`1–10`) controlling the maximum number of returned passages.
  4. **Abstention Gate:** Toggle `Enable Abstention Gate` on or off.
  5. **Advanced Settings:** Adjust abstention mode (`dual_gate` vs `single_threshold`), thresholds (`min_top_score`, `min_clause_score`, `min_doc_match_score`), hybrid fusion (`linear` vs `rrf`), and optional Cross-Encoder reranking (`rerank.enabled`).
  6. **Active Configuration:** Shows the active strategy, Top-K, abstention status, SHA-256 config/index fingerprints, and whether any temporary UI overrides are active.
- **Four Workspace Tabs:**
  - **1. Clause Search:** Enter a question or choose a sample query, click **Search**, and inspect ranked clauses with verified `[char_start:char_end]` offsets, highlighted query terms, surrounding contract context (`±350` chars), detected cross-references, or abstention diagnostics.
  - **2. Evaluation:** View measured character-level `Precision@K`, `Recall@K`, and `F1@K` ($K = 1, 3, 5, 10$) and bidirectional abstention tables across the **Train Split** (`166` queries) and **Held-Out Test Split** (`64` queries).
  - **3. Experiments:** Inspect the baseline (`Iteration 0`) and all `7` post-baseline improvement iterations from `experiments/experiment_log.jsonl`.
  - **4. Active Configuration:** Compare the active in-memory configuration side-by-side against the saved YAML file on disk.

---

## 8. How to Use the CLI (With Working Examples)

The CLI is invoked via `python -m contract_intelligence <command>` (implemented in [`contract_intelligence/cli.py`](./contract_intelligence/cli.py)).

```powershell
# Show all available CLI commands and options
python -m contract_intelligence --help

# 1. Download LegalBench-RAG CUAD benchmark and 29 referenced contracts
python -m contract_intelligence download-data

# 2. Validate ground-truth character offsets and split counts
python -m contract_intelligence validate

# 3. Export the 194-query CUAD-mini manifest to outputs/evaluation/cuad_mini_manifest.json
python -m contract_intelligence prepare-mini

# 4. Build and warm the retrieval index cache for a configuration
python -m contract_intelligence build-index --config configs/final.yaml

# 5. Search for an answerable contract question (returns ranked passages + offsets)
python -m contract_intelligence search "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; What is the governing law for this contract?" --config configs/final.yaml --top-k 3

# 6. Search with a specific strategy override (tfidf, bm25, dense, or hybrid)
python -m contract_intelligence search "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?" --strategy bm25 --top-k 2

# 7. Search an unanswerable question (demonstrates honest abstention)
python -m contract_intelligence search "Consider the Content Distribution Agreement between Netflix, Inc. and Sony Pictures Television Inc.; Does this agreement allow sublicensing to third-party streaming platforms?" --config configs/final.yaml --top-k 3

# 8. Output machine-readable JSON with --json
python -m contract_intelligence search "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; What is the governing law for this contract?" --top-k 2 --json
```

---

## 9. How to Run Evaluation

Evaluation computes exact character-level `Precision@k`, `Recall@k`, and `F1@k` ($k \in \{1, 3, 5, 10\}$) along with bidirectional abstention metrics (`contract_intelligence/evaluation.py`).

```powershell
# Evaluate the baseline TF-IDF configuration on the train split (166 queries)
python -m contract_intelligence evaluate --split train --config configs/baseline.yaml --prefix baseline_train

# Evaluate the frozen final hybrid configuration on the train split (166 queries)
python -m contract_intelligence evaluate --split train --config configs/final.yaml --prefix final_train

# Regenerate comparison Markdown tables and Matplotlib charts in outputs/figures/
python -m contract_intelligence export-report
```

> **Note on Held-Out Test Split (`--split test`):** Every evaluation with `--split test` is automatically recorded in `outputs/evaluation/test_run_ledger.json`. The test split was evaluated **exactly 2 times** (once for `configs/baseline.yaml` and once for `configs/final.yaml`), and its saved outputs are already committed in `outputs/evaluation/baseline_test_metrics.json` and `outputs/evaluation/final_test_metrics.json`.

---

## 10. How to Run the Improvement Loop

The automated self-improvement loop ([`contract_intelligence/improve.py`](./contract_intelligence/improve.py)) operates **strictly on the train split** (`split="train"` is enforced with a runtime guard that raises `RuntimeError` if `test` is requested).

```powershell
# Run via module entrypoint:
python -m contract_intelligence.improve

# Or equivalently via the Typer CLI:
python -m contract_intelligence improve --baseline configs/baseline.yaml --final configs/final.yaml --iterations 7
```

This single command:
1. Evaluates `configs/baseline.yaml` on the 166 train queries (`Iteration 0`).
2. Runs 7 candidate iterations (`Iterations 1–7`) on the train split, diagnosing error categories (`ranked_too_low`, `right_document_wrong_clause`, `chunk_boundary_truncation`, `cross_reference_or_defined_term`, `failed_to_abstain`, `false_abstention`) after each run.
3. Enforces the primary objective and 5 guardrails, accepting Iterations `1, 2, 3, 5, 7` and rejecting Iterations `4` (oversized chunk precision bloat) and `6` (MS-MARCO cross-encoder domain mismatch).
4. Writes `experiments/experiment_log.jsonl`, `experiments/experiment_log.csv`, `experiments/improvement_summary.json`, and saves the winning configuration to `configs/final.yaml`.

---

## 11. Configuration Options and Their Actual Locations

All pipeline configurations are validated by [`PipelineConfig`](./contract_intelligence/config.py) and stored as YAML files in [`configs/baseline.yaml`](./configs/baseline.yaml) and [`configs/final.yaml`](./configs/final.yaml):

| Section & Key | Baseline (`configs/baseline.yaml`) | Final (`configs/final.yaml`) | Description |
| :--- | :--- | :--- | :--- |
| `chunking.strategy` | `fixed` | `clause_aware` | Fixed window vs. section/clause-aware splitting |
| `chunking.chunk_size` | `1000` | `750` | Maximum characters per chunk |
| `chunking.chunk_overlap` | `100` | `120` | Overlapping characters between adjacent chunks |
| `chunking.respect_section_boundaries` | `false` | `true` | Prefer splitting at detected section headings |
| `legal_context.include_doc_header` | `false` | `true` | Index contract title/preamble metadata alongside chunk |
| `legal_context.doc_routing_weight` | `0.0` | `0.85` | Weight for hierarchical contract-to-clause routing |
| `legal_context.enable_legal_query_expansion` | `false` | `true` | Deterministic legal synonym expansion |
| `legal_context.resolve_cross_references` | `false` | `true` | Extract and link intra-document section references |
| `retrieval.strategy` | `tfidf` | `hybrid` | `tfidf`, `bm25`, `dense`, or `hybrid` |
| `retrieval.top_k` | `10` | `10` | Default retrieval cutoff depth |
| `retrieval.candidate_pool_size` | `30` | `30` | Candidate pool size before top-K / reranking |
| `retrieval.fusion_method` | `linear` | `linear` | Hybrid score combination (`linear` or `rrf`) |
| `retrieval.lexical_weight` / `dense_weight` | `0.65` / `0.35` | `0.82` / `0.18` | Hybrid lexical (`BM25`) and dense (`MiniLM`) weights |
| `rerank.enabled` | `false` | `false` | Optional `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker |
| `abstention.enabled` | `true` | `true` | Enable or disable abstention gating |
| `abstention.mode` | `single_threshold` | `dual_gate` | `single_threshold` or 3-check `dual_gate` |
| `abstention.min_top_score` | `0.14` | `0.178` | Minimum combined retrieval score threshold |
| `abstention.min_clause_score` | `0.0` | `0.2092` | Minimum raw clause relevance threshold (`dual_gate`) |
| `abstention.min_doc_match_score` | `0.0` | `0.457` | Minimum target contract alignment threshold (`dual_gate`) |

---

## 12. Testing Instructions

Run the complete automated test suite (`20` tests across `7` test modules in `tests/`) using `pytest` or the CLI:

```powershell
# Run via pytest
python -m pytest -v

# Or via the CLI wrapper
python -m contract_intelligence test
```

The test suite verifies:
- `tests/test_dataset.py` — 194-query `mini()` selection, 29-contract scope, UTF-8 loading, and `assign_split()` determinism.
- `tests/test_offsets.py` — `272/272` ground-truth snippet offset validation, CRLF drift detection, and `fixed`/`clause_aware` chunk offset invariants.
- `tests/test_retrieval.py` — Pydantic config validation, empty/malformed query handling, and `tfidf`/`bm25` ranking consistency.
- `tests/test_abstention.py` — Single-threshold and dual-gate abstention decisions and train-only threshold calibration.
- `tests/test_evaluation.py` — Character-level `Precision@k` and `Recall@k` math and overlapping span deduplication (`sort_and_merge_spans`).
- `tests/test_improvement.py` — Strict train-only isolation (`RuntimeError` on `split="test"`) and guardrail acceptance/rejection logic.
- `tests/test_ui_helpers.py` — Streamlit sidebar configuration builder, all 4 retrieval strategies × `Top-K` in `{1, 3, 5}`, abstention toggle ON/OFF, and HTML-safe query term highlighting.

---

## 13. Final Evaluation Results (Summary)

Below is the headline comparison between `configs/baseline.yaml` and `configs/final.yaml` on both the **Train Split (`166` queries)** and **Held-Out Test Split (`64` queries)**. Full analysis, charts, and 12 concrete failure cases are in [`REPORT.md`](./REPORT.md).

| Metric | Baseline (Train) | Final (Train) | Baseline (Test) | Final (Test) |
| :--- | ---: | ---: | ---: | ---: |
| **Answerable / Unanswerable Queries** | `140` / `26` | `140` / `26` | `54` / `10` | `54` / `10` |
| **Primary Objective Score** | `0.1298` | **`0.6177`** | `0.1057` | **`0.5406`** |
| **Char Precision@1** | `0.0087` | **`0.3617`** | `0.0000` | **`0.2534`** |
| **Char Recall@1** | `0.0147` | **`0.5491`** | `0.0000` | **`0.4572`** |
| **Char Precision@3** | `0.0155` | **`0.1830`** | `0.0100` | **`0.1506`** |
| **Char Recall@3** | `0.1044` | **`0.7325`** | `0.0932` | **`0.6625`** |
| **Char Precision@5** | `0.0227` | **`0.1228`** | `0.0161` | **`0.1073`** |
| **Char Recall@5** | `0.2276` | **`0.8013`** | `0.1785` | **`0.7080`** |
| **Char F1@5** | `0.0397` | **`0.1992`** | `0.0258` | **`0.1676`** |
| **Char Precision@10** | `0.0196` | **`0.0720`** | `0.0142` | **`0.0590`** |
| **Char Recall@10** | `0.3893` | **`0.8784`** | `0.2586` | **`0.7545`** |
| **Correct Abstention Rate (Unanswerable)** | `0.0385` (`1/26`) | **`0.8846`** (`23/26`) | `0.1000` (`1/10`) | **`0.9000`** (`9/10`) |
| **False Abstention Rate (Answerable)** | `0.0000` (`0/140`) | **`0.0143`** (`2/140`) | `0.0000` (`0/54`) | **`0.0556`** (`3/54`) |
| **Answer Coverage (Answerable)** | `1.0000` (`140/140`) | **`0.9857`** (`138/140`) | `1.0000` (`54/54`) | **`0.9444`** (`51/54`) |
| **Avg Latency (ms/query)** | `8.03 ms` | **`36.04 ms`** | `7.64 ms` | **`40.39 ms`** |

---

## 14. System Architecture and Retrieval Pipeline

```text
Raw Corpus (29 UTF-8 Contracts) + Benchmark Queries (194 Answerable + 36 Unanswerable)
        │
        ▼
1. Offset Validation & Split Assignment (`dataset.py`, `validation.py`)
   - Verifies doc[start:end] == snippet["answer"] (272/272 exact matches)
   - Applies prescribed SHA-256 assign_split(query) -> 166 Train / 64 Test
        │
        ▼
2. Structural Chunking & Legal Context (`chunking.py`)
   - Splits into clause_aware (750 chars, 120 overlap) or fixed (1000/100) chunks
   - Preserves exact [char_start, char_end) into original UTF-8 text
   - Extracts section headings and heuristic intra-contract cross-references
        │
        ▼
3. Multi-Strategy Indexing & Hierarchical Routing (`indexing.py`, `retrieval.py`, `hybrid.py`)
   - Decomposes "Consider the <Contract>; <Question>" into contract routing + clause query
   - Computes Lexical (TF-IDF / BM25Okapi) and Dense (all-MiniLM-L6-v2 + FAISS) scores
   - Fuses scores (Linear 0.82 BM25 + 0.18 Dense, or RRF) modulated by contract alignment
        │
        ▼
4. Calibrated Dual-Gate Abstention (`abstention.py`)
   - Gate 1: Combined score >= min_top_score (0.1780)
   - Gate 2: Contract alignment >= min_doc_match_score (0.4570)
   - Gate 3: Raw clause & concept relevance >= min_clause_score (0.2092)
```

---

## 15. Abstention Mechanism and Evidence Retrieval

- **Evidence Object Integrity:** Every returned [`EvidencePassage`](./contract_intelligence/retrieval.py) contains `doc_id`, `file_path`, `char_start`, `char_end`, `passage_text`, `retrieval_method`, `rank`, `retrieval_score`, component `scores`, `section_heading`, `clause_number`, `cross_references`, and `referenced_context`. Any linked cross-reference preview is stored in `referenced_context` rather than concatenated into `passage_text`, ensuring `original_document[char_start:char_end] == passage_text` always holds.
- **Similarity Scores vs. Probabilities:** Retrieval scores are uncalibrated similarity measures in `[0, 1]` used for ranking and threshold gating; they are never presented as legal probabilities.
- **Abstention Gate Behavior:**
  - When **Enabled (`dual_gate`)**: Returns `passages = []` with status `abstained_low_overall_score`, `abstained_unmatched_contract`, or `abstained_missing_clause_evidence` if any gate fails.
  - When **Disabled (`abstention.enabled = False`)**: Relevance thresholds are bypassed so top-$K$ candidate passages are returned. Only the input-validation check (`len(query.strip()) < 3` or no alphanumeric characters, returning `abstained_invalid_query`) remains active to prevent empty or punctuation-only queries from failing.

---

## 16. Assumptions and Limitations

1. **Character-Level Span Deduplication & Macro-Averaging:** Following the LegalBench-RAG paper, overlapping retrieved spans from the same contract are merged via `sort_and_merge_spans` before computing character precision and recall, and per-query `Precision@k`, `Recall@k`, and `F1@k` are macro-averaged across all answerable queries in the split.
2. **Heuristic Cross-Reference Linking:** Intra-contract references (e.g., `Section 4.2`, `Article VII`) are matched heuristically to section headings in the same document. Multi-hop cross-reference graph traversal (needed when an exception clause delegates its entire substance to another numbered section, as in `CUAD-MINI-175`) is not yet integrated into the abstention scorer.
3. **General-Domain Cross-Encoder Mismatch:** The optional `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker is disabled in `configs/final.yaml` because testing on the training split (Iteration 6) showed that web-search pretraining downweighted formal contract phrasing (`Recall@5` dropped from `0.8072` to `0.7548`).
4. **Plain-Text Input Scope:** LegalBench-RAG provides pre-extracted `.txt` files. Deploying on scanned or multi-column PDF agreements would require layout-aware OCR/PDF parsing that preserves character and bounding-box offsets.

---

## 17. AI Tool Usage and External Model/API Usage

- **AI Coding Tool Usage:** Google Antigravity (IDE AI coding assistant) was used to assist with modular Python package scaffolding (`contract_intelligence/`), Pydantic configuration schemas, Streamlit custom CSS styling (`app.py`), Matplotlib plotting scripts (`reporting.py`), and pytest fixtures (`tests/`). All retrieval algorithms, character-offset validations, unanswerable queries, improvement-loop guardrails, and benchmark evaluations were verified against actual code execution.
- **External Models & Network Usage:**
  - **No Paid or Generative LLM APIs:** ClauseTrace uses **zero** hosted LLM APIs (no OpenAI, Anthropic, or Gemini API calls) and requires no API keys.
  - **Local Open-Weights Encoders:** Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2` (and optional reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2`), downloaded once from Hugging Face Hub on first use and cached locally on disk (`outputs/cache/`). After initial download of the dataset and encoder weights, all indexing, search, evaluation, and self-improvement runs execute 100% locally offline on CPU.
