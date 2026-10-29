# ClauseTrace | Contract Intelligence Assistant

**ClauseTrace** is an evaluation-driven legal clause retrieval and evidence verification system built on the **LegalBench-RAG** benchmark (specifically the 194-query **CUAD-mini** subset and its 29 referenced commercial agreements).

Given a natural-language question about a contract (for example, *"Does this agreement allow sublicensing to third-party platforms?"*, *"When does the exclusivity period end?"*, or *"Who owns the intellectual property in derivative works?"*), ClauseTrace returns:
- Ranked supporting contract passages anchored to **exact character start and end offsets** (`original_document[start:end] == returned_passage`).
- Original document identifiers, relative file paths, and detected section/clause metadata.
- Separate lexical, semantic, clause-relevance, and document-alignment scores (explicitly distinguished from calibrated probabilities).
- An **honest abstention decision** with transparent diagnostic codes when no passage in the indexed corpus sufficiently supports an answer.

ClauseTrace is strictly an evidence-retrieval system — it never fabricates clauses or generates unsupported legal advice.

> **Note:** All quantitative benchmark results, baseline-vs-final tables, self-improvement iteration logs, and concrete failure analyses are documented in [`REPORT.md`](./REPORT.md).

---

## 1. Architecture and Data Flow

```
                       +---------------------------------------------+
                       |  Raw Corpus & Benchmark (LegalBench-RAG)    |
                       |  - benchmarks/cuad.json (4,042 queries)     |
                       |  - mini(tests, n=194) -> 194 CUAD queries   |
                       |  - 29 referenced contracts (UTF-8 text)     |
                       |  - data/unanswerable_queries.json (36 set)  |
                       +----------------------+----------------------+
                                              |
                   Prescribed SHA-256 assign_split(query)
                   (Train: 140 ans + 26 unans | Test: 54 ans + 10 unans)
                                              |
                       +----------------------v----------------------+
                       |  1. Ingestion & Offset Validation           |
                       |     - UTF-8 default text mode reading       |
                       |     - Verifies doc[start:end] == answer     |
                       +----------------------+----------------------+
                                              |
                       +----------------------v----------------------+
                       |  2. Chunking & Legal Context Enrichment     |
                       |     - Fixed-window or Clause-Aware chunking |
                       |     - Exact original char offset tracking   |
                       |     - Document preamble/header extraction   |
                       |     - Heuristic cross-reference detection   |
                       +----------------------+----------------------+
                                              |
                       +----------------------v----------------------+
                       |  3. Multi-Strategy Indexing & Retrieval     |
                       |     - Baseline TF-IDF cosine similarity     |
                       |     - BM25Okapi with legal tokenization     |
                       |     - Dense FAISS (all-MiniLM-L6-v2 cached) |
                       |     - Hybrid linear / RRF fusion            |
                       |     - Hierarchical contract-to-clause route |
                       |     - Optional Cross-Encoder reranking      |
                       +----------------------+----------------------+
                                              |
                       +----------------------v----------------------+
                       |  4. Honest Abstention Gate                  |
                       |     - Single-threshold or Dual-Gate mode    |
                       |     - Verifies both target contract match   |
                       |       and raw-passage clause evidence       |
                       +----------------------+----------------------+
                                              |
             +--------------------------------+--------------------------------+
             |                                |                                |
+------------v-------------+    +-------------v--------------+   +-------------v--------------+
| Train Self-Improvement   |    | Evaluation & Reporting     |   | User Interfaces            |
| (improve.py)             |    | (evaluation.py)            |   | - Streamlit Web UI (app.py)|
| - Train split ONLY       |    | - LegalBench-RAG char P@k  |   | - Typer CLI (cli.py)       |
| - 7 logged iterations    |    |   and R@k at k=1,3,5,10    |   | - FastAPI REST (api.py)    |
| - 5 guardrails enforced  |    | - Bidirectional abstention |   +----------------------------+
+--------------------------+    +----------------------------+
```

### Repository Layout

```text
.
├── app.py                              # Streamlit web application (ClauseTrace UI)
├── README.md                           # Setup, architecture, and reproducibility guide
├── REPORT.md                           # Full evaluation report, tables, charts, and error analysis
├── pyproject.toml                      # Python packaging and dependencies
├── .gitignore                          # Excludes raw corpora, cached embeddings, and secrets
├── .env.example                        # Example environment variable configuration
├── configs/
│   ├── baseline.yaml                   # Initial fixed-window TF-IDF baseline configuration
│   └── final.yaml                      # Frozen configuration selected by the train-only improvement loop
├── contract_intelligence/
│   ├── __init__.py
│   ├── __main__.py                     # Entrypoint for `python -m contract_intelligence`
│   ├── api.py                          # FastAPI backend service
│   ├── cli.py                          # Typer CLI commands
│   ├── config.py                       # Pydantic config schemas and SHA-256 fingerprinting
│   ├── dataset.py                      # Exact mini(194) selector, assign_split, and UTF-8 loader
│   ├── validation.py                   # Ground-truth span and runtime offset verification
│   ├── chunking.py                     # Fixed and clause-aware chunking with offset preservation
│   ├── indexing.py                     # TF-IDF, BM25, and cached Dense FAISS indexers
│   ├── retrieval.py                    # Core retrieval, contract routing, and evidence assembly
│   ├── hybrid.py                       # RRF, linear score fusion, and Cross-Encoder reranking
│   ├── abstention.py                   # Single-threshold & dual-gate abstention + train calibration
│   ├── evaluation.py                   # LegalBench-RAG character-level P@k, R@k, and test ledger
│   ├── diagnostics.py                  # Automated failure classification taxonomy
│   ├── improve.py                      # Single-command train-only self-improvement loop
│   └── reporting.py                    # Matplotlib chart and Markdown table generators
├── data/
│   ├── README.md
│   └── unanswerable_queries.json       # Committed 36-query unanswerable dataset (26 train, 10 test)
├── experiments/
│   ├── README.md
│   ├── experiment_log.jsonl            # Machine-readable JSONL log of all improvement iterations
│   ├── experiment_log.csv              # Tabular CSV summary of all improvement iterations
│   └── iterations/                     # Per-iteration configs, predictions, and diagnostics
├── outputs/
│   ├── evaluation/                     # Saved train/test metrics JSON, predictions, and error CSVs
│   └── figures/                        # Generated comparison and trajectory charts (.png)
└── tests/
    ├── test_dataset.py                 # Tests mini(194), assign_split, and UTF-8 loading
    ├── test_offsets.py                 # Tests GT span validation and chunk offset invariants
    ├── test_retrieval.py               # Tests strategies, malformed queries, and config validation
    ├── test_abstention.py              # Tests single/dual-gate abstention and train calibration
    ├── test_evaluation.py              # Tests character-level overlap math and span merging
    └── test_improvement.py             # Tests train/test isolation and guardrail enforcement
```

---

## 2. Environment Setup & Installation

### Python Version & Hardware Assumptions
- **Python:** Python `3.11+` (tested on Python `3.13.5` on Windows and compatible with Linux/macOS).
- **Hardware:** Runs 100% locally on CPU. No GPU or paid external API key is required.

### Install Dependencies

```bash
# Optional: create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install package and dependencies in editable mode
pip install -e .
```

---

## 3. Dataset Download & Validation

ClauseTrace uses the open **LegalBench-RAG** dataset mirror (`huggingface.co/datasets/awinml/legalbench-rag`), specifically `benchmarks/cuad.json` and the 29 contract files under `corpus/cuad/` referenced by the authors' 194-query `mini()` function.

### Required Data Paths
- Benchmark JSON: `data/legalbench_rag/benchmarks/cuad.json`
- Referenced Corpus Files: `data/legalbench_rag/corpus/cuad/*.txt` (29 files)
- Committed Unanswerable Queries: `data/unanswerable_queries.json` (36 queries: 26 train, 10 test)

### Download and Validate Data via CLI

```bash
# 1. Download benchmarks/cuad.json and the 29 referenced CUAD-mini contracts (~1.2 MB total)
python -m contract_intelligence download-data

# 2. Validate all 272 ground-truth character spans against UTF-8 contract texts
python -m contract_intelligence validate

# 3. Export the 194-query mini manifest and split breakdown
python -m contract_intelligence prepare-mini
```

---

## 4. How to Run

### A. Build or Warm the Retrieval Index

```bash
python -m contract_intelligence build-index --config configs/final.yaml
```

### B. Search from the Command Line (CLI)

```bash
# Search using the frozen final configuration
python -m contract_intelligence search "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; What is the governing law for this contract?" --top-k 5

# Search an unanswerable question to observe honest abstention
python -m contract_intelligence search "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; Does this agreement allow sublicensing of the mobile games to third-party cloud gaming platforms?" --top-k 5

# Output raw machine-readable JSON
python -m contract_intelligence search "Consider the Sponsorship Agreement between Jacksonville Jaguars, LLC and The ARC Group, Inc.; What licenses are granted under this contract?" --top-k 3 --json
```

### C. Start the Web Application (Streamlit)

```bash
streamlit run app.py
```

### D. Optional FastAPI Backend Server

```bash
uvicorn contract_intelligence.api:app --host 127.0.0.1 --port 8000
```

### E. Run Evaluation, Self-Improvement Loop, and Automated Tests

```bash
# Run the train-only self-improvement loop (establishes baseline + 7 logged iterations on train split)
python -m contract_intelligence.improve
# (or equivalently: python -m contract_intelligence improve)

# Evaluate baseline configuration on train split
python -m contract_intelligence evaluate --split train --config configs/baseline.yaml

# Evaluate frozen final configuration on train split
python -m contract_intelligence evaluate --split train --config configs/final.yaml

# Evaluate frozen final configuration on held-out test split (for final reporting only)
python -m contract_intelligence evaluate --split test --config configs/final.yaml

# Generate evaluation comparison tables and charts
python -m contract_intelligence export-report

# Run automated unit and integration tests (does NOT touch the test split)
python -m pytest -q
```

---

## 5. Exact Procedure for Reproducing Reported Metrics

To reproduce every number, table, and chart in [`REPORT.md`](./REPORT.md) from a fresh clone:

1. **Download & Validate Data (~5–10 seconds):**
   ```bash
   python -m contract_intelligence download-data
   python -m contract_intelligence validate
   ```
2. **Run the Train-Only Self-Improvement Loop (~25–60 seconds on CPU; ~15 seconds when embedding cache is warm):**
   ```bash
   python -m contract_intelligence.improve
   ```
   This command:
   - Evaluates `configs/baseline.yaml` on the 166 train queries (`140 answerable + 26 unanswerable`).
   - Runs 7 post-baseline candidate iterations on the train split only, checking the primary objective and 5 guardrails at each step.
   - Writes `experiments/experiment_log.jsonl`, `experiments/experiment_log.csv`, `outputs/evaluation/baseline_train_metrics.json`, `outputs/evaluation/final_train_metrics.json`, and freezes `configs/final.yaml`.
3. **Run Held-Out Test Split Evaluation (~5–15 seconds on CPU):**
   ```bash
   python -m contract_intelligence evaluate --split test --config configs/baseline.yaml --prefix baseline_test
   python -m contract_intelligence evaluate --split test --config configs/final.yaml --prefix final_test
   ```
4. **Regenerate Charts and Markdown Comparison Tables (~2 seconds):**
   ```bash
   python -m contract_intelligence export-report
   ```

---

## 6. Assumptions and Design Decisions

1. **Exact Character-Level Span Deduplication:** The LegalBench-RAG paper defines character-level Precision@$k$ and Recall@$k$ over the union of retrieved character spans at cutoff $k$. When chunk overlap is positive (`chunk_overlap > 0`), two retrieved chunks from the same contract can overlap. We apply the benchmark authors' exact `sort_and_merge_spans` algorithm per document before computing character intersections so overlapping characters are never double-counted and Recall@$k$ is strictly bounded in $[0, 1]$.
2. **Abstention Scoring Semantics:** When the system abstains on an answerable benchmark query (`false_abstention`), it returns an empty passage list `[]`. Following standard retrieval evaluation, an empty retrieval list receives `Precision@k = 0.0` and `Recall@k = 0.0` for that query, directly penalizing false abstentions in both the retrieval metrics and our primary objective.
3. **Query Structure Decomposition:** Every query in `LegalBench-RAG` CUAD follows the structure `Consider the <Agreement Description>; <Clause Question>`. Our pipeline supports general natural-language queries with or without a semicolon separator: when a semicolon is present, the prefix informs contract-header alignment (`doc_match_score`) while the question clause drives passage scoring (`clause_score`); when no semicolon is present, the full query is used for both.
4. **Cross-Reference Resolution Scope:** When `resolve_cross_references: true` is enabled, ClauseTrace detects intra-document section/article/clause references (e.g., `Section 4.2`, `Article III`) via regular expressions and links them heuristically to matching section headings within the same agreement. Any linked preview is stored in `referenced_context` rather than concatenated into `passage_text`, preserving the exact `original_document[start:end] == passage_text` invariant.
5. **Local Reproducibility over Paid APIs:** All lexical (`scikit-learn` TF-IDF, `rank-bm25`), dense (`sentence-transformers/all-MiniLM-L6-v2` + `faiss-cpu`), and cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) components run 100% locally on CPU with deterministic disk caching. No paid LLM API is required to reproduce any experiment.

---

## 7. AI Coding Tool Usage

- **Tools Used:** Google Antigravity (AI pair-programming assistant in the IDE).
- **How Used:** Assisted with scaffolding modular Python package structure (`contract_intelligence/`), drafting Pydantic configuration schemas, formatting Streamlit CSS panels and Matplotlib plotting routines, and structuring unit test fixtures (`tests/`).
- **Verification:** All retrieval mathematics, offset validation checks, unanswerable query verification against the 29 CUAD-mini contracts, self-improvement loop guardrails, and evaluation metrics were executed, verified, and audited against real benchmark runs.

