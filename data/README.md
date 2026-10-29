# Data Directory (`data/`)

This directory contains version-controlled evaluation metadata and instructions for downloading the raw **LegalBench-RAG** (CUAD) benchmark and corpus.

## Version-Controlled Files

- `unanswerable_queries.json`: The frozen set of **36 unanswerable queries** (`26 train`, `10 test` under the prescribed SHA-256 `assign_split` function), covering five realistic failure categories:
  1. `unindexed_contract` (8 queries)
  2. `missing_clause_or_term` (8 queries)
  3. `unsupported_jurisdiction` (6 queries)
  4. `missing_date_or_renewal` (6 queries)
  5. `unestablished_ip_or_sublicensing` (8 queries)

## Downloaded Raw Benchmark & Corpus (`data/legalbench_rag/`)

Raw LegalBench-RAG files are downloaded into `data/legalbench_rag/` (ignored by Git via `.gitignore`):
- `data/legalbench_rag/benchmarks/cuad.json`
- `data/legalbench_rag/corpus/cuad/*.txt` (the 29 contracts referenced by the 194-query `LegalBench-RAG-mini` CUAD subset)

To download or verify the dataset automatically, run:

```bash
python -m contract_intelligence download-data
python -m contract_intelligence validate-data
```
