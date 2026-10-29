# Experiments Directory (`experiments/`)

This directory stores the machine-readable logs and snapshots produced by the train-only self-improvement loop (`python -m contract_intelligence.improve`):

- `experiment_log.jsonl`: Full structured JSON Lines log of the baseline and every subsequent improvement iteration (accepted and rejected), including parent/candidate configs, failure diagnosis counts, guardrail checks, and before/after train metrics.
- `experiment_log.csv`: Tabular CSV summary of all iterations for quick inspection and plotting.
- `iterations/`: Per-iteration configuration snapshots, train predictions, and diagnostic summaries.

