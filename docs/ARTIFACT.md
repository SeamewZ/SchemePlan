# Artifact Notes

This artifact supports two reproduction modes.

1. Metric recomputation from archived predictions:

```bash
python3 scripts/recompute_metrics.py --out-dir runs/recomputed_metrics
```

2. Fresh extraction runs:

```bash
python3 scripts/run_schemaplan.py --limit 10 --no-require-llm
```

API-backed fresh runs require `DEEPSEEK_API_KEY` and use the same LLM-facing roles as the paper: planner, slot-level candidate generation, and verifier-guided repair.

The code and processed data are anonymized for double-blind review. Local machine paths from the original data-preparation environment are removed from the released JSONL files.
