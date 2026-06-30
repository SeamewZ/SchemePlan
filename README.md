# SchemaPlan-RAG

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Artifact](https://img.shields.io/badge/Artifact-Anonymous-555555)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Metrics%20%2B%20Fresh%20Runs-2E7D32)
![JSON-LD](https://img.shields.io/badge/Output-schema.org%20JSON--LD-0B7285)
![Dependencies](https://img.shields.io/badge/Core%20Dependencies-Stdlib%20Only-6A1B9A)

Anonymous review artifact for **SchemaPlan-RAG**, a planned, evidence-verified framework for generating schema.org JSON-LD from web pages.

SchemaPlan-RAG treats JSON-LD generation as a sequence of auditable field-local decisions rather than a single final-record prompt. The system constructs a DOM Evidence Graph, plans a page-specific schema contract, retrieves evidence for each property, generates evidence-cited slot candidates, verifies schema and evidence support, repairs rejected candidates under verifier feedback, and composes admitted fields into nested JSON-LD.

This repository is prepared for double-blind review. It omits author identities, private paths, API keys, paper build files, historical caches, and exploratory methods that are not part of the reported system.

## Highlights

- **Method:** planned, property-wise, evidence-verified schema.org JSON-LD generation.
- **Benchmark:** WebSchema-JSONLD Bench-3452 over SWDE, public non-SWDE examples, and WDC-PAVE.
- **Reproduction:** metric recomputation works without API keys or third-party packages.
- **Fresh runs:** API-backed SchemaPlan-RAG and Direct DeepSeek scripts are included.
- **Claim control:** archived predictions and reference tables are separated from fresh-run outputs.

## Repository Map

```text
src/schemarag_core.py                 Core SchemaPlan-RAG implementation and evaluator
scripts/recompute_metrics.py          Recompute main paper metrics from archived predictions
scripts/run_schemaplan.py             Fresh SchemaPlan-RAG runs with optional API calls
scripts/run_direct_baseline.py        Fresh direct DeepSeek JSON-LD baseline
tests/smoke_test.py                   Minimal offline import/run test
data/processed/*.jsonl.gz             WebSchema-JSONLD Bench-3452 splits
data/schema/schema_index.json         Cached schema.org property index
data/schema/swde_train_templates.json SWDE train-side template index
results/predictions/                 Archived predictions for reproduction
results/reference/                   Reference CSV tables reported in the paper
docs/ARTIFACT.md                      Short artifact notes
```

## Benchmark

`WebSchema-JSONLD Bench-3452` combines three public-data splits under a unified JSON-LD extraction protocol.

| Split | Examples | Source |
| --- | ---: | --- |
| SWDE-1600 | 1,600 | SWDE pages mapped to schema.org types |
| Legacy non-SWDE-434 | 434 | public web/schema.org examples |
| WDC-PAVE-1418 | 1,418 | WDC-PAVE product pages projected to Product fields |

The processed JSONL files contain the page text or HTML needed by the extractor and normalized gold JSON-LD records for evaluation.

## Quick Start

Core reproduction uses only the Python standard library.

```bash
python3 --version
python3 tests/smoke_test.py
```

Expected output:

```json
{
  "status": "ok"
}
```

## Reproduce Main Results

Recompute the main paper table from the released evaluator and archived predictions:

```bash
python3 scripts/recompute_metrics.py --out-dir runs/recomputed_metrics
```

Generated files:

```text
runs/recomputed_metrics/main_metrics.csv
runs/recomputed_metrics/main_metrics.md
runs/recomputed_metrics/main_metrics.json
```

Expected pooled results:

| Method | Precision | Recall | F1 | Validity | Unsupported / Predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regex baseline | 0.578 | 0.368 | 0.450 | 0.331 | 0/13613 |
| Direct DeepSeek | 0.548 | 0.600 | 0.573 | 0.999 | 589/23587 |
| Direct+Verifier | 0.644 | 0.468 | 0.542 | 1.000 | 0/15559 |
| SchemaPlan-RAG | 0.950 | 0.924 | 0.937 | 1.000 | 16/20784 |

The same command also writes the reference ablation and no-template diagnostic CSVs into the output directory.

## Fresh SchemaPlan-RAG Runs

Set a DeepSeek-compatible key for API-backed runs:

```bash
export SCHEMARAG_LLM_PROVIDER=deepseek
read -s DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
export SCHEMARAG_DEEPSEEK_MODEL=deepseek-v4-flash
export SCHEMARAG_DEEPSEEK_THINKING=disabled
```

Run a small fresh sample:

```bash
python3 scripts/run_schemaplan.py \
  --dataset data/processed/WDC-PAVE-1418.jsonl.gz \
  --limit 10 \
  --workers 4 \
  --out-dir runs/schemaplan_wdc10
```

Run the deterministic diagnostic path without API calls:

```bash
python3 scripts/run_schemaplan.py \
  --dataset data/processed/WDC-PAVE-1418.jsonl.gz \
  --limit 10 \
  --no-require-llm \
  --out-dir runs/schemaplan_wdc10_offline
```

For a full fresh run, omit `--limit`. Full runs invoke the planner, slot-level candidate generator, and verifier-guided repair branch, so runtime depends on API latency, rate limits, and retries.

## Fresh Direct Baseline

```bash
read -s DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY

python3 scripts/run_direct_baseline.py \
  --dataset data/processed/WDC-PAVE-1418.jsonl.gz \
  --limit 10 \
  --workers 4 \
  --out-dir runs/direct_wdc10
```

If no key is set, the script exits cleanly and writes a skipped status file.

## Metrics

The evaluator flattens nested JSON-LD into dot paths and computes micro precision, recall, and F1 over normalized field-value pairs.

| Metric | Meaning |
| --- | --- |
| Precision / Recall / F1 | Field-value matching after normalization |
| Site-Macro F1 | Average F1 across sites |
| Validity | Parseable JSON-LD with target type and admissible schema.org top-level properties |
| Unsupported Fields | Retained predicted fields whose value is not supported by page text |

## Double-Blind and Review Notes

- This artifact is anonymous and contains no author or institution metadata.
- Local run outputs should stay under `runs/`, which is ignored by Git.
- Do not commit `.env` files, API keys, provider logs, or raw local caches.
- Processed benchmark files are included to make review-time reproduction self-contained.

## Minimal Verification Checklist

```bash
python3 tests/smoke_test.py
python3 scripts/recompute_metrics.py --out-dir runs/recomputed_metrics
```

The second command should reproduce the pooled SchemaPlan-RAG row:

```text
precision=0.950, recall=0.924, f1=0.937, validity=1.000, unsupported_fields=16
```
