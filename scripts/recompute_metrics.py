#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemarag_core import evaluate_predictions, metric_rows, write_csv, write_json, write_markdown_table


DATASETS = {
    "SWDE-1600": ROOT / "data/processed/SWDE-1600.jsonl.gz",
    "Legacy non-SWDE-434": ROOT / "data/processed/Legacy_non-SWDE-434.jsonl.gz",
    "WDC-PAVE-1418": ROOT / "data/processed/WDC-PAVE-1418.jsonl.gz",
}


MAIN_PREDICTIONS = {
    "Regex baseline": "baseline_regex.json.gz",
    "Direct DeepSeek": "direct_deepseek.json.gz",
    "Direct+Verifier": "direct_deepseek_verifier.json.gz",
    "SchemaPlan-RAG": "schemaplan_rag.json.gz",
}


ABLATION_PREDICTIONS = {
    "No planner": ROOT / "results/predictions/ablations/no_planner.json.gz",
    "No property-wise retrieval": ROOT / "results/predictions/ablations/no_property_wise_retrieval.json.gz",
    "No repair": ROOT / "results/predictions/ablations/no_repair.json.gz",
}


DIAGNOSTIC_PREDICTIONS = {
    "No fallback": ROOT / "results/predictions/diagnostics/no_fallback.json.gz",
    "No target-slot admission": ROOT / "results/predictions/diagnostics/no_target_slot_admission.json.gz",
}


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(read_text(path))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def macro_f1(examples: list[dict[str, Any]], predictions: dict[str, Any], schema_index: dict[str, Any], key: str) -> float:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        group = example.get(key) or example.get("source") or "unknown"
        grouped.setdefault(str(group), []).append(example)
    if not grouped:
        return 0.0
    return sum(evaluate_predictions(group, predictions, schema_index)["f1"] for group in grouped.values()) / len(grouped)


def add_macro_metrics(
    examples: list[dict[str, Any]], predictions: dict[str, Any], schema_index: dict[str, Any]
) -> dict[str, float]:
    metrics = evaluate_predictions(examples, predictions, schema_index)
    metrics["site_macro_f1"] = macro_f1(examples, predictions, schema_index, "site")
    metrics["type_macro_f1"] = macro_f1(examples, predictions, schema_index, "schema_type")
    return metrics


def filter_predictions(examples: list[dict[str, Any]], predictions: dict[str, Any]) -> dict[str, Any]:
    return {str(ex["id"]): predictions[str(ex["id"])] for ex in examples if str(ex["id"]) in predictions}


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="data/schema/schema_index.json")
    parser.add_argument("--out-dir", default="runs/recomputed_metrics")
    parser.add_argument(
        "--recompute-ablation-predictions",
        action="store_true",
        help=(
            "Recompute ablation metrics from archived prediction files. "
            "By default, the script copies the reference ablation table because the archived ablation "
            "runs use their own WDC diagnostic IDs."
        ),
    )
    args = parser.parse_args()

    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = ROOT / schema_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    schema_index = load_json(schema_path)

    split_examples = {name: load_jsonl(path) for name, path in DATASETS.items()}
    pooled_examples = [ex for examples in split_examples.values() for ex in examples]
    split_metrics: dict[str, dict[str, dict[str, float]]] = {}
    rows: list[dict[str, str]] = []
    pooled_predictions: dict[str, dict[str, Any]] = {method: {} for method in MAIN_PREDICTIONS}

    for split_name, examples in split_examples.items():
        split_metrics[split_name] = {}
        for method, filename in MAIN_PREDICTIONS.items():
            pred_path = ROOT / "results/predictions" / split_name.replace(" ", "_") / filename
            predictions = filter_predictions(examples, load_json(pred_path))
            pooled_predictions[method].update(predictions)
            metrics = add_macro_metrics(examples, predictions, schema_index)
            split_metrics[split_name][method] = metrics
            row = {"split": split_name, "method": method}
            row.update(metric_rows({method: metrics})[0])
            rows.append(row)

    pooled_name = f"WebSchema-JSONLD Bench-{len(pooled_examples)}"
    split_metrics[pooled_name] = {}
    for method, predictions in pooled_predictions.items():
        metrics = add_macro_metrics(pooled_examples, predictions, schema_index)
        split_metrics[pooled_name][method] = metrics
        row = {"split": pooled_name, "method": method}
        row.update(metric_rows({method: metrics})[0])
        rows.append(row)

    write_json(out_dir / "main_metrics.json", split_metrics)
    write_rows(out_dir / "main_metrics.csv", rows)
    write_markdown_table(out_dir / "main_metrics.md", rows)

    if args.recompute_ablation_predictions:
        ablation_metrics = {"Full SchemaPlan-RAG": split_metrics[pooled_name]["SchemaPlan-RAG"]}
        for method, path in ABLATION_PREDICTIONS.items():
            ablation_metrics[method] = add_macro_metrics(pooled_examples, filter_predictions(pooled_examples, load_json(path)), schema_index)
        diagnostic_metrics = {}
        no_template_examples = split_examples["SWDE-1600"]
        diagnostic_metrics["No-template check"] = add_macro_metrics(
            no_template_examples,
            filter_predictions(no_template_examples, load_json(ROOT / "results/predictions/diagnostics/no_template_swde.json.gz")),
            schema_index,
        )
        for method, path in DIAGNOSTIC_PREDICTIONS.items():
            diagnostic_metrics[method] = add_macro_metrics(
                pooled_examples, filter_predictions(pooled_examples, load_json(path)), schema_index
            )
        write_json(out_dir / "ablation_metrics.json", ablation_metrics)
        write_csv(out_dir / "ablation_metrics.csv", metric_rows(ablation_metrics))
        write_markdown_table(out_dir / "ablation_metrics.md", metric_rows(ablation_metrics))
        write_json(out_dir / "diagnostic_metrics.json", diagnostic_metrics)
        write_csv(out_dir / "diagnostic_metrics.csv", metric_rows(diagnostic_metrics))
        write_markdown_table(out_dir / "diagnostic_metrics.md", metric_rows(diagnostic_metrics))
    else:
        ref_ablation = read_csv_rows(ROOT / "results/reference/ablation_metrics.csv")
        ref_no_template = read_csv_rows(ROOT / "results/reference/no_template_swde_metrics.csv")
        write_rows(out_dir / "ablation_reference.csv", ref_ablation)
        write_rows(out_dir / "no_template_swde_reference.csv", ref_no_template)
        write_markdown_table(out_dir / "ablation_reference.md", ref_ablation)
        write_markdown_table(out_dir / "no_template_swde_reference.md", ref_no_template)

    print(json.dumps({"main": str(out_dir / "main_metrics.csv"), "splits": list(split_metrics)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
