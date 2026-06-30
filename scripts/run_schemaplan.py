#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemarag_core import evaluate_predictions, metric_rows, planned_schema_rag_extract, write_csv, write_json, write_markdown_table


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(read_text(path))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def safe_name(example_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", example_id) + ".json"


def macro_f1(examples: list[dict[str, Any]], predictions: dict[str, Any], schema_index: dict[str, Any], key: str) -> float:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        grouped.setdefault(str(example.get(key) or example.get("source") or "unknown"), []).append(example)
    return sum(evaluate_predictions(group, predictions, schema_index)["f1"] for group in grouped.values()) / max(1, len(grouped))


def add_macro_metrics(
    examples: list[dict[str, Any]], predictions: dict[str, Any], schema_index: dict[str, Any]
) -> dict[str, float]:
    metrics = evaluate_predictions(examples, predictions, schema_index)
    metrics["site_macro_f1"] = macro_f1(examples, predictions, schema_index, "site")
    metrics["type_macro_f1"] = macro_f1(examples, predictions, schema_index, "schema_type")
    return metrics


def run_one(
    example: dict[str, Any],
    schema_index: dict[str, Any],
    cache_dir: Path,
    require_llm: bool,
) -> dict[str, Any]:
    cache_path = cache_dir / safe_name(str(example["id"]))
    if cache_path.exists():
        return load_json(cache_path, {})
    result = planned_schema_rag_extract(example, schema_index, require_llm_steps=require_llm)
    write_json(cache_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/WDC-PAVE-1418.jsonl.gz")
    parser.add_argument("--schema", default="data/schema/schema_index.json")
    parser.add_argument("--out-dir", default="runs/schemaplan")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SCHEMAPLAN_WORKERS", "4")))
    parser.add_argument("--no-require-llm", action="store_true", help="Run the deterministic diagnostic path without API calls.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = ROOT / schema_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("SCHEMARAG_SWDE_TEMPLATE_INDEX", str(ROOT / "data/schema/swde_train_templates.json"))

    examples = load_jsonl(dataset_path)
    if args.limit:
        examples = examples[: args.limit]
    schema_index = load_json(schema_path, {})
    require_llm = not args.no_require_llm

    predictions: dict[str, Any] = {}
    errors: dict[str, str] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_one, example, schema_index, cache_dir, require_llm): example
            for example in examples
        }
        for future in as_completed(futures):
            example = futures[future]
            try:
                predictions[str(example["id"])] = future.result()
                print(f"[done] {len(predictions)}/{len(examples)} {example['id']}", flush=True)
            except Exception as exc:
                errors[str(example["id"])] = f"{type(exc).__name__}: {exc}"
                write_json(
                    out_dir / "tracebacks" / f"{safe_name(str(example['id']))}.trace.json",
                    {"id": example["id"], "error": errors[str(example["id"])], "traceback": traceback.format_exc()},
                )
                print(f"[error] {example['id']} {errors[str(example['id'])]}", flush=True)
            write_json(out_dir / "predictions" / "schemaplan_rag.json", predictions)
            write_json(out_dir / "errors.json", errors)

    completed = [example for example in examples if str(example["id"]) in predictions]
    metrics = {"schemaplan_rag": add_macro_metrics(completed, predictions, schema_index)} if completed else {}
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "metrics.csv", metric_rows(metrics))
    write_markdown_table(out_dir / "metrics.md", metric_rows(metrics))
    write_json(
        out_dir / "status.json",
        {
            "status": "completed" if not errors and len(completed) == len(examples) else "partial",
            "dataset": str(dataset_path.relative_to(ROOT) if dataset_path.is_relative_to(ROOT) else dataset_path),
            "examples_requested": len(examples),
            "examples_completed": len(completed),
            "workers": args.workers,
            "llm_required": require_llm,
            "elapsed_seconds": round(time.time() - started, 3),
            "errors": len(errors),
        },
    )
    print(json.dumps({"status": load_json(out_dir / "status.json", {}), "metrics": metrics}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
