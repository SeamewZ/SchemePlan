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
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemarag_core import (
    COMMON_SCHEMA_PROPS,
    compact_evidence_block,
    evidence_graph_for_example,
    evaluate_predictions,
    flatten_fields,
    metric_rows,
    supported_by_text,
    write_csv,
    write_json,
    write_markdown_table,
)


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


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(?s)\{.*\}", cleaned)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), {})
    return parsed if isinstance(parsed, dict) else {}


def direct_prompt(example: dict[str, Any], schema_index: dict[str, Any], max_blocks: int, max_chars: int) -> str:
    schema_type = str(example.get("schema_type", "Thing"))
    allowed = sorted(set(schema_index.get(schema_type, {}).get("properties", [])) | COMMON_SCHEMA_PROPS)
    projection = example.get("selection_notes", {}).get("projection", [])
    projection = [str(item) for item in projection] if isinstance(projection, list) else []
    graph = evidence_graph_for_example(example)
    blocks = [compact_evidence_block(block, graph) for block in graph.get("blocks", [])[:max_blocks]]
    return (
        "You are a direct LLM baseline for schema.org JSON-LD extraction.\n"
        "Input is page DOM evidence plus admissible schema properties. Output exactly one JSON-LD object.\n"
        "Do not output candidate records. Do not cite evidence. Do not add markdown.\n"
        "Only extract values explicitly supported by the evidence blocks. Omit unknown fields.\n"
        f"Target fields for this benchmark: {', '.join(projection) if projection else 'unspecified'}.\n"
        "Use nested schema.org objects such as offers.price or aggregateRating.ratingValue when appropriate.\n\n"
        f"Target @type: {schema_type}\n"
        f"Allowed top-level properties: {', '.join(allowed[:160])}\n"
        "Return compact JSON with @context='https://schema.org' and the target @type.\n\n"
        "DOM evidence blocks:\n"
        f"{json.dumps(blocks, ensure_ascii=False)[:max_chars]}"
    )


def call_chat_completion(prompt: str, api_key: str, base_url: str, model: str, timeout: int) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You extract schema.org JSON-LD from web evidence. Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if "deepseek" in base_url:
        body["thinking"] = {"type": "disabled"}
    retries = int(os.environ.get("SCHEMARAG_LLM_RETRIES", "4"))
    backoff = float(os.environ.get("SCHEMARAG_LLM_RETRY_BACKOFF", "2.0"))
    req = Request(
        base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return data["choices"][0]["message"]["content"]
        except HTTPError as exc:
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= retries:
                raise
        except (URLError, TimeoutError, RemoteDisconnected, IncompleteRead):
            if attempt >= retries:
                raise
        time.sleep(backoff * (2**attempt))
    raise RuntimeError("chat completion failed after retries")


def verify_jsonld(obj: dict[str, Any], example: dict[str, Any], schema_index: dict[str, Any]) -> dict[str, Any]:
    schema_type = str(example["schema_type"])
    allowed = set(schema_index.get(schema_type, {}).get("properties", [])) | COMMON_SCHEMA_PROPS
    verified: dict[str, Any] = {"@context": "https://schema.org", "@type": schema_type}
    for key, value in obj.items():
        root = str(key).split(":")[-1]
        if root in {"@context", "@type"} or root not in allowed:
            continue
        flat = flatten_fields({root: value})
        if flat and all(supported_by_text(v, example["text"]) for v in flat.values()):
            verified[root] = value
    return verified


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
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_blocks: int,
    max_chars: int,
) -> dict[str, Any]:
    cache_path = cache_dir / safe_name(str(example["id"]))
    if cache_path.exists():
        return load_json(cache_path, {})
    raw = call_chat_completion(direct_prompt(example, schema_index, max_blocks, max_chars), api_key, base_url, model, timeout)
    obj = parse_json_object(raw)
    obj.setdefault("@context", "https://schema.org")
    obj["@type"] = example.get("schema_type", obj.get("@type", "Thing"))
    result = {"jsonld": obj, "raw_output": raw}
    write_json(cache_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/WDC-PAVE-1418.jsonl.gz")
    parser.add_argument("--schema", default="data/schema/schema_index.json")
    parser.add_argument("--out-dir", default="runs/direct_deepseek")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_CHAT_URL", "https://api.deepseek.com/chat/completions"))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-blocks", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=24000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("SCHEMARAG_DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        status = {"status": "skipped", "reason": "Set DEEPSEEK_API_KEY to run the direct baseline."}
        write_json(out_dir / "status.json", status)
        print(json.dumps(status, indent=2))
        return 0

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = ROOT / schema_path
    examples = load_jsonl(dataset_path)
    if args.limit:
        examples = examples[: args.limit]
    schema_index = load_json(schema_path, {})
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, Any] = {}
    errors: dict[str, str] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_one,
                example,
                schema_index,
                cache_dir,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                max_blocks=args.max_blocks,
                max_chars=args.max_chars,
            ): example
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
            write_json(out_dir / "predictions" / "direct_deepseek.json", predictions)

    completed = [example for example in examples if str(example["id"]) in predictions]
    verified = {
        ex_id: {
            "jsonld": verify_jsonld(
                pred.get("jsonld", {}),
                next(example for example in completed if str(example["id"]) == ex_id),
                schema_index,
            )
        }
        for ex_id, pred in predictions.items()
    }
    all_predictions = {"direct_deepseek": predictions, "direct_deepseek_verifier": verified}
    metrics = {method: add_macro_metrics(completed, pred, schema_index) for method, pred in all_predictions.items()}
    write_json(out_dir / "predictions" / "direct_deepseek_verifier.json", verified)
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
            "model": args.model,
            "elapsed_seconds": round(time.time() - started, 3),
            "errors": len(errors),
        },
    )
    print(json.dumps({"status": load_json(out_dir / "status.json", {}), "metrics": metrics}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
