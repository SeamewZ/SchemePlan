#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemarag_core import planned_schema_rag_extract


def load_first_jsonl_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"no rows in {path}")


def main() -> int:
    schema = json.loads((ROOT / "data/schema/schema_index.json").read_text(encoding="utf-8"))
    example = load_first_jsonl_gz(ROOT / "data/processed/WDC-PAVE-1418.jsonl.gz")
    result = planned_schema_rag_extract(example, schema, require_llm_steps=False)
    assert isinstance(result.get("jsonld"), dict)
    assert result["jsonld"].get("@type") == example["schema_type"]
    print(json.dumps({"status": "ok", "example_id": example["id"], "fields": sorted(result["jsonld"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
