#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any


RELEASE = Path(__file__).resolve().parents[1]
ROOT = RELEASE.parent


DATASETS = {
    "SWDE-1600": ROOT / "data/processed/final_eval_swde_balanced.jsonl",
    "Legacy_non-SWDE-434": ROOT / "data/processed/final_eval_original_non_swde.jsonl",
    "WDC-PAVE-1418": ROOT / "data/processed/final_eval_wdc_pave_evidence9_1418.jsonl",
}


PREDICTIONS = {
    "SWDE-1600/baseline_regex.json.gz": ROOT / "results/final_eval_swde_balanced/predictions/baseline_regex.json",
    "SWDE-1600/direct_deepseek.json.gz": ROOT
    / "results/direct_deepseek_baseline_webschema3452_full/predictions/direct_deepseek.json",
    "SWDE-1600/direct_deepseek_verifier.json.gz": ROOT
    / "results/direct_deepseek_baseline_webschema3452_full/predictions/direct_deepseek_verifier.json",
    "SWDE-1600/schemaplan_rag.json.gz": ROOT
    / "results/webschema_jsonld_bench_deepseek_step17_20260616/predictions/schemaplan_rag.json",
    "Legacy_non-SWDE-434/baseline_regex.json.gz": ROOT
    / "results/final_eval_original_non_swde/predictions/baseline_regex.json",
    "Legacy_non-SWDE-434/direct_deepseek.json.gz": ROOT
    / "results/direct_deepseek_baseline_webschema3452_full/predictions/direct_deepseek.json",
    "Legacy_non-SWDE-434/direct_deepseek_verifier.json.gz": ROOT
    / "results/direct_deepseek_baseline_webschema3452_full/predictions/direct_deepseek_verifier.json",
    "Legacy_non-SWDE-434/schemaplan_rag.json.gz": ROOT
    / "results/final_eval_original_non_swde/predictions/schemaplan_rag_deepseek_final.json",
    "WDC-PAVE-1418/baseline_regex.json.gz": ROOT
    / "results/final_eval_wdc_pave_evidence9_1418_main_offline_rerun/predictions/baseline_regex.json",
    "WDC-PAVE-1418/direct_deepseek.json.gz": ROOT
    / "results/direct_deepseek_baseline_evidence9_1418_full/predictions/direct_deepseek.json",
    "WDC-PAVE-1418/direct_deepseek_verifier.json.gz": ROOT
    / "results/direct_deepseek_baseline_evidence9_1418_full/predictions/direct_deepseek_verifier.json",
    "WDC-PAVE-1418/schemaplan_rag.json.gz": ROOT
    / "results/final_eval_wdc_pave_evidence9_1418_admission_main/predictions/schemaplan_rag.json",
    "ablations/no_planner.json.gz": ROOT
    / "results/acceptance_supplement_webschema3452_full/predictions/no_planner.json",
    "ablations/no_property_wise_retrieval.json.gz": ROOT
    / "results/acceptance_supplement_webschema3452_full/predictions/no_property_wise_retrieval.json",
    "ablations/no_repair.json.gz": ROOT
    / "results/acceptance_supplement_webschema3452_full/predictions/no_repair.json",
    "diagnostics/no_fallback.json.gz": ROOT
    / "results/acceptance_supplement_webschema3452_full/predictions/no_fallback.json",
    "diagnostics/no_target_slot_admission.json.gz": ROOT
    / "results/acceptance_supplement_webschema3452_full/predictions/no_target_slot_admission.json",
    "diagnostics/no_template_swde.json.gz": ROOT
    / "results/acceptance_supplement/no_template_swde_replay/predictions/schemaplan_rag.json",
}


RESULT_SUMMARIES = {
    "main_results_by_split.csv": ROOT / "results/webschema_jsonld_bench_3452_summary/tables/metrics_by_split.csv",
    "main_results_pooled.csv": ROOT
    / "results/webschema_jsonld_bench_3452_summary/tables/WebSchema-JSONLD_Bench-3452_metrics.csv",
    "direct_deepseek_by_split.csv": ROOT
    / "results/direct_deepseek_baseline_webschema3452_full/tables/metrics_by_split.csv",
    "recent_baselines_by_split.csv": ROOT / "results/fair_recent_baselines_deepseek/comparison_by_split.csv",
    "recent_baselines_pooled.csv": ROOT / "results/fair_recent_baselines_deepseek/pooled_metrics.csv",
    "ablation_metrics.csv": ROOT / "results/acceptance_supplement_webschema3452_full/tables/ablation_metrics.csv",
    "no_template_swde_metrics.csv": ROOT / "results/acceptance_supplement/no_template_swde_replay/tables/metrics.csv",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def copy_gzip(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as in_fh, gzip.open(dst, "wb") as out_fh:
        shutil.copyfileobj(in_fh, out_fh)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sanitize_example(example: dict[str, Any]) -> dict[str, Any]:
    ex = dict(example)
    raw_path = ex.pop("raw_html_path", "")
    if raw_path and "html" not in ex:
        path = Path(raw_path)
        if path.exists():
            ex["html"] = path.read_text(encoding="utf-8", errors="replace")
    ex.pop("source_url", None)
    return ex


def main() -> int:
    (RELEASE / "src").mkdir(parents=True, exist_ok=True)
    (RELEASE / "data/processed").mkdir(parents=True, exist_ok=True)
    (RELEASE / "data/schema").mkdir(parents=True, exist_ok=True)
    (RELEASE / "results/reference").mkdir(parents=True, exist_ok=True)
    (RELEASE / "results/predictions").mkdir(parents=True, exist_ok=True)

    copy_if_exists(ROOT / "src/schemarag_core.py", RELEASE / "src/schemarag_core.py")
    copy_if_exists(ROOT / "data/processed/schema_index.json", RELEASE / "data/schema/schema_index.json")
    copy_if_exists(
        ROOT / "results/swde_official_train_template_probe_cap200/compiled_templates.json",
        RELEASE / "data/schema/swde_train_templates.json",
    )

    manifest: dict[str, Any] = {"datasets": {}, "predictions": {}, "results": {}}
    for name, src in DATASETS.items():
        rows = [sanitize_example(ex) for ex in load_jsonl(src)]
        dst = RELEASE / "data/processed" / f"{name}.jsonl.gz"
        write_jsonl_gz(dst, rows)
        manifest["datasets"][name] = {
            "path": str(dst.relative_to(RELEASE)),
            "examples": len(rows),
            "schema_types": sorted({str(row.get("schema_type", "")) for row in rows}),
        }

    for rel, src in PREDICTIONS.items():
        dst = RELEASE / "results/predictions" / rel
        copy_gzip(src, dst)
        manifest["predictions"][rel] = str(dst.relative_to(RELEASE))

    for rel, src in RESULT_SUMMARIES.items():
        dst = RELEASE / "results/reference" / rel
        copy_if_exists(src, dst)
        manifest["results"][rel] = str(dst.relative_to(RELEASE))

    (RELEASE / "data/manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"release": str(RELEASE), "datasets": manifest["datasets"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
