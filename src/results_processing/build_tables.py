#!/usr/bin/env python3
"""Build canonical TRACE result tables from raw pipeline outputs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.results_processing.io import ensure_list, read_json, to_number_or_empty, write_csv, write_json
from src.results_processing.schema import load_schema, table_columns
from src.results_processing.validators import get_result_input_paths


SCORE_PATTERNS = {
    "final_combined_score": re.compile(r"Final Combined Score:\s*([-+0-9.eE]+)"),
    "final_silhouette_score": re.compile(r"Final Silhouette Score:\s*([-+0-9.eE]+)"),
    "final_davies_bouldin_score": re.compile(r"Final Davies-Bouldin Score:\s*([-+0-9.eE]+)"),
}


def _safe_int(value: Any) -> Any:
    if value in (None, ""):
        return ""
    try:
        return int(value)
    except Exception:
        return ""


def _infer_from_cluster_path(path_value: Any) -> Dict[str, Any]:
    """Infer clusterer, cleaner, and dataset id from a clustered output directory."""
    if not path_value:
        return {}

    p = Path(str(path_value))
    parts = list(p.parts)
    out: Dict[str, Any] = {}

    try:
        idx = parts.index("clustered_data")
        out["clusterer"] = parts[idx + 1]
        out["cleaner"] = parts[idx + 2]
        cluster_dir = parts[idx + 3]
        if cluster_dir.startswith("clustered_"):
            out["dataset_id"] = _safe_int(cluster_dir.replace("clustered_", ""))
    except Exception:
        pass

    return out


def _parse_cluster_text(clustered_path: Any) -> Dict[str, Any]:
    """Parse the conventional clustering text output if it exists."""
    if not clustered_path:
        return {}

    root = Path(str(clustered_path))
    if not root.exists():
        return {}

    txt_files = sorted(root.glob("*.txt"))
    if not txt_files:
        return {}

    text = txt_files[0].read_text(encoding="utf-8-sig", errors="replace")
    out: Dict[str, Any] = {"source_file": str(txt_files[0])}

    for key, pattern in SCORE_PATTERNS.items():
        match = pattern.search(text)
        if match:
            out[key] = to_number_or_empty(match.group(1))

    params_match = re.search(r"Best parameters:\s*(.*)", text)
    if params_match:
        out["best_parameters"] = params_match.group(1).strip()

    return out


def _scalar_items(data: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    """Return flattened scalar values from a nested dictionary."""
    rows: List[Tuple[str, Any]] = []

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            rows.extend(_scalar_items(value, full_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            rows.append((full_key, value))
        else:
            continue

    return rows


def build_trials(eigenvectors: List[dict]) -> List[dict]:
    """Build dirty-instance trial rows."""
    rows = []

    for idx, record in enumerate(eigenvectors):
        rows.append(
            {
                "dataset_id": record.get("dataset_id", idx),
                "dataset_name": record.get("dataset_name", record.get("dataset", "")),
                "csv_file": record.get("csv_file", ""),
                "error_rate": to_number_or_empty(record.get("error_rate")),
                "missing_rate": to_number_or_empty(record.get("missing_rate")),
                "noise_rate": to_number_or_empty(record.get("noise_rate")),
                "source": "eigenvectors.json",
            }
        )

    return rows


def build_cleaning_metrics(cleaned_results: List[dict]) -> List[dict]:
    """Build cleaner-level rows."""
    rows = []

    for record in cleaned_results:
        rows.append(
            {
                "dataset_id": record.get("dataset_id", ""),
                "cleaner": record.get("algorithm", record.get("cleaner", "")),
                "cleaner_id": record.get("algorithm_id", record.get("cleaner_id", "")),
                "cleaning_runtime_sec": to_number_or_empty(record.get("runtime", record.get("cleaning_runtime"))),
                "cleaned_file_path": record.get("cleaned_file_path", ""),
            }
        )

    return rows


def build_result_metrics(clustered_results: List[dict]) -> List[dict]:
    """Build final result rows."""
    rows = []

    for record in clustered_results:
        parsed = _parse_cluster_text(record.get("clustered_file_path"))
        inferred = _infer_from_cluster_path(record.get("clustered_file_path"))

        rows.append(
            {
                "dataset_id": record.get("dataset_id", inferred.get("dataset_id", "")),
                "cleaner": record.get("cleaning_algorithm", record.get("cleaner", inferred.get("cleaner", ""))),
                "clusterer": record.get("clustering_name", record.get("clusterer", inferred.get("clusterer", ""))),
                "clusterer_id": record.get("clustering_algorithm", record.get("clusterer_id", "")),
                "cleaning_runtime_sec": to_number_or_empty(record.get("cleaning_runtime")),
                "clustering_runtime_sec": to_number_or_empty(record.get("clustering_runtime")),
                "final_combined_score": parsed.get("final_combined_score", ""),
                "final_silhouette_score": parsed.get("final_silhouette_score", ""),
                "final_davies_bouldin_score": parsed.get("final_davies_bouldin_score", ""),
                "clustered_file_path": record.get("clustered_file_path", ""),
                "source_file": parsed.get("source_file", ""),
            }
        )

    return rows


def _candidate_lists_from_analyzed_record(record: dict) -> List[dict]:
    """Extract candidate config rows from several possible analyzed-result layouts."""
    possible_keys = [
        "top_k",
        "top_k_results",
        "results",
        "configs",
        "best_configs",
        "top_configs",
    ]

    for key in possible_keys:
        value = record.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            return [x for x in value.values() if isinstance(x, dict)]

    if any(k in record for k in ["final_score", "combined_score", "score", "best_params", "params"]):
        return [record]

    return []


def build_best_configs(analyzed_results: List[dict]) -> List[dict]:
    """Build ranked configuration rows."""
    rows = []

    for record in analyzed_results:
        dataset_id = record.get("dataset_id", "")
        candidates = _candidate_lists_from_analyzed_record(record)

        for rank, item in enumerate(candidates, 1):
            params = item.get("best_params", item.get("params", item.get("best_parameters", {})))

            rows.append(
                {
                    "dataset_id": item.get("dataset_id", dataset_id),
                    "rank": item.get("rank", rank),
                    "cleaner": item.get("cleaning_algorithm", item.get("cleaner", "")),
                    "clusterer": item.get("clustering_name", item.get("clusterer", "")),
                    "final_score": to_number_or_empty(
                        item.get("final_score", item.get("combined_score", item.get("score")))
                    ),
                    "params_json": json.dumps(params, ensure_ascii=False),
                    "source_file": "analyzed_results.json",
                }
            )

    return rows


def build_process_metrics(results_dir: Path) -> List[dict]:
    """Collect process-level scalar metrics from clustering JSON traces."""
    rows = []
    clustered_root = results_dir / "clustered_data"

    if not clustered_root.exists():
        return rows

    for json_path in sorted(clustered_root.rglob("*.json")):
        if json_path.name.endswith("_param_shift.json"):
            continue

        data = read_json(json_path, default=None)
        if not isinstance(data, dict):
            continue

        inferred = _infer_from_cluster_path(json_path.parent)
        dataset_id = inferred.get("dataset_id", "")
        cleaner = inferred.get("cleaner", "")
        clusterer = inferred.get("clusterer", "")

        for metric_name, metric_value in _scalar_items(data):
            if metric_name in {"clean_state"}:
                continue

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "cleaner": cleaner,
                    "clusterer": clusterer,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "source_file": str(json_path),
                }
            )

    return rows


def build_parameter_shifts(results_dir: Path) -> List[dict]:
    """Collect parameter-shift rows from *_param_shift.json files."""
    rows = []
    clustered_root = results_dir / "clustered_data"

    if not clustered_root.exists():
        return rows

    for json_path in sorted(clustered_root.rglob("*param_shift.json")):
        data = read_json(json_path, default={})
        if not isinstance(data, dict):
            continue

        inferred = _infer_from_cluster_path(json_path.parent)

        for key, value in data.items():
            if key in {"dataset_id"}:
                continue
            if not isinstance(value, (int, float, str)):
                continue

            rows.append(
                {
                    "dataset_id": data.get("dataset_id", inferred.get("dataset_id", "")),
                    "cleaner": inferred.get("cleaner", ""),
                    "clusterer": inferred.get("clusterer", ""),
                    "parameter_name": key,
                    "delta_value": value,
                    "source_file": str(json_path),
                }
            )

    return rows


def build_canonical_results(
    results_dir: Path,
    output_dir: Path,
    schema_path: Path = Path("configs/results_schema.yaml"),
) -> dict:
    """Build canonical CSV tables from raw pipeline outputs."""
    schema = load_schema(schema_path)
    paths = get_result_input_paths(results_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eigenvectors = ensure_list(read_json(paths.eigenvectors, []), "eigenvectors.json")
    cleaned_results = ensure_list(read_json(paths.cleaned_results, []), "cleaned_results.json")
    clustered_results = ensure_list(read_json(paths.clustered_results, []), "clustered_results.json")
    analyzed_results = ensure_list(read_json(paths.analyzed_results, []), "analyzed_results.json")

    tables = {
        "trials": build_trials(eigenvectors),
        "cleaning_metrics": build_cleaning_metrics(cleaned_results),
        "result_metrics": build_result_metrics(clustered_results),
        "best_configs": build_best_configs(analyzed_results),
        "process_metrics": build_process_metrics(paths.results_dir),
        "parameter_shifts": build_parameter_shifts(paths.results_dir),
    }

    for table_name, rows in tables.items():
        write_csv(out / f"{table_name}.csv", rows, table_columns(schema, table_name))

    manifest = {
        "schema_version": schema.get("version", ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(paths.results_dir),
        "output_dir": str(out),
        "input_files": {
            "eigenvectors": str(paths.eigenvectors),
            "cleaned_results": str(paths.cleaned_results),
            "clustered_results": str(paths.clustered_results),
            "analyzed_results": str(paths.analyzed_results),
            "pipeline_manifest": str(paths.pipeline_manifest),
        },
        "row_counts": {name: len(rows) for name, rows in tables.items()},
    }

    write_json(out / "run_manifest.json", manifest)
    return manifest

