#!/usr/bin/env python3
"""Select a small stratified TRACE replay subset by q_tot/error-rate bands.

This script helps build the 2 datasets x 3 error bands smoke replay plan.  It
reads eigenvectors.json, groups records by dataset_name, and picks the record
closest to each requested error band for each selected dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _normalize_rate(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    if parsed < 0:
        return 0.0
    if parsed > 1.0:
        return parsed / 100.0
    return parsed


def _q_tot(record: dict[str, Any]) -> float:
    if record.get("error_rate") not in (None, ""):
        return _normalize_rate(record.get("error_rate"))
    return _normalize_rate(record.get("missing_rate")) + _normalize_rate(record.get("noise_rate"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset_id",
        "dataset_name",
        "csv_file",
        "band",
        "q_tot",
        "distance_to_band",
        "missing_rate",
        "noise_rate",
        "error_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select stratified dataset ids for TRACE replay.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1], help="TRACE repository root.")
    parser.add_argument("--eigenvectors", type=Path, default=None, help="Path to eigenvectors.json. Default: <project-root>/results/eigenvectors.json.")
    parser.add_argument("--output", type=Path, default=None, help="CSV output path. Default: results/processed/trace/trace_replay_subset.csv.")
    parser.add_argument("--bands", nargs="*", type=float, default=[0.05, 0.15, 0.30], help="Target q_tot bands as fractions or percentages. Default: 0.05 0.15 0.30.")
    parser.add_argument("--num-datasets", type=int, default=2, help="Number of dataset_name groups to select. Default: 2.")
    parser.add_argument("--dataset-names", nargs="*", default=None, help="Optional explicit dataset names. If omitted, choose datasets with best band coverage.")
    parser.add_argument("--exclude-clean-like", action="store_true", help="Drop records with q_tot <= 0.011.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    eigenvectors_path = (args.eigenvectors or (project_root / "results" / "eigenvectors.json")).resolve()
    output_path = (args.output or (project_root / "results" / "processed" / "trace" / "trace_replay_subset.csv")).resolve()
    bands = [_normalize_rate(x) for x in args.bands]

    data = _read_json(eigenvectors_path)
    if not isinstance(data, list):
        raise SystemExit(f"eigenvectors.json must contain a list: {eigenvectors_path}")

    by_dataset: dict[str, list[tuple[int, dict[str, Any], float]]] = defaultdict(list)
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        q = _q_tot(record)
        if args.exclude_clean_like and q <= 0.011:
            continue
        dataset_name = str(record.get("dataset_name") or record.get("dataset") or "").strip()
        if not dataset_name:
            continue
        dataset_id = int(record.get("dataset_id", idx))
        by_dataset[dataset_name].append((dataset_id, record, q))

    if args.dataset_names:
        selected_names = [name for name in args.dataset_names if name in by_dataset]
    else:
        coverage_scores: list[tuple[float, str]] = []
        for dataset_name, records in by_dataset.items():
            if not records:
                continue
            total_distance = 0.0
            unique_ids: set[int] = set()
            for band in bands:
                dataset_id, _, q = min(records, key=lambda item: abs(item[2] - band))
                unique_ids.add(dataset_id)
                total_distance += abs(q - band)
            # Prefer datasets with distinct records for the requested bands, then low distance.
            penalty = (len(bands) - len(unique_ids)) * 10.0
            coverage_scores.append((penalty + total_distance, dataset_name))
        coverage_scores.sort()
        selected_names = [name for _, name in coverage_scores[: max(1, int(args.num_datasets))]]

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for dataset_name in selected_names:
        records = by_dataset.get(dataset_name, [])
        for band in bands:
            dataset_id, record, q = min(records, key=lambda item: abs(item[2] - band))
            key = (dataset_name, dataset_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "csv_file": record.get("csv_file", ""),
                    "band": band,
                    "q_tot": q,
                    "distance_to_band": abs(q - band),
                    "missing_rate": _normalize_rate(record.get("missing_rate")),
                    "noise_rate": _normalize_rate(record.get("noise_rate")),
                    "error_rate": _normalize_rate(record.get("error_rate")),
                }
            )

    rows.sort(key=lambda r: (str(r["dataset_name"]), float(r["band"]), int(r["dataset_id"])))
    _write_csv(output_path, rows)

    ids = [str(int(row["dataset_id"])) for row in rows]
    print(f"[TRACE] Wrote subset CSV: {output_path}")
    print(f"[TRACE] Selected {len(rows)} records from datasets: {', '.join(selected_names)}")
    print("[TRACE] PowerShell dataset-id argument:")
    print("--dataset-ids " + " ".join(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
