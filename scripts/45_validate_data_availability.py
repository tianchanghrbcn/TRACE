#!/usr/bin/env python3
"""Validate TRACE data availability for reviewer-facing runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_DATASETS = ["beers", "flights", "hospital", "rayyan"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TRACE data availability.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/train"))
    parser.add_argument("--pre-experiment-csv", type=Path, default=Path("data/pre_experiment/alpha_metrics.csv"))
    return parser.parse_args()


def dataset_report(data_root: Path, dataset: str) -> dict:
    root = data_root / dataset
    dirty_files = sorted(root.glob(f"{dataset}_*.csv"))
    return {
        "dataset": dataset,
        "exists": root.exists(),
        "clean_csv": (root / "clean.csv").exists(),
        "horizon_rules": (root / "dc_rules-validate-fd-horizon.txt").exists(),
        "holoclean_rules": (root / "dc_rules_holoclean.txt").exists(),
        "dirty_file_count": len(dirty_files),
    }


def main() -> None:
    args = parse_args()

    reports = [dataset_report(args.data_root, name) for name in REQUIRED_DATASETS]
    pre_exp_exists = args.pre_experiment_csv.exists()

    failures = []
    for row in reports:
        if not row["exists"]:
            failures.append(f"{row['dataset']}: dataset directory missing")
        if not row["clean_csv"]:
            failures.append(f"{row['dataset']}: clean.csv missing")
        if row["dirty_file_count"] == 0:
            failures.append(f"{row['dataset']}: no dirty CSV files found")

    if not pre_exp_exists:
        failures.append(f"pre-experiment alpha metrics missing: {args.pre_experiment_csv}")

    report = {
        "data_root": str(args.data_root),
        "datasets": reports,
        "pre_experiment_csv": str(args.pre_experiment_csv),
        "pre_experiment_csv_exists": pre_exp_exists,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[TRACE] Data availability status: {report['status']}")

    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

