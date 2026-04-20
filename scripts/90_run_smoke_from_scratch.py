#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run TRACE Mode B smoke test from scratch.

Mode B-smoke is a minimal end-to-end pipeline check. It is not intended to
reproduce the full paper results.

Default smoke configuration:
- dataset: beers
- dirty_id: 1
- cleaner: mode
- clusterer: HC
- cluster_trials: 5
- workers: 1

Mode C-full is represented by running the same pipeline without smoke limits:
all records, all cleaners, all clusterers, and full clustering trial budgets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACE Mode B smoke test.")
    parser.add_argument(
        "--config",
        default="configs/mode_b_smoke.yaml",
        help="Mode B config path. Currently used for record keeping.",
    )
    parser.add_argument(
        "--dataset",
        default="beers",
        help="Dataset name for smoke test.",
    )
    parser.add_argument(
        "--dirty-id",
        type=int,
        default=1,
        help="Dirty dataset id for smoke test.",
    )
    parser.add_argument(
        "--cleaner",
        default="mode",
        help="Cleaner used in smoke test.",
    )
    parser.add_argument(
        "--clusterer",
        default="HC",
        help="Clusterer used in smoke test.",
    )
    parser.add_argument(
        "--cluster-trials",
        type=int,
        default=5,
        help="Clustering trial budget for smoke test.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of pipeline workers for smoke test.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous smoke outputs before running.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print(f"[TRACE] Running: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_dataset_link() -> None:
    """
    Create datasets/train as a compatibility path for the original pipeline.

    The canonical TRACE input location is data/raw/train.
    """
    datasets_dir = PROJECT_ROOT / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    link_path = datasets_dir / "train"
    target_path = PROJECT_ROOT / "data" / "raw" / "train"

    if link_path.exists():
        return

    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    else:
        link_path.symlink_to(target_path, target_is_directory=True)


def clean_outputs(dataset: str, cleaner: str, clusterer: str) -> None:
    paths = [
        PROJECT_ROOT / "results" / "eigenvectors.json",
        PROJECT_ROOT / "results" / "cleaned_results.json",
        PROJECT_ROOT / "results" / "clustered_results.json",
        PROJECT_ROOT / "results" / "analyzed_results.json",
        PROJECT_ROOT / "results" / "cleaned_data" / cleaner,
        PROJECT_ROOT / "results" / "clustered_data" / clusterer,
        PROJECT_ROOT / "results" / "logs" / "pipeline_run_manifest.json",
        PROJECT_ROOT / "results" / "logs" / "pipeline_failures.json",
        PROJECT_ROOT / "results" / "logs" / "mode_b_smoke_summary.json",
        PROJECT_ROOT / "src" / "cleaning" / "Repaired_res" / cleaner / dataset,
    ]

    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def summarize_outputs() -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for name in [
        "eigenvectors.json",
        "cleaned_results.json",
        "clustered_results.json",
        "analyzed_results.json",
    ]:
        path = PROJECT_ROOT / "results" / name
        data = load_json_if_exists(path)

        summary[name] = {
            "exists": path.exists(),
            "records": len(data) if isinstance(data, list) else None,
        }

    manifest_path = PROJECT_ROOT / "results" / "logs" / "pipeline_run_manifest.json"
    failures_path = PROJECT_ROOT / "results" / "logs" / "pipeline_failures.json"

    summary["pipeline_run_manifest.json"] = {
        "exists": manifest_path.exists(),
        "path": str(manifest_path),
    }
    summary["pipeline_failures.json"] = {
        "exists": failures_path.exists(),
        "path": str(failures_path),
    }

    return summary


def write_smoke_summary(args: argparse.Namespace) -> None:
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "B_SMOKE_FROM_SCRATCH",
        "config": args.config,
        "dataset": args.dataset,
        "dirty_id": args.dirty_id,
        "cleaner": args.cleaner,
        "clusterer": args.clusterer,
        "cluster_trials": args.cluster_trials,
        "workers": args.workers,
        "outputs": summarize_outputs(),
    }

    output_path = PROJECT_ROOT / "results" / "logs" / "mode_b_smoke_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[TRACE] Smoke summary written to: {output_path}")


def main() -> None:
    args = parse_args()

    ensure_dataset_link()

    if args.clean:
        clean_outputs(
            dataset=args.dataset,
            cleaner=args.cleaner,
            clusterer=args.clusterer,
        )

    run_command(
        [
            sys.executable,
            "-m",
            "src.pipeline.preprocess",
            "--skip-injection",
            "--data-dir",
            "datasets/train",
            "--output-file",
            "results/eigenvectors.json",
            "--datasets",
            args.dataset,
            "--dirty-ids",
            str(args.dirty_id),
        ]
    )

    run_command(
        [
            sys.executable,
            "-m",
            "src.pipeline.runner",
            "--max-records",
            "1",
            "--workers",
            str(args.workers),
            "--cleaners",
            args.cleaner,
            "--clusterers",
            args.clusterer,
            "--cluster-trials",
            str(args.cluster_trials),
        ]
    )

    write_smoke_summary(args)
    print("[TRACE] Mode B smoke test completed.")


if __name__ == "__main__":
    main()