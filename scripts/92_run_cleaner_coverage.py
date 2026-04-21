#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run TRACE cleaner coverage smoke test.

This script validates the cleaning side of the pipeline with a fixed clusterer.

Default coverage configuration:
- dataset: beers
- dirty_id: 1
- cleaners: group:lightweight from configs/methods.yaml
- clusterers: group:smoke from configs/methods.yaml
- cluster_trials: 5
- workers: 1

This is not a full paper reproduction. It is a method-integration check used to
make sure registered cleaners can be called through the same pipeline contract.
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.method_registry import load_default_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACE cleaner coverage smoke test.")

    parser.add_argument(
        "--config",
        default="configs/mode_b_smoke.yaml",
        help="Config path recorded in the summary. Method lists come from configs/methods.yaml.",
    )
    parser.add_argument(
        "--dataset",
        default="beers",
        help="Dataset name for coverage smoke.",
    )
    parser.add_argument(
        "--dirty-id",
        type=int,
        default=1,
        help="Dirty dataset id for coverage smoke.",
    )
    parser.add_argument(
        "--cleaners",
        nargs="*",
        default=["group:lightweight"],
        help=(
            "Cleaner tokens resolved by configs/methods.yaml. Accepts names, TRACE ids, "
            "legacy ids with legacy:<id>, or groups such as group:lightweight."
        ),
    )
    parser.add_argument(
        "--clusterers",
        nargs="*",
        default=["group:smoke"],
        help="Clusterer tokens resolved by configs/methods.yaml. Default: group:smoke.",
    )
    parser.add_argument(
        "--cluster-trials",
        type=int,
        default=5,
        help="Clustering trial budget for clusterers that support TRACE_N_TRIALS.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of pipeline workers.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous cleaner coverage outputs before running.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit with status 0 even if one or more cleaners fail.",
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


def clean_outputs(dataset: str, cleaner_names: list[str], clusterer_names: list[str]) -> None:
    paths = [
        PROJECT_ROOT / "results" / "eigenvectors.json",
        PROJECT_ROOT / "results" / "cleaned_results.json",
        PROJECT_ROOT / "results" / "clustered_results.json",
        PROJECT_ROOT / "results" / "analyzed_results.json",
        PROJECT_ROOT / "results" / "logs" / "pipeline_run_manifest.json",
        PROJECT_ROOT / "results" / "logs" / "pipeline_failures.json",
        PROJECT_ROOT / "results" / "logs" / "cleaner_coverage_summary.json",
    ]

    for cleaner_name in cleaner_names:
        paths.append(PROJECT_ROOT / "results" / "cleaned_data" / cleaner_name)
        paths.append(PROJECT_ROOT / "src" / "cleaning" / "Repaired_res" / cleaner_name / dataset)

    for clusterer_name in clusterer_names:
        paths.append(PROJECT_ROOT / "results" / "clustered_data" / clusterer_name)

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


def summarize_outputs(cleaner_names: list[str], clusterer_names: list[str]) -> dict[str, Any]:
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

    cleaned_results = load_json_if_exists(PROJECT_ROOT / "results" / "cleaned_results.json")
    clustered_results = load_json_if_exists(PROJECT_ROOT / "results" / "clustered_results.json")

    observed_cleaners: set[str] = set()
    observed_clustered_pairs: set[tuple[str, str]] = set()

    if isinstance(cleaned_results, list):
        for row in cleaned_results:
            name = row.get("algorithm")
            if name:
                observed_cleaners.add(str(name))

    if isinstance(clustered_results, list):
        for row in clustered_results:
            cleaner = row.get("cleaning_algorithm")
            clusterer = row.get("clustering_name")
            if cleaner and clusterer:
                observed_clustered_pairs.add((str(cleaner), str(clusterer)))

    summary["cleaners"] = {
        cleaner_name: {
            "observed_in_cleaned_results": cleaner_name in observed_cleaners,
            "cleaned_data_dir_exists": (
                PROJECT_ROOT / "results" / "cleaned_data" / cleaner_name
            ).exists(),
            "clustered_by": {
                clusterer_name: (cleaner_name, clusterer_name) in observed_clustered_pairs
                for clusterer_name in clusterer_names
            },
        }
        for cleaner_name in cleaner_names
    }

    manifest_path = PROJECT_ROOT / "results" / "logs" / "pipeline_run_manifest.json"
    failures_path = PROJECT_ROOT / "results" / "logs" / "pipeline_failures.json"

    manifest = load_json_if_exists(manifest_path)
    failures = load_json_if_exists(failures_path)

    summary["pipeline_run_manifest.json"] = {
        "exists": manifest_path.exists(),
        "path": str(manifest_path),
        "failure_count": manifest.get("failure_count") if isinstance(manifest, dict) else None,
        "cleaned_result_count": manifest.get("cleaned_result_count") if isinstance(manifest, dict) else None,
        "clustered_result_count": manifest.get("clustered_result_count") if isinstance(manifest, dict) else None,
    }
    summary["pipeline_failures.json"] = {
        "exists": failures_path.exists(),
        "path": str(failures_path),
        "records": len(failures) if isinstance(failures, list) else None,
    }

    return summary


def write_coverage_summary(
    args: argparse.Namespace,
    cleaner_names: list[str],
    clusterer_names: list[str],
    output_summary: dict[str, Any],
) -> Path:
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "CLEANER_COVERAGE_SMOKE",
        "config": args.config,
        "dataset": args.dataset,
        "dirty_id": args.dirty_id,
        "cleaner_tokens": args.cleaners,
        "clusterer_tokens": args.clusterers,
        "cleaners": cleaner_names,
        "clusterers": clusterer_names,
        "cluster_trials": args.cluster_trials,
        "workers": args.workers,
        "outputs": output_summary,
    }

    output_path = PROJECT_ROOT / "results" / "logs" / "cleaner_coverage_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[TRACE] Cleaner coverage summary written to: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()

    registry = load_default_registry(PROJECT_ROOT)
    cleaner_names = registry.names("cleaners", args.cleaners)
    clusterer_names = registry.names("clusterers", args.clusterers)

    print(f"[TRACE] Coverage cleaners: {cleaner_names}")
    print(f"[TRACE] Coverage clusterers: {clusterer_names}")

    ensure_dataset_link()

    if args.clean:
        clean_outputs(
            dataset=args.dataset,
            cleaner_names=cleaner_names,
            clusterer_names=clusterer_names,
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
            *args.cleaners,
            "--clusterers",
            *args.clusterers,
            "--cluster-trials",
            str(args.cluster_trials),
        ]
    )

    output_summary = summarize_outputs(cleaner_names, clusterer_names)
    write_coverage_summary(args, cleaner_names, clusterer_names, output_summary)

    manifest_summary = output_summary.get("pipeline_run_manifest.json", {})
    failures = manifest_summary.get("failure_count")
    cleaned = manifest_summary.get("cleaned_result_count")
    clustered = manifest_summary.get("clustered_result_count")

    print(
        "[TRACE] Cleaner coverage completed: "
        f"cleaned={cleaned}, clustered={clustered}, failures={failures}"
    )

    if failures and not args.allow_failures:
        raise SystemExit(
            "[TRACE] Cleaner coverage detected failures. "
            "Use --allow-failures while developing individual cleaner wrappers or environments."
        )


if __name__ == "__main__":
    main()