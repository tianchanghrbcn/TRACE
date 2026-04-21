#!/usr/bin/env python3
"""Build TRACE pre-experiment outputs from audited legacy artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TRACE_PROJECT_ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()

if str(TRACE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_PROJECT_ROOT))

from src.pre_experiment.alpha_metrics import (
    build_alpha_metric_summary,
    copy_alpha_metrics,
    find_alpha_metrics_source,
)
from src.pre_experiment.alpha_plots import build_alpha_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TRACE pre-experiment outputs.")
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("results/processed/legacy_audit_files.csv"),
        help="Legacy audit CSV.",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help="Optional explicit path to legacy alpha_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pre_experiment"),
        help="Directory for pre-experiment tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/pre_experiment"),
        help="Directory for generated pre-experiment figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_csv = args.source_csv or find_alpha_metrics_source(args.audit_csv)
    output_csv = args.output_dir / "alpha_metrics.csv"

    copy_report = copy_alpha_metrics(source_csv, output_csv)
    summary = build_alpha_metric_summary(output_csv)
    figure_manifest = build_alpha_plots(output_csv, args.figure_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "copy_report": copy_report,
        "alpha_metric_summary": summary,
        "figure_manifest": figure_manifest,
    }

    manifest_path = args.output_dir / "pre_experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[TRACE] Pre-experiment outputs were built.")


if __name__ == "__main__":
    main()

