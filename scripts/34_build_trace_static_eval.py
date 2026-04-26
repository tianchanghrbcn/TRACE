#!/usr/bin/env python3
"""Build TRACE static entry-screening evaluation tables from archived results."""

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

from src.results_processing.trace_static_eval import evaluate_trace_static


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TRACE static entry-screening metrics from archived Stage 3 result tables. "
            "No baseline re-run and no trial archives are required."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("results/processed"),
        help="Directory containing canonical result tables such as trials.csv and result_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/processed/trace_static"),
        help="Output directory for TRACE static evaluation tables and paper snippets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/trace_static.yaml"),
        help="TRACE static evaluation config.",
    )
    parser.add_argument(
        "--dataset-ids",
        nargs="*",
        default=None,
        help="Optional subset of dataset_id values to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = evaluate_trace_static(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        dataset_ids=args.dataset_ids,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[TRACE] Static entry-screening evaluation completed successfully.")


if __name__ == "__main__":
    main()
