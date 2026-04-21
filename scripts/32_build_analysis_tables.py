#!/usr/bin/env python3
"""Build Stage 3 analysis tables from canonical result tables."""

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

from src.results_processing.cleaning_analysis import build_cleaning_analysis
from src.results_processing.clustering_analysis import build_clustering_analysis
from src.results_processing.summary_tables import build_cleaning_clustering_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TRACE Stage 3 analysis tables.")
    parser.add_argument("--processed-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "cleaning_analysis_summary": len(build_cleaning_analysis(args.processed_dir, args.output_dir)),
        "clustering_analysis_summary": len(build_clustering_analysis(args.processed_dir, args.output_dir)),
        "cleaning_clustering_summary": len(
            build_cleaning_clustering_summary(args.processed_dir, args.output_dir)
        ),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[TRACE] Stage 3 analysis tables were built.")


if __name__ == "__main__":
    main()

