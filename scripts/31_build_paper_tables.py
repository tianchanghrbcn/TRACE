#!/usr/bin/env python3
"""Build initial paper-table summaries from canonical TRACE tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import os
import sys

TRACE_PROJECT_ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()

if str(TRACE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_PROJECT_ROOT))


from src.results_processing.paper_tables import build_paper_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TRACE paper-table summaries.")
    parser.add_argument("--processed-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_paper_tables(args.processed_dir, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[TRACE] Paper-table summaries were built.")


if __name__ == "__main__":
    main()

