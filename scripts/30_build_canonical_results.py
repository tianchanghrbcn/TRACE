#!/usr/bin/env python3
"""Build canonical TRACE result tables."""

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


from src.results_processing.build_tables import build_canonical_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TRACE canonical result tables.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--schema", type=Path, default=Path("configs/results_schema.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_canonical_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        schema_path=args.schema,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[TRACE] Canonical result tables were built.")


if __name__ == "__main__":
    main()

