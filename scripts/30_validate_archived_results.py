#!/usr/bin/env python3
"""Validate raw result files for TRACE Mode A replay."""

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


from src.results_processing.validators import validate_result_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TRACE archived result inputs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_result_inputs(args.results_dir, require_all=args.require_all)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["missing"]:
        print("[TRACE] Some result files are missing. This is acceptable for early Stage 3 scaffolding.")
    else:
        print("[TRACE] Archived result validation passed.")


if __name__ == "__main__":
    main()

