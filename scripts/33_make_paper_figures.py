#!/usr/bin/env python3
"""Generate Stage 3 paper-figure scaffolds from TRACE tables."""

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

from src.figures.paper_figures import build_paper_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TRACE paper-figure scaffolds.")
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--output-root", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_paper_figures(args.tables_dir, args.output_root)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[TRACE] Stage 3 paper-figure scaffolds were generated.")


if __name__ == "__main__":
    main()

