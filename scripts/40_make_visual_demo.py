#!/usr/bin/env python3
"""Build reviewer-facing TRACE visual demo."""

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

from src.visual_demo.demo_plots import build_visual_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TRACE visual demo.")
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=Path("results/visual_demo"),
    )
    parser.add_argument(
        "--output-figure-dir",
        type=Path,
        default=Path("figures/visual_demo"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = build_visual_demo(
        output_data_dir=args.output_data_dir,
        output_figure_dir=args.output_figure_dir,
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[TRACE] Visual demo was generated.")


if __name__ == "__main__":
    main()

