#!/usr/bin/env python3
"""Generate the first migrated TRACE figure batch."""

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

from src.figures.data_level_figures import plot_error_profile_heatmap
from src.figures.result_level_figures import (
    plot_score_by_error_rate,
    plot_top_configuration_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate first migrated TRACE figure batch.")
    parser.add_argument("--processed-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("figures"))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs = {
        "result_top_configuration_scores": plot_top_configuration_scores(
            args.processed_dir,
            args.output_root,
            top_k=args.top_k,
        ),
        "result_score_by_error_rate": plot_score_by_error_rate(
            args.processed_dir,
            args.output_root,
        ),
        "data_error_profile_heatmap": plot_error_profile_heatmap(
            args.processed_dir,
            args.output_root,
        ),
    }

    manifest_path = args.output_root / "migrated_figure_batch1_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    print("[TRACE] First migrated TRACE figure batch was generated.")


if __name__ == "__main__":
    main()

