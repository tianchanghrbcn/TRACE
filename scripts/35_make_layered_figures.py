#!/usr/bin/env python3
"""Generate Stage 3.4.2 layered TRACE figure scaffolds."""

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

from src.figures.data_level_figures import plot_error_rate_by_trial
from src.figures.framework_figures import plot_trace_layer_diagram
from src.figures.process_level_figures import plot_process_metric_coverage
from src.figures.result_level_figures import plot_combined_score_by_clusterer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TRACE layered figure scaffolds.")
    parser.add_argument("--processed-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--output-root", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs = {
        "data_error_rate_by_trial": plot_error_rate_by_trial(args.processed_dir, args.output_root),
        "process_metric_coverage": plot_process_metric_coverage(args.processed_dir, args.output_root),
        "result_combined_score_by_clusterer": plot_combined_score_by_clusterer(args.tables_dir, args.output_root),
        "framework_trace_layers": plot_trace_layer_diagram(args.output_root),
    }

    manifest_path = args.output_root / "layered_figure_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    print("[TRACE] Layered TRACE figure scaffolds were generated.")


if __name__ == "__main__":
    main()

