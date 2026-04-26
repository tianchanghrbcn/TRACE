#!/usr/bin/env python3
"""CLI wrapper for TRACE Stage 4 replay.

Run from the repository root, for example:
    python .\\scripts\\30_replay_trace.py --results-dir .\\results\\raw --config .\\configs\\trace.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def _parse_dataset_ids(values: Optional[Iterable[str]]) -> Optional[List[int]]:
    if not values:
        return None
    out: List[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            out.append(int(token))
    return out or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay TRACE faithfully from saved pipeline logs.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Repository root. Defaults to the parent directory of this script.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing eigenvectors.json / cleaned_results.json / clustered_results.json. "
        "You can pass either .\\results or .\\results\\raw.",
    )
    parser.add_argument(
        "--config",
        default="configs/trace.yaml",
        help="TRACE config path, relative to project root unless absolute.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for Stage 4 replay artifacts. Defaults to configs/trace.yaml -> trace.paths.output_dir.",
    )
    parser.add_argument(
        "--dataset-ids",
        nargs="*",
        default=None,
        help="Optional dataset ids, e.g. --dataset-ids 0 1 2 or --dataset-ids 0,1,2",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else script_path.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.analysis.trace_replay import replay_trace

    result = replay_trace(
        project_root=project_root,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        config_path=Path(args.config),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        dataset_ids=_parse_dataset_ids(args.dataset_ids),
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("[TRACE] Stage 4 replay completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
