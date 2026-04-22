#!/usr/bin/env python3
"""Validate Mode A paper-exact archive outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_GROUP_MIN_COUNTS = {
    "raw_results": 4,
    "analysis_summaries": 4,
    "paper_figures_latex": 10,
    "paper_table_scripts": 5,
    "paper_figure_scripts": 5,
    "paper_tex": 1,
}


REQUIRED_FILE_HINTS = [
    "beers_summary.xlsx",
    "flights_summary.xlsx",
    "hospital_summary.xlsx",
    "rayyan_summary.xlsx",
    "latex_paper.tex",
    "TRACE.pdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Mode A paper-exact archive.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("analysis/paper_exact/mode_a_paper_exact_archive_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper_exact/mode_a_paper_exact_validation_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Mode A archive manifest not found: {args.manifest}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    copied = manifest.get("copied", [])
    missing = manifest.get("missing", [])

    group_counts = Counter(row.get("selection_group", "") for row in copied)
    copied_paths = [row.get("archive_path", "") for row in copied]

    failures = []

    if missing:
        failures.append(f"{len(missing)} selected files were missing during copy.")

    for group, min_count in REQUIRED_GROUP_MIN_COUNTS.items():
        actual = group_counts.get(group, 0)
        if actual < min_count:
            failures.append(f"Group `{group}` has {actual} files; expected at least {min_count}.")

    for hint in REQUIRED_FILE_HINTS:
        if not any(hint.lower() in path.lower() for path in copied_paths):
            failures.append(f"Required file hint not found in archive: {hint}")

    for row in copied:
        archive_path = Path(row["archive_path"])
        if not archive_path.exists():
            failures.append(f"Archive path missing after build: {archive_path}")
        if row.get("replay_path"):
            replay_path = Path(row["replay_path"])
            if not replay_path.exists():
                failures.append(f"Replay path missing after build: {replay_path}")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "manifest": str(args.manifest),
        "copied_count": len(copied),
        "missing_count": len(missing),
        "selection_group_counts": dict(group_counts),
        "required_group_min_counts": REQUIRED_GROUP_MIN_COUNTS,
        "required_file_hints": REQUIRED_FILE_HINTS,
        "failures": failures,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[TRACE] Mode A validation report written to: {args.output}")
    print(f"[TRACE] Mode A paper-exact status: {report['status']}")

    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

