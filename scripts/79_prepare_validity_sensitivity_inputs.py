#!/usr/bin/env python3
"""Prepare input summary workbooks for validity-sensitivity checks.

This script stages paper-exact summary workbooks into:

    analysis/validity_sensitivity/inputs/analysis_results/

It does not write into results/analysis_results, because that directory belongs
to generated pipeline outputs and should not be polluted by legacy paper inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASKS = ["beers", "flights", "hospital", "rayyan"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare validity-sensitivity input workbooks.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/validity_sensitivity/inputs/analysis_results"),
    )
    parser.add_argument(
        "--source-roots",
        nargs="*",
        type=Path,
        default=[
            Path("analysis/paper_exact/analysis_summaries/results/analysis_results"),
            Path("analysis/paper_generated/paper_table_workspace/results/analysis_results"),
            Path("results/analysis_results"),
            Path(r"E:\algorithm paper\AutoMLClustering_full\results\analysis_results"),
            Path(r"E:\algorithm paper\AutoMLClustering\results\analysis_results"),
        ],
    )
    return parser.parse_args()


def find_workbook(task: str, source_roots: list[Path]) -> Path | None:
    filename = f"{task}_summary.xlsx"

    for root in source_roots:
        candidate = root / filename
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for task in TASKS:
        src = find_workbook(task, args.source_roots)

        if src is None:
            missing.append(task)
            rows.append(
                {
                    "task": task,
                    "status": "MISSING",
                    "source": "",
                    "destination": str(args.output_dir / f"{task}_summary.xlsx"),
                    "sha256": "",
                    "size_bytes": 0,
                }
            )
            continue

        dst = args.output_dir / f"{task}_summary.xlsx"
        shutil.copy2(src, dst)

        rows.append(
            {
                "task": task,
                "status": "COPIED",
                "source": str(src),
                "destination": str(dst),
                "sha256": sha256_file(dst),
                "size_bytes": dst.stat().st_size,
            }
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not missing else "FAIL",
        "output_dir": str(args.output_dir),
        "source_roots": [str(root) for root in args.source_roots],
        "copied_count": sum(1 for row in rows if row["status"] == "COPIED"),
        "missing": missing,
        "rows": rows,
    }

    manifest_path = args.output_dir.parent / "validity_sensitivity_input_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(
        {
            "status": report["status"],
            "copied_count": report["copied_count"],
            "missing": missing,
            "output_dir": str(args.output_dir),
            "manifest": str(manifest_path),
        },
        indent=2,
        ensure_ascii=False,
    ))

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
