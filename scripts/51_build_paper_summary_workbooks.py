#!/usr/bin/env python3
"""Build generated paper summary workbooks from archived analysis CSV files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

TRACE_PROJECT_ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()

if str(TRACE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_PROJECT_ROOT))

import pandas as pd

from src.paper_replay.workbook_utils import safe_sheet_name, sha256_file, write_json


DATASETS = ["beers", "flights", "hospital", "rayyan"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generated paper summary workbooks.")
    parser.add_argument(
        "--analysis-csv-root",
        type=Path,
        default=Path("analysis/paper_exact/analysis_csv/results/analysis_results"),
    )
    parser.add_argument(
        "--archived-summary-root",
        type=Path,
        default=Path("analysis/paper_exact/analysis_summaries/results/analysis_results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/paper_generated/summary_workbooks"),
    )
    return parser.parse_args()


def find_dataset_csvs(root: Path, dataset: str) -> list[Path]:
    if not root.exists():
        return []

    direct = sorted(root.glob(f"{dataset}_*.csv"))

    # Also include selected stats files only once in the global index, not per dataset.
    return direct


def add_dataframe_sheet(writer, path: Path, sheet_name: str) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    safe_name = safe_sheet_name(sheet_name)
    df.to_excel(writer, sheet_name=safe_name, index=False)
    return {
        "source_csv": str(path),
        "sheet_name": safe_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }


def build_dataset_workbook(dataset: str, csvs: list[Path], output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sheets = []
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        overview = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "source_csv_count": len(csvs),
                    "generated_by": "scripts/51_build_paper_summary_workbooks.py",
                }
            ]
        )
        overview.to_excel(writer, sheet_name="overview", index=False)

        for csv_path in csvs:
            stem = csv_path.stem
            if stem.startswith(dataset + "_"):
                stem = stem[len(dataset) + 1:]
            sheets.append(add_dataframe_sheet(writer, csv_path, stem))

    return {
        "dataset": dataset,
        "output_xlsx": str(output_path),
        "sha256": sha256_file(output_path),
        "source_csv_count": len(csvs),
        "sheets": sheets,
    }


def build_index_workbook(dataset_reports: list[dict], output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for report in dataset_reports:
        rows.append(
            {
                "dataset": report["dataset"],
                "output_xlsx": report["output_xlsx"],
                "source_csv_count": report["source_csv_count"],
                "sha256": report["sha256"],
            }
        )

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="summary_index", index=False)

    return {
        "output_xlsx": str(output_path),
        "sha256": sha256_file(output_path),
        "rows": len(rows),
    }


def main() -> None:
    args = parse_args()

    if not args.analysis_csv_root.exists():
        raise FileNotFoundError(f"Analysis CSV root not found: {args.analysis_csv_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_reports = []
    missing = []

    for dataset in DATASETS:
        csvs = find_dataset_csvs(args.analysis_csv_root, dataset)
        if not csvs:
            missing.append(dataset)
            continue

        output_path = args.output_dir / f"{dataset}_summary.xlsx"
        dataset_reports.append(build_dataset_workbook(dataset, csvs, output_path))

    index_report = build_index_workbook(
        dataset_reports,
        args.output_dir / "paper_summary_index.xlsx",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_csv_root": str(args.analysis_csv_root),
        "archived_summary_root": str(args.archived_summary_root),
        "output_dir": str(args.output_dir),
        "datasets": dataset_reports,
        "index_workbook": index_report,
        "missing_datasets": missing,
    }

    manifest_path = args.output_dir.parent / "generated_summary_manifest.json"
    write_json(manifest_path, manifest)

    print(json.dumps({
        "generated_dataset_workbooks": len(dataset_reports),
        "missing_datasets": missing,
        "manifest": str(manifest_path),
    }, indent=2, ensure_ascii=False))
    print("[TRACE] Generated paper summary workbooks were built.")


if __name__ == "__main__":
    main()

