#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mode imputation cleaner.

This cleaner fills missing values in a dirty CSV file using simple column-wise
statistics:

- Numeric columns: median value.
- Non-numeric columns: mode value.

The cleaning logic is intentionally kept simple and compatible with the original
Mode baseline. TRACE changes are limited to path handling, CLI options, and
English logging.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mode imputation cleaner.")
    parser.add_argument("--clean_path", required=True, help="Path to clean.csv.")
    parser.add_argument("--dirty_path", required=True, help="Path to the dirty CSV file.")
    parser.add_argument("--task_name", required=True, help="Dataset/task name.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for repaired CSV output. "
            "Default: src/cleaning/Repaired_res/mode/<task_name>."
        ),
    )
    return parser.parse_args()


def repair_with_mode(df_dirty: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values without changing the original algorithmic intent.

    This avoids chained inplace assignments so that the cleaner works with
    pandas Copy-on-Write behavior.
    """
    repaired = df_dirty.copy()

    for col in repaired.columns:
        if not repaired[col].isna().any():
            continue

        if pd.api.types.is_numeric_dtype(repaired[col]):
            median_val = repaired[col].median()
            if pd.notna(median_val):
                repaired[col] = repaired[col].fillna(median_val)
        else:
            mode_values = repaired[col].mode(dropna=True)
            if not mode_values.empty:
                repaired[col] = repaired[col].fillna(mode_values.iloc[0])

    return repaired


def main() -> None:
    args = parse_args()

    dirty_path = Path(args.dirty_path).resolve()
    clean_path = Path(args.clean_path).resolve()

    if not dirty_path.exists():
        raise FileNotFoundError(f"Dirty CSV file not found: {dirty_path}")

    if not clean_path.exists():
        raise FileNotFoundError(f"Clean CSV file not found: {clean_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (
            PROJECT_ROOT
            / "src"
            / "cleaning"
            / "Repaired_res"
            / "mode"
            / args.task_name
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    df_dirty = pd.read_csv(dirty_path, encoding="utf-8-sig")
    repaired = repair_with_mode(df_dirty)

    output_path = output_dir / f"repaired_{args.task_name}.csv"
    repaired.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Repaired data saved to {output_path}")


if __name__ == "__main__":
    main()
