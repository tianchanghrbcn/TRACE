#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate feature vectors for dirty CSV files.

This script computes the dataset-level features used by the original
cleaning-clustering pipeline, including missing rate, anomaly/noise rate,
total error rate, row count, column count, and the fixed K value.

Default behavior is kept compatible with the original pipeline:

1. Optionally run the error-injection scripts.
2. Scan datasets under datasets/train.
3. Write results/eigenvectors.json.

TRACE additions:
- CLI arguments for data directory, output file, dataset subset, and dirty-id subset.
- --skip-injection for smoke tests that use already generated dirty tables.
- Path handling based on the TRACE repository root.
- English comments and logs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd


SCRIPT_VERSION = "eigenvectors/trace-compatible-v2.3"

PROJECT_ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()

DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "train"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "results" / "eigenvectors.json"
DEFAULT_INJECT_SCRIPT = Path(__file__).resolve().parent / "inject_all_errors_advanced.py"

K_VALUE = 5

# Primary key column used for aligning dirty and clean tables.
# It can be either an integer column index or a column name.
PK_COL: Union[int, str] = 0

# Alignment policy. These defaults preserve the previous robust fallback behavior.
ALIGN_VERBOSE = True
ALLOW_POSITIONAL_FALLBACK = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate eigenvector-style feature records for dirty CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing dataset folders. Default: datasets/train.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON path. Default: results/eigenvectors.json.",
    )
    parser.add_argument(
        "--skip-injection",
        action="store_true",
        help="Skip error injection and only scan existing dirty CSV files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset subset, for example: --datasets beers flights.",
    )
    parser.add_argument(
        "--dirty-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional dirty-file ids, for example: --dirty-ids 1 2 3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to the error-injection script.",
    )
    parser.add_argument(
        "--inject-script",
        type=Path,
        default=DEFAULT_INJECT_SCRIPT,
        help="Path to inject_all_errors_advanced.py.",
    )
    return parser.parse_args()


def run_inject_scripts(
    data_dir: Path,
    inject_script: Path,
    datasets: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> None:
    """
    Run the error-injection script for each selected dataset.

    The original implementation hard-coded four os.system commands. This version
    keeps the same behavior but builds commands from paths and runs them through
    subprocess for clearer error reporting.
    """
    if datasets is None:
        datasets = ["beers", "flights", "hospital", "rayyan"]

    inject_script = inject_script.resolve()
    if not inject_script.exists():
        raise FileNotFoundError(f"Injection script not found: {inject_script}")

    print("===== Running error injection =====")
    for index, dataset_name in enumerate(datasets, start=1):
        dataset_dir = data_dir / dataset_name
        clean_path = dataset_dir / "clean.csv"

        if not clean_path.exists():
            print(f"[WARN] clean.csv not found, skip injection for {dataset_name}: {clean_path}")
            continue

        command = [
            "python",
            str(inject_script),
            "--input",
            str(clean_path),
            "--output",
            str(dataset_dir),
            "--task_name",
            dataset_name,
            "--seed",
            str(seed),
        ]

        print(f"[{index}] Running: {' '.join(command)}")
        completed = subprocess.run(command, text=True)
        if completed.returncode != 0:
            print(f"[WARN] Injection command returned {completed.returncode}: {dataset_name}")
        else:
            print(f"[INFO] Injection completed: {dataset_name}")

    print("===== Error injection completed =====\n")


def _drop_unnamed_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop temporary columns created by saving DataFrame indices."""
    if df.empty:
        return df
    keep = ~df.columns.astype(str).str.startswith("Unnamed")
    return df.loc[:, keep].copy()


def _strip_pk(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without the first column. Kept for compatibility."""
    return df.iloc[:, 1:].copy() if df.shape[1] > 1 else df.copy()


def _resolve_pk_name(
    df_dirty: pd.DataFrame,
    df_clean: pd.DataFrame,
    pk_col: Union[int, str],
) -> Optional[str]:
    """
    Resolve the primary-key column name.

    If pk_col is an integer, both tables must have the same column name at that
    position. If pk_col is a string, both tables must contain that column.
    """
    if isinstance(pk_col, int):
        if pk_col < df_dirty.shape[1] and pk_col < df_clean.shape[1]:
            dirty_name = df_dirty.columns[pk_col]
            clean_name = df_clean.columns[pk_col]
            return dirty_name if dirty_name == clean_name else None
        return None

    pk_name = str(pk_col)
    if pk_name in df_dirty.columns and pk_name in df_clean.columns:
        return pk_name
    return None


def _align_for_compare(
    df_dirty: pd.DataFrame,
    df_clean: pd.DataFrame,
    pk_col: Union[int, str] = PK_COL,
    dedup_keep: str = "first",
    verbose: bool = ALIGN_VERBOSE,
    allow_positional_fallback: bool = ALLOW_POSITIONAL_FALLBACK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align dirty and clean tables for cell-level comparison.

    The function first tries primary-key alignment. If that fails, it falls back
    to common-column and positional alignment. The returned DataFrames have the
    same row/column labels and exclude the primary-key column when key alignment
    succeeds.
    """
    dirty = _drop_unnamed_cols(df_dirty).copy()
    clean = _drop_unnamed_cols(df_clean).copy()

    pk_name = _resolve_pk_name(dirty, clean, pk_col)
    if pk_name is not None:
        if not dirty[pk_name].is_unique:
            if verbose:
                print(f"[WARN] Dirty primary key `{pk_name}` has duplicates; keep={dedup_keep}.")
            dirty = dirty.drop_duplicates(subset=[pk_name], keep=dedup_keep)

        if not clean[pk_name].is_unique:
            if verbose:
                print(f"[WARN] Clean primary key `{pk_name}` has duplicates; keep={dedup_keep}.")
            clean = clean.drop_duplicates(subset=[pk_name], keep=dedup_keep)

        dirty = dirty.set_index(pk_name, drop=True)
        clean = clean.set_index(pk_name, drop=True)

        common_cols = [col for col in clean.columns if col in dirty.columns]
        extra_dirty = [col for col in dirty.columns if col not in clean.columns]
        extra_clean = [col for col in clean.columns if col not in dirty.columns]

        if verbose and (extra_dirty or extra_clean):
            if extra_dirty:
                print(f"[INFO] Ignore dirty-only columns: {extra_dirty}")
            if extra_clean:
                print(f"[INFO] Ignore clean-only columns: {extra_clean}")

        dirty = dirty[common_cols]
        clean = clean[common_cols]

        common_idx = dirty.index.intersection(clean.index)
        if len(common_idx) > 0:
            dirty = dirty.loc[common_idx].sort_index()
            clean = clean.loc[common_idx].sort_index()

            dirty, clean = dirty.align(clean, join="inner", axis=None)
            dirty = dirty.sort_index(axis=0).sort_index(axis=1)
            clean = clean.sort_index(axis=0).sort_index(axis=1)
            return dirty, clean

        if verbose:
            print("[INFO] Primary keys have no overlap; use positional fallback.")

    if allow_positional_fallback:
        common_cols = dirty.columns.intersection(clean.columns)
        if verbose and len(common_cols) == 0:
            print("[INFO] No common columns; positional fallback returns empty aligned tables.")

        dirty_aligned = dirty[common_cols].reset_index(drop=True)
        clean_aligned = clean[common_cols].reset_index(drop=True)

        n_rows = min(len(dirty_aligned), len(clean_aligned))
        dirty_aligned = dirty_aligned.iloc[:n_rows].copy()
        clean_aligned = clean_aligned.iloc[:n_rows].copy()

        dirty_aligned, clean_aligned = dirty_aligned.align(
            clean_aligned, join="inner", axis=None
        )
        dirty_aligned = dirty_aligned.sort_index(axis=0).sort_index(axis=1)
        clean_aligned = clean_aligned.sort_index(axis=0).sort_index(axis=1)
        return dirty_aligned, clean_aligned

    raise ValueError("Failed to align by primary key and positional fallback is disabled.")


def compute_missing_rate(df_dirty: pd.DataFrame, df_clean: pd.DataFrame) -> float:
    """
    Compute the missing-value rate.

    The denominator is the number of non-missing cells in the clean table.
    The numerator counts cells that were non-missing in clean but became NaN
    in the dirty table.
    """
    dirty, clean = _align_for_compare(df_dirty, df_clean, pk_col=PK_COL)

    clean_non_missing = ~clean.isna()
    denominator = int(clean_non_missing.values.sum())
    if denominator == 0:
        return 0.0

    missing_mask = clean_non_missing.to_numpy() & dirty.isna().to_numpy()
    numerator = int(missing_mask.sum())
    return numerator / denominator


def compute_noise_rate(df_dirty: pd.DataFrame, df_clean: pd.DataFrame) -> float:
    """
    Compute the anomaly/noise rate.

    The denominator is the number of non-missing cells in the clean table.
    The numerator counts cells that were non-missing in clean, are non-missing
    in dirty, but have a different value.
    """
    dirty, clean = _align_for_compare(df_dirty, df_clean, pk_col=PK_COL)

    clean_non_missing = ~clean.isna()
    denominator = int(clean_non_missing.values.sum())
    if denominator == 0:
        return 0.0

    clean_non_missing_arr = clean_non_missing.to_numpy()
    dirty_non_missing_arr = ~dirty.isna().to_numpy()
    value_diff_arr = dirty.to_numpy() != clean.to_numpy()

    anomaly_mask = clean_non_missing_arr & dirty_non_missing_arr & value_diff_arr
    numerator = int(anomaly_mask.sum())
    return numerator / denominator


def process_single_file(
    csv_path: Path,
    dataset_name: str,
    dataset_id: int,
    df_clean: pd.DataFrame,
) -> dict:
    """
    Read one dirty CSV file and return its feature vector.

    clean.csv is intentionally skipped by the caller and should not appear in
    eigenvectors.json.
    """
    file_name = csv_path.name
    if file_name == "clean.csv":
        return {}

    df_dirty = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing_rate = compute_missing_rate(df_dirty, df_clean)
    noise_rate = compute_noise_rate(df_dirty, df_clean)
    error_rate = (missing_rate + noise_rate) * 100.0

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "csv_file": file_name,
        "error_rate": error_rate,
        "K": K_VALUE,
        "missing_rate": missing_rate,
        "noise_rate": noise_rate,
        "m": df_dirty.shape[1],
        "n": df_dirty.shape[0],
    }


def _dirty_id_from_name(dataset_name: str, file_name: str) -> Optional[int]:
    """
    Extract the dirty id from a file name such as beers_1.csv.

    Returns None if the file does not follow the expected pattern.
    """
    prefix = f"{dataset_name}_"
    if not file_name.startswith(prefix) or not file_name.endswith(".csv"):
        return None

    raw_id = file_name[len(prefix) : -len(".csv")]
    try:
        return int(raw_id)
    except ValueError:
        return None


def _file_allowed_by_dirty_ids(
    dataset_name: str,
    file_name: str,
    dirty_ids: Optional[set[int]],
) -> bool:
    if dirty_ids is None:
        return True

    dirty_id = _dirty_id_from_name(dataset_name, file_name)
    return dirty_id in dirty_ids


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    output_file = args.output_file.resolve()

    selected_datasets = args.datasets if args.datasets else None
    selected_dirty_ids = set(args.dirty_ids) if args.dirty_ids else None

    print(f"== Running {Path(__file__).name} · {SCRIPT_VERSION} ==")
    print(f"[TRACE] Project root: {PROJECT_ROOT}")
    print(f"[TRACE] Data directory: {data_dir}")
    print(f"[TRACE] Output file: {output_file}")

    if not args.skip_injection:
        run_inject_scripts(
            data_dir=data_dir,
            inject_script=args.inject_script,
            datasets=selected_datasets,
            seed=args.seed,
        )
    else:
        print("[TRACE] Skip error injection; use existing dirty CSV files.")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory does not exist: {data_dir}")

    all_vectors: list[dict] = []
    dataset_id_counter = 0

    dataset_names = sorted(
        name
        for name in os.listdir(data_dir)
        if (data_dir / name).is_dir()
    )

    if selected_datasets is not None:
        selected = set(selected_datasets)
        dataset_names = [name for name in dataset_names if name in selected]

    for dataset_name in dataset_names:
        dataset_dir = data_dir / dataset_name
        clean_path = dataset_dir / "clean.csv"

        if not clean_path.is_file():
            print(f"[WARN] Missing clean.csv for dataset `{dataset_name}`; skip.")
            continue

        df_clean = pd.read_csv(clean_path, encoding="utf-8-sig")

        # Keep the original lexicographic ordering for compatibility with the
        # historical dataset_id assignment.
        csv_files = sorted(
            file_name
            for file_name in os.listdir(dataset_dir)
            if file_name.endswith(".csv")
        )

        if not csv_files:
            print(f"[WARN] No CSV files found for dataset `{dataset_name}`; skip.")
            continue

        for csv_file in csv_files:
            if csv_file == "clean.csv":
                continue

            if not _file_allowed_by_dirty_ids(dataset_name, csv_file, selected_dirty_ids):
                continue

            csv_path = dataset_dir / csv_file
            vector = process_single_file(
                csv_path=csv_path,
                dataset_name=dataset_name,
                dataset_id=dataset_id_counter,
                df_clean=df_clean,
            )

            if not vector:
                continue

            all_vectors.append(vector)
            print(
                f"[{dataset_id_counter}] Processed {dataset_name}/{csv_file} "
                f"=> error_rate={vector['error_rate']:.2f}%"
            )
            dataset_id_counter += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_vectors, f, indent=4, ensure_ascii=False)

    print(f"\n[TRACE] Completed. Wrote {len(all_vectors)} records to: {output_file}")


if __name__ == "__main__":
    main()

