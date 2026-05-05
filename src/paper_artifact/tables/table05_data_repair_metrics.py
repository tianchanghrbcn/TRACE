from __future__ import annotations

"""
Stage 3 builder for paper Table 5.

This module is intended to live at:
    src/paper_artifact/tables/table05_data_repair_metrics.py

It adapts the original data-repair metrics table script.

Input
-----
Four paper summary workbooks under ctx.input_root:

    beers_summary.xlsx
    flights_summary.xlsx
    hospital_summary.xlsx
    rayyan_summary.xlsx

Each workbook must contain at least:

    task_name, dataset_id, error_rate, missing, anomaly,
    cleaning_method, cluster_method, EDR, F1

Output
------
Exactly one CSV file:

    <ctx.output_dir>/table_5/table5_data_repair_metrics.csv

No TeX, XLSX, long CSV, or auxiliary outputs are generated.

Aggregation logic follows the original script:
1. M = pure missing corruption: missing > 0 and anomaly == 0.
2. E = pure anomaly corruption: anomaly > 0 and missing == 0.
3. Mixed-error rows are excluded.
4. Since EDR/F1 repeat across clustering algorithms, values are first
   deduplicated by dirty instance and cleaning method.
5. The table reports medians across pure error-rate settings within each
   dataset, error family, and cleaning method.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.paper_artifact.io import ArtifactResult, BuildContext


ARTIFACT = {
    "id": "table05_data_repair_metrics",
    "paper_id": "Table 5",
    "label": "Table 5: Cell-level data repair metrics",
    "description": "Build Table 5 from the four *_summary.xlsx workbooks under results/.",
    "enabled": True,
}


DATASET_ORDER = ["beers", "flights", "hospital", "rayyan"]

METHOD_ORDER = [
    "Mode",
    "Baran",
    "HoloClean",
    "BigDansing",
    "BoostClean",
    "Horizon",
    "SCAReD",
    "Unified",
    "UniClean",
]

METHOD_ALIASES = {
    "mode": "Mode",
    "modeimpute": "Mode",
    "modeimputation": "Mode",
    "baran": "Baran",
    "holoclean": "HoloClean",
    "holo": "HoloClean",
    "bigdansing": "BigDansing",
    "boostclean": "BoostClean",
    "horizon": "Horizon",
    "scared": "SCAReD",
    "unified": "Unified",
    "uniclean": "UniClean",
    "groundtruth": "GroundTruth",
    "groundtruthclean": "GroundTruth",
    "gt": "GroundTruth",
    "oracle": "GroundTruth",
}

COLUMN_ALIASES = {
    "task_name": ["task_name", "task", "dataset", "dataset_name"],
    "dataset_id": ["dataset_id", "dirty_id", "instance_id", "id"],
    "error_rate": ["error_rate", "q_tot", "qtot", "total_error_rate"],
    "missing": ["missing", "q_missing", "q_miss", "qmiss", "missing_rate"],
    "anomaly": ["anomaly", "q_anomaly", "q_anom", "qanom", "anomaly_rate"],
    "cleaning_method": ["cleaning_method", "cleaner", "method", "cleaning"],
    "cluster_method": ["cluster_method", "clusterer", "clustering_method", "algorithm"],
    "EDR": ["EDR", "edr", "Error Drop Rate", "error_drop_rate"],
    "F1": ["F1", "f1", "F1 Score", "F1_score", "f1_score", "F_1"],
}

# Internal labels are M/A, but the paper table displays anomaly rows as E.
ERROR_DISPLAY = {
    "M": "M",
    "A": "E",
}


def _norm_key(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def normalize_method(name: object) -> str:
    key = _norm_key(name)
    return METHOD_ALIASES.get(key, str(name).strip())


def find_column(df: pd.DataFrame, canonical: str) -> str | None:
    norm_to_col = {_norm_key(c): c for c in df.columns}
    for alias in COLUMN_ALIASES[canonical]:
        key = _norm_key(alias)
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def require_column(df: pd.DataFrame, canonical: str, file_path: Path) -> str:
    col = find_column(df, canonical)
    if col is None:
        raise ValueError(
            f"Missing required column for {canonical!r} in {file_path}. "
            f"Available columns: {list(df.columns)}"
        )
    return col


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_task_from_path(path: Path) -> str:
    stem = path.stem.lower()
    stem = stem.replace("_summary", "").replace("-summary", "")
    for task in DATASET_ORDER:
        if task in stem:
            return task
    return stem


def read_summary_workbooks(input_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    frames = []
    inputs: list[Path] = []

    for task in DATASET_ORDER:
        candidates = [
            input_dir / f"{task}_summary.xlsx",
            input_dir / f"{task}-summary.xlsx",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"Missing summary workbook for task={task!r}. "
                f"Expected one of: {candidates}"
            )

        df0 = pd.read_excel(path)

        col_task = find_column(df0, "task_name")
        col_dataset_id = require_column(df0, "dataset_id", path)
        col_error_rate = require_column(df0, "error_rate", path)
        col_missing = require_column(df0, "missing", path)
        col_anomaly = require_column(df0, "anomaly", path)
        col_cleaning = require_column(df0, "cleaning_method", path)
        col_cluster = find_column(df0, "cluster_method")
        col_edr = require_column(df0, "EDR", path)
        col_f1 = require_column(df0, "F1", path)

        task_from_file = infer_task_from_path(path)

        tmp = pd.DataFrame()
        if col_task is not None:
            tmp["task_name"] = df0[col_task].astype(str).str.strip().str.lower()
        else:
            tmp["task_name"] = task_from_file

        tmp.loc[tmp["task_name"].isna() | (tmp["task_name"] == ""), "task_name"] = task_from_file
        tmp["task_name"] = tmp["task_name"].str.lower().replace({"beer": "beers"})

        tmp["dataset_id"] = df0[col_dataset_id]
        tmp["error_rate"] = to_numeric(df0[col_error_rate])
        tmp["missing"] = to_numeric(df0[col_missing])
        tmp["anomaly"] = to_numeric(df0[col_anomaly])
        tmp["cleaning_method_raw"] = df0[col_cleaning].astype(str).str.strip()
        tmp["cleaning_method"] = tmp["cleaning_method_raw"].map(normalize_method)
        tmp["cluster_method"] = df0[col_cluster].astype(str).str.strip() if col_cluster else ""
        tmp["EDR"] = to_numeric(df0[col_edr])
        tmp["F1"] = to_numeric(df0[col_f1])
        tmp["source_file"] = path.name

        frames.append(tmp)
        inputs.append(path)

    out = pd.concat(frames, ignore_index=True)
    return out, inputs


def classify_error_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    eps = 1e-12
    miss_pos = out["missing"].fillna(0) > eps
    anom_pos = out["anomaly"].fillna(0) > eps

    out["error_family"] = pd.NA
    out.loc[miss_pos & ~anom_pos, "error_family"] = "M"
    out.loc[anom_pos & ~miss_pos, "error_family"] = "A"

    out = out[out["error_family"].isin(["M", "A"])].copy()
    return out


def compute_repair_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["cleaning_method"].isin(METHOD_ORDER)].copy()
    df = classify_error_family(df)

    if df.empty:
        raise RuntimeError("No pure missing or pure anomaly rows found after filtering.")

    # Repair metrics are repeated across clustering algorithms. Deduplicate first
    # so methods with more clustering rows are not overweighted.
    dedup_keys = [
        "task_name",
        "dataset_id",
        "error_rate",
        "missing",
        "anomaly",
        "error_family",
        "cleaning_method",
    ]

    dedup = (
        df.groupby(dedup_keys, dropna=False, as_index=False)
        .agg(
            EDR=("EDR", "median"),
            F1=("F1", "median"),
            n_repeated_rows=("EDR", "size"),
        )
    )

    long = (
        dedup.groupby(["task_name", "error_family", "cleaning_method"], as_index=False)
        .agg(
            EDR=("EDR", "median"),
            F1=("F1", "median"),
            n_dirty_instances=("dataset_id", "nunique"),
            n_rows_after_dedup=("EDR", "size"),
        )
    )

    long["task_name"] = pd.Categorical(long["task_name"], categories=DATASET_ORDER, ordered=True)
    long["error_family"] = pd.Categorical(long["error_family"], categories=["M", "A"], ordered=True)
    long["cleaning_method"] = pd.Categorical(long["cleaning_method"], categories=METHOD_ORDER, ordered=True)
    long = long.sort_values(["task_name", "error_family", "cleaning_method"]).reset_index(drop=True)
    return long


def fmt_num(v: float | int | None) -> str:
    if v is None or pd.isna(v):
        return "–"
    x = float(v)
    if abs(x) < 0.0005:
        x = 0.0
    return f"{x:.3f}"


def make_paper_table_csv(long: pd.DataFrame) -> pd.DataFrame:
    lookup: dict[tuple[str, str, str, str], float] = {}
    for _, row in long.iterrows():
        task = str(row["task_name"])
        err = str(row["error_family"])
        method = str(row["cleaning_method"])
        lookup[(task, err, method, "EDR")] = row["EDR"]
        lookup[(task, err, method, "F1")] = row["F1"]

    rows = []

    for task in DATASET_ORDER:
        first_task_row = True

        for err in ["M", "A"]:
            first_error_row = True

            for metric in ["EDR", "F1"]:
                rec = {
                    "Dataset": task if first_task_row else "",
                    "Error": ERROR_DISPLAY[err] if first_error_row else "",
                    "Metric": metric,
                }

                for method in METHOD_ORDER:
                    value = lookup.get((task, err, method, metric), np.nan)
                    rec[method] = fmt_num(value)

                rows.append(rec)

                first_task_row = False
                first_error_row = False

    return pd.DataFrame(rows, columns=["Dataset", "Error", "Metric", *METHOD_ORDER])


def build(ctx: BuildContext) -> ArtifactResult:
    out_dir = ctx.output_dir / "table_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep output directory minimal.
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()

    df, inputs = read_summary_workbooks(ctx.input_root)
    long = compute_repair_long(df)
    table = make_paper_table_csv(long)

    out_path = out_dir / "table5_data_repair_metrics.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")

    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="success",
        outputs=[out_path],
        inputs=inputs,
        message=f"Built Table 5 CSV under {out_dir}.",
        metadata={
            "output_subdir": "table_5",
            "expected_output_count": 1,
            "actual_output_count": 1,
            "datasets": DATASET_ORDER,
            "methods": METHOD_ORDER,
        },
    )
