#!/usr/bin/env python3
"""Utilities for TRACE alpha/weight pre-experiment tables."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    p = Path(path)
    if not p.exists():
        return []

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write CSV rows with stable column order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def find_alpha_metrics_source(audit_csv: Path) -> Path:
    """Find the legacy alpha_metrics.csv file from the Stage 3 audit CSV."""
    rows = read_csv_rows(audit_csv)

    candidates = []
    for row in rows:
        if row.get("category") != "pre_experiment_candidate":
            continue
        if row.get("file_name", "").lower() != "alpha_metrics.csv":
            continue

        root = Path(row.get("legacy_root", ""))
        rel = row.get("relative_path", "")
        path = root / rel
        if path.exists():
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Could not locate legacy alpha_metrics.csv from the audit CSV."
        )

    return candidates[0]


def copy_alpha_metrics(source_csv: Path, output_csv: Path) -> dict[str, Any]:
    """Copy the legacy alpha metrics table into TRACE results/pre_experiment."""
    source = Path(source_csv)
    target = Path(output_csv)

    if not source.exists():
        raise FileNotFoundError(f"Alpha metrics source not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)

    rows = read_csv_rows(target)
    columns = list(rows[0].keys()) if rows else []

    return {
        "source_csv": str(source),
        "output_csv": str(target),
        "row_count": len(rows),
        "columns": columns,
    }


def detect_alpha_column(rows: list[dict[str, str]]) -> str:
    """Detect the alpha column in alpha_metrics.csv."""
    if not rows:
        raise ValueError("Cannot detect alpha column from an empty table.")

    columns = list(rows[0].keys())

    for column in columns:
        lower = column.lower()
        if lower == "alpha" or lower.startswith("alpha"):
            return column

    for column in columns:
        values = [to_float(row.get(column)) for row in rows]
        values = [value for value in values if value is not None]
        if values and min(values) >= 0 and max(values) <= 1:
            return column

    raise ValueError("Could not detect an alpha column.")


def detect_metric_columns(rows: list[dict[str, str]], alpha_column: str) -> list[str]:
    """Detect numeric metric columns to plot against alpha."""
    if not rows:
        return []

    columns = [column for column in rows[0].keys() if column != alpha_column]
    metric_columns = []

    preferred_keywords = [
        "median",
        "variance",
        "var",
        "mean",
        "score",
        "combined",
        "sil",
        "db",
    ]

    for column in columns:
        lower = column.lower()
        values = [to_float(row.get(column)) for row in rows]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            continue

        if any(keyword in lower for keyword in preferred_keywords):
            metric_columns.append(column)

    if metric_columns:
        return metric_columns

    for column in columns:
        values = [to_float(row.get(column)) for row in rows]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            metric_columns.append(column)

    return metric_columns


def load_alpha_metrics(path: Path) -> tuple[list[dict[str, str]], str, list[str]]:
    """Load alpha metrics and return rows, alpha column, and metric columns."""
    rows = read_csv_rows(path)
    alpha_column = detect_alpha_column(rows)
    metric_columns = detect_metric_columns(rows, alpha_column)

    return rows, alpha_column, metric_columns


def build_alpha_metric_summary(path: Path) -> dict[str, Any]:
    """Summarize the alpha metrics table."""
    rows, alpha_column, metric_columns = load_alpha_metrics(path)

    alpha_values = [to_float(row.get(alpha_column)) for row in rows]
    alpha_values = [value for value in alpha_values if value is not None]

    return {
        "path": str(path),
        "row_count": len(rows),
        "alpha_column": alpha_column,
        "metric_columns": metric_columns,
        "alpha_min": min(alpha_values) if alpha_values else "",
        "alpha_max": max(alpha_values) if alpha_values else "",
    }

