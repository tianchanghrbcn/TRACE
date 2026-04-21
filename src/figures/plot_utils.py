#!/usr/bin/env python3
"""Shared helpers for TRACE figure generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    p = Path(path)
    if not p.exists():
        return []

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def numeric_points(rows: list[dict[str, str]], label_col: str, value_col: str) -> list[tuple[str, float]]:
    """Extract label-value pairs with numeric values."""
    points = []

    for row in rows:
        label = row.get(label_col, "")
        value = to_float(row.get(value_col))
        if value is None:
            continue
        points.append((label, value))

    return points


def combined_label(row: dict[str, str], columns: list[str], sep: str = " / ") -> str:
    """Build a combined label from several columns."""
    values = [row.get(column, "") for column in columns]
    values = [value for value in values if value]
    return sep.join(values)

