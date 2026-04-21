#!/usr/bin/env python3
"""Input and output helpers for TRACE result replay."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, List


def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file with UTF-8/BOM tolerance."""
    p = Path(path)
    if not p.exists():
        return default

    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Write JSON with stable formatting."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: Path) -> List[dict]:
    """Read a CSV file as a list of dictionaries."""
    p = Path(path)
    if not p.exists():
        return []

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], columns: List[str]) -> None:
    """Write rows as CSV with a fixed column order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def ensure_list(value: Any, name: str) -> List[Any]:
    """Ensure that a JSON value is a list."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list.")
    return value


def to_number_or_empty(value: Any) -> Any:
    """Convert numeric-looking values to float; otherwise return an empty string."""
    if value in (None, ""):
        return ""
    try:
        return float(value)
    except Exception:
        return ""

