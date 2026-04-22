#!/usr/bin/env python3
"""Workbook helpers for TRACE paper-exact replay."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return SHA-256 digest for a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with stable UTF-8 settings."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_sheet_name(name: str) -> str:
    """Return a valid Excel sheet name."""
    value = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
    value = value[:31]
    return value or "sheet"


def safe_stem(name: str) -> str:
    """Normalize a file stem for generated output names."""
    value = name.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "output"

