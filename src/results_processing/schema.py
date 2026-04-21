#!/usr/bin/env python3
"""Schema helpers for TRACE canonical result tables."""

from __future__ import annotations

from pathlib import Path
from typing import List


DEFAULT_SCHEMA_PATH = Path("configs/results_schema.yaml")


def load_schema(schema_path: Path = DEFAULT_SCHEMA_PATH) -> dict:
    """Load the YAML schema file."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configs/results_schema.yaml.") from exc

    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid schema file: {path}")

    return data


def table_columns(schema: dict, table_name: str) -> List[str]:
    """Return the declared columns for a canonical table."""
    tables = schema.get("tables", {})
    if table_name not in tables:
        raise KeyError(f"Unknown table in schema: {table_name}")

    columns = tables[table_name].get("columns", [])
    if not isinstance(columns, list):
        raise ValueError(f"Invalid column list for table: {table_name}")

    return [str(col) for col in columns]


def table_names(schema: dict) -> List[str]:
    """Return all canonical table names."""
    tables = schema.get("tables", {})
    return list(tables.keys())

