#!/usr/bin/env python3
"""Cleaning-level analysis built from canonical TRACE result tables."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.results_processing.io import read_csv_rows, write_csv


def _to_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def build_cleaning_analysis(processed_dir: Path, output_dir: Path) -> list[dict]:
    """Build cleaning-level summary rows from canonical cleaning metrics."""
    rows = read_csv_rows(processed_dir / "cleaning_metrics.csv")
    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        grouped[row.get("cleaner", "")].append(row)

    out_rows = []
    for cleaner, items in sorted(grouped.items()):
        runtimes = [
            value for value in (_to_float(item.get("cleaning_runtime_sec")) for item in items)
            if value is not None
        ]

        out_rows.append(
            {
                "cleaner": cleaner,
                "n_runs": len(items),
                "mean_cleaning_runtime_sec": sum(runtimes) / len(runtimes) if runtimes else "",
                "min_cleaning_runtime_sec": min(runtimes) if runtimes else "",
                "max_cleaning_runtime_sec": max(runtimes) if runtimes else "",
            }
        )

    columns = [
        "cleaner",
        "n_runs",
        "mean_cleaning_runtime_sec",
        "min_cleaning_runtime_sec",
        "max_cleaning_runtime_sec",
    ]

    write_csv(output_dir / "cleaning_analysis_summary.csv", out_rows, columns)
    return out_rows

