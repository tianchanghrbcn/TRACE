#!/usr/bin/env python3
"""Combined cleaning-clustering summary tables."""

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


def build_cleaning_clustering_summary(processed_dir: Path, output_dir: Path) -> list[dict]:
    """Build summary rows grouped by cleaner and clusterer."""
    rows = read_csv_rows(processed_dir / "result_metrics.csv")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in rows:
        key = (row.get("cleaner", ""), row.get("clusterer", ""))
        grouped[key].append(row)

    out_rows = []
    for (cleaner, clusterer), items in sorted(grouped.items()):
        combined = [
            value for value in (_to_float(item.get("final_combined_score")) for item in items)
            if value is not None
        ]

        runtimes = [
            value for value in (_to_float(item.get("clustering_runtime_sec")) for item in items)
            if value is not None
        ]

        out_rows.append(
            {
                "cleaner": cleaner,
                "clusterer": clusterer,
                "n_runs": len(items),
                "mean_final_combined_score": sum(combined) / len(combined) if combined else "",
                "mean_clustering_runtime_sec": sum(runtimes) / len(runtimes) if runtimes else "",
            }
        )

    columns = [
        "cleaner",
        "clusterer",
        "n_runs",
        "mean_final_combined_score",
        "mean_clustering_runtime_sec",
    ]

    write_csv(output_dir / "cleaning_clustering_summary.csv", out_rows, columns)
    return out_rows

