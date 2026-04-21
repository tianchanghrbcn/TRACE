#!/usr/bin/env python3
"""Clustering-level analysis built from canonical TRACE result tables."""

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


def _mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def build_clustering_analysis(processed_dir: Path, output_dir: Path) -> list[dict]:
    """Build clusterer-level summary rows from canonical result metrics."""
    rows = read_csv_rows(processed_dir / "result_metrics.csv")
    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        grouped[row.get("clusterer", "")].append(row)

    out_rows = []
    for clusterer, items in sorted(grouped.items()):
        combined = [
            value for value in (_to_float(item.get("final_combined_score")) for item in items)
            if value is not None
        ]
        silhouette = [
            value for value in (_to_float(item.get("final_silhouette_score")) for item in items)
            if value is not None
        ]
        davies_bouldin = [
            value for value in (_to_float(item.get("final_davies_bouldin_score")) for item in items)
            if value is not None
        ]
        runtimes = [
            value for value in (_to_float(item.get("clustering_runtime_sec")) for item in items)
            if value is not None
        ]

        out_rows.append(
            {
                "clusterer": clusterer,
                "n_runs": len(items),
                "mean_final_combined_score": _mean(combined),
                "mean_final_silhouette_score": _mean(silhouette),
                "mean_final_davies_bouldin_score": _mean(davies_bouldin),
                "mean_clustering_runtime_sec": _mean(runtimes),
            }
        )

    columns = [
        "clusterer",
        "n_runs",
        "mean_final_combined_score",
        "mean_final_silhouette_score",
        "mean_final_davies_bouldin_score",
        "mean_clustering_runtime_sec",
    ]

    write_csv(output_dir / "clustering_analysis_summary.csv", out_rows, columns)
    return out_rows

