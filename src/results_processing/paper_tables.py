#!/usr/bin/env python3
"""Build lightweight paper-table summaries from canonical TRACE tables."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.results_processing.io import read_csv_rows, write_csv


def _mean(values: List[float]):
    if not values:
        return ""
    return sum(values) / len(values)


def _numeric(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def build_cleaner_summary(processed_dir: Path, output_dir: Path) -> List[dict]:
    """Build a cleaner-level summary table."""
    rows = read_csv_rows(processed_dir / "cleaning_metrics.csv")
    groups: Dict[str, List[dict]] = defaultdict(list)

    for row in rows:
        groups[row.get("cleaner", "")].append(row)

    out_rows = []
    for cleaner, items in sorted(groups.items()):
        runtimes = [
            _numeric(item.get("cleaning_runtime_sec"))
            for item in items
            if item.get("cleaning_runtime_sec") not in ("", None)
        ]
        runtimes = [x for x in runtimes if x == x]

        out_rows.append(
            {
                "cleaner": cleaner,
                "n_runs": len(items),
                "mean_cleaning_runtime_sec": _mean(runtimes),
            }
        )

    write_csv(
        output_dir / "summary_by_cleaner.csv",
        out_rows,
        ["cleaner", "n_runs", "mean_cleaning_runtime_sec"],
    )

    return out_rows


def build_clusterer_summary(processed_dir: Path, output_dir: Path) -> List[dict]:
    """Build a clusterer-level summary table."""
    rows = read_csv_rows(processed_dir / "result_metrics.csv")
    groups: Dict[str, List[dict]] = defaultdict(list)

    for row in rows:
        groups[row.get("clusterer", "")].append(row)

    out_rows = []
    for clusterer, items in sorted(groups.items()):
        combined = [
            _numeric(item.get("final_combined_score"))
            for item in items
            if item.get("final_combined_score") not in ("", None)
        ]
        combined = [x for x in combined if x == x]

        runtimes = [
            _numeric(item.get("clustering_runtime_sec"))
            for item in items
            if item.get("clustering_runtime_sec") not in ("", None)
        ]
        runtimes = [x for x in runtimes if x == x]

        out_rows.append(
            {
                "clusterer": clusterer,
                "n_runs": len(items),
                "mean_final_combined_score": _mean(combined),
                "mean_clustering_runtime_sec": _mean(runtimes),
            }
        )

    write_csv(
        output_dir / "summary_by_clusterer.csv",
        out_rows,
        [
            "clusterer",
            "n_runs",
            "mean_final_combined_score",
            "mean_clustering_runtime_sec",
        ],
    )

    return out_rows


def build_run_counts(processed_dir: Path, output_dir: Path) -> List[dict]:
    """Build a simple row-count summary for canonical tables."""
    table_names = [
        "trials",
        "cleaning_metrics",
        "result_metrics",
        "best_configs",
        "process_metrics",
        "parameter_shifts",
    ]

    out_rows = []
    for table in table_names:
        rows = read_csv_rows(processed_dir / f"{table}.csv")
        out_rows.append({"table": table, "n_rows": len(rows)})

    write_csv(output_dir / "run_counts.csv", out_rows, ["table", "n_rows"])
    return out_rows


def build_paper_tables(processed_dir: Path, output_dir: Path) -> dict:
    """Build initial paper-table summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaner_summary = build_cleaner_summary(processed_dir, output_dir)
    clusterer_summary = build_clusterer_summary(processed_dir, output_dir)
    run_counts = build_run_counts(processed_dir, output_dir)

    return {
        "summary_by_cleaner": len(cleaner_summary),
        "summary_by_clusterer": len(clusterer_summary),
        "run_counts": len(run_counts),
    }

