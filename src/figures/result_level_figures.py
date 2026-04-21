#!/usr/bin/env python3
"""Result-level TRACE figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.figures.plot_utils import read_csv_rows, to_float
from src.figures.style import DEFAULT_FIGSIZE, apply_axis_style, save_figure


def plot_combined_score_by_clusterer(tables_dir: Path, output_root: Path) -> dict[str, str]:
    """Plot mean final combined score by clusterer."""
    rows = read_csv_rows(tables_dir / "clustering_analysis_summary.csv")

    labels = []
    values = []
    for row in rows:
        value = to_float(row.get("mean_final_combined_score"))
        if value is None:
            continue
        labels.append(row.get("clusterer", "clusterer"))
        values.append(value)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if values:
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=25)
        apply_axis_style(
            ax,
            "Mean Combined Score by Clusterer",
            "Clusterer",
            "Mean combined score",
        )
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No result-level score data available.", ha="center", va="center")

    return save_figure(fig, output_root, "result_combined_score_by_clusterer")

