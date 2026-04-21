#!/usr/bin/env python3
"""Process-level TRACE figures."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from src.figures.plot_utils import read_csv_rows
from src.figures.style import DEFAULT_FIGSIZE, apply_axis_style, save_figure


def plot_process_metric_coverage(processed_dir: Path, output_root: Path) -> dict[str, str]:
    """Plot the number of extracted process metrics by clusterer."""
    rows = read_csv_rows(processed_dir / "process_metrics.csv")
    counter = Counter(row.get("clusterer", "unknown") for row in rows)

    labels = list(counter.keys())
    values = [counter[label] for label in labels]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if values:
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=25)
        apply_axis_style(
            ax,
            "Extracted Process Metrics by Clusterer",
            "Clusterer",
            "Metric count",
        )
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No process-level metrics available.", ha="center", va="center")

    return save_figure(fig, output_root, "process_metric_coverage")

