#!/usr/bin/env python3
"""Data-level TRACE figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.figures.plot_utils import read_csv_rows, to_float
from src.figures.style import DEFAULT_FIGSIZE, apply_axis_style, save_figure


def plot_error_rate_by_trial(processed_dir: Path, output_root: Path) -> dict[str, str]:
    """Plot error rate by dirty dataset instance."""
    rows = read_csv_rows(processed_dir / "trials.csv")

    labels = []
    values = []
    for row in rows:
        value = to_float(row.get("error_rate"))
        if value is None:
            continue
        dataset = row.get("dataset_name", "dataset")
        dataset_id = row.get("dataset_id", "")
        labels.append(f"{dataset}-{dataset_id}")
        values.append(value)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if values:
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=25)
        apply_axis_style(
            ax,
            "Injected Error Rate by Trial",
            "Trial",
            "Error rate (%)",
        )
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No trial-level error-rate data available.", ha="center", va="center")

    return save_figure(fig, output_root, "data_error_rate_by_trial")

