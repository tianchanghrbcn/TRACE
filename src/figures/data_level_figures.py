#!/usr/bin/env python3
"""Data-level TRACE figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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



def plot_error_profile_heatmap(processed_dir: Path, output_root: Path) -> dict[str, str]:
    """Plot a compact heatmap of trial-level error descriptors.

    This is the first Stage 3 migration target for legacy error-type heatmap
    figures. It uses canonical trial descriptors and remains stable for smoke
    and full archived results.
    """
    rows = read_csv_rows(processed_dir / "trials.csv")
    metric_names = ["missing_rate", "noise_rate", "error_rate"]

    labels = []
    matrix = []

    for row in rows:
        values = []
        for metric in metric_names:
            value = to_float(row.get(metric))
            values.append(0.0 if value is None else value)

        dataset = row.get("dataset_name", "dataset")
        dataset_id = row.get("dataset_id", "")
        labels.append(f"{dataset}-{dataset_id}")
        matrix.append(values)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if matrix:
        data = np.asarray(matrix, dtype=float)
        image = ax.imshow(data, aspect="auto")
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=25, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title("Trial-Level Error Profile")
        ax.set_xlabel("Error descriptor")
        ax.set_ylabel("Trial")
        fig.colorbar(image, ax=ax, label="Rate (%)")
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No trial-level error descriptors available.",
            ha="center",
            va="center",
            wrap=True,
        )

    return save_figure(fig, output_root, "data_error_profile_heatmap")
