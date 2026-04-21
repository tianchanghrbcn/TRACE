#!/usr/bin/env python3
"""Result-level TRACE figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.figures.plot_utils import read_csv_rows, to_float
from src.figures.style import DEFAULT_FIGSIZE, apply_axis_style, save_figure, shorten_labels


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



def plot_top_configuration_scores(processed_dir: Path, output_root: Path, top_k: int = 10) -> dict[str, str]:
    """Plot top cleaner-clusterer configurations by final combined score.

    This is the TRACE migration target for the old top-10 configuration bar
    chart. The migrated version reads canonical Stage 3 result tables instead
    of legacy analysis_result files.
    """
    rows = read_csv_rows(processed_dir / "result_metrics.csv")

    scored = []
    for row in rows:
        score = to_float(row.get("final_combined_score"))
        if score is None:
            continue
        cleaner = row.get("cleaner", "")
        clusterer = row.get("clusterer", "")
        dataset_id = row.get("dataset_id", "")
        label = f"{cleaner} / {clusterer} / {dataset_id}"
        scored.append((label, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    scored = scored[:top_k]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if scored:
        labels = shorten_labels([label for label, _ in scored], max_len=30)
        values = [value for _, value in scored]
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=25)
        apply_axis_style(
            ax,
            "Top Cleaner-Clusterer Configurations",
            "Configuration",
            "Final combined score",
        )
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No final combined-score data available.",
            ha="center",
            va="center",
            wrap=True,
        )

    return save_figure(fig, output_root, "result_top_configuration_scores")


def plot_score_by_error_rate(processed_dir: Path, output_root: Path) -> dict[str, str]:
    """Plot final combined score against injected error rate.

    This is a lightweight result-level migration target for the old error-rate
    curve figures. Full CEGR and breakpoint analysis will be added after Mode A
    archived result tables expose EDR and binned full-search summaries.
    """
    trials = read_csv_rows(processed_dir / "trials.csv")
    results = read_csv_rows(processed_dir / "result_metrics.csv")

    error_by_dataset = {
        str(row.get("dataset_id", "")): to_float(row.get("error_rate"))
        for row in trials
    }

    points = []
    for row in results:
        dataset_id = str(row.get("dataset_id", ""))
        error_rate = error_by_dataset.get(dataset_id)
        score = to_float(row.get("final_combined_score"))
        if error_rate is None or score is None:
            continue
        points.append((error_rate, score))

    points.sort(key=lambda item: item[0])

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if points:
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(xs, ys, marker="o")
        apply_axis_style(
            ax,
            "Combined Score by Error Rate",
            "Injected error rate (%)",
            "Final combined score",
        )
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No score-by-error-rate data available.",
            ha="center",
            va="center",
            wrap=True,
        )

    return save_figure(fig, output_root, "result_score_by_error_rate")
