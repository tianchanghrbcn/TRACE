#!/usr/bin/env python3
"""Build first-batch TRACE paper-figure scaffolds from Stage 3 tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from src.figures.plot_utils import combined_label, numeric_points, read_csv_rows, to_float
from src.figures.style import DEFAULT_FIGSIZE, apply_axis_style, save_figure, shorten_labels


@dataclass
class FigureRecord:
    """One generated figure record."""

    name: str
    description: str
    files: dict[str, str]


def _placeholder_figure(title: str, message: str):
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_title(title)
    return fig


def _bar_figure(
    points: list[tuple[str, float]],
    title: str,
    xlabel: str,
    ylabel: str,
    horizontal: bool = False,
):
    if not points:
        return _placeholder_figure(title, "No numeric data are available for this figure.")

    labels = shorten_labels([label for label, _ in points])
    values = [value for _, value in points]

    if horizontal:
        fig_height = max(DEFAULT_FIGSIZE[1], 0.35 * len(points) + 1.2)
        fig, ax = plt.subplots(figsize=(DEFAULT_FIGSIZE[0], fig_height))
        ax.barh(labels, values)
        ax.invert_yaxis()
    else:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=25)

    apply_axis_style(ax, title, xlabel, ylabel)
    return fig


def plot_cleaner_runtime(tables_dir: Path, output_root: Path) -> FigureRecord:
    """Plot mean cleaning runtime by cleaner."""
    rows = read_csv_rows(tables_dir / "cleaning_analysis_summary.csv")

    if not rows:
        rows = read_csv_rows(tables_dir / "summary_by_cleaner.csv")
        label_col = "cleaner"
        value_col = "mean_cleaning_runtime_sec"
    else:
        label_col = "cleaner"
        value_col = "mean_cleaning_runtime_sec"

    points = numeric_points(rows, label_col, value_col)
    fig = _bar_figure(
        points,
        title="Mean Cleaning Runtime by Cleaner",
        xlabel="Cleaner",
        ylabel="Runtime (seconds)",
    )
    files = save_figure(fig, output_root, "cleaner_runtime_summary")

    return FigureRecord(
        name="cleaner_runtime_summary",
        description="Mean cleaning runtime by cleaner.",
        files=files,
    )


def plot_clusterer_quality(tables_dir: Path, output_root: Path) -> FigureRecord:
    """Plot mean final combined score by clusterer."""
    rows = read_csv_rows(tables_dir / "clustering_analysis_summary.csv")

    if not rows:
        rows = read_csv_rows(tables_dir / "summary_by_clusterer.csv")
        value_col = "mean_final_combined_score"
    else:
        value_col = "mean_final_combined_score"

    points = numeric_points(rows, "clusterer", value_col)
    fig = _bar_figure(
        points,
        title="Mean Final Combined Score by Clusterer",
        xlabel="Clusterer",
        ylabel="Combined score",
    )
    files = save_figure(fig, output_root, "clusterer_quality_summary")

    return FigureRecord(
        name="clusterer_quality_summary",
        description="Mean final combined clustering score by clusterer.",
        files=files,
    )


def plot_cleaner_clusterer_quality(tables_dir: Path, output_root: Path) -> FigureRecord:
    """Plot mean final combined score by cleaner-clusterer pair."""
    rows = read_csv_rows(tables_dir / "cleaning_clustering_summary.csv")

    points: list[tuple[str, float]] = []
    for row in rows:
        value = to_float(row.get("mean_final_combined_score"))
        if value is None:
            continue
        label = combined_label(row, ["cleaner", "clusterer"])
        points.append((label, value))

    points = points[:20]

    fig = _bar_figure(
        points,
        title="Mean Final Combined Score by Cleaner-Clusterer Pair",
        xlabel="Combined score",
        ylabel="Cleaner / Clusterer",
        horizontal=True,
    )
    files = save_figure(fig, output_root, "cleaner_clusterer_quality_summary")

    return FigureRecord(
        name="cleaner_clusterer_quality_summary",
        description="Mean final combined score by cleaner-clusterer pair.",
        files=files,
    )


def build_paper_figures(tables_dir: Path, output_root: Path) -> dict:
    """Build the first batch of Stage 3 paper-figure scaffolds."""
    tables_dir = Path(tables_dir)
    output_root = Path(output_root)

    records = [
        plot_cleaner_runtime(tables_dir, output_root),
        plot_clusterer_quality(tables_dir, output_root),
        plot_cleaner_clusterer_quality(tables_dir, output_root),
    ]

    manifest = {
        "figures": [
            {
                "name": record.name,
                "description": record.description,
                "files": record.files,
            }
            for record in records
        ]
    }

    manifest_path = output_root / "figure_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest

