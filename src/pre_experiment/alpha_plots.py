#!/usr/bin/env python3
"""English alpha/weight pre-experiment plots for TRACE."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from src.pre_experiment.alpha_metrics import load_alpha_metrics, to_float


def _safe_stem(name: str) -> str:
    value = name.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "metric"


def _pretty_metric_name(name: str) -> str:
    value = name.replace("_", " ").replace("-", " ").strip()
    return value[:1].upper() + value[1:]


def _save_figure(fig, figure_dir: Path, stem: str) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)

    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"

    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def plot_alpha_metric(
    metrics_csv: Path,
    figure_dir: Path,
    metric_column: str,
    output_stem: str | None = None,
) -> dict[str, str]:
    """Plot one metric against alpha."""
    rows, alpha_column, _ = load_alpha_metrics(metrics_csv)

    points = []
    for row in rows:
        alpha = to_float(row.get(alpha_column))
        metric = to_float(row.get(metric_column))
        if alpha is None or metric is None:
            continue
        points.append((alpha, metric))

    points.sort(key=lambda item: item[0])

    stem = output_stem or f"alpha_vs_{_safe_stem(metric_column)}"

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    if points:
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(xs, ys, marker="o")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_xlabel("Alpha")
        ax.set_ylabel(_pretty_metric_name(metric_column))
        ax.set_title(f"Alpha vs. {_pretty_metric_name(metric_column)}")
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"No numeric data available for metric: {metric_column}",
            ha="center",
            va="center",
            wrap=True,
        )

    return _save_figure(fig, figure_dir, stem)


def build_alpha_plots(metrics_csv: Path, figure_dir: Path) -> dict:
    """Build English alpha pre-experiment figures from alpha_metrics.csv."""
    rows, _, metric_columns = load_alpha_metrics(metrics_csv)

    outputs = {}
    selected_columns = []

    for column in metric_columns:
        lower = column.lower()
        if "median" in lower or "variance" in lower or "var" in lower:
            selected_columns.append(column)

    if not selected_columns:
        selected_columns = metric_columns[:2]

    for column in selected_columns:
        lower = column.lower()
        if "median" in lower:
            stem = "alpha_vs_median"
        elif "variance" in lower or "var" in lower:
            stem = "alpha_vs_variance"
        else:
            stem = f"alpha_vs_{_safe_stem(column)}"

        outputs[stem] = {
            "metric_column": column,
            "files": plot_alpha_metric(metrics_csv, figure_dir, column, stem),
        }

    manifest = {
        "metrics_csv": str(metrics_csv),
        "row_count": len(rows),
        "generated_figures": outputs,
    }

    manifest_path = figure_dir / "pre_experiment_figure_manifest.json"
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest

