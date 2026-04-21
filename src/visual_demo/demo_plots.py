#!/usr/bin/env python3
"""Reviewer-facing visual-demo plots for TRACE."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.visual_demo.demo_data import build_demo_datasets, write_visual_demo_outputs


DISPLAY_NAMES = {
    "clean": "Clean data",
    "dirty": "Dirty data",
    "statistical_impute": "Statistical imputation",
    "constraint_repair": "Constraint/statistical repair",
    "context_repair": "Context-aware repair",
}


def _prepare_points_for_plot(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace missing values only for display."""
    display = points.copy()
    missing = np.isnan(display[:, 1])
    display[missing, 1] = 0.0
    return display, missing


def _simple_kmeans(points: np.ndarray, k: int = 4, seed: int = 0, max_iter: int = 50) -> np.ndarray:
    """Small deterministic k-means used only for visual-demo plotting."""
    rng = np.random.default_rng(seed)
    display, _ = _prepare_points_for_plot(points)

    indices = rng.choice(display.shape[0], size=k, replace=False)
    centers = display[indices].copy()

    labels = np.zeros(display.shape[0], dtype=int)

    for _ in range(max_iter):
        dists = np.linalg.norm(display[:, None, :] - centers[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)

        new_centers = []
        for j in range(k):
            if np.any(new_labels == j):
                new_centers.append(display[new_labels == j].mean(axis=0))
            else:
                new_centers.append(display[rng.integers(display.shape[0])])

        new_centers = np.asarray(new_centers)

        if np.array_equal(labels, new_labels):
            break

        labels = new_labels
        centers = new_centers

    return labels


def _set_common_limits(ax) -> None:
    ax.set_xlim(-120, 120)
    ax.set_ylim(-130, 130)


def _plot_data_panel(ax, dataset) -> None:
    display, missing = _prepare_points_for_plot(dataset.points)

    ax.scatter(display[:, 0], display[:, 1], s=18, alpha=0.85)

    if missing.any():
        ax.scatter(
            display[missing, 0],
            display[missing, 1],
            marker="x",
            s=45,
            linewidths=1.2,
        )

    if dataset.outlier_mask.any():
        outlier_points = display[dataset.outlier_mask]
        ax.scatter(
            outlier_points[:, 0],
            outlier_points[:, 1],
            marker="^",
            s=35,
            alpha=0.9,
        )

    ax.set_title(DISPLAY_NAMES.get(dataset.name, dataset.name))
    ax.set_xlabel("Centered age")
    ax.set_ylabel("Centered income")
    _set_common_limits(ax)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)


def _plot_cluster_panel(ax, dataset) -> None:
    display, missing = _prepare_points_for_plot(dataset.points)
    labels = _simple_kmeans(dataset.points, k=4, seed=13)

    ax.scatter(display[:, 0], display[:, 1], c=labels, s=18, alpha=0.85)

    if missing.any():
        ax.scatter(
            display[missing, 0],
            display[missing, 1],
            marker="x",
            s=45,
            linewidths=1.2,
        )

    ax.set_title(DISPLAY_NAMES.get(dataset.name, dataset.name))
    ax.set_xlabel("Centered age")
    ax.set_ylabel("Centered income")
    _set_common_limits(ax)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)


def _save(fig, figure_dir: Path, stem: str) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)

    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"

    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def plot_data_rewrite_demo(figure_dir: Path) -> dict[str, str]:
    """Plot clean, dirty, and repaired data views."""
    datasets = build_demo_datasets()

    fig, axes = plt.subplots(1, len(datasets), figsize=(17.5, 3.7), sharex=True, sharey=True)

    for ax, dataset in zip(axes, datasets):
        _plot_data_panel(ax, dataset)

    fig.suptitle(
        "Clean, Dirty, and Repaired Data Views",
        y=1.04,
        fontsize=12,
    )

    return _save(fig, figure_dir, "data_rewrite_demo")


def plot_clustering_demo(figure_dir: Path) -> dict[str, str]:
    """Plot clustering outcomes for clean, dirty, and repaired data."""
    datasets = build_demo_datasets()

    fig, axes = plt.subplots(1, len(datasets), figsize=(17.5, 3.7), sharex=True, sharey=True)

    for ax, dataset in zip(axes, datasets):
        _plot_cluster_panel(ax, dataset)

    fig.suptitle(
        "Clustering Outcomes after Different Data Treatments",
        y=1.04,
        fontsize=12,
    )

    return _save(fig, figure_dir, "clustering_demo")


def build_visual_demo(output_data_dir: Path, output_figure_dir: Path) -> dict[str, Any]:
    """Build visual-demo data and figures."""
    data_manifest = write_visual_demo_outputs(output_data_dir)

    figure_outputs = {
        "data_rewrite_demo": plot_data_rewrite_demo(output_figure_dir),
        "clustering_demo": plot_clustering_demo(output_figure_dir),
    }

    manifest = {
        "description": "Reviewer-facing TRACE visual demo with English labels.",
        "data_manifest": data_manifest,
        "figure_outputs": figure_outputs,
    }

    output_figure_dir.mkdir(parents=True, exist_ok=True)
    figure_manifest_path = output_figure_dir / "visual_demo_figure_manifest.json"
    figure_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest

