#!/usr/bin/env python3
"""Synthetic visual-demo data for TRACE.

The demo is explanatory. It is not used as an experimental result.
It recreates the clean / dirty / cleaned comparison used in the paper narrative
with stable, reproducible, English-labeled outputs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DemoDataset:
    """One visual-demo dataset."""

    name: str
    points: np.ndarray
    labels: np.ndarray
    missing_mask: np.ndarray
    outlier_mask: np.ndarray


def _make_clean(seed: int = 42) -> DemoDataset:
    rng = np.random.default_rng(seed)

    centers = np.asarray(
        [
            [-22.0, -26.0],
            [-7.0, 6.0],
            [9.0, 23.0],
            [24.0, -8.0],
        ],
        dtype=float,
    )

    counts = [36, 32, 30, 28]
    points = []
    labels = []

    for label, (center, count) in enumerate(zip(centers, counts)):
        cov = np.asarray([[9.0, 2.0], [2.0, 13.0]], dtype=float)
        sample = rng.multivariate_normal(center, cov, size=count)
        points.append(sample)
        labels.extend([label] * count)

    clean_points = np.vstack(points)
    clean_labels = np.asarray(labels, dtype=int)

    missing_mask = np.zeros(clean_points.shape[0], dtype=bool)
    outlier_mask = np.zeros(clean_points.shape[0], dtype=bool)

    return DemoDataset(
        name="clean",
        points=clean_points,
        labels=clean_labels,
        missing_mask=missing_mask,
        outlier_mask=outlier_mask,
    )


def _make_dirty(clean: DemoDataset, seed: int = 7) -> DemoDataset:
    rng = np.random.default_rng(seed)
    points = clean.points.copy()

    n = points.shape[0]
    missing_idx = rng.choice(n, size=12, replace=False)
    remaining = np.setdiff1d(np.arange(n), missing_idx)
    outlier_idx = rng.choice(remaining, size=10, replace=False)

    points[missing_idx, 1] = np.nan

    # Create two visible outlier bands.
    points[outlier_idx[:5], 0] += rng.normal(82.0, 8.0, size=5)
    points[outlier_idx[:5], 1] += rng.normal(92.0, 10.0, size=5)
    points[outlier_idx[5:], 0] -= rng.normal(72.0, 8.0, size=5)
    points[outlier_idx[5:], 1] -= rng.normal(80.0, 10.0, size=5)

    missing_mask = np.zeros(n, dtype=bool)
    outlier_mask = np.zeros(n, dtype=bool)
    missing_mask[missing_idx] = True
    outlier_mask[outlier_idx] = True

    return DemoDataset(
        name="dirty",
        points=points,
        labels=clean.labels.copy(),
        missing_mask=missing_mask,
        outlier_mask=outlier_mask,
    )


def _column_medians(points: np.ndarray) -> np.ndarray:
    return np.nanmedian(points, axis=0)


def _clip_by_clean_range(points: np.ndarray, clean_points: np.ndarray, factor: float = 1.25) -> np.ndarray:
    clipped = points.copy()
    center = np.nanmedian(clean_points, axis=0)
    span = np.nanpercentile(clean_points, 95, axis=0) - np.nanpercentile(clean_points, 5, axis=0)
    lower = center - factor * span
    upper = center + factor * span

    for j in range(points.shape[1]):
        clipped[:, j] = np.clip(clipped[:, j], lower[j], upper[j])

    return clipped


def _make_statistical_impute(clean: DemoDataset, dirty: DemoDataset) -> DemoDataset:
    points = dirty.points.copy()
    medians = _column_medians(points)

    missing_locations = np.where(np.isnan(points))
    points[missing_locations] = np.take(medians, missing_locations[1])

    # Deliberately weak baseline: impute missing values but leave most outliers.
    return DemoDataset(
        name="statistical_impute",
        points=points,
        labels=clean.labels.copy(),
        missing_mask=dirty.missing_mask.copy(),
        outlier_mask=dirty.outlier_mask.copy(),
    )


def _make_constraint_repair(clean: DemoDataset, dirty: DemoDataset) -> DemoDataset:
    points = dirty.points.copy()
    medians = _column_medians(points)

    missing_locations = np.where(np.isnan(points))
    points[missing_locations] = np.take(medians, missing_locations[1])

    points = _clip_by_clean_range(points, clean.points, factor=1.05)

    return DemoDataset(
        name="constraint_repair",
        points=points,
        labels=clean.labels.copy(),
        missing_mask=dirty.missing_mask.copy(),
        outlier_mask=dirty.outlier_mask.copy(),
    )


def _make_context_repair(clean: DemoDataset, dirty: DemoDataset, seed: int = 17) -> DemoDataset:
    rng = np.random.default_rng(seed)
    points = dirty.points.copy()

    # Explanatory near-oracle repair for the visual demo.
    corrupted = dirty.missing_mask | dirty.outlier_mask
    noise = rng.normal(0.0, 1.2, size=points[corrupted].shape)
    points[corrupted] = clean.points[corrupted] + noise

    return DemoDataset(
        name="context_repair",
        points=points,
        labels=clean.labels.copy(),
        missing_mask=dirty.missing_mask.copy(),
        outlier_mask=dirty.outlier_mask.copy(),
    )


def build_demo_datasets() -> list[DemoDataset]:
    """Build all visual-demo datasets."""
    clean = _make_clean()
    dirty = _make_dirty(clean)
    statistical = _make_statistical_impute(clean, dirty)
    constraint = _make_constraint_repair(clean, dirty)
    context = _make_context_repair(clean, dirty)

    return [clean, dirty, statistical, constraint, context]


def _row_dict(dataset: DemoDataset, index: int) -> dict[str, Any]:
    x, y = dataset.points[index]
    return {
        "dataset": dataset.name,
        "row_id": index,
        "x": "" if np.isnan(x) else float(x),
        "y": "" if np.isnan(y) else float(y),
        "reference_cluster": int(dataset.labels[index]),
        "is_missing": bool(dataset.missing_mask[index]),
        "is_outlier": bool(dataset.outlier_mask[index]),
    }


def write_demo_dataset(path: Path, dataset: DemoDataset) -> None:
    """Write one demo dataset as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "dataset",
        "row_id",
        "x",
        "y",
        "reference_cluster",
        "is_missing",
        "is_outlier",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(dataset.points.shape[0]):
            writer.writerow(_row_dict(dataset, i))


def write_visual_demo_outputs(output_dir: Path) -> dict[str, Any]:
    """Write demo data files and return a manifest."""
    datasets = build_demo_datasets()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    for dataset in datasets:
        path = output_dir / f"{dataset.name}_points.csv"
        write_demo_dataset(path, dataset)
        files[dataset.name] = str(path)

    manifest = {
        "description": "Synthetic TRACE visual demo. Not used as an experimental result.",
        "datasets": files,
        "n_points": int(datasets[0].points.shape[0]),
        "dataset_order": [dataset.name for dataset in datasets],
    }

    manifest_path = output_dir / "visual_demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest

