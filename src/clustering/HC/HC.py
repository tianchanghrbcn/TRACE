#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agglomerative clustering (HC) with merge-tree tracking.

This script is used by the original cleaning-clustering pipeline. It keeps the
historical input/output contract:

Input environment variables:
- CSV_FILE_PATH: path to the cleaned CSV file.
- DATASET_ID: legacy dataset id.
- ALGO: cleaning algorithm name.
- OUTPUT_DIR: output directory for clustering results.
- CLEAN_STATE: optional label, usually "raw" or "cleaned".

TRACE-compatible environment variables:
- TRACE_N_TRIALS: number of Optuna trials. Default: 100.
- TRACE_OPTUNA_SEED: optional Optuna sampler seed.
- TRACE_OPTUNA_VERBOSE: set to 1 to show Optuna trial logs.

Outputs:
- repaired_<dataset_id>.txt
- repaired_<dataset_id>_<clean_state>_merge_history.json
- repaired_<dataset_id>_<clean_state>_summary.json
- repaired_<dataset_id>_param_shift.json, if the paired raw/cleaned summary exists.

The objective function is kept compatible with the original implementation:
a harmonic-style combination of normalized Silhouette and Davies-Bouldin scores.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

CSV_FILE_PATH = os.getenv("CSV_FILE_PATH")
DATASET_ID = os.getenv("DATASET_ID")
ALGORITHM_NAME = os.getenv("ALGO", "unknown_cleaner")
CLEAN_STATE = os.getenv("CLEAN_STATE", "raw")

if not CSV_FILE_PATH:
    raise SystemExit("CSV_FILE_PATH is not provided.")

CSV_FILE_PATH = str(Path(CSV_FILE_PATH).resolve())
DATASET_ID = str(DATASET_ID if DATASET_ID is not None else "unknown")

PROJECT_ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()

OUTPUT_DIR_ENV = os.getenv("OUTPUT_DIR")
if OUTPUT_DIR_ENV:
    OUTPUT_DIR = Path(OUTPUT_DIR_ENV).resolve()
else:
    OUTPUT_DIR = (
        PROJECT_ROOT
        / "results"
        / "clustered_data"
        / "HC"
        / ALGORITHM_NAME
        / f"clustered_{DATASET_ID}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = int(os.getenv("TRACE_N_TRIALS", "100"))
OPTUNA_SEED_RAW = os.getenv("TRACE_OPTUNA_SEED")
OPTUNA_SEED = int(OPTUNA_SEED_RAW) if OPTUNA_SEED_RAW not in (None, "") else None

if os.getenv("TRACE_OPTUNA_VERBOSE", "0") != "1":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

# Avoid noisy scikit-learn warnings in smoke runs.
warnings.filterwarnings(
    "ignore",
    message="Attribute `affinity` was deprecated.*",
    category=FutureWarning,
)


# ---------------------------------------------------------------------------
# Input loading and preprocessing
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_FILE_PATH, encoding="utf-8-sig")

# Keep the historical convention: columns whose name contains "id" are excluded.
excluded_columns = [column for column in df.columns if "id" in column.lower()]
feature_columns = df.columns.difference(excluded_columns)
X = df[feature_columns].copy()

for column in X.columns:
    if X[column].dtype in ("object", "category"):
        value_freq = X[column].value_counts(normalize=True)
        X[column] = X[column].map(value_freq)

X = X.dropna()

if X.empty:
    raise SystemExit("No valid rows remain after preprocessing.")

if len(X) < 3:
    raise SystemExit("HC requires at least 3 rows after preprocessing.")

X_scaled = StandardScaler().fit_transform(X)

# Objective weights. gamma is kept for compatibility with the earlier docstring,
# but the historical combined score uses alpha and beta only.
ALPHA = 0.47
BETA = 1.0 - ALPHA
GAMMA = 0.0

global_centroid = X_scaled.mean(axis=0)
SSE_MAX = float(np.sum((X_scaled - global_centroid) ** 2))

start_time = time.time()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_sse(labels: np.ndarray) -> float:
    """Compute SSE for a clustering assignment."""
    sse = 0.0
    for label in np.unique(labels):
        points = X_scaled[labels == label]
        centroid = points.mean(axis=0)
        sse += np.sum((points - centroid) ** 2)
    return float(sse)


def combined_score(db_score: float, sil_score: float, sse: float) -> float:
    """
    Compute the historical combined clustering score.

    Silhouette is mapped from [-1, 1] to [0, 1].
    Davies-Bouldin is mapped from [0, +inf) to (0, 1].
    """
    normalized_sil = (sil_score + 1.0) / 2.0
    normalized_db = 1.0 / (1.0 + db_score)
    eps = 1e-12

    return 1.0 / (
        ALPHA / max(normalized_sil, eps)
        + BETA / max(normalized_db, eps)
    )


def make_agglomerative_model(
    n_clusters: int,
    linkage: str,
    metric: str,
) -> AgglomerativeClustering:
    """
    Create an AgglomerativeClustering model compatible with multiple
    scikit-learn versions.

    scikit-learn 1.2 introduced `metric` and deprecated `affinity`.
    Older versions require `affinity`.
    """
    kwargs: dict[str, Any] = {
        "n_clusters": n_clusters,
        "linkage": linkage,
        "compute_distances": True,
    }

    signature = inspect.signature(AgglomerativeClustering)
    if "metric" in signature.parameters:
        kwargs["metric"] = metric
    else:
        kwargs["affinity"] = metric

    return AgglomerativeClustering(**kwargs)


def run_hc_tracking(
    n_clusters: int,
    linkage: str,
    metric: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    """
    Run HC and collect merge-tree and separation statistics.

    Returns:
        labels, merge_history, core_stats
    """
    model = make_agglomerative_model(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
    )
    labels = model.fit_predict(X_scaled)

    merge_history: list[dict[str, Any]] = []
    if hasattr(model, "children_") and hasattr(model, "distances_"):
        for step, (left, right, distance) in enumerate(
            zip(model.children_[:, 0], model.children_[:, 1], model.distances_),
            start=1,
        ):
            merge_history.append(
                {
                    "step": int(step),
                    "cluster_i": int(left),
                    "cluster_j": int(right),
                    "dist": float(distance),
                }
            )

    distance_matrix = pairwise_distances(X_scaled, metric="euclidean")
    intra_sum = 0.0
    inter_sum = 0.0
    intra_count = 0
    inter_count = 0

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_sum += distance_matrix[i, j]
                intra_count += 1
            else:
                inter_sum += distance_matrix[i, j]
                inter_count += 1

    intra_mean = intra_sum / max(intra_count, 1)
    inter_mean = inter_sum / max(inter_count, 1)

    core_stats = {
        "intra_dist_mean": float(intra_mean),
        "inter_dist_mean": float(inter_mean),
        "ratio_intra_inter": float(intra_mean / (inter_mean + 1e-12)),
    }

    return labels, merge_history, core_stats


# ---------------------------------------------------------------------------
# Optuna search
# ---------------------------------------------------------------------------

optuna_trials: list[dict[str, Any]] = []


def objective(trial: optuna.Trial) -> float:
    """Optuna objective for HC hyperparameter search."""
    n_rows = X_scaled.shape[0]
    max_k = max(5, math.isqrt(n_rows))
    max_k = min(max_k, n_rows - 1)

    if max_k < 2:
        raise optuna.exceptions.TrialPruned()

    min_k = min(5, max_k)
    n_clusters = trial.suggest_int("n_clusters", min_k, max_k)

    linkage = trial.suggest_categorical(
        "linkage",
        ["ward", "complete", "average", "single"],
    )
    metric = trial.suggest_categorical(
        "metric",
        ["euclidean", "manhattan", "cosine"],
    )

    # Ward linkage is only valid with Euclidean distance.
    if linkage == "ward" and metric != "euclidean":
        raise optuna.exceptions.TrialPruned()

    try:
        labels, merge_history, core_stats = run_hc_tracking(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
            raise optuna.exceptions.TrialPruned()

        sil = float(silhouette_score(X_scaled, labels))
        db = float(davies_bouldin_score(X_scaled, labels))
        sse = compute_sse(labels)
        score = combined_score(db, sil, sse)
        h_max = merge_history[-1]["dist"] if merge_history else 0.0

    except optuna.exceptions.TrialPruned:
        raise
    except Exception:
        raise optuna.exceptions.TrialPruned()

    optuna_trials.append(
        {
            "trial_number": int(trial.number),
            "n_clusters": int(n_clusters),
            "linkage": str(linkage),
            "metric": str(metric),
            "combined_score": float(score),
            "silhouette": float(sil),
            "davies_bouldin": float(db),
            "sse": float(sse),
            "n_merge_steps": int(len(merge_history)),
            "h_max": float(h_max),
            **core_stats,
        }
    )

    return score


if OPTUNA_SEED is None:
    sampler = None
else:
    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)

study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

if not optuna_trials:
    raise SystemExit("No valid HC trial completed.")

best_trial = max(optuna_trials, key=lambda item: item["combined_score"])


# ---------------------------------------------------------------------------
# Final model and output
# ---------------------------------------------------------------------------

final_best_k = int(best_trial["n_clusters"])
final_linkage = str(best_trial["linkage"])
final_metric = str(best_trial["metric"])
final_cov_type = f"{final_linkage}-{final_metric}"

labels_final, merge_history, core_stats_final = run_hc_tracking(
    n_clusters=final_best_k,
    linkage=final_linkage,
    metric=final_metric,
)

final_db = float(davies_bouldin_score(X_scaled, labels_final))
final_sil = float(silhouette_score(X_scaled, labels_final))
final_sse = compute_sse(labels_final)
final_combined = combined_score(final_db, final_sil, final_sse)
final_hmax = merge_history[-1]["dist"] if merge_history else 0.0
total_runtime_sec = time.time() - start_time

base_name = Path(CSV_FILE_PATH).stem

# Text output: keep the historical four-line format.
text_output = "\n".join(
    [
        f"Best parameters: n_components={final_best_k}, covariance type={final_cov_type}",
        f"Final Combined Score: {final_combined}",
        f"Final Silhouette Score: {final_sil}",
        f"Final Davies-Bouldin Score: {final_db}",
    ]
)

(OUTPUT_DIR / f"{base_name}.txt").write_text(text_output, encoding="utf-8")

# Merge-tree output.
with (OUTPUT_DIR / f"{base_name}_{CLEAN_STATE}_merge_history.json").open(
    "w",
    encoding="utf-8",
) as fp:
    json.dump(merge_history, fp, indent=4)

summary = {
    "clean_state": CLEAN_STATE,
    "best_k": final_best_k,
    "linkage": final_linkage,
    "metric": final_metric,
    "combined": float(final_combined),
    "silhouette": float(final_sil),
    "davies_bouldin": float(final_db),
    "sse": float(final_sse),
    "sse_max": float(SSE_MAX),
    "weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
    **core_stats_final,
    "n_merge_steps": int(len(merge_history)),
    "h_max": float(final_hmax),
    "n_trials_requested": int(N_TRIALS),
    "n_trials_completed": int(len(optuna_trials)),
    "total_runtime_sec": float(total_runtime_sec),
}

summary_path = OUTPUT_DIR / f"{base_name}_{CLEAN_STATE}_summary.json"
with summary_path.open("w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=4)

# Save trial-level search records for later TRACE/analysis adapters.
trials_path = OUTPUT_DIR / f"{base_name}_{CLEAN_STATE}_optuna_trials.json"
with trials_path.open("w", encoding="utf-8") as fp:
    json.dump(optuna_trials, fp, indent=4)

# Parameter shift output if a paired raw/cleaned summary already exists.
other_state = "cleaned" if CLEAN_STATE == "raw" else "raw"
other_summary_path = OUTPUT_DIR / f"{base_name}_{other_state}_summary.json"

if other_summary_path.exists():
    other = json.loads(other_summary_path.read_text(encoding="utf-8"))
    shift = {
        "dataset_id": DATASET_ID,
        "delta_k": summary["best_k"] - other["best_k"],
        "delta_combined": summary["combined"] - other["combined"],
        "rel_shift": abs(summary["best_k"] - other["best_k"]) / max(other["best_k"], 1),
    }
    with (OUTPUT_DIR / f"{base_name}_param_shift.json").open("w", encoding="utf-8") as fp:
        json.dump(shift, fp, indent=4)

print(f"All files saved in: {OUTPUT_DIR}")
print(f"Program completed in {total_runtime_sec:.2f} sec")