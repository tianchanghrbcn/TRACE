#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBSCAN clustering with core / border / noise statistics.

Input environment variables:
- CSV_FILE_PATH: path to the cleaned CSV file.
- DATASET_ID: legacy dataset id.
- ALGO: cleaning algorithm name.
- OUTPUT_DIR: output directory for clustering results.
- CLEAN_STATE: optional state label, usually "raw" or "cleaned".

TRACE-compatible environment variables:
- TRACE_PROJECT_ROOT: repository root.
- TRACE_N_TRIALS: number of Optuna trials. Default: 150.
- TRACE_OPTUNA_SEED: optional Optuna sampler seed.
- TRACE_OPTUNA_VERBOSE: set to 1 to show Optuna trial logs.

Outputs:
- repaired_<dataset_id>.txt
- repaired_<dataset_id>_<clean_state>_core_stats.json
- repaired_<dataset_id>_<clean_state>_optuna_trials.json
- repaired_<dataset_id>_param_shift.json, if paired raw/cleaned stats exist.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler


METHOD_NAME = "DBSCAN"
ALPHA = 0.47
BETA = 1.0 - ALPHA
GAMMA = 0.0


def getenv_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is not provided.")
    return value


CSV_FILE_PATH = str(Path(getenv_required("CSV_FILE_PATH")).resolve())
DATASET_ID = str(os.getenv("DATASET_ID", "unknown"))
ALGORITHM_NAME = os.getenv("ALGO", "unknown_cleaner")
CLEAN_STATE = os.getenv("CLEAN_STATE", "cleaned")

PROJECT_ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()

OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        str(PROJECT_ROOT / "results" / "clustered_data" / METHOD_NAME / ALGORITHM_NAME / f"clustered_{DATASET_ID}"),
    )
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = int(os.getenv("TRACE_N_TRIALS", "150"))
OPTUNA_SEED_RAW = os.getenv("TRACE_OPTUNA_SEED")
OPTUNA_SEED = int(OPTUNA_SEED_RAW) if OPTUNA_SEED_RAW not in (None, "") else None

if os.getenv("TRACE_OPTUNA_VERBOSE", "0") != "1":
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_features(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    excluded_cols = [column for column in df.columns if "id" in column.lower()]
    feature_columns = df.columns.difference(excluded_cols)

    features = df[feature_columns].copy()
    for column in features.columns:
        if features[column].dtype in ("object", "category"):
            frequencies = features[column].value_counts(normalize=True)
            features[column] = features[column].map(frequencies)

    features = features.dropna()
    if features.empty:
        raise SystemExit("No valid rows remain after preprocessing.")

    if len(features) < 3:
        raise SystemExit("DBSCAN requires at least 3 valid rows.")

    return StandardScaler().fit_transform(features)


X_SCALED = load_features(CSV_FILE_PATH)
GLOBAL_CENTROID = X_SCALED.mean(axis=0)
SSE_MAX = float(np.sum((X_SCALED - GLOBAL_CENTROID) ** 2))
START_TIME = time.time()


def compute_sse(labels: np.ndarray) -> float:
    """Compute SSE and ignore DBSCAN noise labels."""
    sse = 0.0
    for label in np.unique(labels):
        if label == -1:
            continue

        points = X_SCALED[labels == label]
        if len(points) == 0:
            continue

        centroid = points.mean(axis=0)
        sse += np.sum((points - centroid) ** 2)

    return float(sse)


def combined_score(db_score: float, sil_score: float, sse: float) -> float:
    """Compute the TRACE-compatible clustering score."""
    normalized_sil = (sil_score + 1.0) / 2.0
    normalized_db = 1.0 / (1.0 + max(db_score, 1e-12))
    eps = 1e-12

    return 1.0 / (
        ALPHA / max(normalized_sil, eps)
        + BETA / max(normalized_db, eps)
    )


def evaluate(labels: np.ndarray) -> Optional[dict[str, Any]]:
    """Return clustering metrics, or None if labels are invalid."""
    unique_labels = np.unique(labels)
    non_noise_clusters = [label for label in unique_labels if label != -1]
    noise_ratio = float((labels == -1).mean())

    if len(non_noise_clusters) < 2:
        return None

    if len(unique_labels) >= len(labels):
        return None

    try:
        sil = float(silhouette_score(X_SCALED, labels))
        db = float(davies_bouldin_score(X_SCALED, labels))
    except Exception:
        return None

    sse = compute_sse(labels)
    score = combined_score(db, sil, sse)

    return {
        "combined_score": float(score),
        "silhouette": sil,
        "davies_bouldin": db,
        "noise_ratio": noise_ratio,
        "sse": float(sse),
    }


OPTUNA_TRIALS: list[dict[str, Any]] = []


def objective(trial: optuna.Trial) -> float:
    eps = trial.suggest_float("eps", 0.1, 2.0, step=0.05)
    min_samples = trial.suggest_int("min_samples", 5, 50)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_SCALED)
    metrics = evaluate(labels)

    if metrics is None:
        raise optuna.exceptions.TrialPruned()

    penalized_score = metrics["combined_score"] * (1.0 - metrics["noise_ratio"])

    OPTUNA_TRIALS.append(
        {
            "trial_number": int(trial.number),
            "eps": float(eps),
            "min_samples": int(min_samples),
            "penalized_score": float(penalized_score),
            **metrics,
        }
    )

    return penalized_score


sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED) if OPTUNA_SEED is not None else None
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

if not OPTUNA_TRIALS:
    raise SystemExit("No valid DBSCAN trial completed.")

best_trial = max(OPTUNA_TRIALS, key=lambda item: item["penalized_score"])
best_params = {
    "eps": float(best_trial["eps"]),
    "min_samples": int(best_trial["min_samples"]),
}

labels_final = DBSCAN(
    eps=best_params["eps"],
    min_samples=best_params["min_samples"],
).fit_predict(X_SCALED)

final_metrics = evaluate(labels_final)
if final_metrics is None:
    raise SystemExit("Final DBSCAN model produced invalid labels.")

distance_matrix = pairwise_distances(X_SCALED, metric="euclidean")
neighbor_counts = np.sum(distance_matrix <= best_params["eps"], axis=1)

core_mask = neighbor_counts >= best_params["min_samples"]
noise_mask = labels_final == -1
border_mask = ~core_mask & ~noise_mask

neighbor_hist = np.bincount(np.clip(neighbor_counts.astype(int), 0, 49), minlength=50)

base_name = Path(CSV_FILE_PATH).stem
total_runtime_sec = time.time() - START_TIME

text_output = "\n".join(
    [
        f"Best parameters: {best_params}",
        f"Final Combined Score: {final_metrics['combined_score']}",
        f"Final Silhouette Score: {final_metrics['silhouette']}",
        f"Final Davies-Bouldin Score: {final_metrics['davies_bouldin']}",
    ]
)
(OUTPUT_DIR / f"{base_name}.txt").write_text(text_output, encoding="utf-8")

core_stats = {
    "clean_state": CLEAN_STATE,
    "best_params": best_params,
    "core_count": int(core_mask.sum()),
    "border_count": int(border_mask.sum()),
    "noise_count": int(noise_mask.sum()),
    "neighbor_hist": neighbor_hist.tolist(),
    "weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
    "sse_max": float(SSE_MAX),
    "n_trials_requested": int(N_TRIALS),
    "n_trials_completed": int(len(OPTUNA_TRIALS)),
    "total_runtime_sec": float(total_runtime_sec),
    **final_metrics,
}

core_path = OUTPUT_DIR / f"{base_name}_{CLEAN_STATE}_core_stats.json"
with core_path.open("w", encoding="utf-8") as fp:
    json.dump(core_stats, fp, indent=4)

trials_path = OUTPUT_DIR / f"{base_name}_{CLEAN_STATE}_optuna_trials.json"
with trials_path.open("w", encoding="utf-8") as fp:
    json.dump(OPTUNA_TRIALS, fp, indent=4)

other_state = "cleaned" if CLEAN_STATE == "raw" else "raw"
other_path = OUTPUT_DIR / f"{base_name}_{other_state}_core_stats.json"

if other_path.exists():
    other = json.loads(other_path.read_text(encoding="utf-8"))
    shift = {
        "dataset_id": DATASET_ID,
        "delta_eps": best_params["eps"] - float(other.get("best_params", {}).get("eps", 0.0)),
        "delta_min_samples": best_params["min_samples"] - int(other.get("best_params", {}).get("min_samples", 0)),
        "delta_combined": core_stats["combined_score"] - float(other.get("combined_score", 0.0)),
    }

    with (OUTPUT_DIR / f"{base_name}_param_shift.json").open("w", encoding="utf-8") as fp:
        json.dump(shift, fp, indent=4)

print(f"All files saved in: {OUTPUT_DIR}")
print(f"Program completed in {total_runtime_sec:.2f} sec")