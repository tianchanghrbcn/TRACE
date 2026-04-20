#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means clustering with centroid-trajectory tracking.

Input environment variables:
- CSV_FILE_PATH
- DATASET_ID
- ALGO
- OUTPUT_DIR
- CLEAN_STATE

TRACE-compatible environment variables:
- TRACE_PROJECT_ROOT
- TRACE_N_TRIALS: number of Optuna trials. Default: 30.
- TRACE_OPTUNA_SEED
- TRACE_OPTUNA_VERBOSE

Outputs:
- repaired_<dataset_id>.txt
- repaired_<dataset_id>_<clean_state>_centroid_history.json
- repaired_<dataset_id>_<clean_state>_summary.json
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


METHOD_NAME = "KMEANS"
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

N_TRIALS = int(os.getenv("TRACE_N_TRIALS", "30"))
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
        raise SystemExit("KMEANS requires at least 3 valid rows.")

    return StandardScaler().fit_transform(features)


X_SCALED = load_features(CSV_FILE_PATH)
GLOBAL_CENTROID = X_SCALED.mean(axis=0)
SSE_MAX = float(np.sum((X_SCALED - GLOBAL_CENTROID) ** 2))
START_TIME = time.time()


def combined_score(db_score: float, sil_score: float, sse: float) -> float:
    normalized_sil = (sil_score + 1.0) / 2.0
    normalized_db = 1.0 / (1.0 + max(db_score, 1e-12))
    eps = 1e-12

    return 1.0 / (
        ALPHA / max(normalized_sil, eps)
        + BETA / max(normalized_db, eps)
    )


def validate_labels(labels: np.ndarray) -> bool:
    unique_labels = np.unique(labels)
    return 2 <= len(unique_labels) < len(labels)


def run_kmeans_tracking(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 300,
    tolerance: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, list[dict[str, float]], int, float]:
    rng = np.random.default_rng(seed)
    centers, _ = kmeans_plusplus(data, n_clusters=n_clusters, random_state=seed)

    history: list[dict[str, float]] = []
    labels = np.zeros(data.shape[0], dtype=int)
    sse = 0.0

    for iteration in range(1, max_iter + 1):
        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)

        new_centers = np.vstack(
            [
                data[labels == cluster_id].mean(axis=0)
                if np.any(labels == cluster_id)
                else data[rng.integers(0, data.shape[0])]
                for cluster_id in range(n_clusters)
            ]
        )

        delta = float(np.linalg.norm(new_centers - centers))
        relative_delta = delta / (float(np.linalg.norm(centers)) + 1e-12)
        sse = float(np.sum((data - new_centers[labels]) ** 2))

        history.append(
            {
                "iter": float(iteration),
                "delta": delta,
                "relative_delta": relative_delta,
                "sse": sse,
            }
        )

        centers = new_centers
        if delta < tolerance:
            break

    return labels, history, len(history), sse


def add_convergence_stats(record: dict[str, Any]) -> None:
    deltas = [item["delta"] for item in record["history"] if item["delta"] > 1e-12]

    if len(deltas) > 1:
        record["auc_delta"] = float(np.sum(deltas))
        record["geo_decay"] = float((deltas[-1] / deltas[0]) ** (1 / (len(deltas) - 1)))
    else:
        record["auc_delta"] = 0.0
        record["geo_decay"] = 0.0


def make_record(
    trial_number: int,
    n_clusters: int,
    labels: np.ndarray,
    history: list[dict[str, float]],
    iterations: int,
    sse: float,
) -> dict[str, Any]:
    if not validate_labels(labels):
        raise optuna.exceptions.TrialPruned()

    db_score = float(davies_bouldin_score(X_SCALED, labels))
    sil_score = float(silhouette_score(X_SCALED, labels))
    ch_score = float(calinski_harabasz_score(X_SCALED, labels))
    score = combined_score(db_score, sil_score, sse)

    record = {
        "trial_number": int(trial_number),
        "n_clusters": int(n_clusters),
        "sse": float(sse),
        "iterations": int(iterations),
        "combined_score": float(score),
        "silhouette": sil_score,
        "davies_bouldin": db_score,
        "calinski_harabasz": ch_score,
        "history": history,
    }
    add_convergence_stats(record)
    return record


OPTUNA_TRIALS: list[dict[str, Any]] = []


def objective(trial: optuna.Trial) -> float:
    n_rows = X_SCALED.shape[0]
    max_k = min(max(5, math.isqrt(n_rows)), n_rows - 1)
    min_k = min(5, max_k)

    if max_k < 2:
        raise optuna.exceptions.TrialPruned()

    n_clusters = trial.suggest_int("n_clusters", min_k, max_k)

    labels, history, iterations, sse = run_kmeans_tracking(
        X_SCALED,
        n_clusters=n_clusters,
        seed=trial.number,
    )

    record = make_record(
        trial_number=trial.number,
        n_clusters=n_clusters,
        labels=labels,
        history=history,
        iterations=iterations,
        sse=sse,
    )

    OPTUNA_TRIALS.append(record)
    return record["combined_score"]


sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED) if OPTUNA_SEED is not None else None
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

if not OPTUNA_TRIALS:
    raise SystemExit("No valid KMEANS trial completed.")

best_record = max(OPTUNA_TRIALS, key=lambda item: item["combined_score"])
final_k = int(best_record["n_clusters"])

labels_final, history_final, final_iterations, final_sse = run_kmeans_tracking(
    X_SCALED,
    n_clusters=final_k,
    seed=42,
)

final_record = make_record(
    trial_number=-1,
    n_clusters=final_k,
    labels=labels_final,
    history=history_final,
    iterations=final_iterations,
    sse=final_sse,
)

final_db = final_record["davies_bouldin"]
final_sil = final_record["silhouette"]
final_ch = final_record["calinski_harabasz"]
final_combined = final_record["combined_score"]
total_runtime_sec = time.time() - START_TIME

base_name = Path(CSV_FILE_PATH).stem
best_params = {"n_clusters": final_k}

text_output = "\n".join(
    [
        f"Best parameters: {best_params}",
        f"Final Combined Score: {final_combined}",
        f"Final Silhouette Score: {final_sil}",
        f"Final Davies-Bouldin Score: {final_db}",
        f"Calinski-Harabasz: {final_ch}",
        f"Iterations to converge: {final_iterations}",
        f"Final SSE: {final_sse}",
    ]
)
(OUTPUT_DIR / f"{base_name}.txt").write_text(text_output, encoding="utf-8")

history_path = OUTPUT_DIR / f"{base_name}_{CLEAN_STATE}_centroid_history.json"
with history_path.open("w", encoding="utf-8") as fp:
    json.dump(OPTUNA_TRIALS + [final_record], fp, indent=4)

summary = {
    "clean_state": CLEAN_STATE,
    "best_params": best_params,
    "best_k": final_k,
    "combined_score": float(final_combined),
    "silhouette": float(final_sil),
    "davies_bouldin": float(final_db),
    "calinski_harabasz": float(final_ch),
    "sse": float(final_sse),
    "sse_max": float(SSE_MAX),
    "weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
    "n_trials_requested": int(N_TRIALS),
    "n_trials_completed": int(len(OPTUNA_TRIALS)),
    "avg_iterations": float(np.mean([item["iterations"] for item in OPTUNA_TRIALS])),
    "median_iterations": float(np.median([item["iterations"] for item in OPTUNA_TRIALS])),
    "avg_auc_delta": float(np.mean([item["auc_delta"] for item in OPTUNA_TRIALS])),
    "avg_geo_decay": float(np.mean([item["geo_decay"] for item in OPTUNA_TRIALS])),
    "total_runtime_sec": float(total_runtime_sec),
}

summary_path = OUTPUT_DIR / f"{base_name}_{CLEAN_STATE}_summary.json"
with summary_path.open("w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=4)

print(f"All files saved in: {OUTPUT_DIR}")
print(f"Program completed in {total_runtime_sec:.2f} sec")