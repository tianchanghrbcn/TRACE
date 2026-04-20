#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian Mixture Model clustering with EM-iteration tracking.

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
- repaired_<dataset_id>_<clean_state>_gmm_history.json
- repaired_<dataset_id>_<clean_state>_summary.json
- repaired_<dataset_id>_param_shift.json, if paired raw/cleaned summary exists.
"""

from __future__ import annotations

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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


METHOD_NAME = "GMM"
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

warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
        raise SystemExit("GMM requires at least 3 valid rows.")

    return StandardScaler().fit_transform(features)


X_SCALED = load_features(CSV_FILE_PATH)
GLOBAL_CENTROID = X_SCALED.mean(axis=0)
SSE_MAX = float(np.sum((X_SCALED - GLOBAL_CENTROID) ** 2))
START_TIME = time.time()


def compute_sse(labels: np.ndarray) -> float:
    sse = 0.0
    for label in np.unique(labels):
        points = X_SCALED[labels == label]
        if len(points) == 0:
            continue
        centroid = points.mean(axis=0)
        sse += np.sum((points - centroid) ** 2)
    return float(sse)


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


def gmm_with_tracking(
    n_components: int,
    covariance_type: str,
) -> tuple[np.ndarray, int, list[float], GaussianMixture]:
    """
    Fit GMM and track lower-bound values.

    The repeated one-iteration warm-start loop preserves the original idea of
    recording the EM trajectory.
    """
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=1,
        warm_start=True,
        random_state=0,
    )

    lower_bounds: list[float] = []

    for _ in range(300):
        model.fit(X_SCALED)
        lower_bounds.append(float(model.lower_bound_))
        if model.converged_:
            break

    labels = model.predict(X_SCALED)
    return labels, len(lower_bounds), lower_bounds, model


def make_trial_record(
    trial_number: int,
    n_components: int,
    covariance_type: str,
    labels: np.ndarray,
    n_iter: int,
    lower_bounds: list[float],
) -> dict[str, Any]:
    if not validate_labels(labels):
        raise optuna.exceptions.TrialPruned()

    db_score = float(davies_bouldin_score(X_SCALED, labels))
    sil_score = float(silhouette_score(X_SCALED, labels))
    sse = compute_sse(labels)
    score = combined_score(db_score, sil_score, sse)

    if len(lower_bounds) > 1:
        auc_ll = float(np.trapz(lower_bounds))
        ll_geo_decay = float(
            abs(lower_bounds[-1] - lower_bounds[0]) / max(abs(lower_bounds[0]), 1e-12)
        )
    else:
        auc_ll = 0.0
        ll_geo_decay = 0.0

    return {
        "trial_number": int(trial_number),
        "n_components": int(n_components),
        "covariance_type": str(covariance_type),
        "combined_score": float(score),
        "silhouette": sil_score,
        "davies_bouldin": db_score,
        "sse": float(sse),
        "n_iter": int(n_iter),
        "ll_start": float(lower_bounds[0]) if lower_bounds else 0.0,
        "ll_end": float(lower_bounds[-1]) if lower_bounds else 0.0,
        "auc_ll": auc_ll,
        "ll_geo_decay": ll_geo_decay,
        "ll_curve": lower_bounds,
    }


OPTUNA_TRIALS: list[dict[str, Any]] = []


def objective(trial: optuna.Trial) -> float:
    n_rows = X_SCALED.shape[0]
    max_k = min(max(5, math.isqrt(n_rows)), n_rows - 1)
    min_k = min(5, max_k)

    if max_k < 2:
        raise optuna.exceptions.TrialPruned()

    n_components = trial.suggest_int("n_components", min_k, max_k)
    covariance_type = trial.suggest_categorical(
        "covariance_type",
        ["full", "tied", "diag", "spherical"],
    )

    try:
        labels, n_iter, lower_bounds, _ = gmm_with_tracking(n_components, covariance_type)
        record = make_trial_record(
            trial_number=trial.number,
            n_components=n_components,
            covariance_type=covariance_type,
            labels=labels,
            n_iter=n_iter,
            lower_bounds=lower_bounds,
        )
    except optuna.exceptions.TrialPruned:
        raise
    except Exception:
        raise optuna.exceptions.TrialPruned()

    OPTUNA_TRIALS.append(record)
    return record["combined_score"]


sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED) if OPTUNA_SEED is not None else None
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

if not OPTUNA_TRIALS:
    raise SystemExit("No valid GMM trial completed.")

best_record = max(OPTUNA_TRIALS, key=lambda item: item["combined_score"])
final_best_k = int(best_record["n_components"])
best_covariance_type = str(best_record["covariance_type"])

labels_final, n_iter_final, lower_bounds_final, _ = gmm_with_tracking(
    final_best_k,
    best_covariance_type,
)

if not validate_labels(labels_final):
    raise SystemExit("Final GMM model produced invalid labels.")

final_db = float(davies_bouldin_score(X_SCALED, labels_final))
final_sil = float(silhouette_score(X_SCALED, labels_final))
final_sse = compute_sse(labels_final)
final_combined = combined_score(final_db, final_sil, final_sse)
total_runtime_sec = time.time() - START_TIME

base_name = Path(CSV_FILE_PATH).stem
best_params = {
    "n_components": final_best_k,
    "covariance_type": best_covariance_type,
}

text_output = "\n".join(
    [
        f"Best parameters: {best_params}",
        f"Final Combined Score: {final_combined}",
        f"Final Silhouette Score: {final_sil}",
        f"Final Davies-Bouldin Score: {final_db}",
    ]
)
(OUTPUT_DIR / f"{base_name}.txt").write_text(text_output, encoding="utf-8")

history_path = OUTPUT_DIR / f"{base_name}_{CLEAN_STATE}_gmm_history.json"
with history_path.open("w", encoding="utf-8") as fp:
    json.dump(OPTUNA_TRIALS, fp, indent=4)

summary = {
    "clean_state": CLEAN_STATE,
    "best_params": best_params,
    "best_k": final_best_k,
    "best_covariance_type": best_covariance_type,
    "combined_score": float(final_combined),
    "silhouette": final_sil,
    "davies_bouldin": final_db,
    "sse": float(final_sse),
    "sse_max": float(SSE_MAX),
    "weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
    "n_iter_final": int(n_iter_final),
    "ll_curve_final": lower_bounds_final,
    "n_trials_requested": int(N_TRIALS),
    "n_trials_completed": int(len(OPTUNA_TRIALS)),
    "total_runtime_sec": float(total_runtime_sec),
}

summary_path = OUTPUT_DIR / f"{base_name}_{CLEAN_STATE}_summary.json"
with summary_path.open("w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=4)

other_state = "cleaned" if CLEAN_STATE == "raw" else "raw"
other_path = OUTPUT_DIR / f"{base_name}_{other_state}_summary.json"

if other_path.exists():
    other = json.loads(other_path.read_text(encoding="utf-8"))
    shift = {
        "dataset_id": DATASET_ID,
        "delta_k": summary["best_k"] - int(other.get("best_k", 0)),
        "delta_combined": summary["combined_score"] - float(other.get("combined_score", 0.0)),
        "rel_shift": abs(summary["best_k"] - int(other.get("best_k", 1))) / max(int(other.get("best_k", 1)), 1),
    }

    with (OUTPUT_DIR / f"{base_name}_param_shift.json").open("w", encoding="utf-8") as fp:
        json.dump(shift, fp, indent=4)

print(f"All files saved in: {OUTPUT_DIR}")
print(f"Program completed in {total_runtime_sec:.2f} sec")