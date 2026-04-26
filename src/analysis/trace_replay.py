#!/usr/bin/env python3
"""TRACE Stage 4 replay implementation.

This module replays TRACE from saved raw pipeline logs, following the paper's
TRACE section as closely as possible:

1. Freeze the paper definitions first:
   - EDR = (TP - FP) / (TP + FN)
   - H = harmonic aggregation of Sil* and DB*
   - H_full(D_q) = full-search best score across non-GroundTruth cleaners
   - X = first progress ratio that reaches 95% of H_full
   - Y = score-retention ratio under TRACE's actually consumed budget

2. Replay on saved logs rather than rerunning cleaners/clusterers.

3. Keep TRACE as an outer policy layer:
   - mode/c0 is the fixed baseline cleaner reference;
   - candidate cleaners are screened by entry rules and runtime transitions;
   - the underlying trial definitions and score function are not altered.

The code prefers transparency over cleverness. It emits trial-level replay logs,
path-level decisions, dataset-level summaries, and an aggregate JSON summary.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN as SKDBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from src.results_processing.io import read_json, write_csv, write_json


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TraceWarning:
    dataset_id: Optional[int]
    cleaner: str
    clusterer: str
    kind: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id if self.dataset_id is not None else "",
            "cleaner": self.cleaner,
            "clusterer": self.clusterer,
            "kind": self.kind,
            "message": self.message,
        }


@dataclass
class DatasetMeta:
    dataset_id: int
    dataset_name: str
    csv_file: str
    dirty_csv_path: Path
    clean_csv_path: Optional[Path]
    missing_rate: Optional[float]
    noise_rate: Optional[float]
    error_rate: Optional[float]
    q_tot: float
    dirty_id: Optional[int]

    @property
    def error_dominance(self) -> str:
        miss = self.missing_rate or 0.0
        noise = self.noise_rate or 0.0
        if miss > noise:
            return "missing"
        if noise > miss:
            return "anomaly"
        return "balanced"


@dataclass
class TrialRecord:
    dataset_id: int
    dataset_name: str
    cleaner: str
    clusterer: str
    path_dir: Path
    cleaned_file_path: Optional[Path]
    order_in_path: int
    trial_number: int
    score: float
    silhouette: float
    davies_bouldin: float
    sil_star: float
    db_star: float
    params: dict[str, Any]
    process: dict[str, Any]
    raw_record: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    valid: bool = True
    status: str = "complete"

    def as_brief_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "cleaner": self.cleaner,
            "clusterer": self.clusterer,
            "trial_number": self.trial_number,
            "order_in_path": self.order_in_path,
            "score": self.score,
            "silhouette": self.silhouette,
            "davies_bouldin": self.davies_bouldin,
            "params_json": json.dumps(self.params, ensure_ascii=False, sort_keys=True),
        }


@dataclass
class PathLedger:
    dataset_id: int
    dataset_name: str
    cleaner: str
    clusterer: str
    clusterer_family: str
    path_dir: Path
    cleaned_file_path: Optional[Path]
    original_path_order: int
    trials: list[TrialRecord]
    edr: Optional[float]
    cleaned_metrics: dict[str, Any]

    @property
    def full_best_trial(self) -> Optional[TrialRecord]:
        scored_trials = [
            trial for trial in self.trials
            if trial.valid and np.isfinite(float(trial.score))
        ]
        if not scored_trials:
            return None
        return max(scored_trials, key=lambda item: item.score)

    @property
    def full_trial_count(self) -> int:
        """Budget-denominator trial count, including pruned/missing Optuna trials."""
        return len(self.trials)

    @property
    def scored_trial_count(self) -> int:
        return sum(1 for trial in self.trials if trial.valid and np.isfinite(float(trial.score)))


@dataclass
class PathDecision:
    dataset_id: int
    dataset_name: str
    cleaner: str
    clusterer: str
    phase: str
    evaluation_index: int
    consumed_trials_in_path: int
    consumed_global_trials: int
    action: str
    reason: str
    edr: Optional[float]
    delta_h: Optional[float]
    gp: Optional[bool]
    gr: Optional[bool]
    delta_lambda_nonzero: Optional[bool]
    current_best_score: Optional[float]
    current_best_params_json: str
    process_ref_kind: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "cleaner": self.cleaner,
            "clusterer": self.clusterer,
            "phase": self.phase,
            "evaluation_index": self.evaluation_index,
            "consumed_trials_in_path": self.consumed_trials_in_path,
            "consumed_global_trials": self.consumed_global_trials,
            "action": self.action,
            "reason": self.reason,
            "edr": "" if self.edr is None else self.edr,
            "delta_h": "" if self.delta_h is None else self.delta_h,
            "gp": "" if self.gp is None else int(bool(self.gp)),
            "gr": "" if self.gr is None else int(bool(self.gr)),
            "delta_lambda_nonzero": "" if self.delta_lambda_nonzero is None else int(bool(self.delta_lambda_nonzero)),
            "current_best_score": "" if self.current_best_score is None else self.current_best_score,
            "current_best_params_json": self.current_best_params_json,
            "process_ref_kind": self.process_ref_kind,
        }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


DEFAULT_CONFIG: dict[str, Any] = {
    "trace": {
        "name": "TRACE",
        "baseline_cleaner": "mode",
        "skip_cleaners": ["groundtruth"],
        "alpha": 0.47,
        "tau_turn": 0.10,
        "tau_hi": 0.20,
        "count_mode_reference_trials": True,
        "miss_hit95_as_full_progress": True,
        "budget_count_uses_requested_trials": True,
        "missing_trial_status": "pruned_or_missing",
        "cleaner_groups": {
            "norm": ["unified", "horizon", "bigdansing"],
            "sem": ["baran", "holoclean"],
            "strong": ["boostclean", "scared"],
        },
        "entry_bias": {
            "missing_norm_first": True,
            "anomaly_sem_first_in_mid": True,
            "anomaly_strong_early_in_high": True,
        },
        "eval": {
            "chunk_size": 5,
            "ledger_progress_every": 25,
            "recompute_dbscan_process_features": True,
        },
        "process_reference": {
            "use_same_lambda_reference_when_available": True,
            "fallback": "nearest",
        },
        "param_shift": {
            "dbscan_eps_zero_tol": 0.025,
            "compare_gmm_covariance_type": True,
            "compare_hc_linkage_metric": True,
        },
        "retune": {
            "centroid_k_radius": 1,
            "gmm_n_components_radius": 1,
            "dbscan_eps_window": 0.10,
            "dbscan_min_samples_window": 5,
            "hc_k_radius": 1,
            "hc_fix_linkage": True,
            "hc_fix_metric": True,
        },
        "edr": {
            "key_columns_override": {},
            "key_column_name_contains": ["id"],
            "numeric_tol": 1e-9,
        },
        "paths": {
            "dataset_manifest": "data/manifest.json",
            "methods_config": "configs/methods.yaml",
            "output_dir": "results/processed/trace",
            "trial_log_csv": "trace_replay_trials.csv",
            "path_decisions_csv": "trace_path_decisions.csv",
            "dataset_summary_csv": "trace_dataset_summary.csv",
            "baseline_csv": "trace_baseline_sequence.csv",
            "aggregate_json": "trace_aggregate_summary.json",
            "warnings_json": "trace_warnings.json",
        },
    }
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_trace_config(config_path: Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_CONFIG
    with path.open("r", encoding="utf-8-sig") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid TRACE config: {path}")
    return _deep_merge(DEFAULT_CONFIG, loaded)


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_rate(value: Any) -> Optional[float]:
    parsed = to_float(value)
    if parsed is None:
        return None
    if parsed < 0:
        return 0.0
    if parsed > 1.0:
        return parsed / 100.0
    return parsed


def sil_star(value: float) -> float:
    return (float(value) + 1.0) / 2.0


def db_star(value: float) -> float:
    return 1.0 / (1.0 + max(float(value), 1e-12))


def score_h(alpha: float, silhouette: float, davies_bouldin: float) -> float:
    s = max(sil_star(silhouette), 1e-12)
    d = max(db_star(davies_bouldin), 1e-12)
    return 1.0 / ((float(alpha) / s) + ((1.0 - float(alpha)) / d))


def ensure_fraction_from_meta(error_rate: Optional[float], missing_rate: Optional[float], noise_rate: Optional[float]) -> float:
    if error_rate is not None:
        return normalize_rate(error_rate) or 0.0
    miss = normalize_rate(missing_rate) or 0.0
    noise = normalize_rate(noise_rate) or 0.0
    return miss + noise


def parse_dirty_id_from_filename(value: str) -> Optional[int]:
    match = re.search(r"_(\d+)\.csv$", str(value))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def normalize_text_name(value: Any) -> str:
    return str(value or "").strip()


def normalize_cleaner_name(value: Any) -> str:
    return normalize_text_name(value).lower()


def normalize_clusterer_name(value: Any) -> str:
    return normalize_text_name(value).upper()


def path_or_none(value: Any) -> Optional[Path]:
    if value in (None, ""):
        return None
    return Path(str(value))


def resolve_path(candidate: Any, roots: Sequence[Path]) -> Optional[Path]:
    p = path_or_none(candidate)
    if p is None:
        return None
    if p.is_absolute() and p.exists():
        return p.resolve()
    for root in roots:
        q = (root / p).resolve()
        if q.exists():
            return q
    if p.exists():
        return p.resolve()
    return (roots[0] / p).resolve() if roots else p.resolve()


def values_equal(left: Any, right: Any, numeric_tol: float) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    left_s = str(left).strip()
    right_s = str(right).strip()
    if left_s == right_s:
        return True
    if _NUMERIC_RE.match(left_s) and _NUMERIC_RE.match(right_s):
        try:
            return math.isclose(float(left_s), float(right_s), abs_tol=numeric_tol, rel_tol=0.0)
        except Exception:
            return left_s == right_s
    return left_s == right_s


# ---------------------------------------------------------------------------
# Manifest / registry loading
# ---------------------------------------------------------------------------


def autodetect_results_root(project_root: Path, requested: Optional[Path]) -> Path:
    if requested is not None:
        root = Path(requested)
        if not root.is_absolute():
            root = (project_root / root).resolve()
        if (root / "eigenvectors.json").exists():
            return root
        if (root / "raw" / "eigenvectors.json").exists():
            return (root / "raw").resolve()
        return root
    raw_root = (project_root / "results" / "raw").resolve()
    if (raw_root / "eigenvectors.json").exists():
        return raw_root
    return (project_root / "results").resolve()


def load_dataset_manifest(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    rel = config["trace"]["paths"]["dataset_manifest"]
    path = resolve_path(rel, [project_root])
    if path is None or not path.exists():
        return {}
    data = read_json(path, default={})
    if not isinstance(data, dict):
        return {}
    return data.get("datasets", {}) if isinstance(data.get("datasets"), dict) else {}


def load_methods_config(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    rel = config["trace"]["paths"]["methods_config"]
    path = resolve_path(rel, [project_root])
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def clusterer_family_map(methods_config: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    clusterers = methods_config.get("clusterers", {})
    if not isinstance(clusterers, dict):
        return out
    for name, spec in clusterers.items():
        if not isinstance(spec, dict):
            continue
        group = normalize_text_name(spec.get("group"))
        family = "other"
        if group == "density":
            family = "density"
        elif group == "hierarchical":
            family = "hierarchical"
        elif group == "model":
            family = "model"
        elif group == "centroid":
            family = "centroid"
        out[normalize_clusterer_name(name)] = family
    return out


# ---------------------------------------------------------------------------
# Dataset metadata / EDR
# ---------------------------------------------------------------------------


def build_dataset_meta(
    project_root: Path,
    results_root: Path,
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> tuple[dict[int, DatasetMeta], list[TraceWarning]]:
    warnings: list[TraceWarning] = []
    eigenvectors = read_json(results_root / "eigenvectors.json", default=[])
    if not isinstance(eigenvectors, list):
        raise ValueError(f"eigenvectors.json must be a JSON list: {results_root / 'eigenvectors.json'}")

    meta_by_id: dict[int, DatasetMeta] = {}
    roots = [project_root, results_root, results_root.parent]

    for idx, record in enumerate(eigenvectors):
        if not isinstance(record, dict):
            continue
        dataset_id = int(record.get("dataset_id", idx))
        dataset_name = normalize_text_name(record.get("dataset_name") or record.get("dataset"))
        csv_file = normalize_text_name(record.get("csv_file"))
        dirty_path = resolve_path(csv_file, roots)
        dirty_id = parse_dirty_id_from_filename(csv_file)
        manifest_entry = dataset_manifest.get(dataset_name, {}) if dataset_name else {}
        clean_rel = manifest_entry.get("clean") if isinstance(manifest_entry, dict) else None
        root_rel = manifest_entry.get("root") if isinstance(manifest_entry, dict) else None
        clean_path: Optional[Path] = None
        if clean_rel and root_rel:
            clean_path = resolve_path(Path(root_rel) / clean_rel, [project_root])
        elif dataset_name and dirty_id is not None:
            guessed = project_root / "data" / "raw" / "train" / dataset_name / "clean.csv"
            clean_path = guessed.resolve()

        missing_rate = normalize_rate(record.get("missing_rate"))
        noise_rate = normalize_rate(record.get("noise_rate"))
        error_rate = normalize_rate(record.get("error_rate"))
        q_tot = ensure_fraction_from_meta(error_rate, missing_rate, noise_rate)

        meta_by_id[dataset_id] = DatasetMeta(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            csv_file=csv_file,
            dirty_csv_path=dirty_path if dirty_path is not None else Path(csv_file),
            clean_csv_path=clean_path,
            missing_rate=missing_rate,
            noise_rate=noise_rate,
            error_rate=error_rate,
            q_tot=q_tot,
            dirty_id=dirty_id,
        )

        if dirty_path is None or not dirty_path.exists():
            warnings.append(
                TraceWarning(
                    dataset_id=dataset_id,
                    cleaner="",
                    clusterer="",
                    kind="missing_dirty_csv",
                    message=f"Dirty CSV not found for dataset_id={dataset_id}: {csv_file}",
                )
            )
        if clean_path is None or not clean_path.exists():
            warnings.append(
                TraceWarning(
                    dataset_id=dataset_id,
                    cleaner="",
                    clusterer="",
                    kind="missing_clean_csv",
                    message=(
                        f"Clean reference CSV not found for dataset_id={dataset_id}, dataset={dataset_name}."
                    ),
                )
            )
    return meta_by_id, warnings


def _choose_key_columns(columns: Sequence[str], dataset_name: str, config: dict[str, Any]) -> set[str]:
    edr_cfg = config["trace"]["edr"]
    overrides = edr_cfg.get("key_columns_override", {})
    if isinstance(overrides, dict) and dataset_name in overrides and isinstance(overrides[dataset_name], list):
        return {str(col) for col in overrides[dataset_name]}
    contains = [str(x).lower() for x in edr_cfg.get("key_column_name_contains", ["id"])]
    out = set()
    for column in columns:
        lower = str(column).lower()
        if any(token in lower for token in contains):
            out.add(str(column))
    return out


def compute_edr_metrics(
    clean_csv: Path,
    dirty_csv: Path,
    repaired_csv: Path,
    dataset_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    numeric_tol = float(config["trace"]["edr"].get("numeric_tol", 1e-9))

    clean_df = pd.read_csv(clean_csv, encoding="utf-8-sig")
    dirty_df = pd.read_csv(dirty_csv, encoding="utf-8-sig")
    repaired_df = pd.read_csv(repaired_csv, encoding="utf-8-sig")

    shared_columns = [col for col in clean_df.columns if col in dirty_df.columns and col in repaired_df.columns]
    key_columns = _choose_key_columns(shared_columns, dataset_name, config)
    compare_columns = [col for col in shared_columns if col not in key_columns]

    n_rows = min(len(clean_df), len(dirty_df), len(repaired_df))
    clean_df = clean_df.iloc[:n_rows][compare_columns].reset_index(drop=True)
    dirty_df = dirty_df.iloc[:n_rows][compare_columns].reset_index(drop=True)
    repaired_df = repaired_df.iloc[:n_rows][compare_columns].reset_index(drop=True)

    tp = fp = fn = changed = 0
    total = int(n_rows * len(compare_columns))

    for row_idx in range(n_rows):
        for column in compare_columns:
            clean_value = clean_df.at[row_idx, column]
            dirty_value = dirty_df.at[row_idx, column]
            repaired_value = repaired_df.at[row_idx, column]
            dirty_is_error = not values_equal(dirty_value, clean_value, numeric_tol)
            repaired_equals_clean = values_equal(repaired_value, clean_value, numeric_tol)
            repaired_equals_dirty = values_equal(repaired_value, dirty_value, numeric_tol)
            if not repaired_equals_dirty:
                changed += 1
            if dirty_is_error and repaired_equals_clean:
                tp += 1
            elif (not dirty_is_error) and (not repaired_equals_clean):
                fp += 1
            elif dirty_is_error and (not repaired_equals_clean):
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    edr = (tp - fp) / (tp + fn) if (tp + fn) > 0 else 0.0
    rho_chg = changed / total if total > 0 else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "changed": int(changed),
        "total_cells": int(total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "edr": float(edr),
        "rho_chg": float(rho_chg),
        "n_rows_compared": int(n_rows),
        "n_columns_compared": int(len(compare_columns)),
        "key_columns_json": json.dumps(sorted(key_columns), ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Trial-log parsing
# ---------------------------------------------------------------------------


def load_clustering_features(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    excluded_cols = [column for column in df.columns if "id" in str(column).lower()]
    feature_columns = df.columns.difference(excluded_cols)
    features = df[feature_columns].copy()
    for column in features.columns:
        if str(features[column].dtype) in ("object", "category"):
            frequencies = features[column].value_counts(normalize=True)
            features[column] = features[column].map(frequencies)
    features = features.dropna()
    if features.empty:
        raise ValueError(f"No valid rows remain after preprocessing: {csv_path}")
    if len(features) < 3:
        raise ValueError(f"At least 3 valid rows are required: {csv_path}")
    return StandardScaler().fit_transform(features)


def _param_json(params: Mapping[str, Any]) -> str:
    return json.dumps(dict(params), ensure_ascii=False, sort_keys=True)


def _search_file(path_dir: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(path_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _read_trial_budget_from_path(path_dir: Path) -> Optional[int]:
    """Return n_trials_requested when a clusterer summary/core file records it.

    The current clustering scripts append only successful Optuna trials to their
    public trial lists. Pruned trials still consume Optuna budget and must be
    counted for X. We therefore read n_trials_requested from the per-path summary
    whenever available and inflate missing trial numbers as scoreless budget
    placeholders.
    """
    candidate_patterns = [
        "*_summary.json",
        "*_core_stats.json",
        "*_centroid_summary.json",
        "*_kmeans_summary.json",
    ]
    for pattern in candidate_patterns:
        for file_path in sorted(path_dir.glob(pattern)):
            data = read_json(file_path, default={})
            if isinstance(data, dict):
                value = data.get("n_trials_requested") or data.get("trial_budget") or data.get("n_trials")
                parsed = to_float(value)
                if parsed is not None and parsed > 0:
                    return int(parsed)
    return None


def _make_scoreless_budget_trial(
    *,
    dataset_id: int,
    dataset_name: str,
    cleaner: str,
    clusterer: str,
    path_dir: Path,
    cleaned_file_path: Optional[Path],
    trial_number: int,
    status: str = "pruned_or_missing",
) -> TrialRecord:
    return TrialRecord(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        cleaner=cleaner,
        clusterer=clusterer,
        path_dir=path_dir,
        cleaned_file_path=cleaned_file_path,
        order_in_path=int(trial_number),
        trial_number=int(trial_number),
        score=float("nan"),
        silhouette=float("nan"),
        davies_bouldin=float("nan"),
        sil_star=float("nan"),
        db_star=float("nan"),
        params={},
        process={"status": status},
        raw_record={},
        source_file=f"budget_placeholder:{path_dir}",
        valid=False,
        status=status,
    )


def _inflate_missing_budget_trials(
    *,
    path_dir: Path,
    valid_trials: list[TrialRecord],
    dataset_id: int,
    dataset_name: str,
    cleaner: str,
    clusterer: str,
    cleaned_file_path: Optional[Path],
    config: Optional[dict[str, Any]] = None,
) -> list[TrialRecord]:
    """Insert scoreless placeholders for pruned/missing trial numbers.

    For old logs this cannot recover the missing trial parameters, but it fixes
    the budget accounting: step 17 means the 17th Optuna trial, not the 17th
    successful trial. If no n_trials_requested is available, the function still
    fills gaps up to max(valid trial_number)+1.
    """
    by_number: dict[int, TrialRecord] = {}
    for trial in valid_trials:
        trial_number = int(trial.trial_number)
        # Keep the first record if a malformed log has duplicates.
        by_number.setdefault(trial_number, trial)

    max_valid_budget = (max(by_number.keys()) + 1) if by_number else 0
    requested_budget = _read_trial_budget_from_path(path_dir)
    budget = max(max_valid_budget, int(requested_budget or 0))
    if budget <= 0:
        return []

    status = "pruned_or_missing"
    if config is not None:
        status = str(config["trace"].get("missing_trial_status", status))

    inflated: list[TrialRecord] = []
    for trial_number in range(budget):
        trial = by_number.get(trial_number)
        if trial is None:
            inflated.append(
                _make_scoreless_budget_trial(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    cleaner=cleaner,
                    clusterer=clusterer,
                    path_dir=path_dir,
                    cleaned_file_path=cleaned_file_path,
                    trial_number=trial_number,
                    status=status,
                )
            )
            continue
        trial.order_in_path = int(trial_number)
        trial.valid = True
        trial.status = str(trial.raw_record.get("status", "complete"))
        inflated.append(trial)

    # If a log contains a trial number beyond n_trials_requested, keep it rather
    # than silently dropping evidence.
    for trial_number in sorted(num for num in by_number if num >= budget):
        trial = by_number[trial_number]
        trial.order_in_path = int(trial_number)
        trial.valid = True
        trial.status = str(trial.raw_record.get("status", "complete"))
        inflated.append(trial)
    return inflated


def _build_centroid_trials(
    *,
    alpha: float,
    clusterer: str,
    dataset_id: int,
    dataset_name: str,
    cleaner: str,
    path_dir: Path,
    cleaned_file_path: Optional[Path],
    history_file: Path,
) -> list[TrialRecord]:
    history = read_json(history_file, default=[])
    if not isinstance(history, list):
        return []
    trials: list[TrialRecord] = []
    order = 0
    for item in history:
        if not isinstance(item, dict):
            continue
        trial_number = int(item.get("trial_number", -1))
        if trial_number < 0:
            continue
        silhouette = float(item["silhouette"])
        davies = float(item["davies_bouldin"])
        params = {"n_clusters": int(item["n_clusters"])}
        trials.append(
            TrialRecord(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                cleaner=cleaner,
                clusterer=clusterer,
                path_dir=path_dir,
                cleaned_file_path=cleaned_file_path,
                order_in_path=order,
                trial_number=trial_number,
                score=score_h(alpha, silhouette, davies),
                silhouette=silhouette,
                davies_bouldin=davies,
                sil_star=sil_star(silhouette),
                db_star=db_star(davies),
                params=params,
                process={
                    "sse": float(item.get("sse", np.nan)),
                    "iterations": int(item.get("iterations", 0)),
                    "auc_delta": to_float(item.get("auc_delta")),
                    "geo_decay": to_float(item.get("geo_decay")),
                },
                raw_record=dict(item),
                source_file=str(history_file),
            )
        )
        order += 1
    trials.sort(key=lambda x: (x.order_in_path, x.trial_number))
    return _inflate_missing_budget_trials(
        path_dir=path_dir,
        valid_trials=trials,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        cleaner=cleaner,
        clusterer=clusterer,
        cleaned_file_path=cleaned_file_path,
    )


def _build_gmm_trials(
    *,
    alpha: float,
    clusterer: str,
    dataset_id: int,
    dataset_name: str,
    cleaner: str,
    path_dir: Path,
    cleaned_file_path: Optional[Path],
    history_file: Path,
) -> list[TrialRecord]:
    history = read_json(history_file, default=[])
    if not isinstance(history, list):
        return []
    trials: list[TrialRecord] = []
    order = 0
    for item in history:
        if not isinstance(item, dict):
            continue
        trial_number = int(item.get("trial_number", -1))
        if trial_number < 0:
            continue
        silhouette = float(item["silhouette"])
        davies = float(item["davies_bouldin"])
        params = {
            "n_components": int(item["n_components"]),
            "covariance_type": str(item["covariance_type"]),
        }
        trials.append(
            TrialRecord(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                cleaner=cleaner,
                clusterer=clusterer,
                path_dir=path_dir,
                cleaned_file_path=cleaned_file_path,
                order_in_path=order,
                trial_number=trial_number,
                score=score_h(alpha, silhouette, davies),
                silhouette=silhouette,
                davies_bouldin=davies,
                sil_star=sil_star(silhouette),
                db_star=db_star(davies),
                params=params,
                process={
                    "sse": float(item.get("sse", np.nan)),
                    "ll_end": to_float(item.get("ll_end")),
                    "ll_start": to_float(item.get("ll_start")),
                    "auc_ll": to_float(item.get("auc_ll")),
                    "ll_geo_decay": to_float(item.get("ll_geo_decay")),
                    "n_iter": int(item.get("n_iter", 0)),
                },
                raw_record=dict(item),
                source_file=str(history_file),
            )
        )
        order += 1
    trials.sort(key=lambda x: (x.order_in_path, x.trial_number))
    return _inflate_missing_budget_trials(
        path_dir=path_dir,
        valid_trials=trials,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        cleaner=cleaner,
        clusterer=clusterer,
        cleaned_file_path=cleaned_file_path,
    )


def _dbscan_trial_process_features(
    cleaned_file_path: Path,
    eps_value: float,
    min_samples: int,
    cache: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Recompute lightweight DBSCAN process signals for one trial.

    DBSCAN trial logs already contain the trial-level noise ratio. Re-running
    sklearn.DBSCAN.fit_predict for every trial is prohibitively slow during
    full TRACE replay, so we only recompute average-neighbor and core-count
    signals from a cached distance matrix.
    """
    cache_key = str(cleaned_file_path)
    if cache_key not in cache:
        features = load_clustering_features(cleaned_file_path)
        distances = pairwise_distances(features, metric="euclidean")
        cache[cache_key] = {
            "distances": distances,
        }
    distances = cache[cache_key]["distances"]
    neighbor_counts = np.sum(distances <= float(eps_value), axis=1)
    avg_neighbor = float(np.mean(neighbor_counts))
    core_count = int(np.sum(neighbor_counts >= int(min_samples)))
    return {
        "avg_neighbor_count": avg_neighbor,
        "core_count_recomputed": core_count,
    }

def _build_dbscan_trials(
    *,
    alpha: float,
    clusterer: str,
    dataset_id: int,
    dataset_name: str,
    cleaner: str,
    path_dir: Path,
    cleaned_file_path: Optional[Path],
    trials_file: Path,
    dbscan_cache: MutableMapping[str, Any],
    config: Optional[dict[str, Any]] = None,
) -> list[TrialRecord]:
    history = read_json(trials_file, default=[])
    if not isinstance(history, list):
        return []
    trials: list[TrialRecord] = []
    eval_cfg = (config or {}).get("trace", {}).get("eval", {})
    recompute_process = bool(eval_cfg.get("recompute_dbscan_process_features", True))
    order = 0
    for item in history:
        if not isinstance(item, dict):
            continue
        trial_number = int(item.get("trial_number", -1))
        if trial_number < 0:
            continue
        silhouette = float(item["silhouette"])
        davies = float(item["davies_bouldin"])
        params = {
            "eps": float(item["eps"]),
            "min_samples": int(item["min_samples"]),
        }
        process: dict[str, Any] = {
            "noise_ratio": float(item.get("noise_ratio", np.nan)),
            "sse": float(item.get("sse", np.nan)),
        }
        if recompute_process and cleaned_file_path is not None and cleaned_file_path.exists():
            process.update(_dbscan_trial_process_features(cleaned_file_path, params["eps"], params["min_samples"], dbscan_cache))
        trials.append(
            TrialRecord(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                cleaner=cleaner,
                clusterer=clusterer,
                path_dir=path_dir,
                cleaned_file_path=cleaned_file_path,
                order_in_path=order,
                trial_number=trial_number,
                score=score_h(alpha, silhouette, davies),
                silhouette=silhouette,
                davies_bouldin=davies,
                sil_star=sil_star(silhouette),
                db_star=db_star(davies),
                params=params,
                process=process,
                raw_record=dict(item),
                source_file=str(trials_file),
            )
        )
        order += 1
    trials.sort(key=lambda x: (x.order_in_path, x.trial_number))
    # Keep DBSCAN replay memory bounded. The distance matrix is useful only
    # within this one path.
    dbscan_cache.clear()
    return _inflate_missing_budget_trials(
        path_dir=path_dir,
        valid_trials=trials,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        cleaner=cleaner,
        clusterer=clusterer,
        cleaned_file_path=cleaned_file_path,
    )


def _build_hc_trials(
    *,
    alpha: float,
    clusterer: str,
    dataset_id: int,
    dataset_name: str,
    cleaner: str,
    path_dir: Path,
    cleaned_file_path: Optional[Path],
    trials_file: Path,
) -> list[TrialRecord]:
    history = read_json(trials_file, default=[])
    if not isinstance(history, list):
        return []
    trials: list[TrialRecord] = []
    order = 0
    for item in history:
        if not isinstance(item, dict):
            continue
        trial_number = int(item.get("trial_number", -1))
        if trial_number < 0:
            continue
        silhouette = float(item["silhouette"])
        davies = float(item["davies_bouldin"])
        params = {
            "n_clusters": int(item["n_clusters"]),
            "linkage": str(item["linkage"]),
            "metric": str(item["metric"]),
        }
        process = {
            "ratio_intra_inter": float(item.get("ratio_intra_inter", np.nan)),
            "h_max": to_float(item.get("h_max")),
            "n_merge_steps": int(item.get("n_merge_steps", 0)),
            "sse": float(item.get("sse", np.nan)),
        }
        trials.append(
            TrialRecord(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                cleaner=cleaner,
                clusterer=clusterer,
                path_dir=path_dir,
                cleaned_file_path=cleaned_file_path,
                order_in_path=order,
                trial_number=trial_number,
                score=score_h(alpha, silhouette, davies),
                silhouette=silhouette,
                davies_bouldin=davies,
                sil_star=sil_star(silhouette),
                db_star=db_star(davies),
                params=params,
                process=process,
                raw_record=dict(item),
                source_file=str(trials_file),
            )
        )
        order += 1
    trials.sort(key=lambda x: (x.order_in_path, x.trial_number))
    return _inflate_missing_budget_trials(
        path_dir=path_dir,
        valid_trials=trials,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        cleaner=cleaner,
        clusterer=clusterer,
        cleaned_file_path=cleaned_file_path,
    )


def load_trials_for_path(
    *,
    alpha: float,
    clusterer: str,
    dataset_id: int,
    dataset_name: str,
    cleaner: str,
    path_dir: Path,
    cleaned_file_path: Optional[Path],
    dbscan_cache: MutableMapping[str, Any],
    config: Optional[dict[str, Any]] = None,
) -> tuple[list[TrialRecord], Optional[str]]:
    clusterer_norm = normalize_clusterer_name(clusterer)
    if clusterer_norm in {"KMEANS", "KMEANSNF", "KMEANSPPS"}:
        history_file = _search_file(path_dir, ["*_centroid_history.json"])
        if history_file is None:
            return [], f"Missing centroid history in {path_dir}"
        return _build_centroid_trials(
            alpha=alpha,
            clusterer=clusterer_norm,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            cleaner=cleaner,
            path_dir=path_dir,
            cleaned_file_path=cleaned_file_path,
            history_file=history_file,
        ), None
    if clusterer_norm == "GMM":
        history_file = _search_file(path_dir, ["*_gmm_history.json"])
        if history_file is None:
            return [], f"Missing GMM history in {path_dir}"
        return _build_gmm_trials(
            alpha=alpha,
            clusterer=clusterer_norm,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            cleaner=cleaner,
            path_dir=path_dir,
            cleaned_file_path=cleaned_file_path,
            history_file=history_file,
        ), None
    if clusterer_norm == "DBSCAN":
        trials_file = _search_file(path_dir, ["*_optuna_trials.json"])
        if trials_file is None:
            return [], f"Missing DBSCAN trial log in {path_dir}"
        return _build_dbscan_trials(
            alpha=alpha,
            clusterer=clusterer_norm,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            cleaner=cleaner,
            path_dir=path_dir,
            cleaned_file_path=cleaned_file_path,
            trials_file=trials_file,
            dbscan_cache=dbscan_cache,
            config=config,
        ), None
    if clusterer_norm == "HC":
        trials_file = _search_file(path_dir, ["*_optuna_trials.json"])
        if trials_file is None:
            return [], f"Missing HC trial log in {path_dir}"
        return _build_hc_trials(
            alpha=alpha,
            clusterer=clusterer_norm,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            cleaner=cleaner,
            path_dir=path_dir,
            cleaned_file_path=cleaned_file_path,
            trials_file=trials_file,
        ), None
    return [], f"Unsupported clusterer for TRACE replay: {clusterer_norm}"


# ---------------------------------------------------------------------------
# Ledger construction
# ---------------------------------------------------------------------------


def build_path_ledgers(
    project_root: Path,
    results_root: Path,
    config: dict[str, Any],
    dataset_meta: dict[int, DatasetMeta],
) -> tuple[dict[int, list[PathLedger]], list[TraceWarning]]:
    warnings: list[TraceWarning] = []
    alpha = float(config["trace"]["alpha"])

    cleaned_results = read_json(results_root / "cleaned_results.json", default=[])
    clustered_results = read_json(results_root / "clustered_results.json", default=[])
    if not isinstance(cleaned_results, list):
        raise ValueError(f"cleaned_results.json must be a JSON list: {results_root / 'cleaned_results.json'}")
    if not isinstance(clustered_results, list):
        raise ValueError(f"clustered_results.json must be a JSON list: {results_root / 'clustered_results.json'}")

    methods_config = load_methods_config(project_root, config)
    family_map = clusterer_family_map(methods_config)

    roots = [project_root, results_root, results_root.parent]

    cleaned_map: dict[tuple[int, str], Path] = {}
    for item in cleaned_results:
        if not isinstance(item, dict):
            continue
        dataset_id = int(item.get("dataset_id", -1))
        cleaner = normalize_cleaner_name(item.get("algorithm") or item.get("cleaner"))
        cleaned_path = resolve_path(item.get("cleaned_file_path"), roots)
        if dataset_id >= 0 and cleaner and cleaned_path is not None:
            cleaned_map[(dataset_id, cleaner)] = cleaned_path

    # Cache EDR by (dataset_id, cleaner)
    edr_cache: dict[tuple[int, str], dict[str, Any]] = {}

    def get_edr(dataset_id: int, cleaner: str, cleaned_path: Optional[Path]) -> tuple[Optional[float], dict[str, Any]]:
        key = (dataset_id, cleaner)
        if key in edr_cache:
            metrics = edr_cache[key]
            return to_float(metrics.get("edr")), metrics
        meta = dataset_meta.get(dataset_id)
        if meta is None or cleaned_path is None or not cleaned_path.exists() or meta.clean_csv_path is None or not meta.clean_csv_path.exists() or not meta.dirty_csv_path.exists():
            metrics = {}
            edr_cache[key] = metrics
            return None, metrics
        try:
            metrics = compute_edr_metrics(meta.clean_csv_path, meta.dirty_csv_path, cleaned_path, meta.dataset_name, config)
        except Exception as exc:
            warnings.append(
                TraceWarning(
                    dataset_id=dataset_id,
                    cleaner=cleaner,
                    clusterer="",
                    kind="edr_failed",
                    message=f"Failed to compute EDR: {exc}",
                )
            )
            metrics = {}
        edr_cache[key] = metrics
        return to_float(metrics.get("edr")), metrics

    ledgers_by_dataset: dict[int, list[PathLedger]] = defaultdict(list)
    dbscan_cache: dict[str, Any] = {}

    skip_cleaners = {normalize_cleaner_name(x) for x in config["trace"].get("skip_cleaners", [])}
    eval_cfg = config["trace"].get("eval", {})
    progress_every = int(eval_cfg.get("ledger_progress_every", 25) or 0)
    total_clustered = len(clustered_results)

    for original_order, item in enumerate(clustered_results):
        if progress_every > 0 and (original_order == 0 or (original_order + 1) % progress_every == 0):
            print(f"[TRACE] Loading trial ledgers: {original_order + 1}/{total_clustered}", flush=True)
        if not isinstance(item, dict):
            continue
        dataset_id = int(item.get("dataset_id", -1))
        if dataset_id < 0 or dataset_id not in dataset_meta:
            continue
        cleaner = normalize_cleaner_name(
            item.get("cleaning_algorithm") or item.get("cleaner") or item.get("algorithm")
        )
        if cleaner in skip_cleaners:
            continue
        clusterer = normalize_clusterer_name(item.get("clustering_name") or item.get("clusterer"))
        path_dir = resolve_path(item.get("clustered_file_path"), roots)
        if path_dir is None:
            warnings.append(
                TraceWarning(
                    dataset_id=dataset_id,
                    cleaner=cleaner,
                    clusterer=clusterer,
                    kind="missing_cluster_path",
                    message="clustered_file_path is missing",
                )
            )
            continue
        cleaned_path = cleaned_map.get((dataset_id, cleaner))
        trials, error_message = load_trials_for_path(
            alpha=alpha,
            clusterer=clusterer,
            dataset_id=dataset_id,
            dataset_name=dataset_meta[dataset_id].dataset_name,
            cleaner=cleaner,
            path_dir=path_dir,
            cleaned_file_path=cleaned_path,
            dbscan_cache=dbscan_cache,
            config=config,
        )
        if error_message:
            warnings.append(
                TraceWarning(
                    dataset_id=dataset_id,
                    cleaner=cleaner,
                    clusterer=clusterer,
                    kind="trial_log_missing",
                    message=error_message,
                )
            )
            continue
        if not trials:
            warnings.append(
                TraceWarning(
                    dataset_id=dataset_id,
                    cleaner=cleaner,
                    clusterer=clusterer,
                    kind="empty_trial_log",
                    message=f"No replayable trials were found in {path_dir}",
                )
            )
            continue
        edr_value, edr_metrics = get_edr(dataset_id, cleaner, cleaned_path)
        ledger = PathLedger(
            dataset_id=dataset_id,
            dataset_name=dataset_meta[dataset_id].dataset_name,
            cleaner=cleaner,
            clusterer=clusterer,
            clusterer_family=family_map.get(clusterer, "other"),
            path_dir=path_dir,
            cleaned_file_path=cleaned_path,
            original_path_order=original_order,
            trials=trials,
            edr=edr_value,
            cleaned_metrics=edr_metrics,
        )
        ledgers_by_dataset[dataset_id].append(ledger)

    for dataset_id in list(ledgers_by_dataset.keys()):
        ledgers_by_dataset[dataset_id].sort(key=lambda x: x.original_path_order)

    return ledgers_by_dataset, warnings


# ---------------------------------------------------------------------------
# Param matching / delta definitions
# ---------------------------------------------------------------------------


def _dbscan_eps_equal(left: Any, right: Any, config: dict[str, Any]) -> bool:
    tol = float(config["trace"]["param_shift"].get("dbscan_eps_zero_tol", 0.025))
    lf = to_float(left)
    rf = to_float(right)
    if lf is None or rf is None:
        return False
    return abs(lf - rf) <= tol


def params_equal(clusterer: str, left: Mapping[str, Any], right: Mapping[str, Any], config: dict[str, Any]) -> bool:
    clusterer_norm = normalize_clusterer_name(clusterer)
    if clusterer_norm == "DBSCAN":
        return _dbscan_eps_equal(left.get("eps"), right.get("eps"), config) and int(left.get("min_samples", -1)) == int(right.get("min_samples", -2))
    if clusterer_norm == "GMM":
        same_k = int(left.get("n_components", -1)) == int(right.get("n_components", -2))
        if not same_k:
            return False
        if bool(config["trace"]["param_shift"].get("compare_gmm_covariance_type", True)):
            return str(left.get("covariance_type")) == str(right.get("covariance_type"))
        return True
    if clusterer_norm == "HC":
        same_k = int(left.get("n_clusters", -1)) == int(right.get("n_clusters", -2))
        if not same_k:
            return False
        if bool(config["trace"]["param_shift"].get("compare_hc_linkage_metric", True)):
            return str(left.get("linkage")) == str(right.get("linkage")) and str(left.get("metric")) == str(right.get("metric"))
        return True
    if "n_clusters" in left or "n_clusters" in right:
        return int(left.get("n_clusters", -1)) == int(right.get("n_clusters", -2))
    return _param_json(left) == _param_json(right)


def param_distance(clusterer: str, left: Mapping[str, Any], right: Mapping[str, Any], config: dict[str, Any]) -> float:
    clusterer_norm = normalize_clusterer_name(clusterer)
    if clusterer_norm == "DBSCAN":
        left_eps = to_float(left.get("eps")) or 0.0
        right_eps = to_float(right.get("eps")) or 0.0
        left_ms = int(left.get("min_samples", 0))
        right_ms = int(right.get("min_samples", 0))
        return abs(left_eps - right_eps) + 0.01 * abs(left_ms - right_ms)
    if clusterer_norm == "GMM":
        dist = abs(int(left.get("n_components", 0)) - int(right.get("n_components", 0)))
        if str(left.get("covariance_type")) != str(right.get("covariance_type")):
            dist += 0.25
        return float(dist)
    if clusterer_norm == "HC":
        dist = abs(int(left.get("n_clusters", 0)) - int(right.get("n_clusters", 0)))
        if str(left.get("linkage")) != str(right.get("linkage")):
            dist += 0.25
        if str(left.get("metric")) != str(right.get("metric")):
            dist += 0.25
        return float(dist)
    return float(abs(int(left.get("n_clusters", 0)) - int(right.get("n_clusters", 0))))


def pick_mode_process_reference(
    mode_path: PathLedger,
    candidate_trial: TrialRecord,
    config: dict[str, Any],
) -> tuple[Optional[TrialRecord], str]:
    if mode_path.full_best_trial is None:
        return None, "missing"
    use_same_lambda = bool(config["trace"]["process_reference"].get("use_same_lambda_reference_when_available", True))
    if use_same_lambda:
        exact_matches = [trial for trial in mode_path.trials if params_equal(candidate_trial.clusterer, trial.params, candidate_trial.params, config)]
        if exact_matches:
            return max(exact_matches, key=lambda item: item.score), "same_lambda"
    fallback = str(config["trace"]["process_reference"].get("fallback", "nearest")).lower()
    if fallback == "nearest":
        nearest = min(mode_path.trials, key=lambda item: param_distance(candidate_trial.clusterer, item.params, candidate_trial.params, config))
        return nearest, "nearest"
    return mode_path.full_best_trial, "mode_best"


# ---------------------------------------------------------------------------
# TRACE entry ordering
# ---------------------------------------------------------------------------


def cleaner_order_for_dataset(
    ledgers: Sequence[PathLedger],
    meta: DatasetMeta,
    config: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    baseline_cleaner = normalize_cleaner_name(config["trace"].get("baseline_cleaner", "mode"))
    groups = config["trace"]["cleaner_groups"]
    group_map: dict[str, list[str]] = {
        key: [normalize_cleaner_name(x) for x in value]
        for key, value in groups.items()
        if isinstance(value, list)
    }

    available_cleaners = []
    seen = set()
    for ledger in ledgers:
        if ledger.cleaner == baseline_cleaner:
            continue
        if ledger.cleaner not in seen:
            available_cleaners.append(ledger.cleaner)
            seen.add(ledger.cleaner)

    tau_turn = float(config["trace"]["tau_turn"])
    tau_hi = float(config["trace"]["tau_hi"])
    bias_cfg = config["trace"]["entry_bias"]

    if meta.q_tot < tau_turn:
        primary_group_order = ["norm"]
        secondary_group_order = ["sem", "strong"]
    elif meta.q_tot < tau_hi:
        if meta.error_dominance == "anomaly" and bool(bias_cfg.get("anomaly_sem_first_in_mid", True)):
            primary_group_order = ["sem", "norm"]
        else:
            primary_group_order = ["norm", "sem"]
        secondary_group_order = ["strong"]
    else:
        primary_group_order = ["sem", "strong"]
        secondary_group_order = ["norm"]

    if meta.error_dominance == "missing" and bool(bias_cfg.get("missing_norm_first", True)):
        if "norm" in secondary_group_order and "norm" not in primary_group_order and meta.q_tot < tau_hi:
            secondary_group_order = ["norm"] + [x for x in secondary_group_order if x != "norm"]
    if meta.error_dominance == "anomaly" and bool(bias_cfg.get("anomaly_strong_early_in_high", True)) and meta.q_tot >= tau_hi:
        primary_group_order = ["strong", "sem"] if "strong" in primary_group_order else primary_group_order

    def expand_groups(group_order: Sequence[str]) -> list[str]:
        ordered: list[str] = []
        for group_name in group_order:
            for cleaner in group_map.get(group_name, []):
                if cleaner in available_cleaners and cleaner not in ordered:
                    ordered.append(cleaner)
        return ordered

    primary_cleaners = expand_groups(primary_group_order)
    secondary_cleaners = expand_groups(secondary_group_order)

    # Keep any cleaner not present in groups at the tail, preserving original availability order.
    leftovers = [cleaner for cleaner in available_cleaners if cleaner not in primary_cleaners and cleaner not in secondary_cleaners]
    secondary_cleaners.extend(leftovers)
    start_cleaners = list(primary_cleaners)
    deferred_cleaners = [cleaner for cleaner in secondary_cleaners if cleaner not in start_cleaners]
    return start_cleaners, deferred_cleaners, list(primary_group_order), list(secondary_group_order)


# ---------------------------------------------------------------------------
# Runtime gating
# ---------------------------------------------------------------------------


def process_gate(
    clusterer_family: str,
    candidate_trial: TrialRecord,
    process_reference_trial: Optional[TrialRecord],
) -> bool:
    if process_reference_trial is None:
        return False
    cand = candidate_trial.process
    ref = process_reference_trial.process
    if clusterer_family == "centroid":
        cand_sse = to_float(cand.get("sse"))
        ref_sse = to_float(ref.get("sse"))
        if cand_sse is None or ref_sse is None:
            return False
        return cand_sse < ref_sse
    if clusterer_family == "model":
        cand_ll = to_float(cand.get("ll_end"))
        ref_ll = to_float(ref.get("ll_end"))
        if cand_ll is None or ref_ll is None:
            return False
        return cand_ll > ref_ll
    if clusterer_family == "density":
        cand_nbr = to_float(cand.get("avg_neighbor_count"))
        ref_nbr = to_float(ref.get("avg_neighbor_count"))
        cand_noise = to_float(cand.get("noise_ratio"))
        ref_noise = to_float(ref.get("noise_ratio"))
        if cand_nbr is None or ref_nbr is None or cand_noise is None or ref_noise is None:
            return False
        return (cand_nbr > ref_nbr) and (cand_noise < ref_noise)
    if clusterer_family == "hierarchical":
        cand_ratio = to_float(cand.get("ratio_intra_inter"))
        ref_ratio = to_float(ref.get("ratio_intra_inter"))
        if cand_ratio is None or ref_ratio is None:
            return False
        return cand_ratio < ref_ratio
    return False


def result_gate(meta: DatasetMeta, candidate_trial: TrialRecord, mode_best_trial: TrialRecord, config: dict[str, Any]) -> bool:
    tau_turn = float(config["trace"]["tau_turn"])
    if meta.q_tot < tau_turn:
        return True
    return (candidate_trial.sil_star > mode_best_trial.sil_star) and (candidate_trial.db_star > mode_best_trial.db_star)


def delta_h(candidate_trial: TrialRecord, mode_best_trial: TrialRecord) -> float:
    return float(candidate_trial.score - mode_best_trial.score)


def delta_lambda_nonzero(candidate_trial: TrialRecord, mode_best_trial: TrialRecord, config: dict[str, Any]) -> bool:
    return not params_equal(candidate_trial.clusterer, candidate_trial.params, mode_best_trial.params, config)


def decide_action(
    *,
    meta: DatasetMeta,
    path: PathLedger,
    current_best_trial: TrialRecord,
    mode_best_trial: TrialRecord,
    process_reference_trial: Optional[TrialRecord],
    process_reference_kind: str,
    config: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    edr_value = path.edr
    if edr_value is not None and edr_value <= 0.0:
        return "HALT", "edr_nonpositive", {
            "gp": None,
            "gr": None,
            "delta_h": None,
            "delta_lambda_nonzero": None,
            "process_ref_kind": process_reference_kind,
        }

    gp = process_gate(path.clusterer_family, current_best_trial, process_reference_trial)
    dh = delta_h(current_best_trial, mode_best_trial)
    gr = result_gate(meta, current_best_trial, mode_best_trial, config)
    dlnz = delta_lambda_nonzero(current_best_trial, mode_best_trial, config)

    if gp and dh > 0.0 and gr and (not dlnz):
        return "ADVANCE", "data_process_result_pass", {
            "gp": gp,
            "gr": gr,
            "delta_h": dh,
            "delta_lambda_nonzero": dlnz,
            "process_ref_kind": process_reference_kind,
        }
    if gp and dh > 0.0 and gr and dlnz:
        return "RETUNE", "parameter_shift_detected", {
            "gp": gp,
            "gr": gr,
            "delta_h": dh,
            "delta_lambda_nonzero": dlnz,
            "process_ref_kind": process_reference_kind,
        }
    tau_turn = float(config["trace"]["tau_turn"])
    if ((dh <= 0.0) and (not gp)) or ((meta.q_tot >= tau_turn) and (dh > 0.0) and (not gr)):
        return "HALT", "runtime_gate_failed", {
            "gp": gp,
            "gr": gr,
            "delta_h": dh,
            "delta_lambda_nonzero": dlnz,
            "process_ref_kind": process_reference_kind,
        }
    return "EVAL", "insufficient_evidence", {
        "gp": gp,
        "gr": gr,
        "delta_h": dh,
        "delta_lambda_nonzero": dlnz,
        "process_ref_kind": process_reference_kind,
    }


# ---------------------------------------------------------------------------
# Retune neighborhood
# ---------------------------------------------------------------------------


def in_retune_neighborhood(clusterer: str, candidate_params: Mapping[str, Any], center_params: Mapping[str, Any], meta: DatasetMeta, config: dict[str, Any]) -> bool:
    clusterer_norm = normalize_clusterer_name(clusterer)
    retune_cfg = config["trace"]["retune"]
    if clusterer_norm in {"KMEANS", "KMEANSNF", "KMEANSPPS"}:
        radius = int(retune_cfg.get("centroid_k_radius", 1))
        return abs(int(candidate_params.get("n_clusters", 0)) - int(center_params.get("n_clusters", 0))) <= radius
    if clusterer_norm == "GMM":
        radius = int(retune_cfg.get("gmm_n_components_radius", 1))
        return abs(int(candidate_params.get("n_components", 0)) - int(center_params.get("n_components", 0))) <= radius
    if clusterer_norm == "DBSCAN":
        eps_window = float(retune_cfg.get("dbscan_eps_window", 0.10))
        ms_window = int(retune_cfg.get("dbscan_min_samples_window", 5))
        eps_ok = abs((to_float(candidate_params.get("eps")) or 0.0) - (to_float(center_params.get("eps")) or 0.0)) <= eps_window
        ms_ok = abs(int(candidate_params.get("min_samples", 0)) - int(center_params.get("min_samples", 0))) <= ms_window
        return eps_ok and ms_ok
    if clusterer_norm == "HC":
        radius = int(retune_cfg.get("hc_k_radius", 1))
        k_ok = abs(int(candidate_params.get("n_clusters", 0)) - int(center_params.get("n_clusters", 0))) <= radius
        if not k_ok:
            return False
        if bool(retune_cfg.get("hc_fix_linkage", True)) and str(candidate_params.get("linkage")) != str(center_params.get("linkage")):
            return False
        if bool(retune_cfg.get("hc_fix_metric", True)) and str(candidate_params.get("metric")) != str(center_params.get("metric")):
            return False
        return True
    return False


def retune_sort_key(trial: TrialRecord, center_params: Mapping[str, Any], meta: DatasetMeta, config: dict[str, Any]) -> tuple[Any, ...]:
    clusterer_norm = normalize_clusterer_name(trial.clusterer)
    if clusterer_norm == "DBSCAN":
        eps_value = to_float(trial.params.get("eps")) or 0.0
        center_eps = to_float(center_params.get("eps")) or 0.0
        smaller_bias = 0 if (meta.q_tot >= float(config["trace"]["tau_hi"])) and (eps_value <= center_eps) else 1
        return (smaller_bias, abs(eps_value - center_eps), abs(int(trial.params.get("min_samples", 0)) - int(center_params.get("min_samples", 0))), trial.order_in_path)
    return (param_distance(trial.clusterer, trial.params, center_params, config), trial.order_in_path)


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def build_baseline_sequence(paths: Sequence[PathLedger]) -> list[TrialRecord]:
    sequence: list[TrialRecord] = []
    for path in sorted(paths, key=lambda x: x.original_path_order):
        sequence.extend(path.trials)
    return sequence


def _append_trial_row(
    rows: list[dict[str, Any]],
    *,
    dataset_id: int,
    dataset_name: str,
    cleaner: str,
    clusterer: str,
    phase: str,
    global_step: int,
    path_step: int,
    trial: TrialRecord,
    current_best_score: float,
    hit95_reached: bool,
    action_hint: str,
) -> None:
    rows.append(
        {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "cleaner": cleaner,
            "clusterer": clusterer,
            "phase": phase,
            "global_step": global_step,
            "path_step": path_step,
            "trial_number": trial.trial_number,
            "order_in_path": trial.order_in_path,
            "valid_trial": int(bool(trial.valid)),
            "trial_status": trial.status,
            "score": "" if not trial.valid or not np.isfinite(float(trial.score)) else trial.score,
            "silhouette": "" if not trial.valid or not np.isfinite(float(trial.silhouette)) else trial.silhouette,
            "davies_bouldin": "" if not trial.valid or not np.isfinite(float(trial.davies_bouldin)) else trial.davies_bouldin,
            "params_json": _param_json(trial.params),
            "current_best_score": "" if not np.isfinite(float(current_best_score)) else current_best_score,
            "hit95_reached": int(bool(hit95_reached)),
            "action_hint": action_hint,
            "source_file": trial.source_file,
        }
    )


def replay_dataset(
    dataset_id: int,
    meta: DatasetMeta,
    paths: Sequence[PathLedger],
    config: dict[str, Any],
    warnings: list[TraceWarning],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_cleaner = normalize_cleaner_name(config["trace"].get("baseline_cleaner", "mode"))
    full_candidate_paths = [path for path in paths if path.cleaner != normalize_cleaner_name("groundtruth")]
    if not full_candidate_paths:
        raise ValueError(f"No replayable paths for dataset_id={dataset_id}")

    h_full = max((path.full_best_trial.score for path in full_candidate_paths if path.full_best_trial is not None), default=float("nan"))
    if not np.isfinite(h_full):
        raise ValueError(f"H_full is not finite for dataset_id={dataset_id}")
    full_trial_count = sum(path.full_trial_count for path in full_candidate_paths)
    full_scored_trial_count = sum(path.scored_trial_count for path in full_candidate_paths)
    threshold_95 = 0.95 * h_full

    baseline_sequence = build_baseline_sequence(full_candidate_paths)
    baseline_hit95_step: Optional[int] = None
    baseline_best_at_trace_budget: Optional[float] = None
    baseline_rows: list[dict[str, Any]] = []
    baseline_best = -float("inf")
    for step, trial in enumerate(baseline_sequence, start=1):
        if trial.valid and np.isfinite(float(trial.score)):
            baseline_best = max(baseline_best, trial.score)
        if baseline_hit95_step is None and np.isfinite(float(baseline_best)) and baseline_best >= threshold_95:
            baseline_hit95_step = step
        baseline_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": meta.dataset_name,
                "global_step": step,
                "cleaner": trial.cleaner,
                "clusterer": trial.clusterer,
                "trial_number": trial.trial_number,
                "valid_trial": int(bool(trial.valid)),
                "trial_status": trial.status,
                "score": "" if not trial.valid or not np.isfinite(float(trial.score)) else trial.score,
                "best_so_far": "" if not np.isfinite(float(baseline_best)) else baseline_best,
                "hit95_reached": int(np.isfinite(float(baseline_best)) and baseline_best >= threshold_95),
            }
        )

    mode_paths = [path for path in full_candidate_paths if path.cleaner == baseline_cleaner]
    if not mode_paths:
        warnings.append(
            TraceWarning(
                dataset_id=dataset_id,
                cleaner=baseline_cleaner,
                clusterer="",
                kind="missing_mode_reference",
                message="No mode/c0 path exists for this dataset. Candidate replay will be skipped.",
            )
        )
    mode_best_by_clusterer = {
        path.clusterer: path.full_best_trial for path in mode_paths if path.full_best_trial is not None
    }

    clusterer_order_by_cleaner: dict[str, list[PathLedger]] = defaultdict(list)
    for path in sorted(full_candidate_paths, key=lambda x: x.original_path_order):
        clusterer_order_by_cleaner[path.cleaner].append(path)

    start_cleaners, deferred_cleaners, primary_group_order, secondary_group_order = cleaner_order_for_dataset(full_candidate_paths, meta, config)

    trace_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    global_step = 0
    current_best = -float("inf")
    hit95_step: Optional[int] = None
    consumed_trials = 0

    # Step 0: consume the full mode/c0 reference paths first.
    if bool(config["trace"].get("count_mode_reference_trials", True)):
        for mode_path in sorted(mode_paths, key=lambda x: x.original_path_order):
            for path_step, trial in enumerate(mode_path.trials, start=1):
                global_step += 1
                consumed_trials += 1
                if trial.valid and np.isfinite(float(trial.score)):
                    current_best = max(current_best, trial.score)
                if hit95_step is None and np.isfinite(float(current_best)) and current_best >= threshold_95:
                    hit95_step = global_step
                _append_trial_row(
                    trace_rows,
                    dataset_id=dataset_id,
                    dataset_name=meta.dataset_name,
                    cleaner=mode_path.cleaner,
                    clusterer=mode_path.clusterer,
                    phase="MODE_REFERENCE",
                    global_step=global_step,
                    path_step=path_step,
                    trial=trial,
                    current_best_score=current_best,
                    hit95_reached=np.isfinite(float(current_best)) and current_best >= threshold_95,
                    action_hint="MODE_REFERENCE",
                )
            decision = PathDecision(
                dataset_id=dataset_id,
                dataset_name=meta.dataset_name,
                cleaner=mode_path.cleaner,
                clusterer=mode_path.clusterer,
                phase="MODE_REFERENCE",
                evaluation_index=1,
                consumed_trials_in_path=len(mode_path.trials),
                consumed_global_trials=global_step,
                action="REFERENCE_FULL",
                reason="mode_c0_required_for_deltas",
                edr=mode_path.edr,
                delta_h=0.0,
                gp=None,
                gr=None,
                delta_lambda_nonzero=False,
                current_best_score=mode_path.full_best_trial.score if mode_path.full_best_trial else None,
                current_best_params_json=_param_json(mode_path.full_best_trial.params) if mode_path.full_best_trial else "",
                process_ref_kind="mode_best",
            )
            decision_rows.append(decision.as_dict())

    def replay_candidate_path(path: PathLedger, phase: str) -> None:
        nonlocal global_step, consumed_trials, current_best, hit95_step
        if path.clusterer not in mode_best_by_clusterer:
            warnings.append(
                TraceWarning(
                    dataset_id=dataset_id,
                    cleaner=path.cleaner,
                    clusterer=path.clusterer,
                    kind="missing_mode_clusterer_reference",
                    message=(
                        f"Mode/c0 path is missing for clusterer={path.clusterer}; candidate path skipped."
                    ),
                )
            )
            return
        mode_best_trial = mode_best_by_clusterer[path.clusterer]
        if mode_best_trial is None:
            return

        chunk_size = max(1, int(config["trace"]["eval"].get("chunk_size", 5)))
        remaining_indices = list(range(len(path.trials)))
        consumed_indices: list[int] = []
        evaluation_index = 0
        selector_mode = "sequential"
        center_params: Optional[dict[str, Any]] = None

        while remaining_indices:
            if selector_mode == "retune" and center_params is not None:
                candidate_indices = [
                    idx for idx in remaining_indices
                    if (not path.trials[idx].valid)
                    or in_retune_neighborhood(path.clusterer, path.trials[idx].params, center_params, meta, config)
                ]
                candidate_indices.sort(
                    key=lambda idx: (
                        1 if not path.trials[idx].valid else 0,
                        retune_sort_key(path.trials[idx], center_params, meta, config) if path.trials[idx].valid else (path.trials[idx].order_in_path,),
                    )
                )
            else:
                candidate_indices = list(remaining_indices)

            if not candidate_indices:
                break

            batch_indices = candidate_indices[:chunk_size]
            consumed_indices.extend(batch_indices)
            remaining_indices = [idx for idx in remaining_indices if idx not in set(batch_indices)]
            consumed_trials_for_path = [path.trials[idx] for idx in sorted(consumed_indices, key=lambda i: path.trials[i].order_in_path)]

            for path_step, idx in enumerate(batch_indices, start=(len(consumed_indices) - len(batch_indices) + 1)):
                trial = path.trials[idx]
                global_step += 1
                consumed_trials += 1
                if trial.valid and np.isfinite(float(trial.score)):
                    current_best = max(current_best, trial.score)
                if hit95_step is None and np.isfinite(float(current_best)) and current_best >= threshold_95:
                    hit95_step = global_step
                _append_trial_row(
                    trace_rows,
                    dataset_id=dataset_id,
                    dataset_name=meta.dataset_name,
                    cleaner=path.cleaner,
                    clusterer=path.clusterer,
                    phase=phase,
                    global_step=global_step,
                    path_step=path_step,
                    trial=trial,
                    current_best_score=current_best,
                    hit95_reached=np.isfinite(float(current_best)) and current_best >= threshold_95,
                    action_hint=selector_mode.upper(),
                )

            evaluation_index += 1
            valid_consumed_trials = [
                item for item in consumed_trials_for_path
                if item.valid and np.isfinite(float(item.score))
            ]
            if not valid_consumed_trials:
                decision = PathDecision(
                    dataset_id=dataset_id,
                    dataset_name=meta.dataset_name,
                    cleaner=path.cleaner,
                    clusterer=path.clusterer,
                    phase=phase,
                    evaluation_index=evaluation_index,
                    consumed_trials_in_path=len(consumed_indices),
                    consumed_global_trials=global_step,
                    action="ADVANCE",
                    reason="no_valid_scored_trial_yet",
                    edr=path.edr,
                    delta_h=None,
                    gp=None,
                    gr=None,
                    delta_lambda_nonzero=None,
                    current_best_score=None,
                    current_best_params_json="",
                    process_ref_kind="missing",
                )
                decision_rows.append(decision.as_dict())
                continue
            current_best_trial = max(valid_consumed_trials, key=lambda item: item.score)
            process_ref_trial, process_ref_kind = pick_mode_process_reference(next(p for p in mode_paths if p.clusterer == path.clusterer), current_best_trial, config)
            action, reason, detail = decide_action(
                meta=meta,
                path=path,
                current_best_trial=current_best_trial,
                mode_best_trial=mode_best_trial,
                process_reference_trial=process_ref_trial,
                process_reference_kind=process_ref_kind,
                config=config,
            )
            decision = PathDecision(
                dataset_id=dataset_id,
                dataset_name=meta.dataset_name,
                cleaner=path.cleaner,
                clusterer=path.clusterer,
                phase=phase,
                evaluation_index=evaluation_index,
                consumed_trials_in_path=len(consumed_indices),
                consumed_global_trials=global_step,
                action=action,
                reason=reason,
                edr=path.edr,
                delta_h=detail.get("delta_h"),
                gp=detail.get("gp"),
                gr=detail.get("gr"),
                delta_lambda_nonzero=detail.get("delta_lambda_nonzero"),
                current_best_score=current_best_trial.score,
                current_best_params_json=_param_json(current_best_trial.params),
                process_ref_kind=detail.get("process_ref_kind", process_ref_kind),
            )
            decision_rows.append(decision.as_dict())

            if action == "HALT":
                break
            if action == "RETUNE":
                selector_mode = "retune"
                center_params = dict(current_best_trial.params)
                continue
            selector_mode = "sequential"
            center_params = None

    # Candidate cleaners follow TRACE entry priority.
    for cleaner_name in start_cleaners:
        for path in clusterer_order_by_cleaner.get(cleaner_name, []):
            replay_candidate_path(path, phase="PRIMARY")
    for cleaner_name in deferred_cleaners:
        for path in clusterer_order_by_cleaner.get(cleaner_name, []):
            replay_candidate_path(path, phase="DEFERRED")

    if consumed_trials <= len(baseline_rows):
        baseline_best_at_trace_budget = baseline_rows[consumed_trials - 1]["best_so_far"] if consumed_trials > 0 else -float("inf")
    else:
        baseline_best_at_trace_budget = baseline_rows[-1]["best_so_far"] if baseline_rows else -float("inf")
    baseline_best_at_trace_budget = to_float(baseline_best_at_trace_budget)
    if baseline_best_at_trace_budget is None or not np.isfinite(float(baseline_best_at_trace_budget)):
        baseline_best_at_trace_budget = 0.0

    if not np.isfinite(float(current_best)):
        current_best = 0.0
    trace_retention = current_best / h_full if h_full > 0 else float("nan")
    trace_hit95_progress = (hit95_step / full_trial_count) if hit95_step is not None else (1.0 if bool(config["trace"].get("miss_hit95_as_full_progress", True)) else float("nan"))
    baseline_hit95_progress = (baseline_hit95_step / full_trial_count) if baseline_hit95_step is not None else float("nan")
    baseline_retention_at_trace_budget = baseline_best_at_trace_budget / h_full if h_full > 0 else float("nan")

    summary = {
        "dataset_id": dataset_id,
        "dataset_name": meta.dataset_name,
        "dirty_csv": str(meta.dirty_csv_path),
        "clean_csv": str(meta.clean_csv_path) if meta.clean_csv_path else "",
        "dirty_id": meta.dirty_id if meta.dirty_id is not None else "",
        "q_tot": meta.q_tot,
        "missing_rate": "" if meta.missing_rate is None else meta.missing_rate,
        "noise_rate": "" if meta.noise_rate is None else meta.noise_rate,
        "error_rate": "" if meta.error_rate is None else meta.error_rate,
        "error_dominance": meta.error_dominance,
        "tau_turn": float(config["trace"]["tau_turn"]),
        "tau_hi": float(config["trace"]["tau_hi"]),
        "start_cleaners_json": json.dumps(start_cleaners, ensure_ascii=False),
        "deferred_cleaners_json": json.dumps(deferred_cleaners, ensure_ascii=False),
        "primary_group_order_json": json.dumps(primary_group_order, ensure_ascii=False),
        "secondary_group_order_json": json.dumps(secondary_group_order, ensure_ascii=False),
        "h_full": h_full,
        "full_trial_count": full_trial_count,
        "full_scored_trial_count": full_scored_trial_count,
        "trace_trials_consumed": consumed_trials,
        "trace_best_score": current_best,
        "trace_hit95_step": "" if hit95_step is None else hit95_step,
        "trace_hit95_progress": trace_hit95_progress,
        "trace_score_retention": trace_retention,
        "baseline_hit95_step": "" if baseline_hit95_step is None else baseline_hit95_step,
        "baseline_hit95_progress": baseline_hit95_progress,
        "baseline_best_at_trace_budget": baseline_best_at_trace_budget,
        "baseline_retention_at_trace_budget": baseline_retention_at_trace_budget,
        "mode_paths_count": len(mode_paths),
        "candidate_paths_count": len(full_candidate_paths) - len(mode_paths),
    }
    return summary, trace_rows, decision_rows, baseline_rows


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------


def _median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return None
    return float(np.median(vals))


def replay_trace(
    *,
    project_root: Path,
    results_dir: Optional[Path] = None,
    config_path: Path = Path("configs/trace.yaml"),
    output_dir: Optional[Path] = None,
    dataset_ids: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config = load_trace_config(project_root / config_path if not Path(config_path).is_absolute() else Path(config_path))
    results_root = autodetect_results_root(project_root, results_dir)
    output_root = resolve_path(output_dir or config["trace"]["paths"]["output_dir"], [project_root])
    assert output_root is not None
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_manifest = load_dataset_manifest(project_root, config)
    dataset_meta, warnings = build_dataset_meta(project_root, results_root, config, dataset_manifest)
    ledgers_by_dataset, ledger_warnings = build_path_ledgers(project_root, results_root, config, dataset_meta)
    warnings.extend(ledger_warnings)

    selected_ids = sorted(ledgers_by_dataset.keys())
    if dataset_ids:
        selected = set(int(x) for x in dataset_ids)
        selected_ids = [dataset_id for dataset_id in selected_ids if dataset_id in selected]

    dataset_summaries: list[dict[str, Any]] = []
    trace_rows_all: list[dict[str, Any]] = []
    decision_rows_all: list[dict[str, Any]] = []
    baseline_rows_all: list[dict[str, Any]] = []

    for dataset_id in selected_ids:
        summary, trace_rows, decision_rows, baseline_rows = replay_dataset(
            dataset_id,
            dataset_meta[dataset_id],
            ledgers_by_dataset[dataset_id],
            config,
            warnings,
        )
        dataset_summaries.append(summary)
        trace_rows_all.extend(trace_rows)
        decision_rows_all.extend(decision_rows)
        baseline_rows_all.extend(baseline_rows)

    summary_columns = [
        "dataset_id",
        "dataset_name",
        "dirty_csv",
        "clean_csv",
        "dirty_id",
        "q_tot",
        "missing_rate",
        "noise_rate",
        "error_rate",
        "error_dominance",
        "tau_turn",
        "tau_hi",
        "start_cleaners_json",
        "deferred_cleaners_json",
        "primary_group_order_json",
        "secondary_group_order_json",
        "h_full",
        "full_trial_count",
        "full_scored_trial_count",
        "trace_trials_consumed",
        "trace_best_score",
        "trace_hit95_step",
        "trace_hit95_progress",
        "trace_score_retention",
        "baseline_hit95_step",
        "baseline_hit95_progress",
        "baseline_best_at_trace_budget",
        "baseline_retention_at_trace_budget",
        "mode_paths_count",
        "candidate_paths_count",
    ]
    write_csv(output_root / config["trace"]["paths"]["dataset_summary_csv"], dataset_summaries, summary_columns)

    trace_columns = [
        "dataset_id",
        "dataset_name",
        "cleaner",
        "clusterer",
        "phase",
        "global_step",
        "path_step",
        "trial_number",
        "order_in_path",
        "valid_trial",
        "trial_status",
        "score",
        "silhouette",
        "davies_bouldin",
        "params_json",
        "current_best_score",
        "hit95_reached",
        "action_hint",
        "source_file",
    ]
    write_csv(output_root / config["trace"]["paths"]["trial_log_csv"], trace_rows_all, trace_columns)

    decision_columns = [
        "dataset_id",
        "dataset_name",
        "cleaner",
        "clusterer",
        "phase",
        "evaluation_index",
        "consumed_trials_in_path",
        "consumed_global_trials",
        "action",
        "reason",
        "edr",
        "delta_h",
        "gp",
        "gr",
        "delta_lambda_nonzero",
        "current_best_score",
        "current_best_params_json",
        "process_ref_kind",
    ]
    write_csv(output_root / config["trace"]["paths"]["path_decisions_csv"], decision_rows_all, decision_columns)

    baseline_columns = [
        "dataset_id",
        "dataset_name",
        "global_step",
        "cleaner",
        "clusterer",
        "trial_number",
        "valid_trial",
        "trial_status",
        "score",
        "best_so_far",
        "hit95_reached",
    ]
    write_csv(output_root / config["trace"]["paths"]["baseline_csv"], baseline_rows_all, baseline_columns)

    aggregate = {
        "project_root": str(project_root),
        "results_root": str(results_root),
        "output_root": str(output_root),
        "config_path": str(config_path),
        "n_datasets": len(dataset_summaries),
        "median_trace_hit95_progress": _median(to_float(row.get("trace_hit95_progress")) for row in dataset_summaries),
        "median_trace_score_retention": _median(to_float(row.get("trace_score_retention")) for row in dataset_summaries),
        "median_baseline_hit95_progress": _median(to_float(row.get("baseline_hit95_progress")) for row in dataset_summaries),
        "median_baseline_retention_at_trace_budget": _median(to_float(row.get("baseline_retention_at_trace_budget")) for row in dataset_summaries),
        "datasets_missing_hit95": [
            int(row["dataset_id"])
            for row in dataset_summaries
            if (row.get("trace_hit95_step") in ("", None))
        ],
        "warning_count": len(warnings),
    }

    write_json(output_root / config["trace"]["paths"]["aggregate_json"], aggregate)
    write_json(output_root / config["trace"]["paths"]["warnings_json"], [item.as_dict() for item in warnings])

    return {
        "aggregate": aggregate,
        "dataset_summaries_path": str(output_root / config["trace"]["paths"]["dataset_summary_csv"]),
        "trace_trials_path": str(output_root / config["trace"]["paths"]["trial_log_csv"]),
        "path_decisions_path": str(output_root / config["trace"]["paths"]["path_decisions_csv"]),
        "baseline_path": str(output_root / config["trace"]["paths"]["baseline_csv"]),
        "warnings_path": str(output_root / config["trace"]["paths"]["warnings_json"]),
        "aggregate_path": str(output_root / config["trace"]["paths"]["aggregate_json"]),
    }
