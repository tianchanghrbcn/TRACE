#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run clustering methods for a cleaned CSV file.

This module keeps the original clustering contract:

- Each clustering algorithm is implemented as a standalone script under
  src/clustering/<METHOD>/<METHOD>.py.
- Inputs and outputs are passed through environment variables expected by
  those scripts.
- Clustered outputs are written under results/clustered_data/.

TRACE changes:
- Resolve script and output paths from the project root.
- Use OUTPUT_DIR consistently so clustering scripts do not infer paths from cwd.
- Do not mutate os.environ globally; pass a subprocess-specific environment.
- Support TRACE_N_TRIALS for smoke tests.
- Use English logs and comments.
- Preserve the original ClusterMethod numeric ids.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()


class ClusterMethod(Enum):
    """Legacy clustering method ids used by the original pipeline."""

    HC = 0
    DBSCAN = 1
    GMM = 2
    KMEANS = 3
    KMEANSNF = 4
    KMEANSPPS = 5


def _resolve_project_root(project_root: Optional[Path | str] = None) -> Path:
    if project_root is None:
        return PROJECT_ROOT
    return Path(project_root).resolve()


def _parse_runtime(stdout: str, fallback_runtime: float) -> float:
    """
    Parse runtime printed by clustering scripts.

    Supported formats include:
    - Program completed in: 12.34 seconds
    - Program completed in 12.34 sec
    - Program completed in 12.34 seconds

    If no known line is found, fall back to wall-clock time measured here.
    """
    patterns = [
        r"Program completed in:\s*([\d.]+)\s*seconds",
        r"Program completed in\s*([\d.]+)\s*seconds",
        r"Program completed in\s*([\d.]+)\s*sec",
        r"total_runtime_sec[\"']?\s*[:=]\s*([\d.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    return fallback_runtime


def run_clustering(
    dataset_id: int,
    algorithm: str,
    cluster_method_id: int,
    cleaned_file_path: str,
    project_root: Optional[Path | str] = None,
    results_dir: Optional[Path | str] = None,
    python_executable: Optional[str] = None,
    cluster_trials: Optional[int] = None,
    clean_state: str = "cleaned",
) -> tuple[Optional[str], Optional[float]]:
    """
    Run one clustering algorithm on one cleaned CSV file.

    Parameters
    ----------
    dataset_id:
        Legacy numeric id of the dirty dataset record.
    algorithm:
        Name of the cleaning algorithm that produced the cleaned CSV.
    cluster_method_id:
        Numeric id from ClusterMethod.
    cleaned_file_path:
        Path to the cleaned CSV produced by a cleaner.
    project_root:
        Optional TRACE repository root. Defaults to auto-detected root.
    results_dir:
        Optional results directory. Defaults to <project_root>/results.
    python_executable:
        Optional Python executable for clustering scripts. Defaults to the
        current interpreter.
    cluster_trials:
        Optional number of Optuna trials for clustering scripts that support
        TRACE_N_TRIALS. Useful for Mode B smoke tests.
    clean_state:
        Optional state label passed to clustering scripts.

    Returns
    -------
    (output_dir, runtime) on success, (None, None) on failure.
    """
    root = _resolve_project_root(project_root)
    result_root = Path(results_dir).resolve() if results_dir else root / "results"
    python_bin = python_executable or sys.executable

    try:
        cluster_method = ClusterMethod(cluster_method_id).name
    except ValueError as exc:
        print(f"[ERROR] Invalid clustering method id: {cluster_method_id} ({exc})")
        return None, None

    cluster_script_path = root / "src" / "clustering" / cluster_method / f"{cluster_method}.py"
    if not cluster_script_path.exists():
        print(f"[ERROR] Clustering script not found: {cluster_script_path}")
        return None, None

    cleaned_path = Path(cleaned_file_path).resolve()
    if not cleaned_path.exists():
        print(f"[ERROR] Cleaned CSV file not found: {cleaned_path}")
        return None, None

    output_dir = (
        result_root
        / "clustered_data"
        / cluster_method
        / algorithm
        / f"clustered_{dataset_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TRACE_PROJECT_ROOT"] = str(root)
    env["CSV_FILE_PATH"] = str(cleaned_path)
    env["DATASET_ID"] = str(dataset_id)
    env["ALGO"] = algorithm
    env["OUTPUT_DIR"] = str(output_dir)
    env["CLEAN_STATE"] = clean_state

    if cluster_trials is not None:
        env["TRACE_N_TRIALS"] = str(int(cluster_trials))

    command = [python_bin, str(cluster_script_path)]

    print(
        "[INFO] Running clustering: "
        f"method={cluster_method}, dataset_id={dataset_id}, cleaner={algorithm}"
    )
    if cluster_trials is not None:
        print(f"[INFO] Clustering trial budget: {cluster_trials}")

    start_time = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            cwd=str(cluster_script_path.parent),
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Clustering failed: method={cluster_method}, dataset_id={dataset_id}")
        if exc.stdout:
            print(f"[STDOUT]\n{exc.stdout}")
        if exc.stderr:
            print(f"[STDERR]\n{exc.stderr}")
        return None, None
    except Exception as exc:
        print(f"[ERROR] Unexpected clustering error: {exc}")
        return None, None

    fallback_runtime = time.perf_counter() - start_time
    runtime = _parse_runtime(completed.stdout, fallback_runtime)

    if completed.stdout:
        print(f"[INFO] Clustering stdout:\n{completed.stdout}")
    if completed.stderr:
        # stderr may contain warnings from dependencies. Keep it visible, but
        # successful execution should not be treated as an error.
        print(f"[INFO] Clustering stderr:\n{completed.stderr}")

    print(
        "[INFO] Clustering completed: "
        f"method={cluster_method}, dataset_id={dataset_id}, runtime={runtime:.2f}s"
    )

    return str(output_dir), runtime