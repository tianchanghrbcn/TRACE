#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run data-cleaning methods for one dirty dataset.

This module keeps the original cleaner logic and result contract:

- Cleaning outputs are first produced under src/cleaning/Repaired_res/.
- The final repaired file is copied to results/cleaned_data/<cleaner>/.
- The returned path points to results/cleaned_data/<cleaner>/repaired_<dataset_id>.csv.

TRACE changes:
- Resolve all project paths from the repository root.
- Avoid hard-coded absolute paths.
- Keep the original cleaner ids and command structure.
- Use English logs and comments.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(
    os.environ.get(
        "TRACE_PROJECT_ROOT",
        os.environ.get("CLEANING_CLUSTERING_ROOT", Path(__file__).resolve().parents[2]),
    )
).resolve()

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_SUBPROCESS_CWD = PROJECT_ROOT


CLEANER_ID_TO_NAME = {
    1: "mode",
    2: "baran",
    3: "holoclean",
    4: "bigdansing",
    5: "boostclean",
    6: "horizon",
    7: "scared",
    8: "Unified",
    9: "google_gemini",
}


def conda_run_python(env_name: str) -> list[str]:
    """
    Build a `conda run -n <env> python` command prefix.

    This avoids hard-coded interpreter paths while preserving the original
    multi-environment setup.
    """
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_exe:
        raise RuntimeError(
            "Cannot find the conda executable. Run:\n"
            "  source ~/miniconda3/etc/profile.d/conda.sh\n"
            "or make sure conda is available in PATH."
        )
    return [conda_exe, "run", "-n", env_name, "python"]


def _print_process_output(stdout: Optional[str], stderr: Optional[str]) -> None:
    if stdout:
        print(stdout.strip())
    if stderr:
        cleaned_stderr = re.sub(r"RUNTIME=\d+(\.\d+)?", "", stderr).strip()
        if cleaned_stderr:
            print(cleaned_stderr)


def run_with_time(
    command: list[str],
    *,
    cwd: Optional[Path | str] = DEFAULT_SUBPROCESS_CWD,
    text: bool = True,
) -> tuple[subprocess.CompletedProcess, float]:
    """
    Run a command and return (CompletedProcess, runtime_seconds).

    On Linux, /usr/bin/time is used to record wall-clock time in stderr.
    On systems without /usr/bin/time, this function falls back to Python's
    perf_counter. Runtime is recorded for logging only and is not a paper claim.
    """
    command = [str(item) for item in command]
    cwd_str = str(cwd) if cwd is not None else None

    time_bin = Path("/usr/bin/time")
    if time_bin.exists():
        timed_command = [str(time_bin), "-f", "RUNTIME=%e"] + command
        completed = subprocess.run(
            timed_command,
            cwd=cwd_str,
            text=text,
            capture_output=True,
        )

        match = re.search(r"RUNTIME=(\d+(\.\d+)?)", completed.stderr or "")
        runtime = float(match.group(1)) if match else 0.0

        _print_process_output(completed.stdout, completed.stderr)

        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )

        return completed, runtime

    start_time = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd_str,
        text=text,
        capture_output=True,
    )
    runtime = time.perf_counter() - start_time

    _print_process_output(completed.stdout, completed.stderr)

    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    return completed, runtime


def _cleaning_script(*parts: str) -> str:
    return str(PROJECT_ROOT / "src" / "cleaning" / Path(*parts))


def _dataset_rule_path(dataset_name: str, rule_file: str) -> str:
    return str(PROJECT_ROOT / "datasets" / "train" / dataset_name / rule_file)


def _find_repaired_csv(repaired_res_dir: Path) -> Optional[Path]:
    pattern = re.compile(r"^repaired_.*\.csv$")
    if not repaired_res_dir.exists():
        return None

    for file_name in os.listdir(repaired_res_dir):
        if pattern.match(file_name):
            return repaired_res_dir / file_name
    return None


def run_error_correction(
    dataset_path: str,
    dataset_id: int,
    algorithm_id: int,
    clean_csv_path: str,
    output_dir: str,
    project_root: Optional[Path | str] = None,
    results_dir: Optional[Path | str] = None,
) -> tuple[Optional[str], Optional[float]]:
    """
    Run one cleaning algorithm on one dirty CSV file.

    Parameters are kept compatible with the original pipeline. Additional
    project_root and results_dir arguments are optional and are used by TRACE
    scripts to avoid cwd-dependent behavior.
    """
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    result_root = Path(results_dir).resolve() if results_dir else root / "results"

    dirty_path = str(Path(dataset_path).resolve())
    clean_path = str(Path(clean_csv_path).resolve())
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    algo_name = CLEANER_ID_TO_NAME.get(algorithm_id)
    if algo_name is None:
        print(f"[ERROR] Unsupported cleaner id: {algorithm_id}")
        return None, None

    task_name_1 = f"dataset_{dataset_id}_algo_{algorithm_id}"
    dataset_name = Path(clean_path).parent.name

    rule_path_holoclean = _dataset_rule_path(dataset_name, "dc_rules_holoclean.txt")
    rule_path_horizon = _dataset_rule_path(dataset_name, "dc_rules-validate-fd-horizon.txt")

    runtime: Optional[float] = None

    try:
        if algorithm_id == 1:
            mode_output_dir = (
                    root
                    / "src"
                    / "cleaning"
                    / "Repaired_res"
                    / "mode"
                    / dataset_name
            )

            command = [
                sys.executable,
                _cleaning_script("mode", "mode.py"),
                "--clean_path",
                clean_path,
                "--dirty_path",
                dirty_path,
                "--task_name",
                dataset_name,
                "--output_dir",
                str(mode_output_dir),
            ]
            _, runtime = run_with_time(command, cwd=DEFAULT_SUBPROCESS_CWD)
            print(f"[INFO] Mode completed for `{dataset_name}` ({runtime:.2f}s).")

        elif algorithm_id == 2:
            try:
                df_dirty = pd.read_csv(dirty_path, encoding="utf-8-sig")
                index_attribute = df_dirty.columns[0]
            except Exception as exc:
                print(f"[ERROR] Failed to read dirty dataset: {exc}")
                return None, None

            command = [
                sys.executable,
                _cleaning_script("baran", "correction_with_baran.py"),
                "--dirty_path",
                dirty_path,
                "--clean_path",
                clean_path,
                "--task_name",
                task_name_1,
                "--output_path",
                str(output_path),
                "--index_attribute",
                str(index_attribute),
            ]

            print(
                "[INFO] Running cleaner: "
                f"id={algorithm_id}, name={algo_name}, dataset_id={dataset_id}"
            )

            start_time = time.perf_counter()
            process = subprocess.Popen(
                command,
                cwd=str(DEFAULT_SUBPROCESS_CWD),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            repaired_file: Optional[str] = None
            stdout_lines: list[str] = []

            while process.poll() is None:
                line = process.stdout.readline() if process.stdout else ""
                if not line:
                    continue

                stripped = line.strip()
                stdout_lines.append(stripped)
                print(stripped)

                if "Repaired data saved to" in line:
                    match = re.search(r"Repaired data saved to\s+(.+\.csv)", line)
                    if match:
                        repaired_file = match.group(1).strip()
                        process.terminate()
                        break

            stdout, stderr = process.communicate()
            if stdout:
                stdout_lines.extend(stdout.splitlines())
            if stderr:
                print(stderr.strip())

            full_stdout = "\n".join(stdout_lines)
            if not repaired_file:
                match = re.search(
                    r"Repaired data saved to\s+(.+\.csv)",
                    full_stdout,
                    re.MULTILINE,
                )
                if match:
                    repaired_file = match.group(1).strip()

            if not repaired_file:
                print("[ERROR] Baran did not report a repaired CSV path.")
                return None, None

            baran_res_dir = root / "src" / "cleaning" / "Repaired_res" / "baran" / dataset_name
            baran_res_dir.mkdir(parents=True, exist_ok=True)

            destination = baran_res_dir / Path(repaired_file).name
            try:
                shutil.copy(repaired_file, destination)
                print(f"[INFO] Copied Baran result to {destination}")
            except Exception as exc:
                print(f"[ERROR] Failed to copy Baran result: {exc}")
                return None, None

            runtime = time.perf_counter() - start_time
            print(f"[INFO] Baran completed for `{dataset_name}` ({runtime:.2f}s).")

        elif algorithm_id == 3:
            command = (
                ["timeout", "1d"]
                + conda_run_python("hc37")
                + [
                    _cleaning_script("holoclean-master", "holoclean_run.py"),
                    "--dirty_path",
                    dirty_path,
                    "--clean_path",
                    clean_path,
                    "--task_name",
                    dataset_name,
                    "--rule_path",
                    rule_path_holoclean,
                    "--onlyed",
                    "0",
                    "--perfected",
                    "0",
                ]
            )
            _, runtime = run_with_time(command, cwd=DEFAULT_SUBPROCESS_CWD)
            print(f"[INFO] HoloClean completed for `{dataset_name}` ({runtime:.2f}s).")

        elif algorithm_id == 4:
            command = (
                conda_run_python("torch110")
                + [
                    _cleaning_script("BigDansing_Holistic", "bigdansing.py"),
                    "--task_name",
                    dataset_name,
                    "--rule_path",
                    rule_path_holoclean,
                    "--onlyed",
                    "0",
                    "--perfected",
                    "0",
                    "--dirty_path",
                    dirty_path,
                    "--clean_path",
                    clean_path,
                ]
            )
            _, runtime = run_with_time(command, cwd=DEFAULT_SUBPROCESS_CWD)
            print(f"[INFO] BigDansing completed for `{dataset_name}` ({runtime:.2f}s).")

        elif algorithm_id == 5:
            command = (
                conda_run_python("activedetect")
                + [
                    _cleaning_script("BoostClean", "activedetect", "experiments", "Experiment.py"),
                    "--task_name",
                    dataset_name,
                    "--rule_path",
                    rule_path_holoclean,
                    "--onlyed",
                    "0",
                    "--perfected",
                    "0",
                    "--dirty_path",
                    dirty_path,
                    "--clean_path",
                    clean_path,
                ]
            )
            _, runtime = run_with_time(command, cwd=DEFAULT_SUBPROCESS_CWD)
            print(f"[INFO] BoostClean completed for `{dataset_name}` ({runtime:.2f}s).")

        elif algorithm_id == 6:
            command = (
                conda_run_python("torch110")
                + [
                    _cleaning_script("horizon", "horizon.py"),
                    "--task_name",
                    dataset_name,
                    "--rule_path",
                    rule_path_horizon,
                    "--onlyed",
                    "0",
                    "--perfected",
                    "0",
                    "--dirty_path",
                    dirty_path,
                    "--clean_path",
                    clean_path,
                ]
            )
            _, runtime = run_with_time(command, cwd=DEFAULT_SUBPROCESS_CWD)
            print(f"[INFO] Horizon completed for `{dataset_name}` ({runtime:.2f}s).")

        elif algorithm_id == 7:
            command = (
                conda_run_python("torch110")
                + [
                    _cleaning_script("SCAREd", "scared.py"),
                    "--task_name",
                    dataset_name,
                    "--rule_path",
                    rule_path_horizon,
                    "--onlyed",
                    "0",
                    "--perfected",
                    "0",
                    "--dirty_path",
                    dirty_path,
                    "--clean_path",
                    clean_path,
                ]
            )
            _, runtime = run_with_time(command, cwd=DEFAULT_SUBPROCESS_CWD)
            print(f"[INFO] SCAReD completed for `{dataset_name}` ({runtime:.2f}s).")

        elif algorithm_id == 8:
            command = (
                conda_run_python("torch110")
                + [
                    _cleaning_script("Unified", "Unified.py"),
                    "--task_name",
                    dataset_name,
                    "--rule_path",
                    rule_path_horizon,
                    "--onlyed",
                    "0",
                    "--perfected",
                    "0",
                    "--dirty_path",
                    dirty_path,
                    "--clean_path",
                    clean_path,
                ]
            )
            _, runtime = run_with_time(command, cwd=DEFAULT_SUBPROCESS_CWD)
            print(f"[INFO] Unified completed for `{dataset_name}` ({runtime:.2f}s).")

        elif algorithm_id == 9:
            command = (
                conda_run_python("gm39")
                + [
                    "gemini_cleaning.py",
                    "--input_csv",
                    dirty_path,
                    "--output_dir",
                    str(output_path),
                ]
            )
            _, runtime = run_with_time(
                command,
                cwd=root / "src" / "cleaning" / "google_gemini",
            )
            print(f"[INFO] Google Gemini completed for `{dataset_name}` ({runtime:.2f}s).")

    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Cleaner failed: id={algorithm_id}, name={algo_name}")
        if exc.output:
            print(f"[STDOUT]\n{exc.output}")
        if exc.stderr:
            print(f"[STDERR]\n{exc.stderr}")
        return None, None
    except Exception as exc:
        print(f"[ERROR] Unexpected cleaner error: {exc}")
        return None, None

    repaired_res_dir = root / "src" / "cleaning" / "Repaired_res" / algo_name / dataset_name
    repaired_csv = _find_repaired_csv(repaired_res_dir)

    if repaired_csv is None:
        print(f"[ERROR] No repaired CSV found in {repaired_res_dir}")
        return None, None

    cleaned_data_dir = result_root / "cleaned_data" / algo_name
    cleaned_data_dir.mkdir(parents=True, exist_ok=True)

    final_path = cleaned_data_dir / f"repaired_{dataset_id}.csv"
    shutil.copy(str(repaired_csv), str(final_path))

    print(
        "[INFO] Cleaner output copied: "
        f"id={algorithm_id}, name={algo_name}, path={final_path}"
    )

    if runtime is None:
        runtime = 0.0

    return str(final_path), runtime