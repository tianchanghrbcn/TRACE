#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze clustered results produced by the original pipeline.

This module reads:
- results/eigenvectors.json
- results/clustered_results.json
- clustering output files under each clustered_file_path

It then selects top-K clustering results for each dataset_id and writes:
- results/analyzed_results.json

The structure and purpose are kept compatible with the original pipeline.
TRACE changes are limited to English logs, path robustness, and safer parsing.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Optional


DEFAULT_K_VALUE = 5
INVALID_SCORE_THRESHOLD = 3.0


def read_k_value(preprocessing_file_path: str | Path) -> int:
    """
    Read K_VALUE from pre-processing.py.

    If K_VALUE cannot be found, return DEFAULT_K_VALUE.
    """
    path = Path(preprocessing_file_path)

    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as exc:
        print(f"[WARNING] Failed to read K_VALUE from {path}: {exc}")
        return DEFAULT_K_VALUE

    match = re.search(r"^\s*K_VALUE\s*=\s*(\d+)", text, flags=re.MULTILINE)
    if not match:
        print(f"[WARNING] K_VALUE was not found in {path}; use default {DEFAULT_K_VALUE}.")
        return DEFAULT_K_VALUE

    return int(match.group(1))


def _extract_best_parameters(text: str) -> dict[str, Any]:
    """
    Extract the 'Best parameters' dictionary from a clustering output file.
    """
    patterns = [
        r"Best parameters:\s*(\{.*?\})",
        r"Best Parameters:\s*(\{.*?\})",
        r"best_params\s*=\s*(\{.*?\})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue

        raw = match.group(1).strip()
        try:
            value = ast.literal_eval(raw)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    return {}


def _extract_final_score(text: str) -> Optional[float]:
    """
    Extract the final combined score from a clustering output file.
    """
    patterns = [
        r"Final Combined Score:\s*([+-]?\d+(?:\.\d+)?)",
        r"Final combined score:\s*([+-]?\d+(?:\.\d+)?)",
        r"Combined Score:\s*([+-]?\d+(?:\.\d+)?)",
        r"final_score\s*=\s*([+-]?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))

    return None


def _candidate_result_files(clustered_file_path: str | Path, dataset_id: int) -> list[Path]:
    """
    Return possible clustering result text files.

    Historical clustering scripts usually write repaired_<dataset_id>.txt.
    If that exact file is missing, fall back to all txt files in the output
    directory so that minor naming differences do not break analysis.
    """
    output_dir = Path(clustered_file_path)

    if output_dir.is_file():
        return [output_dir]

    if not output_dir.exists():
        return []

    expected = output_dir / f"repaired_{dataset_id}.txt"
    if expected.exists():
        return [expected]

    return sorted(output_dir.glob("*.txt"))


def parse_clustering_result(
    clustered_file_path: str | Path,
    dataset_id: int,
) -> tuple[dict[str, Any], Optional[float], Optional[str]]:
    """
    Parse one clustering result directory.

    Returns:
        (best_params, final_score, parsed_file_path)

    If no valid result file is found, final_score is None.
    """
    candidates = _candidate_result_files(clustered_file_path, dataset_id)
    if not candidates:
        print(f"[WARNING] No clustering result file found under {clustered_file_path}.")
        return {}, None, None

    for file_path in candidates:
        try:
            text = file_path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception as exc:
            print(f"[WARNING] Failed to read clustering result file {file_path}: {exc}")
            continue

        best_params = _extract_best_parameters(text)
        final_score = _extract_final_score(text)

        if final_score is not None:
            return best_params, final_score, str(file_path)

    print(f"[WARNING] Failed to parse a final score under {clustered_file_path}.")
    return {}, None, None


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_analyzed_results(
    preprocessing_file_path: str,
    eigenvectors_path: str,
    clustered_results_path: str,
    output_path: str,
) -> None:
    """
    Select top-K clustering results for each dataset_id.

    The original logic is preserved:
    - Read K_VALUE from pre-processing.py.
    - Iterate over dataset_id values from eigenvectors.json.
    - Match candidate results from clustered_results.json.
    - Parse best parameters and final score from clustering output files.
    - Ignore invalid scores greater than or equal to INVALID_SCORE_THRESHOLD.
    - Sort candidates by final_score descending and keep top-K.
    """
    k_value = read_k_value(preprocessing_file_path)
    print(f"[INFO] Top-K value: {k_value}")

    try:
        eigenvectors = _load_json(eigenvectors_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read eigenvectors file {eigenvectors_path}: {exc}")
        _write_json(output_path, [])
        return

    try:
        clustered_results = _load_json(clustered_results_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read clustered results file {clustered_results_path}: {exc}")
        _write_json(output_path, [])
        return

    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in clustered_results:
        dataset_id = int(item.get("dataset_id", -1))
        grouped.setdefault(dataset_id, []).append(item)

    analyzed_results: list[dict[str, Any]] = []

    for vector in eigenvectors:
        dataset_id = int(vector.get("dataset_id"))
        dataset_name = vector.get("dataset_name")
        csv_file = vector.get("csv_file")

        candidates = grouped.get(dataset_id, [])
        if not candidates:
            print(
                f"[WARNING] dataset_id={dataset_id} was not found in "
                "clustered_results.json; skipped."
            )
            continue

        parsed_candidates: list[dict[str, Any]] = []

        for candidate in candidates:
            best_params, final_score, parsed_file = parse_clustering_result(
                clustered_file_path=candidate.get("clustered_file_path", ""),
                dataset_id=dataset_id,
            )

            if final_score is None:
                continue

            if final_score >= INVALID_SCORE_THRESHOLD:
                continue

            parsed_candidates.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "csv_file": csv_file,
                    "cleaning_algorithm": candidate.get("cleaning_algorithm"),
                    "cleaning_runtime": candidate.get("cleaning_runtime"),
                    "clustering_algorithm": candidate.get("clustering_algorithm"),
                    "clustering_name": candidate.get("clustering_name"),
                    "clustering_runtime": candidate.get("clustering_runtime"),
                    "clustered_file_path": candidate.get("clustered_file_path"),
                    "parsed_result_file": parsed_file,
                    "best_params": best_params,
                    "final_score": final_score,
                }
            )

        parsed_candidates.sort(key=lambda item: item["final_score"], reverse=True)
        top_candidates = parsed_candidates[:k_value]

        analyzed_results.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "csv_file": csv_file,
                "top_k": k_value,
                "num_candidates": len(parsed_candidates),
                "top_results": top_candidates,
            }
        )

    _write_json(output_path, analyzed_results)
    print(f"[INFO] Analysis results saved to {output_path}")
