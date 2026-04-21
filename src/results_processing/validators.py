#!/usr/bin/env python3
"""Validation helpers for archived TRACE result inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.results_processing.io import read_json


@dataclass(frozen=True)
class ResultInputPaths:
    """Conventional result-file locations."""

    results_dir: Path
    eigenvectors: Path
    cleaned_results: Path
    clustered_results: Path
    analyzed_results: Path
    pipeline_manifest: Path


def get_result_input_paths(results_dir: Path) -> ResultInputPaths:
    """Return conventional result-file locations."""
    root = Path(results_dir)

    return ResultInputPaths(
        results_dir=root,
        eigenvectors=root / "eigenvectors.json",
        cleaned_results=root / "cleaned_results.json",
        clustered_results=root / "clustered_results.json",
        analyzed_results=root / "analyzed_results.json",
        pipeline_manifest=root / "logs" / "pipeline_run_manifest.json",
    )


def validate_result_inputs(results_dir: Path, require_all: bool = False) -> Dict[str, Any]:
    """Validate raw result files used by Stage 3 replay."""
    paths = get_result_input_paths(results_dir)

    checks = {
        "eigenvectors": paths.eigenvectors.exists(),
        "cleaned_results": paths.cleaned_results.exists(),
        "clustered_results": paths.clustered_results.exists(),
        "analyzed_results": paths.analyzed_results.exists(),
        "pipeline_manifest": paths.pipeline_manifest.exists(),
    }

    json_types = {}
    for name, exists in checks.items():
        if not exists:
            json_types[name] = "missing"
            continue

        path = getattr(paths, name)
        value = read_json(path)
        json_types[name] = type(value).__name__

    missing = [name for name, exists in checks.items() if not exists]

    if require_all and missing:
        raise FileNotFoundError(f"Missing required result files: {missing}")

    return {
        "results_dir": str(paths.results_dir),
        "checks": checks,
        "json_types": json_types,
        "missing": missing,
    }

