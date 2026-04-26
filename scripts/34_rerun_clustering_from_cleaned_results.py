#!/usr/bin/env python3
r"""Rerun clustering from cached cleaned CSVs.

This is the maintainer-side Stage 4 bridge for TRACE validation.  It does not
run cleaners.  Instead, it reads an existing exhaustive-baseline
cleaned_results.json, resolves the cached cleaned CSV files, reruns the current
TRACE clustering scripts, and writes a fresh results directory with trial-level
clustering logs.

Typical Windows use from E:\TRACE:
    python .\scripts\34_rerun_clustering_from_cleaned_results.py `
        --source-results-dir E:\AutoMLClustering_full\results `
        --output-results-dir E:\TRACE\results\trace_cluster_replay `
        --dataset-ids 0 7 14 21 28 35 `
        --cleaners group:full `
        --clusterers group:full `
        --cluster-trials 150 `
        --workers 4

Typical Linux use from /home/changtian/TRACE:
    python scripts/34_rerun_clustering_from_cleaned_results.py \
        --source-results-dir /home/changtian/AutoMLClustering_full/results \
        --output-results-dir /home/changtian/TRACE/results/trace_cluster_replay \
        --dataset-ids 0 7 14 21 28 35 \
        --cleaners group:full \
        --clusterers group:full \
        --cluster-trials 150 \
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


def _project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_project_on_path(project_root: Path) -> None:
    root_text = str(project_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    id: int
    legacy_id: int
    display_name: str
    group: str


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _parse_int_list(values: Optional[Iterable[str]]) -> Optional[list[int]]:
    if not values:
        return None
    out: list[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                out.append(int(token))
    return out or None


def _method_registry_path(project_root: Path, methods_config: Optional[Path]) -> Path:
    if methods_config is None:
        return project_root / "configs" / "methods.yaml"
    return methods_config if methods_config.is_absolute() else project_root / methods_config


def _load_runtime_specs(project_root: Path, methods_config: Optional[Path], kind: str, tokens: Optional[list[str]]) -> list[RuntimeSpec]:
    _ensure_project_on_path(project_root)
    from src.pipeline.method_registry import MethodRegistry

    registry = MethodRegistry(_method_registry_path(project_root, methods_config))
    specs = registry.resolve(kind, tokens)
    return [
        RuntimeSpec(
            name=str(spec.name),
            id=int(spec.id),
            legacy_id=int(spec.legacy_id),
            display_name=str(spec.display_name),
            group=str(spec.group),
        )
        for spec in specs
    ]


def _normalize_cleaner(value: Any) -> str:
    return str(value or "").strip().lower()


def _resolve_cached_cleaned_path(
    raw_value: Any,
    *,
    source_results_dir: Path,
    dataset_id: int,
    cleaner: str,
) -> Optional[Path]:
    """Resolve cleaned_file_path from archived results on the current machine."""
    candidates: list[Path] = []
    if raw_value not in (None, ""):
        raw_path = Path(str(raw_value))
        candidates.append(raw_path)
        if not raw_path.is_absolute():
            candidates.append(source_results_dir / raw_path)
        else:
            # Old JSON files may contain absolute paths from another machine.
            # Re-anchor anything after a `results` path component to the current
            # --source-results-dir.
            parts = list(raw_path.parts)
            for idx, part in enumerate(parts):
                if str(part).lower() == "results":
                    suffix = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
                    candidates.append(source_results_dir / suffix)
                    break

    candidates.extend(
        [
            source_results_dir / "cleaned_data" / cleaner / f"repaired_{dataset_id}.csv",
            source_results_dir / "cleaned_data" / cleaner / f"repaired_{dataset_id}.CSV",
        ]
    )

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.exists() and resolved.is_file():
            return resolved

    cleaned_root = source_results_dir / "cleaned_data"
    if cleaned_root.exists():
        matches = list(cleaned_root.glob(f"**/repaired_{dataset_id}.csv"))
        if matches:
            cleaner_matches = [p for p in matches if cleaner in str(p).lower().split(os.sep)]
            return sorted(cleaner_matches or matches)[0].resolve()
    return None


def _select_eigenvectors(eigenvectors: list[dict[str, Any]], dataset_ids: Optional[list[int]], max_records: Optional[int], record_offset: int) -> list[tuple[int, dict[str, Any]]]:
    items: list[tuple[int, dict[str, Any]]] = []
    wanted = set(dataset_ids) if dataset_ids else None
    for idx, record in enumerate(eigenvectors):
        if not isinstance(record, dict):
            continue
        dataset_id = int(record.get("dataset_id", idx))
        if wanted is not None and dataset_id not in wanted:
            continue
        items.append((dataset_id, dict(record)))
    if record_offset:
        items = items[max(0, int(record_offset)) :]
    if max_records is not None:
        items = items[: int(max_records)]
    return items


def _build_cleaned_jobs(
    *,
    cleaned_results: list[dict[str, Any]],
    selected_dataset_ids: set[int],
    cleaner_specs: list[RuntimeSpec],
    source_results_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cleaner_by_name = {_normalize_cleaner(spec.name): spec for spec in cleaner_specs}
    cleaned_out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for item in cleaned_results:
        if not isinstance(item, dict):
            continue
        dataset_id = int(item.get("dataset_id", -1))
        if dataset_id not in selected_dataset_ids:
            continue
        cleaner = _normalize_cleaner(item.get("algorithm") or item.get("cleaner") or item.get("cleaning_algorithm"))
        if cleaner not in cleaner_by_name:
            continue
        cleaned_path = _resolve_cached_cleaned_path(
            item.get("cleaned_file_path"),
            source_results_dir=source_results_dir,
            dataset_id=dataset_id,
            cleaner=cleaner,
        )
        if cleaned_path is None:
            failures.append(
                {
                    "stage": "resolve_cleaned_csv",
                    "dataset_id": dataset_id,
                    "dataset_name": item.get("dataset_name"),
                    "cleaner_name": cleaner,
                    "original_cleaned_file_path": item.get("cleaned_file_path"),
                    "message": "Cached cleaned CSV was not found.",
                }
            )
            continue
        spec = cleaner_by_name[cleaner]
        normalized = dict(item)
        normalized.update(
            {
                "algorithm": spec.name,
                "algorithm_id": int(spec.id),
                "algorithm_legacy_id": int(spec.legacy_id),
                "algorithm_display_name": spec.display_name,
                "cleaned_file_path": str(cleaned_path),
                "cached_cleaning": True,
                "source_cleaned_file_path": str(item.get("cleaned_file_path", "")),
            }
        )
        cleaned_out.append(normalized)
    cleaned_out.sort(key=lambda r: (int(r.get("dataset_id", -1)), str(r.get("algorithm", ""))))
    return cleaned_out, failures


def _cluster_worker(
    job: dict[str, Any],
    project_root: str,
    output_results_dir: str,
    cluster_trials: Optional[int],
    clean_state: str,
    fail_fast: bool,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    root = Path(project_root).resolve()
    _ensure_project_on_path(root)
    from src.pipeline.clustering_runner import run_clustering

    dataset_id = int(job["dataset_id"])
    cleaner = str(job["cleaner"])
    cleaner_id = int(job["cleaner_id"])
    cleaner_legacy_id = int(job["cleaner_legacy_id"])
    cleaner_display = job.get("cleaner_display_name")
    clusterer = str(job["clusterer"])
    clusterer_id = int(job["clusterer_id"])
    clusterer_legacy_id = int(job["clusterer_legacy_id"])
    clusterer_display = job.get("clusterer_display_name")
    cleaned_file_path = str(job["cleaned_file_path"])

    output_dir, runtime = run_clustering(
        dataset_id=dataset_id,
        algorithm=cleaner,
        cluster_method_id=clusterer_legacy_id,
        cleaned_file_path=cleaned_file_path,
        project_root=root,
        results_dir=Path(output_results_dir).resolve(),
        cluster_trials=cluster_trials,
        clean_state=clean_state,
    )
    if output_dir is None or runtime is None:
        failure = {
            "stage": "clustering",
            "dataset_id": dataset_id,
            "dataset_name": job.get("dataset_name"),
            "csv_file": job.get("csv_file"),
            "cleaner_name": cleaner,
            "cleaner_id": cleaner_id,
            "cleaner_legacy_id": cleaner_legacy_id,
            "clusterer_name": clusterer,
            "clusterer_id": clusterer_id,
            "clusterer_legacy_id": clusterer_legacy_id,
            "message": "Clustering failed.",
        }
        if fail_fast:
            raise RuntimeError(json.dumps(failure, ensure_ascii=False))
        return None, failure

    clustered = {
        "dataset_id": dataset_id,
        "dataset_name": job.get("dataset_name"),
        "csv_file": job.get("csv_file"),
        "cleaning_algorithm": cleaner,
        "cleaning_algorithm_id": cleaner_id,
        "cleaning_algorithm_legacy_id": cleaner_legacy_id,
        "cleaning_display_name": cleaner_display,
        "cleaning_runtime": job.get("cleaning_runtime", job.get("runtime", 0.0)),
        "cached_cleaning": True,
        "cleaned_file_path": cleaned_file_path,
        "clustering_algorithm": clusterer_id,
        "clustering_algorithm_legacy_id": clusterer_legacy_id,
        "clustering_name": clusterer,
        "clustering_display_name": clusterer_display,
        "clustering_runtime": float(runtime),
        "clustered_file_path": str(output_dir),
    }
    return clustered, None



def _expected_cluster_output_dir(output_results_dir: Path, job: dict[str, Any]) -> Path:
    """Return the deterministic output directory used by clustering_runner."""
    return (
        output_results_dir
        / "clustered_data"
        / str(job["clusterer"])
        / str(job["cleaner"])
        / f"clustered_{int(job['dataset_id'])}"
    )


def _looks_like_complete_cluster_output(output_dir: Path) -> bool:
    """Heuristic completion check for cached clustering outputs.

    The current clusterer scripts do not write a single global DONE marker.  A
    successful run does, however, write at least one final text file plus one of
    the method-specific JSON logs such as *_summary.json, *_optuna_trials.json,
    *_core_stats.json, *_history.json, or *_merge_history.json.  We deliberately
    do *not* treat an empty directory as complete, because Ctrl+C can leave a
    partially-created output directory behind.
    """
    if not output_dir.exists() or not output_dir.is_dir():
        return False
    try:
        files = [p for p in output_dir.iterdir() if p.is_file() and p.stat().st_size > 0]
    except OSError:
        return False
    if not files:
        return False
    names = [p.name.lower() for p in files]
    has_text = any(name.endswith(".txt") for name in names)
    has_json_marker = any(
        name.endswith("_summary.json")
        or name.endswith("_optuna_trials.json")
        or name.endswith("_core_stats.json")
        or name.endswith("_centroid_history.json")
        or name.endswith("_gmm_history.json")
        or name.endswith("_merge_history.json")
        or name.endswith("_history.json")
        for name in names
    )
    return bool(has_text and has_json_marker)


def _runtime_from_existing_output(output_dir: Path) -> float:
    """Best-effort runtime recovery from existing output files."""
    for summary_path in sorted(output_dir.glob("*_summary.json")):
        data = _read_json(summary_path, default={})
        if isinstance(data, dict):
            for key in ("total_runtime_sec", "runtime", "runtime_sec", "clustering_runtime"):
                value = data.get(key)
                try:
                    if value not in (None, ""):
                        return float(value)
                except Exception:
                    pass
    for txt_path in sorted(output_dir.glob("*.txt")):
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        import re
        for pattern in (
            r"Program completed in:\s*([\d.]+)\s*seconds",
            r"Program completed in\s*([\d.]+)\s*seconds",
            r"Program completed in\s*([\d.]+)\s*sec",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
    return 0.0


def _clustered_record_from_existing_job(job: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "dataset_id": int(job["dataset_id"]),
        "dataset_name": job.get("dataset_name"),
        "csv_file": job.get("csv_file"),
        "cleaning_algorithm": str(job["cleaner"]),
        "cleaning_algorithm_id": int(job["cleaner_id"]),
        "cleaning_algorithm_legacy_id": int(job["cleaner_legacy_id"]),
        "cleaning_display_name": job.get("cleaner_display_name"),
        "cleaning_runtime": job.get("cleaning_runtime", 0.0),
        "cached_cleaning": True,
        "cleaned_file_path": str(job["cleaned_file_path"]),
        "clustering_algorithm": int(job["clusterer_id"]),
        "clustering_algorithm_legacy_id": int(job["clusterer_legacy_id"]),
        "clustering_name": str(job["clusterer"]),
        "clustering_display_name": job.get("clusterer_display_name"),
        "clustering_runtime": float(_runtime_from_existing_output(output_dir)),
        "clustered_file_path": str(output_dir),
        "resumed_from_existing_output": True,
    }


def _write_progress_checkpoint(
    output_results_dir: Path,
    *,
    selected_eigenvectors: list[dict[str, Any]],
    cleaned_out: list[dict[str, Any]],
    clustered_out: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    """Write recoverable partial metadata while a long replay is running."""
    clustered_sorted = sorted(
        clustered_out,
        key=lambda r: (int(r.get("dataset_id", -1)), str(r.get("cleaning_algorithm", "")), int(r.get("clustering_algorithm", -1))),
    )
    _write_json(output_results_dir / "eigenvectors.partial.json", selected_eigenvectors)
    _write_json(output_results_dir / "cleaned_results.partial.json", cleaned_out)
    _write_json(output_results_dir / "clustered_results.partial.json", clustered_sorted)
    _write_json(output_results_dir / "logs" / "cluster_replay_failures.partial.json", failures)


def _build_cluster_jobs(cleaned_records: list[dict[str, Any]], cleaner_specs: list[RuntimeSpec], clusterer_specs: list[RuntimeSpec]) -> list[dict[str, Any]]:
    cleaner_by_name = {_normalize_cleaner(spec.name): spec for spec in cleaner_specs}
    jobs: list[dict[str, Any]] = []
    for record in cleaned_records:
        cleaner_name = _normalize_cleaner(record.get("algorithm"))
        cleaner_spec = cleaner_by_name.get(cleaner_name)
        if cleaner_spec is None:
            continue
        for clusterer_spec in clusterer_specs:
            jobs.append(
                {
                    "dataset_id": int(record["dataset_id"]),
                    "dataset_name": record.get("dataset_name"),
                    "csv_file": record.get("csv_file"),
                    "cleaner": cleaner_spec.name,
                    "cleaner_id": int(cleaner_spec.id),
                    "cleaner_legacy_id": int(cleaner_spec.legacy_id),
                    "cleaner_display_name": cleaner_spec.display_name,
                    "cleaning_runtime": record.get("runtime", record.get("cleaning_runtime", 0.0)),
                    "cleaned_file_path": record["cleaned_file_path"],
                    "clusterer": clusterer_spec.name,
                    "clusterer_id": int(clusterer_spec.id),
                    "clusterer_legacy_id": int(clusterer_spec.legacy_id),
                    "clusterer_display_name": clusterer_spec.display_name,
                }
            )
    jobs.sort(key=lambda r: (int(r["dataset_id"]), str(r["cleaner"]), int(r["clusterer_id"])))
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerun TRACE clustering from cached cleaned CSV outputs.")
    parser.add_argument("--project-root", type=Path, default=_project_root_from_script(), help="TRACE repository root.")
    parser.add_argument("--methods-config", type=Path, default=None, help="Method registry YAML. Default: configs/methods.yaml.")
    parser.add_argument("--source-results-dir", type=Path, required=True, help="Archived exhaustive-baseline results directory containing cleaned_results.json and eigenvectors.json.")
    parser.add_argument("--output-results-dir", type=Path, default=None, help="Fresh output directory for clustering replay results. Default: results/trace_cluster_replay.")
    parser.add_argument("--eigenvectors", type=Path, default=None, help="Optional explicit eigenvectors.json. Default: <source-results-dir>/eigenvectors.json.")
    parser.add_argument("--cleaned-results", type=Path, default=None, help="Optional explicit cleaned_results.json. Default: <source-results-dir>/cleaned_results.json.")
    parser.add_argument("--dataset-ids", nargs="*", default=None, help="Dataset ids, e.g. --dataset-ids 0 7 14 or --dataset-ids 0,7,14.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap after dataset-id filtering.")
    parser.add_argument("--record-offset", type=int, default=0, help="Optional offset after dataset-id filtering.")
    parser.add_argument("--cleaners", nargs="*", default=None, help="Cleaner subset resolved by configs/methods.yaml. Default: all enabled.")
    parser.add_argument("--clusterers", nargs="*", default=None, help="Clusterer subset resolved by configs/methods.yaml. Default: all enabled.")
    parser.add_argument("--cluster-trials", type=int, default=None, help="Optional TRACE_N_TRIALS budget. Omit to use each clusterer default/full budget.")
    parser.add_argument("--clean-state", default="cleaned", help="CLEAN_STATE passed to clusterers. Default: cleaned.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel clustering workers. Use 1 for debugging.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analyzed_results.json generation.")
    parser.add_argument("--resume", action="store_true", help="Skip clustering jobs whose output directories already look complete and include them in clustered_results.json. Useful after Ctrl+C or reboot.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Write *.partial.json progress metadata every N completed jobs. Set 0 to disable. Default: 25.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after first clustering failure.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start_time = time.perf_counter()

    project_root = args.project_root.resolve()
    _ensure_project_on_path(project_root)
    source_results_dir = args.source_results_dir.resolve()
    output_results_dir = (args.output_results_dir or (project_root / "results" / "trace_cluster_replay")).resolve()
    output_results_dir.mkdir(parents=True, exist_ok=True)

    eigenvectors_path = (args.eigenvectors or (source_results_dir / "eigenvectors.json")).resolve()
    cleaned_results_path = (args.cleaned_results or (source_results_dir / "cleaned_results.json")).resolve()

    eigenvectors = _read_json(eigenvectors_path, default=[])
    cleaned_results = _read_json(cleaned_results_path, default=[])
    if not isinstance(eigenvectors, list):
        raise SystemExit(f"eigenvectors.json must contain a list: {eigenvectors_path}")
    if not isinstance(cleaned_results, list):
        raise SystemExit(f"cleaned_results.json must contain a list: {cleaned_results_path}")

    dataset_ids = _parse_int_list(args.dataset_ids)
    selected_records = _select_eigenvectors(eigenvectors, dataset_ids, args.max_records, args.record_offset)
    selected_ids = {dataset_id for dataset_id, _ in selected_records}
    if not selected_ids:
        raise SystemExit("No dataset records were selected.")

    cleaner_specs = _load_runtime_specs(project_root, args.methods_config, "cleaners", args.cleaners)
    clusterer_specs = _load_runtime_specs(project_root, args.methods_config, "clusterers", args.clusterers)

    cleaned_out, failures = _build_cleaned_jobs(
        cleaned_results=cleaned_results,
        selected_dataset_ids=selected_ids,
        cleaner_specs=cleaner_specs,
        source_results_dir=source_results_dir,
    )
    if not cleaned_out:
        _write_json(output_results_dir / "logs" / "cluster_replay_failures.json", failures)
        raise SystemExit("No cached cleaned CSVs were resolved. See logs/cluster_replay_failures.json.")

    cluster_jobs = _build_cluster_jobs(cleaned_out, cleaner_specs, clusterer_specs)

    selected_eigenvectors = []
    selected_by_id = {dataset_id: record for dataset_id, record in selected_records}
    for dataset_id in sorted(selected_by_id):
        record = dict(selected_by_id[dataset_id])
        record["dataset_id"] = dataset_id
        selected_eigenvectors.append(record)

    print(f"[TRACE] Project root: {project_root}")
    print(f"[TRACE] Source results: {source_results_dir}")
    print(f"[TRACE] Output results: {output_results_dir}")
    print(f"[TRACE] Eigenvectors: {eigenvectors_path}")
    print(f"[TRACE] Cleaned results: {cleaned_results_path}")
    print(f"[TRACE] Selected dataset ids: {sorted(selected_ids)}")
    print("[TRACE] Cleaners: " + ", ".join(spec.name for spec in cleaner_specs))
    print("[TRACE] Clusterers: " + ", ".join(spec.name for spec in clusterer_specs))
    print(f"[TRACE] Resolved cleaned records: {len(cleaned_out)}")
    print(f"[TRACE] Clustering jobs: {len(cluster_jobs)}")
    if args.cluster_trials is not None:
        print(f"[TRACE] Cluster trial budget: {args.cluster_trials}")

    clustered_out: list[dict[str, Any]] = []
    pending_jobs = cluster_jobs
    if args.resume:
        resumed_records: list[dict[str, Any]] = []
        remaining_jobs: list[dict[str, Any]] = []
        for job in cluster_jobs:
            expected_dir = _expected_cluster_output_dir(output_results_dir, job)
            if _looks_like_complete_cluster_output(expected_dir):
                resumed_records.append(_clustered_record_from_existing_job(job, expected_dir))
            else:
                remaining_jobs.append(job)
        clustered_out.extend(resumed_records)
        pending_jobs = remaining_jobs
        print(
            "[TRACE] Resume mode: "
            f"found {len(resumed_records)} completed jobs on disk; "
            f"remaining jobs={len(pending_jobs)}"
        )
        _write_progress_checkpoint(
            output_results_dir,
            selected_eigenvectors=selected_eigenvectors,
            cleaned_out=cleaned_out,
            clustered_out=clustered_out,
            failures=failures,
        )

    workers = max(1, int(args.workers))
    if workers == 1:
        for idx, job in enumerate(pending_jobs, start=1):
            print(f"[TRACE] Job {idx}/{len(pending_jobs)}: dataset={job['dataset_id']} cleaner={job['cleaner']} clusterer={job['clusterer']}")
            clustered, failure = _cluster_worker(job, str(project_root), str(output_results_dir), args.cluster_trials, args.clean_state, args.fail_fast)
            if clustered is not None:
                clustered_out.append(clustered)
            if failure is not None:
                failures.append(failure)
            if args.checkpoint_every and idx % max(1, int(args.checkpoint_every)) == 0:
                _write_progress_checkpoint(
                    output_results_dir,
                    selected_eigenvectors=selected_eigenvectors,
                    cleaned_out=cleaned_out,
                    clustered_out=clustered_out,
                    failures=failures,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            future_to_job = {
                executor.submit(_cluster_worker, job, str(project_root), str(output_results_dir), args.cluster_trials, args.clean_state, args.fail_fast): job
                for job in pending_jobs
            }
            for idx, future in enumerate(as_completed(future_to_job), start=1):
                job = future_to_job[future]
                try:
                    clustered, failure = future.result()
                except Exception as exc:
                    failure = {
                        "stage": "clustering",
                        "dataset_id": job.get("dataset_id"),
                        "dataset_name": job.get("dataset_name"),
                        "cleaner_name": job.get("cleaner"),
                        "clusterer_name": job.get("clusterer"),
                        "message": f"Worker failed: {exc}",
                    }
                    if args.fail_fast:
                        raise
                    clustered = None
                if clustered is not None:
                    clustered_out.append(clustered)
                if failure is not None:
                    failures.append(failure)
                print(f"[TRACE] Completed {idx}/{len(pending_jobs)} pending jobs; total_successes={len(clustered_out)} failures={len(failures)}")
                if args.checkpoint_every and idx % max(1, int(args.checkpoint_every)) == 0:
                    _write_progress_checkpoint(
                        output_results_dir,
                        selected_eigenvectors=selected_eigenvectors,
                        cleaned_out=cleaned_out,
                        clustered_out=clustered_out,
                        failures=failures,
                    )

    clustered_out.sort(key=lambda r: (int(r.get("dataset_id", -1)), str(r.get("cleaning_algorithm", "")), int(r.get("clustering_algorithm", -1))))
    runtime_sec = time.perf_counter() - start_time

    _write_json(output_results_dir / "eigenvectors.json", selected_eigenvectors)
    _write_json(output_results_dir / "cleaned_results.json", cleaned_out)
    _write_json(output_results_dir / "clustered_results.json", clustered_out)
    _write_json(output_results_dir / "logs" / "cluster_replay_failures.json", failures)

    if not args.skip_analysis:
        from src.pipeline.analysis import save_analyzed_results

        save_analyzed_results(
            preprocessing_file_path=str(project_root / "src" / "pipeline" / "preprocess.py"),
            eigenvectors_path=str(output_results_dir / "eigenvectors.json"),
            clustered_results_path=str(output_results_dir / "clustered_results.json"),
            output_path=str(output_results_dir / "analyzed_results.json"),
        )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "source_results_dir": str(source_results_dir),
        "output_results_dir": str(output_results_dir),
        "eigenvectors_path": str(eigenvectors_path),
        "cleaned_results_path": str(cleaned_results_path),
        "selected_dataset_ids": sorted(selected_ids),
        "selected_cleaners": [asdict(spec) for spec in cleaner_specs],
        "selected_clusterers": [asdict(spec) for spec in clusterer_specs],
        "cluster_trials": args.cluster_trials,
        "workers": workers,
        "clean_state": args.clean_state,
        "cached_cleaning": True,
        "resolved_cleaned_records": len(cleaned_out),
        "cluster_jobs": len(cluster_jobs),
        "pending_jobs_after_resume_scan": len(pending_jobs),
        "resume": bool(args.resume),
        "clustered_result_count": len(clustered_out),
        "failure_count": len(failures),
        "runtime_sec": runtime_sec,
    }
    _write_json(output_results_dir / "logs" / "cluster_replay_manifest.json", manifest)

    print(
        "[TRACE] Cluster-only replay completed: "
        f"clustered={len(clustered_out)}, failures={len(failures)}, runtime={runtime_sec:.2f}s"
    )
    print(f"[TRACE] Output results ready for Stage 4 replay: {output_results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
