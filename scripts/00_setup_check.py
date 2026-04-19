#!/usr/bin/env python3
"""
TRACE setup checker.

This script validates the repository layout, A/B/C configs, input data files,
source-code directories, and local output directories.

It does not run cleaning, clustering, TRACE replay, or plotting.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("[TRACE][ERROR] Missing dependency: pyyaml")
    print("Install with: pip install pyyaml")
    raise SystemExit(2)


ALLOWED_CLEANERS = {
    "mode",
    "baran",
    "bigdansing",
    "boostclean",
    "holoclean",
    "horizon",
    "scared",
    "unified",
    "groundtruth",
}

ALLOWED_CLUSTERERS = {
    "KMEANS",
    "KMEANSNF",
    "KMEANSPPS",
    "GMM",
    "DBSCAN",
    "HC",
}

REQUIRED_ROOT_FILES = [
    "README.md",
    ".gitignore",
    "environment.yml",
    "requirements.txt",
    "pyproject.toml",
]

SOURCE_CHECKS = {
    "cleaning": [
        "src/cleaning/mode/mode.py",
        "src/cleaning/baran/correction_with_baran.py",
        "src/cleaning/BigDansing_Holistic/bigdansing.py",
        "src/cleaning/BoostClean/setup.py",
        "src/cleaning/holoclean-master/holoclean.py",
        "src/cleaning/horizon/horizon.py",
        "src/cleaning/SCAREd/scared.py",
        "src/cleaning/Unified/Unified.py",
    ],
    "clustering": [
        "src/clustering/KMEANS/KMEANS.py",
        "src/clustering/KMEANSNF/KMEANSNF.py",
        "src/clustering/KMEANSPPS/KMEANSPPS.py",
        "src/clustering/GMM/GMM.py",
        "src/clustering/DBSCAN/DBSCAN.py",
        "src/clustering/HC/HC.py",
    ],
    "pipeline": [
        "src/pipeline/train/train_pipeline.py",
        "src/pipeline/train/error_correction.py",
        "src/pipeline/train/cluster_methods.py",
        "src/pipeline/train/search.py",
        "src/pipeline/utils/analyze_cleaning.py",
        "src/pipeline/utils/analyze_cluster.py",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mode_b_smoke.yaml")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--check-all-data", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Skip checking source-code files. Use this for Stage-1 validation before src is normalized.",
    )
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="For Mode A, require archived results to exist under results/raw.",
    )
    return parser.parse_args()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: Path, root: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = seen or set()
    path = path if path.is_absolute() else root / path
    path = path.resolve()

    if path in seen:
        raise ValueError(f"Circular config inheritance detected: {path}")
    if not path.exists():
        raise FileNotFoundError(path)

    seen.add(path)
    cfg = read_yaml(path)
    parent = cfg.pop("inherits", None)
    if parent is None:
        return cfg

    parent_path = Path(str(parent))
    candidates = [parent_path] if parent_path.is_absolute() else [
        root / parent_path,
        path.parent / parent_path,
    ]
    resolved = next((x for x in candidates if x.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"Cannot resolve inherited config: {parent}")

    return deep_merge(load_config(resolved, root, seen), cfg)


def load_manifest(root: Path, warnings: list[str]) -> dict[str, Any]:
    path = root / "data" / "manifest.json"
    if not path.exists():
        warnings.append("Missing data/manifest.json")
        return {"datasets": {}}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj.get("datasets"), dict):
        raise ValueError("data/manifest.json must contain a top-level 'datasets' object.")
    return obj


def norm_cleaner(value: Any) -> str:
    x = str(value).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "modeimpute": "mode",
        "bigdansingholistic": "bigdansing",
        "groundtruthclean": "groundtruth",
    }
    return aliases.get(x, x)


def norm_clusterer(value: Any) -> str:
    x = str(value).strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "KMEANSBASE": "KMEANS",
        "KMEANSPP": "KMEANSPPS",
        "HIERARCHICAL": "HC",
        "AGGLOMERATIVE": "HC",
    }
    return aliases.get(x, x)


def ensure_dirs(root: Path, cfg: dict[str, Any]) -> list[str]:
    paths = cfg.get("paths", {})
    dirs = [
        paths.get("raw_results_dir", "results/raw"),
        paths.get("processed_results_dir", "results/processed"),
        paths.get("figure_dir", "results/figures"),
        paths.get("log_dir", "results/logs"),
    ]
    ensured = []
    for d in dirs:
        p = root / str(d)
        p.mkdir(parents=True, exist_ok=True)
        ensured.append(rel(p, root))
    return ensured


def check_root_files(root: Path, errors: list[str], warnings: list[str]) -> dict[str, bool]:
    status = {}
    for name in REQUIRED_ROOT_FILES:
        exists = (root / name).exists()
        status[name] = exists
        if not exists:
            errors.append(f"Missing required root file: {name}")

    optional = ["LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md"]
    for name in optional:
        if not (root / name).exists():
            warnings.append(f"Recommended file is missing: {name}")

    if (root / "environments.yml").exists():
        errors.append("Use environment.yml, not environments.yml.")

    return status


def check_source(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for group, files in SOURCE_CHECKS.items():
        group_report = {}
        for name in files:
            path = root / name
            exists = path.exists()
            group_report[name] = exists
            if not exists:
                warnings.append(f"Expected source file not found: {name}")
        report[group] = group_report

    for dirname in ["src/pre_experiment", "src/visual_demo", "src/trace_artifact"]:
        if not (root / dirname).exists():
            warnings.append(f"Reserved source directory is missing: {dirname}")

    return report


def check_methods(cfg: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    cleaners_cfg = cfg.get("cleaners", {})
    clusterers_cfg = cfg.get("clusterers", {})

    cleaner_names = []
    for key in ["baseline", "ground_truth"]:
        if cleaners_cfg.get(key):
            cleaner_names.append(cleaners_cfg[key])
    cleaner_names.extend(cleaners_cfg.get("candidates", []) or [])
    cleaners = [norm_cleaner(x) for x in cleaner_names]

    unknown_cleaners = sorted({x for x in cleaners if x not in ALLOWED_CLEANERS})
    if unknown_cleaners:
        errors.append(f"Unknown cleaners in config: {unknown_cleaners}")

    clusterers = [norm_clusterer(x) for x in (clusterers_cfg.get("names", []) or [])]
    unknown_clusterers = sorted({x for x in clusterers if x not in ALLOWED_CLUSTERERS})
    if unknown_clusterers:
        errors.append(f"Unknown clusterers in config: {unknown_clusterers}")

    if not cleaners:
        warnings.append("No cleaners configured.")
    if not clusterers:
        warnings.append("No clusterers configured.")

    return {
        "cleaners": cleaners,
        "unknown_cleaners": unknown_cleaners,
        "clusterers": clusterers,
        "unknown_clusterers": unknown_clusterers,
    }


def configured_datasets(cfg: dict[str, Any], manifest: dict[str, Any], check_all: bool) -> list[str]:
    if check_all:
        names = list(manifest.get("datasets", {}).keys())
        return names
    names = cfg.get("datasets", {}).get("names")
    return list(names or manifest.get("datasets", {}).keys())


def configured_dirty_ids(name: str, cfg: dict[str, Any], manifest: dict[str, Any], check_all: bool) -> list[int]:
    manifest_ids = manifest.get("datasets", {}).get(name, {}).get("dirty_ids")
    config_ids = cfg.get("datasets", {}).get("dirty_ids")
    ids = manifest_ids if check_all else (config_ids or manifest_ids or [1])
    return [int(x) for x in ids]


def check_data(
    root: Path,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    check_all: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    data_cfg = cfg.get("datasets", {})
    paths_cfg = cfg.get("paths", {})

    raw_train = Path(str(paths_cfg.get("raw_train_dir", "data/raw/train")))
    clean_default = str(data_cfg.get("clean_file", "clean.csv"))
    dirty_default = str(data_cfg.get("dirty_pattern", "{dataset}_{dirty_id}.csv"))
    constraints_default = data_cfg.get("constraint_files", {})

    for name in configured_datasets(cfg, manifest, check_all):
        entry = manifest.get("datasets", {}).get(name, {})
        dataset_dir = root / str(entry.get("root", raw_train / name))
        clean_file = str(entry.get("clean", clean_default))
        dirty_pattern = str(entry.get("dirty_pattern", dirty_default))
        constraints = entry.get("constraints", constraints_default) or {}

        ds_report: dict[str, Any] = {
            "root": rel(dataset_dir, root),
            "exists": dataset_dir.exists(),
            "clean": None,
            "dirty": {},
            "constraints": {},
        }

        if not dataset_dir.exists():
            errors.append(f"Missing dataset directory: {rel(dataset_dir, root)}")
            report[name] = ds_report
            continue

        clean_path = dataset_dir / clean_file
        ds_report["clean"] = {"path": rel(clean_path, root), "exists": clean_path.exists()}
        if not clean_path.exists():
            errors.append(f"Missing clean file: {rel(clean_path, root)}")

        for dirty_id in configured_dirty_ids(name, cfg, manifest, check_all):
            filename = dirty_pattern.format(dataset=name, dirty_id=dirty_id)
            path = dataset_dir / filename
            ds_report["dirty"][str(dirty_id)] = {"path": rel(path, root), "exists": path.exists()}
            if not path.exists():
                errors.append(f"Missing dirty file: {rel(path, root)}")

        for cname, cfile in constraints.items():
            path = dataset_dir / str(cfile)
            ds_report["constraints"][str(cname)] = {"path": rel(path, root), "exists": path.exists()}
            if not path.exists():
                warnings.append(f"Missing constraint file: {rel(path, root)}")

        report[name] = ds_report

    return report


def check_mode_a_results(
    root: Path,
    cfg: dict[str, Any],
    require_results: bool,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    mode = str(cfg.get("run", {}).get("mode", ""))
    if mode != "A_RELEASE_RESULTS":
        return {"checked": False}

    raw_root = root / str(cfg.get("paths", {}).get("raw_results_dir", "results/raw"))
    expected = [
        raw_root / "trials.csv",
        raw_root / "cleaning_metrics.csv",
        raw_root / "process_metrics.csv",
        raw_root / "best_configs.csv",
        raw_root / "run_manifest.json",
    ]

    result = {"checked": True, "files": {}}
    for path in expected:
        exists = path.exists()
        result["files"][rel(path, root)] = exists
        if not exists:
            msg = f"Mode A archived result file not found yet: {rel(path, root)}"
            if require_results:
                errors.append(msg)
            else:
                warnings.append(msg)

    return result


def write_report(root: Path, report: dict[str, Any]) -> Path:
    out = root / "results" / "logs" / "setup_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    config_path = Path(args.config)

    errors: list[str] = []
    warnings: list[str] = []

    print(f"[TRACE] Repository root: {root}")
    print(f"[TRACE] Config: {config_path}")

    try:
        cfg = load_config(config_path, root)
        manifest = load_manifest(root, warnings)
    except Exception as exc:
        print(f"[TRACE][ERROR] {exc}")
        return 1

    ensured_dirs = ensure_dirs(root, cfg)
    root_report = check_root_files(root, errors, warnings)
    if args.skip_source_check:
        source_report = {"checked": False, "reason": "Skipped by --skip-source-check"}
        warnings.append("Source-code checks were skipped for Stage-1 validation.")
    else:
        source_report = check_source(root, errors, warnings)
    methods_report = check_methods(cfg, errors, warnings)
    data_report = check_data(root, cfg, manifest, errors, warnings, args.check_all_data)
    mode_a_report = check_mode_a_results(root, cfg, args.require_results, errors, warnings)

    if args.strict:
        # Missing archived raw results for Mode A are allowed before release unless --require-results is used.
        mode = str(cfg.get("run", {}).get("mode", ""))
        strict_warnings = []
        for w in warnings:
            if mode == "A_RELEASE_RESULTS" and "archived result file not found yet" in w and not args.require_results:
                continue
            if args.skip_source_check and "Source-code checks were skipped" in w:
                continue
            strict_warnings.append(w)
        if strict_warnings:
            errors.extend([f"Strict warning treated as error: {w}" for w in strict_warnings])

    status = "passed" if not errors else "failed"

    report = {
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "config": str((root / config_path).resolve() if not config_path.is_absolute() else config_path),
        "run_mode": cfg.get("run", {}).get("mode"),
        "python": sys.version,
        "platform": platform.platform(),
        "ensured_dirs": ensured_dirs,
        "root_files": root_report,
        "source": source_report,
        "methods": methods_report,
        "data": data_report,
        "mode_a_results": mode_a_report,
        "warnings": warnings,
        "errors": errors,
    }

    report_path = write_report(root, report)

    for w in warnings:
        print(f"[TRACE][WARN] {w}")
    for e in errors:
        print(f"[TRACE][ERROR] {e}")

    print(f"[TRACE] Wrote report: {rel(report_path, root)}")

    if errors:
        print("[TRACE] SETUP CHECK FAILED")
        return 1

    print("[TRACE] SETUP CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
