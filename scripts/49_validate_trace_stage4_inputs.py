#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_CODE_FILES = [
    "configs/trace.yaml",
    "src/analysis/trace_replay.py",
    "scripts/30_replay_trace.py",
    "scripts/36_eval_trace_blind_random.py",
    "scripts/38_lodo_trace_validation.py",
    "scripts/39_run_trace_stage4_paper_repro.py",
]

REQUIRED_TRACE_INPUTS = [
    "eigenvectors.json",
    "cleaned_results.json",
    "clustered_results.json",
    "clustered_data",
]

OPTIONAL_TRACE_INPUTS = [
    "analyzed_results.json",
]

REQUIRED_LODO_OUTPUTS = [
    "lodo_aggregate_summary.json",
    "lodo_folds.csv",
    "lodo_blind_random_dataset_summary.csv",
]


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def count_json_files(path: Path, limit: int = 1) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    count = 0
    for _ in path.rglob("*.json"):
        count += 1
        if count >= limit:
            break
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TRACE Stage 4 paper reproduction inputs.")
    parser.add_argument("--project-root", default=".", help="TRACE repository root.")
    parser.add_argument("--results-dir", required=True, help="TRACE replay snapshot directory, e.g. results/trace_cluster_replay_all.")
    parser.add_argument("--output-dir", default=None, help="Optional TRACE Stage 4 output directory to check for partial outputs.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any required item is missing.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir
    results_dir = results_dir.resolve()

    problems = []
    warnings = []

    print("[TRACE-PREFLIGHT] Project root:", project_root)
    print("[TRACE-PREFLIGHT] Results dir :", results_dir)

    # Code files.
    print("[TRACE-PREFLIGHT] Checking code files...")
    for item in REQUIRED_CODE_FILES:
        p = project_root / item
        if not p.exists():
            problems.append(f"Missing code file: {item}")
        else:
            print("  OK", item)

    # Minimal semantic check for LODO script.
    lodo_script = project_root / "scripts/38_lodo_trace_validation.py"
    if lodo_script.exists():
        text = lodo_script.read_text(encoding="utf-8", errors="replace")
        for token in ["lodo_blind_random_dataset_summary", "heldout_", "blind_random"]:
            if token not in text:
                problems.append(
                    f"scripts/38_lodo_trace_validation.py may be incomplete: missing token {token!r}"
                )

    # Trace input snapshot.
    print("[TRACE-PREFLIGHT] Checking TRACE replay snapshot...")
    if not results_dir.exists():
        problems.append(f"Missing results dir: {results_dir}")
    else:
        for item in REQUIRED_TRACE_INPUTS:
            p = results_dir / item
            if not p.exists():
                problems.append(f"Missing TRACE input: {rel(p, project_root)}")
            else:
                print("  OK", rel(p, project_root))

        for item in OPTIONAL_TRACE_INPUTS:
            p = results_dir / item
            if not p.exists():
                warnings.append(f"Optional TRACE input missing: {rel(p, project_root)}")
            else:
                print("  OK", rel(p, project_root))

        clustered = results_dir / "clustered_data"
        if clustered.exists():
            json_count = count_json_files(clustered, limit=1)
            if json_count == 0:
                problems.append(f"clustered_data exists but contains no JSON logs: {clustered}")
            else:
                print("  OK clustered_data contains JSON logs")

    # Output directory partial-state check.
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        output_dir = output_dir.resolve()

        print("[TRACE-PREFLIGHT] Checking output dir:", output_dir)

        if output_dir.exists():
            lodo_files = [p.name for p in output_dir.glob("lodo_*")]
            heldout_dirs = [p.name for p in output_dir.glob("heldout_*") if p.is_dir()]
            missing_outputs = [
                name for name in REQUIRED_LODO_OUTPUTS
                if not (output_dir / name).exists()
            ]

            if lodo_files or heldout_dirs:
                print("  Existing lodo_* files:", lodo_files)
                print("  Existing heldout_* dirs:", heldout_dirs)

            if missing_outputs and (lodo_files or heldout_dirs):
                problems.append(
                    "Output directory appears partial. Missing "
                    f"{missing_outputs}. Recommended fix: delete this directory before rerun: {output_dir}"
                )
        else:
            print("  OK output dir does not exist yet")

    if warnings:
        print("[TRACE-PREFLIGHT] Warnings:")
        for w in warnings:
            print("  WARN", w)

    if problems:
        print("[TRACE-PREFLIGHT] Problems:")
        for p in problems:
            print("  MISSING", p)
        print("[TRACE-PREFLIGHT] Result: FAILED")
        return 2 if args.strict else 0

    print("[TRACE-PREFLIGHT] Result: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
