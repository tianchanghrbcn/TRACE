#!/usr/bin/env python3
"""Validate the TRACE advisor-review package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_STATIC_PATHS = [
    "scripts/45_validate_data_availability.py",
    "scripts/00_trace_home.py",
    "docs/release_packaging.md",
    "docs/terminal_interface.md",
    "docs/stage1_to_stage4_plan.md",
    "docs/hardware_runtime.md",
    "docs/data_policy.md",
    "data/README.md",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE",
    "README.md",
    "configs/methods.yaml",
    "configs/results_schema.yaml",
    "configs/runtime_reference.yaml",
    "docs/artifact_overview.md",
    "docs/reproducibility_modes.md",
    "docs/release_checklist.md",
    "docs/known_limitations.md",
    "docs/runtime_progress.md",
    "docs/stage2_validation.md",
    "docs/results_schema.md",
    "docs/results_replay.md",
    "docs/analysis_tables.md",
    "docs/figures.md",
    "docs/pre_experiment.md",
    "docs/visual_demo.md",
    "src/results_processing/build_tables.py",
    "src/figures/paper_figures.py",
    "src/pre_experiment/alpha_metrics.py",
    "src/visual_demo/demo_plots.py",
    "data/pre_experiment/alpha_metrics.csv",
]


EXPECTED_GENERATED_PATHS = [
    "results/eigenvectors.json",
    "results/cleaned_results.json",
    "results/clustered_results.json",
    "results/analyzed_results.json",
    "results/processed/run_manifest.json",
    "results/tables/run_counts.csv",
    "figures/figure_manifest.json",
    "figures/layered_figure_manifest.json",
    "figures/migrated_figure_batch1_manifest.json",
    "results/pre_experiment/pre_experiment_manifest.json",
    "figures/pre_experiment/pre_experiment_figure_manifest.json",
    "results/visual_demo/visual_demo_manifest.json",
    "figures/visual_demo/visual_demo_figure_manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TRACE release package v0.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip Mode B smoke execution.")
    parser.add_argument("--strict-stage2-proof", action="store_true", help="Require Stage 2 strict proof files.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per command in seconds.")
    parser.add_argument("--report", type=Path, default=Path("results/logs/release_validation_report.json"))
    return parser.parse_args()


def run_command(name: str, cmd: list[str], timeout: int) -> dict:
    print(f"[TRACE] >>> {name}")
    print("[TRACE] Command:", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())

    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"[TRACE] {status}: {name}")

    return {
        "name": name,
        "status": status,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def check_paths(paths: list[str]) -> list[dict]:
    rows = []
    for path_str in paths:
        path = ROOT / path_str
        rows.append({
            "path": path_str,
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() and path.is_file() else "",
        })
    return rows


def check_stage2_proof() -> dict:
    final_result = ROOT / "results/logs/final_checks/stage2_strict_RESULT"

    if final_result.exists():
        result_text = final_result.read_text(encoding="utf-8-sig", errors="replace").strip()
        return {
            "status": "PASS" if result_text == "PASSED" else "FAIL",
            "source": str(final_result),
            "result": result_text,
        }

    strict_dirs = [
        path for path in (ROOT / "results/logs").glob("stage2_strict_*")
        if path.is_dir()
    ]
    strict_dirs = sorted(strict_dirs, key=lambda p: p.stat().st_mtime, reverse=True)

    for path in strict_dirs:
        result_path = path / "RESULT"
        if result_path.exists():
            result_text = result_path.read_text(encoding="utf-8-sig", errors="replace").strip()
            return {
                "status": "PASS" if result_text == "PASSED" else "FAIL",
                "source": str(result_path),
                "result": result_text,
            }

    return {
        "status": "WARN",
        "source": "",
        "result": "Stage 2 strict proof was not found in this working tree.",
    }


def main() -> None:
    args = parse_args()
    py = sys.executable

    commands = [
        (
            "data_availability",
            [py, "scripts/45_validate_data_availability.py"],
        ),
        (
            "setup_mode_b",
            [py, "scripts/00_setup_check.py", "--config", "configs/mode_b_smoke.yaml", "--strict"],
        ),
        (
            "setup_mode_c",
            [py, "scripts/00_setup_check.py", "--config", "configs/mode_c_full.yaml", "--check-all-data", "--strict"],
        ),
    ]

    if not args.skip_smoke:
        commands.append(
            (
                "mode_b_smoke",
                [py, "scripts/90_run_smoke_from_scratch.py", "--config", "configs/mode_b_smoke.yaml", "--clean"],
            )
        )

    commands.extend([
        (
            "build_canonical_results",
            [py, "scripts/30_build_canonical_results.py", "--results-dir", "results", "--output-dir", "results/processed"],
        ),
        (
            "build_paper_tables",
            [py, "scripts/31_build_paper_tables.py", "--processed-dir", "results/processed", "--output-dir", "results/tables"],
        ),
        (
            "build_analysis_tables",
            [py, "scripts/32_build_analysis_tables.py", "--processed-dir", "results/processed", "--output-dir", "results/tables"],
        ),
        (
            "make_paper_figures",
            [py, "scripts/33_make_paper_figures.py", "--tables-dir", "results/tables", "--output-root", "figures"],
        ),
        (
            "make_layered_figures",
            [py, "scripts/35_make_layered_figures.py", "--processed-dir", "results/processed", "--tables-dir", "results/tables", "--output-root", "figures"],
        ),
        (
            "make_migrated_figure_batch",
            [py, "scripts/36_make_migrated_figure_batch.py", "--processed-dir", "results/processed", "--output-root", "figures"],
        ),
        (
            "build_pre_experiment",
            [py, "scripts/38_build_pre_experiment_outputs.py", "--source-csv", "data/pre_experiment/alpha_metrics.csv", "--output-dir", "results/pre_experiment", "--figure-dir", "figures/pre_experiment"],
        ),
        (
            "make_visual_demo",
            [py, "scripts/40_make_visual_demo.py", "--output-data-dir", "results/visual_demo", "--output-figure-dir", "figures/visual_demo"],
        ),
    ])

    command_results = []
    for name, cmd in commands:
        try:
            command_results.append(run_command(name, cmd, args.timeout))
        except subprocess.TimeoutExpired as exc:
            command_results.append({
                "name": name,
                "status": "FAIL",
                "returncode": "timeout",
                "stdout_tail": exc.stdout[-4000:] if exc.stdout else "",
                "stderr_tail": exc.stderr[-4000:] if exc.stderr else "",
            })
            print(f"[TRACE] FAIL: {name} timed out")

    static_paths = check_paths(REQUIRED_STATIC_PATHS)
    generated_paths = check_paths(EXPECTED_GENERATED_PATHS)
    stage2_proof = check_stage2_proof()

    static_failures = [row for row in static_paths if not row["exists"]]
    generated_failures = [row for row in generated_paths if not row["exists"]]
    command_failures = [row for row in command_results if row["status"] != "PASS"]

    status = "PASS"
    if command_failures or static_failures or generated_failures:
        status = "FAIL"
    if args.strict_stage2_proof and stage2_proof["status"] != "PASS":
        status = "FAIL"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "commands": command_results,
        "static_paths": static_paths,
        "generated_paths": generated_paths,
        "stage2_proof": stage2_proof,
        "failures": {
            "commands": command_failures,
            "static_paths": static_failures,
            "generated_paths": generated_failures,
        },
    }

    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TRACE] Release validation report written to: {report_path}")
    print(f"[TRACE] Release validation status: {status}")

    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()

