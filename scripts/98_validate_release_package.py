#!/usr/bin/env python3
"""Validate the TRACE reviewer-facing release package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_STATIC_PATHS = [
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",

    "configs/paper_replay.yaml",
    "configs/benchmark_smoke.yaml",
    "configs/benchmark_full_audit.yaml",
    "configs/trace.yaml",

    "data/README.md",
    "data/raw/train/beers/clean.csv",
    "data/raw/train/flights/clean.csv",
    "data/raw/train/hospital/clean.csv",
    "data/raw/train/rayyan/clean.csv",
    "data/pre_experiment/alpha_metrics.csv",

    "analysis/validity_sensitivity/inputs/validity_sensitivity_input_manifest.json",
    "analysis/validity_sensitivity/inputs/analysis_results/beers_summary.xlsx",
    "analysis/validity_sensitivity/inputs/analysis_results/flights_summary.xlsx",
    "analysis/validity_sensitivity/inputs/analysis_results/hospital_summary.xlsx",
    "analysis/validity_sensitivity/inputs/analysis_results/rayyan_summary.xlsx",
    "analysis/validity_sensitivity/seed_sensitivity_summary.json",
    "analysis/validity_sensitivity/seed_sensitivity_report.md",

    "src/analysis/trace_replay.py",

    "docs/data_policy.md",
    "docs/hardware_runtime.md",
    "docs/release_packaging.md",
    "docs/terminal_interface.md",
    "docs/workflows.md",
    "docs/pre_experiment_validity.md",
    "docs/trace_stage4_repro.md",
    "docs/uniclean_external.md",
    "docs/stage3_strict_validation.md",
    "docs/paper_output_traceability.md",

    "scripts/trace.py",
    "scripts/00_trace_home.py",
    "scripts/30_replay_trace.py",
    "scripts/36_eval_trace_blind_random.py",
    "scripts/38_lodo_trace_validation.py",
    "scripts/39_run_trace_stage4_paper_repro.py",
    "scripts/45_validate_data_availability.py",
    "scripts/49_validate_trace_stage4_inputs.py",
    "scripts/62_validate_mode_a_paper_replay.py",
    "scripts/63_validate_stage3_strict.py",
    "scripts/81_replay_pre_experiment_validity.py",
    "scripts/90_run_smoke_from_scratch.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TRACE reviewer release package.")
    parser.add_argument(
        "--skip-strict-benchmark-proof",
        action="store_true",
        help="Skip strict workflow proof validation. Not recommended for final release.",
    )
    parser.add_argument(
        "--skip-stage3-strict",
        dest="skip_strict_benchmark_proof",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-benchmark-smoke",
        action="store_true",
        help="Skip benchmark-smoke rerun during release check.",
    )
    parser.add_argument(
        "--skip-preexp-validity",
        action="store_true",
        help="Skip pre-experiment/validity replay during release check.",
    )
    parser.add_argument(
        "--skip-paper-replay",
        action="store_true",
        help="Skip paper-replay validation during release check.",
    )
    parser.add_argument(
        "--run-trace-validation",
        action="store_true",
        help="Run TRACE Stage 4 paper-exact validation.",
    )
    parser.add_argument(
        "--rebuild-paper-replay",
        action="store_true",
        help="Rebuild paper-replay evidence before validating it.",
    )
    parser.add_argument(
        "--rebuild-mode-a",
        dest="rebuild_paper_replay",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-missing-full-audit-proof",
        action="store_true",
        help="Allow missing benchmark-full-audit proof as warning. Not recommended for final release.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/logs/release_validation_report.json"),
    )
    return parser.parse_args()


def run_command(name: str, cmd: list[str]) -> dict[str, Any]:
    print(f"[TRACE] >>> {name}")
    print("[TRACE] Command:", " ".join([sys.executable] + cmd))

    proc = subprocess.run(
        [sys.executable] + cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    return {
        "name": name,
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def static_path_checks() -> list[dict[str, Any]]:
    checks = []
    for rel in REQUIRED_STATIC_PATHS:
        path = ROOT / rel
        checks.append(
            {
                "path": rel,
                "exists": path.exists(),
                "is_file": path.is_file(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
            }
        )
    return checks


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# TRACE Release Validation Report",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Status: {report['status']}",
        "",
        "## Static path checks",
        "",
        "| Path | Exists | Size bytes |",
        "|---|---:|---:|",
    ]

    for row in report["static_path_checks"]:
        lines.append(f"| {row['path']} | {row['exists']} | {row['size_bytes']} |")

    lines += ["", "## Command checks", "", "| Check | Status | Return code |", "|---|---|---:|"]
    for row in report["commands"]:
        lines.append(f"| {row['name']} | {row['status']} | {row['returncode']} |")

    lines += ["", "## Failures", ""]
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("No hard failures.")

    lines += ["", "## Warnings", ""]
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("No warnings.")

    lines += [
        "",
        "## Interpretation",
        "",
        "PASS means the selected reviewer-facing release checks passed.",
        "PASS_WITH_WARNINGS is acceptable only when skipped or warning-level checks are explicitly documented.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_report_status(
    label: str,
    path: Path,
    accepted: set[str],
    failures: list[str],
    warnings: list[str],
) -> str:
    data = read_json_if_exists(path)
    if not data:
        failures.append(f"Missing generated report: {path}")
        return ""
    status = str(data.get("status", ""))
    if status not in accepted:
        failures.append(f"{label} status is not accepted: {status} ({path})")
    elif status == "PASS_WITH_WARNINGS":
        warnings.append(f"{label} passed with accepted warnings.")
    return status


def main() -> None:
    args = parse_args()

    static_checks = static_path_checks()
    failures: list[str] = []
    warnings: list[str] = []
    commands: list[dict[str, Any]] = []

    for row in static_checks:
        if not row["exists"] or not row["is_file"]:
            failures.append(f"Missing required file: {row['path']}")

    commands.append(run_command("data_availability", ["scripts/45_validate_data_availability.py"]))

    commands.append(
        run_command(
            "setup_benchmark_smoke",
            ["scripts/00_setup_check.py", "--config", "configs/benchmark_smoke.yaml", "--strict"],
        )
    )

    commands.append(
        run_command(
            "setup_benchmark_full_audit",
            ["scripts/00_setup_check.py", "--config", "configs/benchmark_full_audit.yaml", "--check-all-data", "--strict"],
        )
    )

    if args.skip_preexp_validity:
        warnings.append("preexp-validity was skipped by user request.")
    else:
        commands.append(run_command("preexp_validity", ["scripts/trace.py", "preexp-validity", "--strict"]))

    if args.skip_paper_replay:
        warnings.append("paper-replay was skipped by user request.")
    else:
        paper_cmd = ["scripts/62_validate_mode_a_paper_replay.py"]
        if args.rebuild_paper_replay:
            paper_cmd.append("--rebuild")
        commands.append(run_command("paper_replay", paper_cmd))

    if args.skip_benchmark_smoke:
        warnings.append("benchmark-smoke was skipped by user request.")
    else:
        commands.append(run_command("benchmark_smoke", ["scripts/trace.py", "benchmark-smoke", "--clean"]))

    if args.run_trace_validation:
        commands.append(run_command("trace_validation_paper_exact", ["scripts/trace.py", "trace-validation", "--paper-exact"]))
    else:
        warnings.append("TRACE Stage 4 paper-exact validation was not run. Use --run-trace-validation for the full TRACE gate.")

    if args.skip_strict_benchmark_proof:
        warnings.append("Strict reviewer workflow validation was skipped by user request.")
    else:
        strict_cmd = ["scripts/63_validate_stage3_strict.py", "--skip-smoke-rerun"]
        if args.rebuild_paper_replay:
            strict_cmd.append("--rebuild-paper-replay")
        if args.allow_missing_full_audit_proof:
            strict_cmd.append("--allow-missing-full-audit-proof")
        commands.append(run_command("strict_reviewer_workflow_validation", strict_cmd))

    for row in commands:
        if row["status"] != "PASS":
            failures.append(f"Command failed: {row['name']}")

    preexp_status = require_report_status(
        "preexp-validity",
        ROOT / "analysis/validity_sensitivity/pre_experiment_validity_report.json",
        {"PASS"},
        failures,
        warnings,
    )

    validity_status = require_report_status(
        "validity-sensitivity summary",
        ROOT / "analysis/validity_sensitivity/validity_sensitivity_summary.json",
        {"PASS", "PASS_WITH_WARNINGS"},
        failures,
        warnings,
    )

    paper_status = ""
    if not args.skip_paper_replay:
        paper_status = require_report_status(
            "paper-replay",
            ROOT / "analysis/paper_generated/paper_replay_validation_report.json",
            {"PASS", "PASS_WITH_WARNINGS"},
            failures,
            warnings,
        )

    strict_status = ""
    if not args.skip_strict_benchmark_proof:
        strict_status = require_report_status(
            "strict reviewer workflow",
            ROOT / "results/logs/stage3_strict_validation_report.json",
            {"PASS", "PASS_WITH_WARNINGS"},
            failures,
            warnings,
        )

    trace_status = ""
    if args.run_trace_validation:
        trace_manifest = ROOT / "results/processed/trace/lodo_paper_repro/trace_stage4_manifest.json"
        trace_report = read_json_if_exists(trace_manifest)
        if not trace_report:
            failures.append(f"Missing TRACE Stage 4 manifest: {trace_manifest}")
        else:
            trace_status = "PASS" if trace_report.get("metric_checks_all_within_tolerance") is True else "FAIL"
            if trace_status != "PASS":
                failures.append(f"TRACE Stage 4 metric checks not accepted: {trace_manifest}")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "selected_options": {
            "skip_strict_benchmark_proof": args.skip_strict_benchmark_proof,
            "skip_benchmark_smoke": args.skip_benchmark_smoke,
            "skip_preexp_validity": args.skip_preexp_validity,
            "skip_paper_replay": args.skip_paper_replay,
            "run_trace_validation": args.run_trace_validation,
            "rebuild_paper_replay": args.rebuild_paper_replay,
            "allow_missing_full_audit_proof": args.allow_missing_full_audit_proof,
        },
        "static_path_checks": static_checks,
        "commands": commands,
        "generated_report_statuses": {
            "preexp_validity": preexp_status,
            "validity_sensitivity": validity_status,
            "paper_replay": paper_status,
            "strict_reviewer_workflow": strict_status,
            "trace_stage4": trace_status,
        },
        "warnings": warnings,
        "failures": failures,
    }

    output = ROOT / args.output
    write_json(output, report)
    write_markdown(output.with_suffix(".md"), report)

    print(json.dumps(
        {
            "status": status,
            "warning_count": len(warnings),
            "failure_count": len(failures),
            "output": str(args.output),
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"[TRACE] Release validation report written to: {args.output}")
    print(f"[TRACE] Release validation status: {status}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()
