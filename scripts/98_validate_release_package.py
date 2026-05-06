#!/usr/bin/env python3
"""Validate the TRACE reviewer-facing release package.

This is the final package-level gate. It connects all reviewer workflows:

  release-check
  paper-replay
  preexp-validity
  trace-validation
  benchmark-smoke
  benchmark-full-audit

TRACE validation is connected but optional by default because the paper-exact
1000-replay check can be longer than the basic package check. Use
--run-trace-validation for the full final gate.
"""

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

    "docs/data_policy.md",
    "docs/hardware_runtime.md",
    "docs/release_packaging.md",
    "docs/terminal_interface.md",
    "docs/workflows.md",
    "docs/pre_experiment_validity.md",
    "docs/stage3_strict_validation.md",
    "docs/paper_output_traceability.md",

    "scripts/trace.py",
    "scripts/00_trace_home.py",
    "scripts/45_validate_data_availability.py",
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
        "--run-trace-validation",
        action="store_true",
        help="Also run TRACE paper-exact validation with 1000 blind-random replays.",
    )
    parser.add_argument(
        "--rebuild-paper-replay",
        action="store_true",
        help="Ask strict validation to rebuild paper-replay outputs.",
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
        "PASS means the reviewer-facing release package passed all selected checks.",
        "PASS_WITH_WARNINGS is acceptable only when skipped/optional checks are explicitly documented.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_generated_reports(failures: list[str], warnings: list[str]) -> dict[str, Any]:
    reports = {
        "preexp_validity": ROOT / "analysis/validity_sensitivity/pre_experiment_validity_report.json",
        "validity_summary": ROOT / "analysis/validity_sensitivity/validity_sensitivity_summary.json",
        "strict_workflow": ROOT / "results/logs/stage3_strict_validation_report.json",
        "release": ROOT / "results/logs/release_validation_report.json",
    }

    status = {}
    for name, path in reports.items():
        data = read_json_if_exists(path)
        status[name] = data.get("status", "") if data else ""
        if name in {"preexp_validity", "validity_summary"}:
            if not data:
                failures.append(f"Missing generated report: {path}")
            elif data.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
                failures.append(f"Generated report not accepted: {path} status={data.get('status')}")

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

    commands.append(
        run_command("data_availability", ["scripts/45_validate_data_availability.py"])
    )

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
        commands.append(
            run_command(
                "preexp_validity",
                ["scripts/trace.py", "preexp-validity", "--strict"],
            )
        )

    if args.skip_benchmark_smoke:
        warnings.append("benchmark-smoke was skipped by user request.")
    else:
        commands.append(
            run_command(
                "benchmark_smoke",
                ["scripts/trace.py", "benchmark-smoke", "--clean"],
            )
        )

    if args.run_trace_validation:
        commands.append(
            run_command(
                "trace_validation_paper_exact",
                ["scripts/trace.py", "trace-validation", "--paper-exact"],
            )
        )

    if args.skip_strict_benchmark_proof:
        warnings.append("Strict workflow validation was skipped by user request.")
    else:
        strict_cmd = ["scripts/63_validate_stage3_strict.py", "--skip-smoke-rerun"]
        if args.rebuild_paper_replay:
            strict_cmd.append("--rebuild-paper-replay")
        if args.allow_missing_full_audit_proof:
            strict_cmd.append("--allow-missing-full-audit-proof")
        commands.append(run_command("strict_workflow_validation", strict_cmd))

    for row in commands:
        if row["status"] != "PASS":
            failures.append(f"Command failed: {row['name']}")

    strict_report_path = ROOT / "results/logs/stage3_strict_validation_report.json"
    strict_report = read_json_if_exists(strict_report_path)

    if not args.skip_strict_benchmark_proof:
        if not strict_report:
            failures.append(f"Missing strict workflow report: {strict_report_path}")
        else:
            strict_status = strict_report.get("status", "")
            if strict_status not in {"PASS", "PASS_WITH_WARNINGS"}:
                failures.append(f"Strict workflow status is not accepted: {strict_status}")
            elif strict_status == "PASS_WITH_WARNINGS":
                warnings.append("Strict workflow validation passed with accepted warnings.")

    generated_report_statuses = validate_generated_reports(failures, warnings)

    if not args.run_trace_validation:
        warnings.append(
            "TRACE paper-exact validation is connected but was not run. "
            "Run with --run-trace-validation for the final full release gate."
        )

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
            "run_trace_validation": args.run_trace_validation,
            "rebuild_paper_replay": args.rebuild_paper_replay,
            "allow_missing_full_audit_proof": args.allow_missing_full_audit_proof,
        },
        "static_path_checks": static_checks,
        "commands": commands,
        "generated_report_statuses": generated_report_statuses,
        "strict_workflow_report": str(strict_report_path),
        "strict_workflow_status": strict_report.get("status", "") if strict_report else "",
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
