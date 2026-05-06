#!/usr/bin/env python3
"""Validate TRACE strict reviewer workflow evidence.

Reviewer-facing workflows:
  paper-replay
  benchmark-smoke
  benchmark-full-audit

Hidden compatibility options:
  --rebuild-mode-a       -> --rebuild-paper-replay
  --skip-mode-b-rerun    -> --skip-smoke-rerun
  --mode-c-proof-dir     -> --full-audit-proof-dir
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

# These names are produced by the historical Linux strict proof script.
# They are kept as proof-row identifiers, not reviewer-facing workflow names.
EXPECTED_FULL_AUDIT_CHECKS = [
    "setup_mode_b",
    "setup_mode_c",
    "method_registry",
    "static_checks",
    "mode_b_smoke",
    "mode_b_smoke_manifest",
    "clusterer_coverage",
    "clusterer_coverage_check",
    "torch110_dependency_probe",
    "torch110_dependency_probe_check",
    "boostclean_import_probe",
    "holoclean_import_probe",
    "holoclean_db_check",
    "cleaner_mode",
    "cleaner_baran",
    "cleaner_holoclean",
    "cleaner_bigdansing",
    "cleaner_boostclean",
    "cleaner_horizon",
    "cleaner_scared",
    "cleaner_unified",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TRACE strict reviewer workflow evidence.")
    parser.add_argument(
        "--full-audit-proof-dir",
        dest="full_audit_proof_dir",
        type=Path,
        default=None,
        help="Directory containing strict Linux full-audit proof files, including RESULT and summary.tsv.",
    )
    parser.add_argument(
        "--mode-c-proof-dir",
        dest="full_audit_proof_dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-smoke-rerun",
        dest="skip_smoke_rerun",
        action="store_true",
        help="Do not rerun benchmark-smoke. Only check existing smoke reports.",
    )
    parser.add_argument(
        "--skip-mode-b-rerun",
        dest="skip_smoke_rerun",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rebuild-paper-replay",
        dest="rebuild_paper_replay",
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
        dest="allow_missing_full_audit_proof",
        action="store_true",
        help="Allow missing benchmark-full-audit proof as a warning. Not recommended for final archival release.",
    )
    parser.add_argument(
        "--allow-missing-mode-c-proof",
        dest="allow_missing_full_audit_proof",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/logs/stage3_strict_validation_report.json"),
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> dict[str, Any]:
    print("[TRACE] >>>", " ".join([sys.executable] + cmd))
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
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


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


def find_full_audit_proof_dir(explicit: Path | None) -> Path | None:
    if explicit:
        return explicit

    candidates = sorted(
        (ROOT / "results/logs").glob("stage2_strict_*"),
        key=lambda p: p.name,
        reverse=True,
    )

    for candidate in candidates:
        if (candidate / "RESULT").exists() and (candidate / "summary.tsv").exists():
            return candidate

    return None


def validate_paper_replay(rebuild: bool) -> dict[str, Any]:
    cmd = ["scripts/62_validate_mode_a_paper_replay.py"]
    if rebuild:
        cmd.append("--rebuild")

    run = run_command(cmd)

    report_path = ROOT / "analysis/paper_generated/paper_replay_validation_report.json"
    legacy_report_path = ROOT / "analysis/paper_generated/mode_a_paper_replay_validation_report.json"

    report = read_json_if_exists(report_path)
    if not report and legacy_report_path.exists():
        report = read_json_if_exists(legacy_report_path)
        report_path = legacy_report_path

    accepted = (
        run["returncode"] == 0
        and report.get("status") in {"PASS", "PASS_WITH_WARNINGS"}
        and not report.get("failures")
    )

    status = "FAIL"
    if accepted:
        status = "PASS_WITH_WARNINGS" if report.get("status") == "PASS_WITH_WARNINGS" else "PASS"

    return {
        "workflow": "paper-replay",
        "description": "Paper-level evidence replay: tables, figures, traceability, and validity checks.",
        "status": status,
        "accepted": accepted,
        "command": run,
        "report_path": str(report_path),
        "report_status": report.get("status", ""),
        "warning_count": len(report.get("warnings", [])),
        "failure_count": len(report.get("failures", [])),
        "failures": [] if accepted else [f"paper-replay report not accepted: {report_path} status={report.get('status', '')}"],
        "warnings": report.get("warnings", []),
    }


def validate_benchmark_smoke(skip_rerun: bool) -> dict[str, Any]:
    commands = []
    failures: list[str] = []

    if not skip_rerun:
        commands.append(
            run_command([
                "scripts/00_setup_check.py",
                "--config",
                "configs/benchmark_smoke.yaml",
                "--strict",
            ])
        )
        commands.append(
            run_command([
                "scripts/90_run_smoke_from_scratch.py",
                "--config",
                "configs/benchmark_smoke.yaml",
                "--clean",
            ])
        )

    manifest_path = ROOT / "results/logs/pipeline_run_manifest.json"
    smoke_summary_path = ROOT / "results/logs/mode_b_smoke_summary.json"

    if not manifest_path.exists():
        failures.append(f"Missing benchmark-smoke pipeline manifest: {manifest_path}")
    else:
        manifest = read_json_if_exists(manifest_path)
        if manifest.get("failure_count", 0) != 0:
            failures.append(f"benchmark-smoke pipeline manifest has failures: {manifest.get('failure_count')}")
        if manifest.get("cleaned_result_count", 0) < 1:
            failures.append("benchmark-smoke cleaned_result_count < 1")
        if manifest.get("clustered_result_count", 0) < 1:
            failures.append("benchmark-smoke clustered_result_count < 1")

    if not smoke_summary_path.exists():
        failures.append(f"Missing benchmark-smoke summary: {smoke_summary_path}")

    for command in commands:
        if command["returncode"] != 0:
            failures.append(f"Command failed: {command['command']}")

    return {
        "workflow": "benchmark-smoke",
        "description": "Lightweight cleaning-clustering smoke pipeline from scratch.",
        "status": "PASS" if not failures else "FAIL",
        "accepted": not failures,
        "commands": commands,
        "manifest_path": str(manifest_path),
        "smoke_summary_path": str(smoke_summary_path),
        "warnings": [],
        "failures": failures,
    }


def parse_summary_tsv(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append(
            {
                "name": parts[0] if len(parts) > 0 else "",
                "status": parts[1] if len(parts) > 1 else "",
                "log": parts[2] if len(parts) > 2 else "",
            }
        )
    return rows


def validate_benchmark_full_audit(proof_dir: Path | None, allow_missing: bool) -> dict[str, Any]:
    if proof_dir is None or not proof_dir.exists():
        status = "WARN_MISSING_PROOF" if allow_missing else "FAIL"
        return {
            "workflow": "benchmark-full-audit",
            "description": "Strict cleaning-clustering execution proof.",
            "status": status,
            "accepted": allow_missing,
            "proof_dir": str(proof_dir) if proof_dir else "",
            "failures": [] if allow_missing else ["benchmark-full-audit proof directory not found."],
            "warnings": ["benchmark-full-audit proof directory not found."] if allow_missing else [],
        }

    result_path = proof_dir / "RESULT"
    summary_path = proof_dir / "summary.tsv"

    failures: list[str] = []
    warnings: list[str] = []

    if not result_path.exists():
        failures.append(f"Missing RESULT file: {result_path}")
        result_value = ""
    else:
        result_value = result_path.read_text(encoding="utf-8-sig", errors="replace").strip()
        if result_value != "PASSED":
            failures.append(f"RESULT is not PASSED: {result_value}")

    if not summary_path.exists():
        failures.append(f"Missing summary.tsv: {summary_path}")
        rows = []
    else:
        rows = parse_summary_tsv(summary_path)

    row_by_name = {row["name"]: row for row in rows}
    missing_checks = [name for name in EXPECTED_FULL_AUDIT_CHECKS if name not in row_by_name]
    fail_rows = [row for row in rows if row.get("status") == "FAIL"]

    if missing_checks:
        failures.append(f"Missing full-audit checks: {missing_checks}")

    if fail_rows:
        failures.append(f"Full-audit summary has FAIL rows: {fail_rows}")

    for name in EXPECTED_FULL_AUDIT_CHECKS:
        if name in row_by_name and row_by_name[name].get("status") != "PASS":
            failures.append(f"Full-audit check not PASS: {name}={row_by_name[name].get('status')}")

    return {
        "workflow": "benchmark-full-audit",
        "description": "Strict cleaning-clustering execution proof.",
        "status": "PASS" if not failures else "FAIL",
        "accepted": not failures,
        "proof_dir": str(proof_dir),
        "result_path": str(result_path),
        "summary_path": str(summary_path),
        "result_value": result_value,
        "summary_row_count": len(rows),
        "expected_check_count": len(EXPECTED_FULL_AUDIT_CHECKS),
        "missing_checks": missing_checks,
        "fail_rows": fail_rows,
        "warnings": warnings,
        "failures": failures,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# TRACE Strict Reviewer Workflow Validation",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Status: {report['status']}",
        "",
        "## Workflow summary",
        "",
        "| Workflow | Status | Accepted | Description |",
        "|---|---|---:|---|",
    ]

    for workflow in report["workflows"]:
        lines.append(
            f"| {workflow['workflow']} | {workflow['status']} | {workflow['accepted']} | {workflow['description']} |"
        )

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
        "PASS means paper-replay, benchmark-smoke, and benchmark-full-audit all passed.",
        "",
        "PASS_WITH_WARNINGS is acceptable only when warnings are documented and do not affect paper-table, paper-figure, TRACE, or benchmark proof validity.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    full_audit_dir = find_full_audit_proof_dir(args.full_audit_proof_dir)

    workflows = [
        validate_paper_replay(args.rebuild_paper_replay),
        validate_benchmark_smoke(args.skip_smoke_rerun),
        validate_benchmark_full_audit(full_audit_dir, args.allow_missing_full_audit_proof),
    ]

    failures: list[str] = []
    warnings: list[str] = []

    for workflow in workflows:
        if not workflow.get("accepted"):
            failures.append(f"{workflow['workflow']} failed: {workflow.get('status')}")
        elif workflow.get("status") != "PASS":
            warnings.append(f"{workflow['workflow']} accepted with status: {workflow.get('status')}")

        for failure in workflow.get("failures", []):
            failures.append(f"{workflow['workflow']}: {failure}")

        for warning in workflow.get("warnings", []):
            warnings.append(f"{workflow['workflow']}: {warning}")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "workflows": workflows,
        "warnings": warnings,
        "failures": failures,
        "scope_note": "This validates strict reviewer workflows: paper-replay, benchmark-smoke, and benchmark-full-audit.",
    }

    output = ROOT / args.output
    write_json(output, report)
    write_markdown(output.with_suffix(".md"), report)

    print(json.dumps(
        {
            "status": status,
            "workflow_statuses": {workflow["workflow"]: workflow["status"] for workflow in workflows},
            "warning_count": len(warnings),
            "failure_count": len(failures),
            "output": str(args.output),
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"[TRACE] Strict reviewer workflow validation report written to: {args.output}")
    print(f"[TRACE] Strict reviewer workflow validation status: {status}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()
