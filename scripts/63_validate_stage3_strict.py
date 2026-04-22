#!/usr/bin/env python3
"""Validate TRACE Stage 3 strict completion.

This wrapper validates the three Stage 3 modes:

Mode A:
    Paper replay from archived/generated outputs.

Mode B:
    Lightweight smoke run from scratch.

Mode C:
    Strict execution-layer proof from Linux Stage 2 validation.

The default behavior reruns Mode B because it is lightweight, validates existing
Mode A reports, and checks Mode C proof files without rerunning the long Linux
strict validation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_MODE_C_CHECKS = [
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
    parser = argparse.ArgumentParser(description="Validate TRACE Stage 3 strict completion.")
    parser.add_argument(
        "--mode-c-proof-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing Linux Stage 2 strict proof files, including "
            "RESULT and summary.tsv. Example: results/logs/stage2_strict_20260421_192851"
        ),
    )
    parser.add_argument(
        "--skip-mode-b-rerun",
        action="store_true",
        help="Do not rerun Mode B smoke. Only check existing reports.",
    )
    parser.add_argument(
        "--rebuild-mode-a",
        action="store_true",
        help="Rebuild Mode A paper replay before validating it.",
    )
    parser.add_argument(
        "--allow-missing-mode-c-proof",
        action="store_true",
        help="Allow missing Mode C proof as a warning. Not recommended for final strict validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/logs/stage3_strict_validation_report.json"),
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> dict[str, Any]:
    print("[TRACE] >>>", " ".join(cmd))
    proc = subprocess.run(
        [sys.executable] + cmd,
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def find_mode_c_proof_dir(explicit: Path | None) -> Path | None:
    if explicit:
        return explicit

    candidates = sorted(
        Path("results/logs").glob("stage2_strict_*"),
        key=lambda p: p.name,
        reverse=True,
    )

    for candidate in candidates:
        if (candidate / "RESULT").exists() and (candidate / "summary.tsv").exists():
            return candidate

    return None


def validate_mode_a(rebuild: bool) -> dict[str, Any]:
    cmd = ["scripts/62_validate_mode_a_paper_replay.py"]
    if rebuild:
        cmd.append("--rebuild")

    run = run_command(cmd)

    report_path = Path("analysis/paper_generated/mode_a_paper_replay_validation_report.json")
    report = read_json(report_path) if report_path.exists() else {}

    accepted = (
        run["returncode"] == 0
        and report.get("status") in {"PASS", "PASS_WITH_WARNINGS"}
        and not report.get("failures")
    )

    return {
        "mode": "Mode A",
        "description": "Paper table and figure replay validation.",
        "status": "PASS_WITH_WARNINGS" if accepted and report.get("status") == "PASS_WITH_WARNINGS" else ("PASS" if accepted else "FAIL"),
        "accepted": accepted,
        "command": run,
        "report_path": str(report_path),
        "report_status": report.get("status", ""),
        "warning_count": len(report.get("warnings", [])),
        "failure_count": len(report.get("failures", [])),
    }


def validate_mode_b(skip_rerun: bool) -> dict[str, Any]:
    commands = []

    if not skip_rerun:
        commands.append(
            run_command([
                "scripts/00_setup_check.py",
                "--config",
                "configs/mode_b_smoke.yaml",
                "--strict",
            ])
        )
        commands.append(
            run_command([
                "scripts/90_run_smoke_from_scratch.py",
                "--config",
                "configs/mode_b_smoke.yaml",
                "--clean",
            ])
        )

    manifest_path = Path("results/logs/pipeline_run_manifest.json")
    smoke_summary_path = Path("results/logs/mode_b_smoke_summary.json")

    failures = []

    if not manifest_path.exists():
        failures.append(f"Missing pipeline manifest: {manifest_path}")
    else:
        manifest = read_json(manifest_path)
        if manifest.get("failure_count", 0) != 0:
            failures.append(f"Mode B pipeline manifest has failures: {manifest.get('failure_count')}")
        if manifest.get("cleaned_result_count", 0) < 1:
            failures.append("Mode B cleaned_result_count < 1")
        if manifest.get("clustered_result_count", 0) < 1:
            failures.append("Mode B clustered_result_count < 1")

    if not smoke_summary_path.exists():
        failures.append(f"Missing Mode B smoke summary: {smoke_summary_path}")

    for command in commands:
        if command["returncode"] != 0:
            failures.append(f"Command failed: {command['command']}")

    return {
        "mode": "Mode B",
        "description": "Smoke pipeline from scratch.",
        "status": "PASS" if not failures else "FAIL",
        "accepted": not failures,
        "commands": commands,
        "manifest_path": str(manifest_path),
        "smoke_summary_path": str(smoke_summary_path),
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


def validate_mode_c(proof_dir: Path | None, allow_missing: bool) -> dict[str, Any]:
    if proof_dir is None or not proof_dir.exists():
        status = "WARN_MISSING_PROOF" if allow_missing else "FAIL"
        return {
            "mode": "Mode C",
            "description": "Strict cleaning-clustering execution-layer proof.",
            "status": status,
            "accepted": allow_missing,
            "proof_dir": str(proof_dir) if proof_dir else "",
            "failures": [] if allow_missing else ["Mode C proof directory not found."],
            "warnings": ["Mode C proof directory not found."] if allow_missing else [],
        }

    result_path = proof_dir / "RESULT"
    summary_path = proof_dir / "summary.tsv"

    failures = []
    warnings = []

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

    missing_checks = [name for name in EXPECTED_MODE_C_CHECKS if name not in row_by_name]
    fail_rows = [row for row in rows if row.get("status") == "FAIL"]

    if missing_checks:
        failures.append(f"Missing Mode C checks: {missing_checks}")

    if fail_rows:
        failures.append(f"Mode C summary has FAIL rows: {fail_rows}")

    for name in EXPECTED_MODE_C_CHECKS:
        if name in row_by_name and row_by_name[name].get("status") != "PASS":
            failures.append(f"Mode C check not PASS: {name}={row_by_name[name].get('status')}")

    return {
        "mode": "Mode C",
        "description": "Strict cleaning-clustering execution-layer proof.",
        "status": "PASS" if not failures else "FAIL",
        "accepted": not failures,
        "proof_dir": str(proof_dir),
        "result_path": str(result_path),
        "summary_path": str(summary_path),
        "result_value": result_value,
        "summary_row_count": len(rows),
        "expected_check_count": len(EXPECTED_MODE_C_CHECKS),
        "missing_checks": missing_checks,
        "fail_rows": fail_rows,
        "warnings": warnings,
        "failures": failures,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# TRACE Stage 3 Strict Validation",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Status: {report['status']}",
        "",
        "## Mode summary",
        "",
        "| Mode | Status | Accepted | Description |",
        "|---|---|---:|---|",
    ]

    for mode in report["modes"]:
        lines.append(
            f"| {mode['mode']} | {mode['status']} | {mode['accepted']} | {mode['description']} |"
        )

    lines += [
        "",
        "## Failures",
        "",
    ]

    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- {failure}")
    else:
        lines.append("No hard failures.")

    lines += [
        "",
        "## Warnings",
        "",
    ]

    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("No warnings.")

    lines += [
        "",
        "## Interpretation",
        "",
        "PASS means Mode A, Mode B, and Mode C all passed without warnings.",
        "",
        "PASS_WITH_WARNINGS is acceptable for the current Stage 3 state when:",
        "",
        "- Mode A paper replay has accepted traceability warnings only;",
        "- Mode B smoke rerun passes;",
        "- Mode C strict Linux proof is present and passed;",
        "- claim-level narrative traceability is deferred.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    mode_c_dir = find_mode_c_proof_dir(args.mode_c_proof_dir)

    mode_a = validate_mode_a(args.rebuild_mode_a)
    mode_b = validate_mode_b(args.skip_mode_b_rerun)
    mode_c = validate_mode_c(mode_c_dir, args.allow_missing_mode_c_proof)

    modes = [mode_a, mode_b, mode_c]

    failures = []
    warnings = []

    for mode in modes:
        if not mode.get("accepted"):
            failures.append(f"{mode['mode']} failed: {mode.get('status')}")
        elif mode.get("status") != "PASS":
            warnings.append(f"{mode['mode']} accepted with status: {mode.get('status')}")

        for failure in mode.get("failures", []):
            failures.append(f"{mode['mode']}: {failure}")

        for warning in mode.get("warnings", []):
            warnings.append(f"{mode['mode']}: {warning}")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "modes": modes,
        "warnings": warnings,
        "failures": failures,
        "scope_note": (
            "This validates Stage 3 core completion for Mode A, Mode B, and Mode C. "
            "Narrative claim traceability is deferred."
        ),
    }

    write_json(args.output, report)
    write_markdown(args.output.with_suffix(".md"), report)

    print(json.dumps(
        {
            "status": status,
            "mode_statuses": {
                mode["mode"]: mode["status"]
                for mode in modes
            },
            "warning_count": len(warnings),
            "failure_count": len(failures),
            "output": str(args.output),
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"[TRACE] Stage 3 strict validation report written to: {args.output}")
    print(f"[TRACE] Stage 3 strict validation status: {status}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()

