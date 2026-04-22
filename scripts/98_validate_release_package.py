#!/usr/bin/env python3
"""Validate the TRACE advisor-review release package."""

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
    "data/README.md",
    "data/raw/train/beers/clean.csv",
    "data/raw/train/flights/clean.csv",
    "data/raw/train/hospital/clean.csv",
    "data/raw/train/rayyan/clean.csv",
    "data/pre_experiment/alpha_metrics.csv",
    "docs/data_policy.md",
    "docs/hardware_runtime.md",
    "docs/release_packaging.md",
    "docs/stage1_to_stage4_plan.md",
    "docs/terminal_interface.md",
    "docs/mode_a_paper_replay_validation.md",
    "docs/stage3_strict_validation.md",
    "docs/paper_output_traceability.md",
    "scripts/00_trace_home.py",
    "scripts/45_validate_data_availability.py",
    "scripts/62_validate_mode_a_paper_replay.py",
    "scripts/63_validate_stage3_strict.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TRACE release package.")
    parser.add_argument(
        "--skip-stage3-strict",
        action="store_true",
        help="Skip Stage 3 strict validation. Not recommended for final advisor release.",
    )
    parser.add_argument(
        "--rebuild-mode-a",
        action="store_true",
        help="Ask Stage 3 strict validation to rebuild Mode A.",
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

    lines += [
        "",
        "## Command checks",
        "",
        "| Check | Status | Return code |",
        "|---|---|---:|",
    ]

    for row in report["commands"]:
        lines.append(f"| {row['name']} | {row['status']} | {row['returncode']} |")

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
        "PASS_WITH_WARNINGS is acceptable for v0.1.2-advisor when Stage 3 strict validation reports accepted Mode A traceability warnings only.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    static_checks = static_path_checks()

    failures = []
    warnings = []

    for row in static_checks:
        if not row["exists"] or not row["is_file"]:
            failures.append(f"Missing required file: {row['path']}")

    commands = []

    commands.append(
        run_command("data_availability", ["scripts/45_validate_data_availability.py"])
    )
    commands.append(
        run_command("setup_mode_b", ["scripts/00_setup_check.py", "--config", "configs/mode_b_smoke.yaml", "--strict"])
    )
    commands.append(
        run_command("setup_mode_c", ["scripts/00_setup_check.py", "--config", "configs/mode_c_full.yaml", "--check-all-data", "--strict"])
    )
    commands.append(
        run_command("mode_b_smoke", ["scripts/90_run_smoke_from_scratch.py", "--config", "configs/mode_b_smoke.yaml", "--clean"])
    )

    if args.skip_stage3_strict:
        warnings.append("Stage 3 strict validation was skipped by user request.")
    else:
        stage3_cmd = ["scripts/63_validate_stage3_strict.py", "--skip-mode-b-rerun"]
        if args.rebuild_mode_a:
            stage3_cmd.append("--rebuild-mode-a")
        commands.append(run_command("stage3_strict_validation", stage3_cmd))

    for row in commands:
        if row["status"] != "PASS":
            failures.append(f"Command failed: {row['name']}")

    stage3_report_path = ROOT / "results/logs/stage3_strict_validation_report.json"
    stage3_report = read_json_if_exists(stage3_report_path)

    if not args.skip_stage3_strict:
        if not stage3_report:
            failures.append(f"Missing Stage 3 strict report: {stage3_report_path}")
        else:
            stage3_status = stage3_report.get("status", "")
            if stage3_status not in {"PASS", "PASS_WITH_WARNINGS"}:
                failures.append(f"Stage 3 strict status is not accepted: {stage3_status}")
            elif stage3_status == "PASS_WITH_WARNINGS":
                warnings.append("Stage 3 strict validation passed with accepted warnings.")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "static_path_checks": static_checks,
        "commands": commands,
        "stage3_strict_report": str(stage3_report_path),
        "stage3_strict_status": stage3_report.get("status", "") if stage3_report else "",
        "warnings": warnings,
        "failures": failures,
    }

    write_json(args.output, report)
    write_markdown(args.output.with_suffix(".md"), report)

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

