#!/usr/bin/env python3
"""Validate TRACE paper-level artifact replay.

Final reviewer-facing paper replay uses the maintained registered builders:

  scripts/50_build_all_paper_figures.py
  scripts/51_build_all_paper_tables.py

Older paper replay helpers are kept for diagnostics, but this validator no
longer depends on the old Mode-A-style table/figure harness.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TRACE paper-level artifact replay.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild paper figure/table artifacts before validation.")
    parser.add_argument("--skip-preexp-validity", action="store_true", help="Skip pre-experiment/validity replay.")
    parser.add_argument("--input-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--figure-output-dir",
        type=Path,
        default=Path("analysis/paper_generated/paper_artifact/figures"),
    )
    parser.add_argument(
        "--table-output-dir",
        type=Path,
        default=Path("analysis/paper_generated/paper_artifact/tables"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper_generated/paper_replay_validation_report.json"),
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_manifest(name: str, path: Path, kind: str) -> dict[str, Any]:
    data = read_json(path)
    row: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "path": str(path),
        "exists": path.exists(),
        "accepted": False,
        "status": "",
        "details": {},
    }

    if not data:
        row["status"] = "MISSING"
        return row

    counts = data.get("counts", {})
    success = int(counts.get("success", 0) or 0)
    failed = int(counts.get("failed", 0) or 0)
    skipped = int(counts.get("skipped", 0) or 0)
    errors = data.get("errors", [])

    row["details"] = {
        "counts": counts,
        "registered_count": len(data.get("registered_specs", [])),
        "result_count": len(data.get("results", [])),
        "copied_output_count": len(data.get("copied_outputs", [])),
        "error_count": len(errors),
    }

    if success > 0 and failed == 0 and not errors:
        row["accepted"] = True
        row["status"] = "PASS"
    else:
        row["status"] = "FAIL"

    return row


def validate_status_report(name: str, path: Path, accepted_statuses: set[str]) -> dict[str, Any]:
    data = read_json(path)
    row: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "accepted": False,
        "status": "",
        "details": {},
    }
    if not data:
        row["status"] = "MISSING"
        return row

    status = str(data.get("status", ""))
    row["status"] = status
    row["accepted"] = status in accepted_statuses
    row["details"] = {
        "failure_count": data.get("failure_count", len(data.get("failures", [])) if isinstance(data.get("failures", []), list) else ""),
        "warning_count": data.get("warning_count", len(data.get("warnings", [])) if isinstance(data.get("warnings", []), list) else ""),
    }
    if data.get("failures"):
        row["accepted"] = False
    return row


def rebuild(args: argparse.Namespace) -> list[dict[str, Any]]:
    commands: list[list[str]] = [
        [
            "scripts/50_build_all_paper_figures.py",
            "--input-root", str(args.input_root),
            "--output-dir", str(args.figure_output_dir),
            "--strict",
            "--clean-output",
        ],
        [
            "scripts/51_build_all_paper_tables.py",
            "--input-root", str(args.input_root),
            "--output-dir", str(args.table_output_dir),
            "--strict",
            "--clean-output",
        ],
    ]

    if not args.skip_preexp_validity:
        commands.append([
            "scripts/81_replay_pre_experiment_validity.py",
            "--generated-data-policy", "fail",
            "--error-model-policy", "fail",
        ])

    results = []
    for cmd in commands:
        result = run_command(cmd)
        results.append(result)
        if result["returncode"] != 0:
            break
    return results


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# TRACE Paper Artifact Replay Validation",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Status: {report['status']}",
        f"- Rebuild requested: {report['rebuild_requested']}",
        "",
        "## Checks",
        "",
        "| Check | Status | Accepted | Path |",
        "|---|---|---:|---|",
    ]

    for row in report["checks"]:
        lines.append(f"| {row['name']} | {row['status']} | {row['accepted']} | {row['path']} |")

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
        "PASS means the maintained registered paper figure/table builders and validity checks passed.",
        "",
        "This validator intentionally uses `50_build_all_paper_figures.py` and `51_build_all_paper_tables.py` as the final reviewer-facing paper artifact checks.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    rebuild_results: list[dict[str, Any]] = []
    if args.rebuild:
        rebuild_results = rebuild(args)

    failures: list[str] = []
    warnings: list[str] = []

    for result in rebuild_results:
        if result["returncode"] != 0:
            failures.append(f"Rebuild command failed: {result['command']}")

    figure_manifest = ROOT / args.figure_output_dir / "figures_manifest.json"
    table_manifest = ROOT / args.table_output_dir / "tables_manifest.json"

    checks = [
        validate_manifest("paper_figures_registered_build", figure_manifest, "figures"),
        validate_manifest("paper_tables_registered_build", table_manifest, "tables"),
    ]

    if not args.skip_preexp_validity:
        checks.append(
            validate_status_report(
                "preexp_validity",
                ROOT / "analysis/validity_sensitivity/pre_experiment_validity_report.json",
                {"PASS"},
            )
        )
        checks.append(
            validate_status_report(
                "validity_sensitivity",
                ROOT / "analysis/validity_sensitivity/validity_sensitivity_summary.json",
                {"PASS", "PASS_WITH_WARNINGS"},
            )
        )

    for row in checks:
        if not row["accepted"]:
            failures.append(f"Check failed: {row['name']} status={row['status']} path={row['path']}")
        elif row["status"] == "PASS_WITH_WARNINGS":
            warnings.append(f"Accepted warning status: {row['name']}")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "rebuild_requested": args.rebuild,
        "scope": "paper artifact replay using registered figure/table builders",
        "rebuild_results": rebuild_results,
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
        "maintained_entry_points": [
            "scripts/50_build_all_paper_figures.py",
            "scripts/51_build_all_paper_tables.py",
            "scripts/81_replay_pre_experiment_validity.py",
        ],
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
    print(f"[TRACE] Paper artifact replay validation report written to: {args.output}")
    print(f"[TRACE] Paper artifact replay validation status: {status}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()
