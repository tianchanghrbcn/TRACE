#!/usr/bin/env python3
"""Validate Mode A paper replay for Stage 3R.

This script is the reviewer-facing Mode A validation wrapper.

It checks the existing Mode A paper-replay reports by default. With
`--rebuild`, it also rebuilds the Mode A archive, summary, table, figure, and
combined paper-output traceability reports before validation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORTS = {
    "archive_replay": Path("analysis/paper_exact/mode_a_paper_exact_validation_report.json"),
    "generated_summaries": Path("analysis/paper_generated/generated_summary_validation_report.json"),
    "paper_table_outputs": Path("analysis/paper_generated/paper_tables/paper_table_validation_report.json"),
    "paper_table_layers": Path("analysis/paper_generated/paper_tables/table_equivalence_layered_report.json"),
    "paper_figure_outputs": Path("analysis/paper_generated/paper_figures/paper_figure_validation_report.json"),
    "paper_figure_traceability": Path("analysis/paper_generated/paper_figures/paper_figure_traceability_report.json"),
    "paper_output_traceability": Path("analysis/paper_generated/paper_output_traceability_report.json"),
}


ACCEPTED_STATUSES = {
    "archive_replay": {"PASS"},
    "generated_summaries": {"PASS"},
    "paper_table_outputs": {"PASS"},
    "paper_table_layers": {"PASS", "PASS_WITH_DIAGNOSTIC_WARNINGS"},
    "paper_figure_outputs": {"PASS"},
    "paper_figure_traceability": {"PASS", "PASS_WITH_WARNINGS"},
    "paper_output_traceability": {"PASS", "PASS_WITH_WARNINGS"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TRACE Mode A paper replay.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild Mode A reports before validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper_generated/mode_a_paper_replay_validation_report.json"),
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
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "status": "PASS" if proc.returncode == 0 else "FAIL",
    }


def rebuild_reports() -> list[dict[str, Any]]:
    """Rebuild the full Mode A table/figure traceability evidence."""
    commands = [
        [
            "scripts/trace.py",
            "mode-a",
            "--audit",
            "--clean",
            "--generated-summaries",
            "--paper-tables",
            "--table-equivalence",
        ],
        [
            "scripts/trace.py",
            "mode-a",
            "--paper-figures",
            "--figure-traceability",
        ],
        [
            "scripts/61_build_paper_output_traceability_report.py",
        ],
    ]

    results = []
    for cmd in commands:
        result = run_command(cmd)
        results.append(result)
        if result["returncode"] != 0:
            break

    return results


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_status(report_name: str, data: dict[str, Any]) -> str:
    return str(data.get("status", ""))


def validate_report(report_name: str, path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": report_name,
        "path": str(path),
        "exists": path.exists(),
        "status": "",
        "accepted_statuses": sorted(ACCEPTED_STATUSES[report_name]),
        "accepted": False,
        "details": {},
    }

    if not path.exists():
        return row

    data = read_json(path)
    status = extract_status(report_name, data)

    row["status"] = status
    row["accepted"] = status in ACCEPTED_STATUSES[report_name]

    if report_name == "paper_table_layers":
        layers = data.get("layers", {})
        paper = layers.get("paper_facing", {})
        row["details"] = {
            "paper_facing_count": paper.get("count", 0),
            "paper_facing_status_counts": paper.get("status_counts", {}),
            "paper_facing_hard_failure_count": paper.get("hard_failure_count", 0),
            "paper_facing_warning_count": paper.get("warning_count", 0),
        }

        if paper.get("hard_failure_count", 1) != 0:
            row["accepted"] = False

    elif report_name == "paper_figure_outputs":
        row["details"] = {
            "script_count": data.get("script_count", 0),
            "failed_script_count": data.get("failed_script_count", 0),
            "collected_figure_count": data.get("collected_figure_count", 0),
            "extension_counts": data.get("extension_counts", {}),
        }

        if data.get("failed_script_count", 1) != 0:
            row["accepted"] = False
        if data.get("collected_figure_count", 0) <= 0:
            row["accepted"] = False

    elif report_name == "paper_figure_traceability":
        status_counts = data.get("tex_reference_status_counts", {})
        row["details"] = {
            "tex_reference_count": data.get("tex_reference_count", 0),
            "archived_figure_count": data.get("archived_figure_count", 0),
            "generated_figure_count": data.get("generated_figure_count", 0),
            "tex_reference_status_counts": status_counts,
        }

        if status_counts.get("FAIL_NO_REFERENCE", 0):
            row["accepted"] = False

    elif report_name == "paper_output_traceability":
        table = data.get("table_traceability", {})
        figure = data.get("figure_traceability", {})

        row["details"] = {
            "table_status": table.get("status", ""),
            "table_paper_facing_hard_failures": table.get("paper_facing_hard_failure_count", ""),
            "figure_traceability_status": figure.get("traceability_status", ""),
            "figure_tex_reference_count": figure.get("tex_reference_count", ""),
            "figure_tex_reference_status_counts": figure.get("tex_reference_status_counts", {}),
        }

        if table.get("paper_facing_hard_failure_count", 1) != 0:
            row["accepted"] = False

        figure_counts = figure.get("tex_reference_status_counts", {})
        if figure_counts.get("FAIL_NO_REFERENCE", 0):
            row["accepted"] = False

    return row


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Mode A Paper Replay Validation",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Status: {report['status']}",
        f"- Rebuild requested: {report['rebuild_requested']}",
        "",
        "## Report checks",
        "",
        "| Check | Status | Accepted | Path |",
        "|---|---|---:|---|",
    ]

    for row in report["checks"]:
        lines.append(
            f"| {row['name']} | {row['status']} | {row['accepted']} | {row['path']} |"
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
        "## Interpretation",
        "",
        "PASS means all Mode A paper-replay report checks passed without warnings.",
        "",
        "PASS_WITH_WARNINGS is acceptable for the current Stage 3R state when:",
        "",
        "- paper-facing tables have no hard mismatches;",
        "- every LaTeX-referenced figure has archived traceability;",
        "- the figure harness has no failed scripts;",
        "- remaining warnings are diagnostic or archived-only cases.",
        "",
        "Narrative claim traceability is intentionally deferred.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    rebuild_results = []
    if args.rebuild:
        rebuild_results = rebuild_reports()

    failures = []
    warnings = []

    for result in rebuild_results:
        if result["returncode"] != 0:
            failures.append(f"Rebuild command failed: {result['command']}")

    checks = []
    for name, path in REPORTS.items():
        row = validate_report(name, path)
        checks.append(row)

        if not row["exists"]:
            failures.append(f"Missing report: {name} -> {path}")
            continue

        if not row["accepted"]:
            failures.append(
                f"Report check failed: {name} status={row['status']} path={path}"
            )
        elif row["status"] != "PASS":
            warnings.append(f"Report has accepted warning status: {name}={row['status']}")

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
        "rebuild_results": rebuild_results,
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
        "scope_note": (
            "This validates Mode A paper-output replay for paper tables and figures. "
            "Narrative claim traceability is deferred."
        ),
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
    print(f"[TRACE] Mode A paper replay validation report written to: {args.output}")
    print(f"[TRACE] Mode A paper replay validation status: {status}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()

