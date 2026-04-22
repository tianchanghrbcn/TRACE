#!/usr/bin/env python3
"""Build the combined paper-output traceability report for Stage 3R.

This report combines:

1. layered paper-table equivalence diagnostics;
2. paper-figure traceability diagnostics.

It is the first consolidated reviewer-facing evidence that paper tables and
figures are traceable to archived and/or generated outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined paper-output traceability report.")
    parser.add_argument(
        "--table-layered-report",
        type=Path,
        default=Path("analysis/paper_generated/paper_tables/table_equivalence_layered_report.json"),
    )
    parser.add_argument(
        "--figure-traceability-report",
        type=Path,
        default=Path("analysis/paper_generated/paper_figures/paper_figure_traceability_report.json"),
    )
    parser.add_argument(
        "--figure-harness-report",
        type=Path,
        default=Path("analysis/paper_generated/paper_figures/paper_figure_validation_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper_generated/paper_output_traceability_report.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def table_summary(report: dict[str, Any]) -> dict[str, Any]:
    layers = report.get("layers", {})
    paper = layers.get("paper_facing", {})

    return {
        "status": report.get("status", ""),
        "raw_status": report.get("raw_status", ""),
        "paper_facing_count": paper.get("count", 0),
        "paper_facing_status_counts": paper.get("status_counts", {}),
        "paper_facing_hard_failure_count": paper.get("hard_failure_count", 0),
        "paper_facing_warning_count": paper.get("warning_count", 0),
        "diagnostic_layers": {
            name: {
                "count": summary.get("count", 0),
                "status_counts": summary.get("status_counts", {}),
                "hard_failure_count": summary.get("hard_failure_count", 0),
                "warning_count": summary.get("warning_count", 0),
            }
            for name, summary in layers.items()
            if name != "paper_facing"
        },
    }


def figure_summary(trace_report: dict[str, Any], harness_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "traceability_status": trace_report.get("status", ""),
        "harness_status": harness_report.get("status", ""),
        "tex_reference_count": trace_report.get("tex_reference_count", 0),
        "archived_figure_count": trace_report.get("archived_figure_count", 0),
        "generated_figure_count": trace_report.get("generated_figure_count", 0),
        "tex_reference_status_counts": trace_report.get("tex_reference_status_counts", {}),
        "generated_match_status_counts": trace_report.get("generated_match_status_counts", {}),
        "harness_script_count": harness_report.get("script_count", 0),
        "harness_failed_script_count": harness_report.get("failed_script_count", 0),
        "harness_collected_figure_count": harness_report.get("collected_figure_count", 0),
    }


def overall_status(table: dict[str, Any], figure: dict[str, Any]) -> str:
    failures = []

    if table["paper_facing_hard_failure_count"]:
        failures.append("paper-facing table hard failures")

    if figure["traceability_status"] == "FAIL":
        failures.append("paper figure traceability failure")

    if figure["harness_status"] == "FAIL":
        failures.append("paper figure harness failure")

    if failures:
        return "FAIL"

    warnings = []

    if table.get("status") not in {"PASS"}:
        warnings.append("table diagnostic warnings")

    if figure.get("traceability_status") not in {"PASS"}:
        warnings.append("figure traceability warnings")

    if figure.get("harness_status") not in {"PASS"}:
        warnings.append("figure harness warnings")

    if warnings:
        return "PASS_WITH_WARNINGS"

    return "PASS"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    table = report["table_traceability"]
    figure = report["figure_traceability"]

    lines = [
        "# Paper Output Traceability Report",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Overall status: {report['status']}",
        "",
        "## Interpretation",
        "",
        "This report combines paper-table and paper-figure traceability evidence.",
        "",
        "A `PASS_WITH_WARNINGS` result is acceptable for the current Stage 3R state when:",
        "",
        "- paper-facing tables have no hard mismatches;",
        "- every LaTeX-referenced paper figure has archived traceability;",
        "- warnings only indicate archived-only figures or upstream intermediate diagnostics.",
        "",
        "Claim-level traceability for narrative conclusions is intentionally deferred.",
        "",
        "## Table traceability",
        "",
        f"- Layered table status: {table['status']}",
        f"- Raw table equivalence status: {table['raw_status']}",
        f"- Paper-facing compared outputs: {table['paper_facing_count']}",
        f"- Paper-facing status counts: {table['paper_facing_status_counts']}",
        f"- Paper-facing hard failures: {table['paper_facing_hard_failure_count']}",
        f"- Paper-facing warnings: {table['paper_facing_warning_count']}",
        "",
        "### Diagnostic table layers",
        "",
        "| Layer | Count | Status counts | Hard failures | Warnings |",
        "|---|---:|---|---:|---:|",
    ]

    for name, summary in sorted(table["diagnostic_layers"].items()):
        lines.append(
            f"| {name} | {summary['count']} | {summary['status_counts']} | "
            f"{summary['hard_failure_count']} | {summary['warning_count']} |"
        )

    lines += [
        "",
        "## Figure traceability",
        "",
        f"- Figure harness status: {figure['harness_status']}",
        f"- Figure traceability status: {figure['traceability_status']}",
        f"- LaTeX figure references: {figure['tex_reference_count']}",
        f"- Archived figure references: {figure['archived_figure_count']}",
        f"- Generated figures: {figure['generated_figure_count']}",
        f"- Figure harness scripts: {figure['harness_script_count']}",
        f"- Failed figure harness scripts: {figure['harness_failed_script_count']}",
        f"- Collected figure files: {figure['harness_collected_figure_count']}",
        "",
        "### LaTeX reference status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]

    for key, value in sorted(figure["tex_reference_status_counts"].items()):
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "### Generated figure match status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]

    for key, value in sorted(figure["generated_match_status_counts"].items()):
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Remaining work",
        "",
        "- Reduce archived-only figure warnings if time permits.",
        "- Add paper-output subset mapping by table/figure number.",
        "- Add claim-level traceability for narrative conclusions after the table/figure layer is stable.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    table_report = read_json(args.table_layered_report)
    figure_trace_report = read_json(args.figure_traceability_report)
    figure_harness_report = read_json(args.figure_harness_report)

    table = table_summary(table_report)
    figure = figure_summary(figure_trace_report, figure_harness_report)

    status = overall_status(table, figure)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "table_layered_report": str(args.table_layered_report),
        "figure_traceability_report": str(args.figure_traceability_report),
        "figure_harness_report": str(args.figure_harness_report),
        "table_traceability": table,
        "figure_traceability": figure,
        "scope_note": (
            "This report validates paper-output traceability for tables and figures. "
            "Narrative claim traceability is deferred."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    write_markdown(args.output.with_suffix(".md"), report)

    print(json.dumps(
        {
            "status": report["status"],
            "table_status": table["status"],
            "table_paper_facing_hard_failures": table["paper_facing_hard_failure_count"],
            "figure_traceability_status": figure["traceability_status"],
            "figure_tex_reference_count": figure["tex_reference_count"],
            "figure_tex_reference_status_counts": figure["tex_reference_status_counts"],
            "output": str(args.output),
        },
        indent=2,
        ensure_ascii=False,
    ))

    print(f"[TRACE] Paper output traceability report written to: {args.output}")
    print(f"[TRACE] Paper output traceability status: {status}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()

