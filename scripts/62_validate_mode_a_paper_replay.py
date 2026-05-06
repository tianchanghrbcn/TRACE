#!/usr/bin/env python3
"""Validate TRACE paper-level evidence replay.

This replaces the old reviewer-facing "Mode A" semantics.

Paper replay now means:
  1. paper summary workbook replay;
  2. paper table replay and table equivalence;
  3. paper figure replay and figure traceability;
  4. combined paper-output traceability;
  5. pre-experiment calibration and validity-sensitivity checks.

The old paper-exact archive validation scripts are kept as source-preparation
or legacy audit helpers, but the reviewer-facing paper replay no longer fails
only because LaTeX source files are not present.
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


REPORTS = {
    "generated_summaries": Path("analysis/paper_generated/generated_summary_validation_report.json"),
    "paper_table_outputs": Path("analysis/paper_generated/paper_tables/paper_table_validation_report.json"),
    "paper_table_layers": Path("analysis/paper_generated/paper_tables/table_equivalence_layered_report.json"),
    "paper_figure_outputs": Path("analysis/paper_generated/paper_figures/paper_figure_validation_report.json"),
    "paper_figure_traceability": Path("analysis/paper_generated/paper_figures/paper_figure_traceability_report.json"),
    "paper_output_traceability": Path("analysis/paper_generated/paper_output_traceability_report.json"),
    "preexp_validity": Path("analysis/validity_sensitivity/pre_experiment_validity_report.json"),
    "validity_sensitivity": Path("analysis/validity_sensitivity/validity_sensitivity_summary.json"),
}


ACCEPTED_STATUSES = {
    "generated_summaries": {"PASS"},
    "paper_table_outputs": {"PASS"},
    "paper_table_layers": {"PASS", "PASS_WITH_DIAGNOSTIC_WARNINGS"},
    "paper_figure_outputs": {"PASS"},
    "paper_figure_traceability": {"PASS", "PASS_WITH_WARNINGS"},
    "paper_output_traceability": {"PASS", "PASS_WITH_WARNINGS"},
    "preexp_validity": {"PASS"},
    "validity_sensitivity": {"PASS", "PASS_WITH_WARNINGS"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TRACE paper-level evidence replay.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild paper-level evidence before validation.",
    )
    parser.add_argument(
        "--skip-source-prep",
        action="store_true",
        help="Skip source selection/archive preparation scripts 46-48.",
    )
    parser.add_argument(
        "--skip-preexp-validity",
        action="store_true",
        help="Skip pre-experiment/validity replay during rebuild.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper_generated/paper_replay_validation_report.json"),
    )
    return parser.parse_args()


def run_command(cmd: list[str], *, allow_failure: bool = False) -> dict[str, Any]:
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

    status = "PASS" if proc.returncode == 0 else ("WARN_ALLOWED_FAILURE" if allow_failure else "FAIL")
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "status": status,
        "allow_failure": allow_failure,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def rebuild_reports(skip_source_prep: bool, skip_preexp_validity: bool) -> list[dict[str, Any]]:
    commands: list[tuple[list[str], bool]] = []

    # 46-48 prepare archived/source workspaces used by later table/figure scripts.
    # 49 is intentionally not part of reviewer-facing paper replay because it
    # requires LaTeX archive hints that are not necessary for paper-output replay.
    if not skip_source_prep:
        commands.extend([
            (["scripts/46_audit_paper_replay_sources.py"], False),
            (["scripts/47_select_paper_exact_sources.py"], False),
            (["scripts/48_build_mode_a_paper_exact_archive.py", "--clean"], False),
        ])

    commands.extend([
        (["scripts/50_audit_paper_table_scripts.py"], False),
        (["scripts/51_build_paper_summary_workbooks.py"], False),
        (["scripts/52_validate_paper_summary_workbooks.py"], False),

        (["scripts/53_run_paper_table_scripts.py", "--clean", "--timeout", "1200", "--include-analysis-scripts"], False),
        (["scripts/54_validate_paper_table_outputs.py"], False),

        # Raw equivalence may report hard mismatches before layered diagnostics.
        # The layered report is the reviewer-facing decision artifact.
        (["scripts/55_validate_paper_table_equivalence.py"], True),
        (["scripts/56_classify_table_equivalence_layers.py"], False),

        (["scripts/57_select_paper_figure_sources.py"], False),
        (["scripts/58_run_paper_figure_scripts.py", "--clean", "--timeout", "1200"], False),
        (["scripts/59_validate_paper_figure_outputs.py"], False),
        (["scripts/60_validate_paper_figure_traceability.py"], False),

        (["scripts/61_build_paper_output_traceability_report.py"], False),
    ])

    if not skip_preexp_validity:
        commands.append((["scripts/81_replay_pre_experiment_validity.py", "--generated-data-policy", "fail", "--error-model-policy", "fail"], False))

    results: list[dict[str, Any]] = []
    for cmd, allow_failure in commands:
        result = run_command(cmd, allow_failure=allow_failure)
        results.append(result)
        if result["returncode"] != 0 and not allow_failure:
            break

    return results


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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
    status = str(data.get("status", ""))

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

    elif report_name == "preexp_validity":
        # The preexp-validity report may expose either explicit
        # failure_count/warning_count fields or failures/warnings lists.
        failures = data.get("failures", [])
        warnings = data.get("warnings", [])

        failure_count = data.get("failure_count")
        if failure_count is None:
            failure_count = len(failures) if isinstance(failures, list) else 0

        warning_count = data.get("warning_count")
        if warning_count is None:
            warning_count = len(warnings) if isinstance(warnings, list) else 0

        row["details"] = {
            "failure_count": failure_count,
            "warning_count": warning_count,
            "report": data.get("report", ""),
            "validity_summary": data.get("validity_summary", ""),
            "top_level_status": data.get("status", ""),
        }

        if data.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
            row["accepted"] = False
        if int(failure_count) != 0:
            row["accepted"] = False

    elif report_name == "validity_sensitivity":
        row["details"] = {
            "scope": data.get("scope", ""),
            "warnings": data.get("warnings", []),
            "failures": data.get("failures", []),
        }
        if data.get("failures"):
            row["accepted"] = False

    return row


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# TRACE Paper Evidence Replay Validation",
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
        "PASS means the paper-level evidence replay passed: summary workbooks, paper tables, paper figures, paper-output traceability, and pre-experiment/validity checks.",
        "",
        "The old Mode A archive validation is not part of this reviewer-facing decision gate.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    rebuild_results: list[dict[str, Any]] = []
    if args.rebuild:
        rebuild_results = rebuild_reports(
            skip_source_prep=args.skip_source_prep,
            skip_preexp_validity=args.skip_preexp_validity,
        )

    failures: list[str] = []
    warnings: list[str] = []

    for result in rebuild_results:
        if result["returncode"] != 0 and not result.get("allow_failure"):
            failures.append(f"Rebuild command failed: {result['command']}")
        elif result["returncode"] != 0 and result.get("allow_failure"):
            warnings.append(f"Allowed diagnostic command returned nonzero: {result['command']}")

    checks = []
    for name, path in REPORTS.items():
        if args.skip_preexp_validity and name in {"preexp_validity", "validity_sensitivity"}:
            continue

        row = validate_report(name, ROOT / path)
        checks.append(row)

        if not row["exists"]:
            failures.append(f"Missing report: {name} -> {path}")
            continue

        if not row["accepted"]:
            failures.append(f"Report check failed: {name} status={row['status']} path={path}")
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
        "scope": "paper-level evidence replay",
        "rebuild_results": rebuild_results,
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
        "scope_note": (
            "This validates paper-level evidence: tables, figures, traceability, "
            "and pre-experiment/validity checks. The old Mode A archive validation "
            "is not required for this reviewer-facing gate."
        ),
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
    print(f"[TRACE] Paper evidence replay validation report written to: {args.output}")
    print(f"[TRACE] Paper evidence replay validation status: {status}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()
