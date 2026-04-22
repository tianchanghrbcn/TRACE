#!/usr/bin/env python3
"""Validate paper figure harness outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FIGURE_EXTENSIONS = {".png", ".pdf", ".svg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paper figure script harness outputs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("analysis/paper_generated/paper_figures/paper_figure_script_run_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper_generated/paper_figures/paper_figure_validation_report.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Run manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Paper Figure Output Validation",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Status: {report['status']}",
        f"- Script count: {report['script_count']}",
        f"- Failed script count: {report['failed_script_count']}",
        f"- Collected output count: {report['collected_output_count']}",
        f"- Collected figure count: {report['collected_figure_count']}",
        "",
        "## Extension counts",
        "",
        "| Extension | Count |",
        "|---|---:|",
    ]

    for key, value in sorted(report["extension_counts"].items()):
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Script runs",
        "",
        "| Script | Status | Changed files | Collected outputs |",
        "|---|---|---:|---:|",
    ]

    for run in report["script_runs"]:
        lines.append(
            f"| {run['script']} | {run['status']} | "
            f"{run['changed_file_count']} | {run['collected_output_count']} |"
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
        "## Scope note",
        "",
        "A PASS_WITH_WARNINGS result is acceptable for the first figure harness run.",
        "It means at least some figure outputs were collected, but one or more legacy figure scripts still require adaptation.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)

    runs = manifest.get("runs", [])
    outputs = manifest.get("collected_outputs", [])
    figures = [
        row for row in outputs
        if Path(row.get("output_path", "")).suffix.lower() in FIGURE_EXTENSIONS
    ]

    failed_runs = [run for run in runs if run.get("status") != "PASS"]
    extension_counts = Counter(
        Path(row.get("output_path", "")).suffix.lower()
        for row in outputs
    )

    failures = []
    warnings = []

    if not runs:
        failures.append("No figure scripts were run.")

    if not outputs:
        failures.append("No outputs were collected from figure scripts.")

    if not figures:
        failures.append("No figure files were collected from figure scripts.")

    if failed_runs:
        warnings.append(f"{len(failed_runs)} figure scripts failed and need adaptation.")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "manifest": str(args.manifest),
        "script_count": len(runs),
        "failed_script_count": len(failed_runs),
        "collected_output_count": len(outputs),
        "collected_figure_count": len(figures),
        "extension_counts": dict(extension_counts),
        "script_runs": [
            {
                "script": run.get("script", ""),
                "status": run.get("status", ""),
                "returncode": run.get("returncode", ""),
                "changed_file_count": run.get("changed_file_count", ""),
                "collected_output_count": len(run.get("copied_outputs", [])),
                "stdout_log": run.get("stdout_log", ""),
                "stderr_log": run.get("stderr_log", ""),
            }
            for run in runs
        ],
        "warnings": warnings,
        "failures": failures,
        "note": (
            "This validates the Stage 3R.5.2 figure-script execution harness. "
            "It does not yet claim figure equivalence."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    write_markdown(args.output.with_suffix(".md"), report)

    print(json.dumps(
        {
            "status": report["status"],
            "script_count": report["script_count"],
            "failed_script_count": report["failed_script_count"],
            "collected_output_count": report["collected_output_count"],
            "collected_figure_count": report["collected_figure_count"],
            "extension_counts": report["extension_counts"],
            "output": str(args.output),
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"[TRACE] Paper figure validation report written to: {args.output}")
    print(f"[TRACE] Paper figure harness validation status: {report['status']}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()

