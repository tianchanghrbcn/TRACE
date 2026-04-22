
#!/usr/bin/env python3
"""Validate outputs from the Mode A paper table script harness.

This validator checks both:
1. files newly generated or changed by the harness; and
2. paper-exact files available in the final compatibility workspace.

This distinction matters because some legacy scripts consume archived
paper-exact outputs without rewriting them when the files already exist.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".png", ".pdf", ".txt"}


EXPECTED_CHECKS = [
    {
        "id": "average_ranks",
        "description": "Average rank table from 6.1.1 rank analysis.",
        "patterns": ["average_ranks.csv"],
        "required": True,
    },
    {
        "id": "cleaning_method_f1_matrix",
        "description": "Cleaning-method F1 matrix from 6.1.1 rank analysis.",
        "patterns": ["cleaning_method_F1_matrix.csv"],
        "required": True,
    },
    {
        "id": "posthoc_or_cd",
        "description": "Post-hoc Nemenyi output or critical-difference fallback.",
        "patterns": ["posthoc_nemenyi_friedman.csv", "critical_difference.txt"],
        "required": True,
    },
    {
        "id": "spearman_rank_correlation",
        "description": "Spearman rank correlation table.",
        "patterns": ["spearman_rank_correlation.csv"],
        "required": True,
    },
    {
        "id": "corr_cov_summary",
        "description": "Correlation and coverage workbook.",
        "patterns": ["corr_cov_summary.xlsx"],
        "required": True,
    },
    {
        "id": "partial_corr_reg",
        "description": "Partial correlation / regression workbook.",
        "patterns": ["partial_corr_reg.xlsx"],
        "required": True,
    },
    {
        "id": "rank_high_noise_top9",
        "description": "High-noise top-9 rank workbook.",
        "patterns": ["rank_high_noise_top9.xlsx"],
        "required": True,
    },
    {
        "id": "q1_cell_level_counts",
        "description": "Cell-level cleaning quality counts.",
        "patterns": ["q1_cell_level_counts.csv"],
        "required": True,
    },
    {
        "id": "q1_metrics_summary",
        "description": "Cleaning quality metric summary.",
        "patterns": ["q1_metrics_summary.csv"],
        "required": True,
    },
    {
        "id": "table6_process_abs",
        "description": "Absolute process-signal table.",
        "patterns": ["table6_process_abs.csv"],
        "required": True,
    },
    {
        "id": "table6_process_delta",
        "description": "Delta process-signal table.",
        "patterns": ["table6_process_delta.csv"],
        "required": True,
    },
    {
        "id": "table6_process_detail",
        "description": "Detailed process-signal workbook or CSV.",
        "patterns": ["table6_process_detail.xlsx", "table6_process_detail.csv"],
        "required": True,
    },
    {
        "id": "table6_process_signature_summary",
        "description": "Process-signal signature summary.",
        "patterns": ["table6_process_signature_summary.xlsx", "table6_process_signature_summary.csv"],
        "required": True,
    },
    {
        "id": "hyper_shift",
        "description": "Hyperparameter-shift workbooks.",
        "patterns": ["*hyper_shift*"],
        "required": True,
    },
    {
        "id": "anova",
        "description": "ANOVA workbooks.",
        "patterns": ["*anova*"],
        "required": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paper table script harness outputs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("analysis/paper_generated/paper_tables/paper_table_script_run_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper_generated/paper_tables/paper_table_validation_report.json"),
    )
    parser.add_argument(
        "--allow-script-failures",
        action="store_true",
        help="Allow script failures while keeping the validation report informative.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Run manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def match_pattern(pattern: str, text: str) -> bool:
    p = pattern.lower()
    t = text.lower()

    if p.startswith("*") and p.endswith("*"):
        return p.strip("*") in t

    return p in t


def iter_workspace_files(workspace: Path):
    if not workspace.exists():
        return

    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in OUTPUT_EXTENSIONS:
            continue
        yield path


def copy_available_file(path: Path, workspace: Path, output_root: Path) -> str:
    rel = path.relative_to(workspace)
    dst = output_root / "final_available_outputs" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return str(dst)


def collected_output_texts(outputs: list[dict]) -> list[str]:
    out = []
    for row in outputs:
        out.append(
            f"{row.get('relative_path', '')} {row.get('output_path', '')}"
        )
    return out


def check_expected_outputs(
    *,
    checks: list[dict],
    collected_outputs: list[dict],
    workspace: Path,
    output_root: Path,
) -> list[dict]:
    collected_text = collected_output_texts(collected_outputs)
    workspace_files = list(iter_workspace_files(workspace))

    rows = []

    for check in checks:
        generated_matches = []
        available_matches = []
        available_copied = []

        for pattern in check["patterns"]:
            for text in collected_text:
                if match_pattern(pattern, text):
                    generated_matches.append(text)

            for path in workspace_files:
                rel_text = str(path.relative_to(workspace))
                haystack = f"{rel_text} {path.name}"
                if match_pattern(pattern, haystack):
                    available_matches.append(str(path))
                    available_copied.append(copy_available_file(path, workspace, output_root))

        generated_matches = sorted(set(generated_matches))
        available_matches = sorted(set(available_matches))
        available_copied = sorted(set(available_copied))

        rows.append(
            {
                "id": check["id"],
                "description": check["description"],
                "patterns": check["patterns"],
                "required": check["required"],
                "found": bool(generated_matches or available_matches),
                "generated_matches": generated_matches,
                "available_matches": available_matches,
                "copied_available_outputs": available_copied,
            }
        )

    return rows


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Paper Table Output Validation",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Status: {report['status']}",
        f"- Script count: {report['script_count']}",
        f"- Failed script count: {report['failed_script_count']}",
        f"- Collected output count: {report['collected_output_count']}",
        f"- Available output count: {report['available_output_count']}",
        "",
        "## Expected output checks",
        "",
        "| ID | Found | Required | Generated matches | Available matches |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in report["expected_output_checks"]:
        lines.append(
            f"| {row['id']} | {row['found']} | {row['required']} | "
            f"{len(row['generated_matches'])} | {len(row['available_matches'])} |"
        )

    lines += [
        "",
        "## Script runs",
        "",
        "| Script | Status | Changed files |",
        "|---|---|---:|",
    ]

    for row in report["script_runs"]:
        lines.append(
            f"| {row['script']} | {row['status']} | {row['changed_file_count']} |"
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
        lines.append("No failures.")

    lines += [
        "",
        "## Scope note",
        "",
        "This validates Stage 3R.4.1 harness output availability.",
        "It checks that legacy scripts execute successfully and that required paper-table outputs are available after the run.",
        "It does not yet claim byte-identical reproduction of every paper table.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)

    runs = manifest.get("runs", [])
    outputs = manifest.get("collected_outputs", [])
    workspace = Path(manifest.get("workspace", ""))

    output_root = args.output.parent
    output_root.mkdir(parents=True, exist_ok=True)

    expected_checks = check_expected_outputs(
        checks=EXPECTED_CHECKS,
        collected_outputs=outputs,
        workspace=workspace,
        output_root=output_root,
    )

    failures = []

    failed_runs = [run for run in runs if run.get("status") != "PASS"]
    if failed_runs and not args.allow_script_failures:
        failures.append(f"{len(failed_runs)} paper table scripts failed.")

    if not runs:
        failures.append("No paper table scripts were run.")

    if not outputs:
        failures.append("No changed outputs were collected from the paper table scripts.")

    for row in expected_checks:
        if row["required"] and not row["found"]:
            failures.append(
                f"Required output group not found: {row['id']} "
                f"(patterns={row['patterns']})"
            )

    available_output_count = sum(len(row["available_matches"]) for row in expected_checks)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "manifest": str(args.manifest),
        "workspace": str(workspace),
        "script_count": len(runs),
        "failed_script_count": len(failed_runs),
        "collected_output_count": len(outputs),
        "available_output_count": available_output_count,
        "expected_output_checks": expected_checks,
        "script_runs": [
            {
                "script": run.get("script", ""),
                "status": run.get("status", ""),
                "returncode": run.get("returncode", ""),
                "changed_file_count": run.get("changed_file_count", ""),
                "stdout_log": run.get("stdout_log", ""),
                "stderr_log": run.get("stderr_log", ""),
            }
            for run in runs
        ],
        "failures": failures,
        "note": (
            "This validates the Stage 3R.4.1 execution harness output availability. "
            "It does not yet claim byte-identical paper table reproduction."
        ),
    }

    write_json(args.output, report)
    md_path = args.output.with_suffix(".md")
    write_markdown(md_path, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[TRACE] Paper table validation report written to: {args.output}")
    print(f"[TRACE] Paper table validation markdown written to: {md_path}")
    print(f"[TRACE] Paper table harness validation status: {report['status']}")

    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
