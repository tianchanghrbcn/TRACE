
#!/usr/bin/env python3
"""Classify paper-table equivalence results into reviewer-facing layers."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PAPER_FACING_PATTERNS = [
    "cleaning_method_f1_matrix.csv",
    "spearman_rank_correlation.csv",
    "q1_cell_level_counts.csv",
    "q1_metrics_summary.csv",
    "table6_process_abs.csv",
    "table6_process_delta.csv",
    "table6_process_detail",
    "table6_process_signature_summary",
    "table10_",
    "hyper_shift",
    "table11_",
    "anova",
]

SUPPORTING_PATTERNS = [
    "average_ranks.csv",
    "critical_difference.txt",
    "posthoc_nemenyi_friedman.csv",
    "corr_cov_summary.xlsx",
    "partial_corr_reg.xlsx",
    "rank_high_noise_top9.xlsx",
]

UPSTREAM_PATTERNS = [
    "_cleaning.csv",
    "_cluster.csv",
    "_summary.xlsx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify table equivalence report into layers.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis/paper_generated/paper_tables/table_equivalence_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper_generated/paper_tables/table_equivalence_layered_report.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Equivalence report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def text_of(row: dict) -> str:
    return (
        row.get("generated_path", "") + " " +
        row.get("file_name", "") + " " +
        row.get("best_reference", "")
    ).lower()


def classify(row: dict) -> str:
    text = text_of(row)

    if any(pattern in text for pattern in PAPER_FACING_PATTERNS):
        return "paper_facing"

    if any(pattern in text for pattern in SUPPORTING_PATTERNS):
        return "supporting_analysis"

    if any(pattern in text for pattern in UPSTREAM_PATTERNS):
        return "upstream_intermediate"

    return "unmapped"


def is_hard_status(status: str) -> bool:
    return status == "FAIL"


def is_warning_status(status: str) -> bool:
    return status in {"WARN", "WARN_NO_REFERENCE", "PASS_WITH_WARNINGS"}


def summarize_layer(rows: list[dict]) -> dict:
    counts = Counter(row.get("status", "") for row in rows)
    hard_failures = [row for row in rows if is_hard_status(row.get("status", ""))]
    warnings = [row for row in rows if is_warning_status(row.get("status", ""))]

    return {
        "count": len(rows),
        "status_counts": dict(counts),
        "hard_failure_count": len(hard_failures),
        "warning_count": len(warnings),
        "hard_failures": hard_failures,
        "warnings": warnings,
    }


def layer_status(layer_summaries: dict) -> str:
    paper = layer_summaries.get("paper_facing", {})
    paper_hard = paper.get("hard_failure_count", 0)
    paper_warn = paper.get("warning_count", 0)

    if paper_hard:
        return "FAIL"

    any_support_hard = any(
        summary.get("hard_failure_count", 0)
        for name, summary in layer_summaries.items()
        if name != "paper_facing"
    )

    any_warn = any(summary.get("warning_count", 0) for summary in layer_summaries.values())

    if paper_warn or any_support_hard or any_warn:
        return "PASS_WITH_DIAGNOSTIC_WARNINGS"

    return "PASS"


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Layered Paper Table Equivalence Report",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Layered status: {report['status']}",
        f"- Raw report status: {report['raw_status']}",
        f"- Compared files: {report['compared_file_count']}",
        "",
        "## Layer summary",
        "",
        "| Layer | Count | Status counts | Hard failures | Warnings |",
        "|---|---:|---|---:|---:|",
    ]

    for layer, summary in sorted(report["layers"].items()):
        lines.append(
            f"| {layer} | {summary['count']} | {summary['status_counts']} | "
            f"{summary['hard_failure_count']} | {summary['warning_count']} |"
        )

    lines += [
        "",
        "## Paper-facing hard failures",
        "",
    ]

    paper_failures = report["layers"].get("paper_facing", {}).get("hard_failures", [])
    if paper_failures:
        for row in paper_failures:
            lines.append(f"- {row.get('generated_path')} -> {row.get('best_reference')}")
    else:
        lines.append("No paper-facing hard failures.")

    lines += [
        "",
        "## Diagnostic failures outside paper-facing layer",
        "",
    ]

    found = False
    for layer, summary in sorted(report["layers"].items()):
        if layer == "paper_facing":
            continue
        failures = summary.get("hard_failures", [])
        if not failures:
            continue
        found = True
        lines.append(f"### {layer}")
        for row in failures[:30]:
            lines.append(f"- {row.get('generated_path')} -> {row.get('best_reference')}")
    if not found:
        lines.append("No non-paper-facing hard failures.")

    lines += [
        "",
        "## Scope note",
        "",
        "This report separates reviewer-facing paper-table outputs from upstream intermediate diagnostics.",
        "A PASS_WITH_DIAGNOSTIC_WARNINGS result means paper-facing outputs have no hard mismatches, but supporting or upstream files still need investigation.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw = read_json(args.input)

    layer_rows = defaultdict(list)
    for row in raw.get("comparisons", []):
        layer = classify(row)
        enriched = dict(row)
        enriched["layer"] = layer
        layer_rows[layer].append(enriched)

    layers = {
        layer: summarize_layer(rows)
        for layer, rows in sorted(layer_rows.items())
    }

    status = layer_status(layers)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "raw_report": str(args.input),
        "raw_status": raw.get("status", ""),
        "compared_file_count": raw.get("compared_file_count", 0),
        "raw_status_counts": raw.get("status_counts", {}),
        "layers": layers,
        "interpretation": (
            "paper_facing decides whether the paper-table layer passes. "
            "supporting_analysis and upstream_intermediate are diagnostics for migration cleanup."
        ),
    }

    write_json(args.output, report)
    write_markdown(args.output.with_suffix(".md"), report)

    print(json.dumps(
        {
            "status": report["status"],
            "raw_status": report["raw_status"],
            "raw_status_counts": report["raw_status_counts"],
            "layers": {
                layer: {
                    "count": summary["count"],
                    "status_counts": summary["status_counts"],
                    "hard_failure_count": summary["hard_failure_count"],
                    "warning_count": summary["warning_count"],
                }
                for layer, summary in layers.items()
            },
            "output": str(args.output),
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"[TRACE] Layered equivalence report written to: {args.output}")
    print(f"[TRACE] Layered paper-table equivalence status: {status}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_DIAGNOSTIC_WARNINGS"} else 1)


if __name__ == "__main__":
    main()
