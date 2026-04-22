#!/usr/bin/env python3
"""Select curated paper-exact replay sources from the Stage 3R audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PREFERRED_LEGACY_MARKER = "AutoMLClustering_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select paper-exact replay sources.")
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("analysis/paper_replay_audit/paper_replay_source_candidates.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/paper_replay_audit"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalized_rel(row: dict[str, str]) -> str:
    return row.get("relative_path", "").replace("\\", "/")


def is_noise(row: dict[str, str]) -> bool:
    root = row.get("root", "")
    rel = normalized_rel(row).lower()

    noise_tokens = [
        ".venv/",
        "/site-packages/",
        "\\site-packages\\",
        "__pycache__",
        ".git/",
        "release/",
        "results/logs/",
        "analysis/paper_replay_audit/",
    ]

    if any(token in rel for token in noise_tokens):
        return True

    # Avoid selecting generated TRACE scaffold outputs as paper-exact sources.
    if root.lower().endswith("trace") and (
        rel.startswith("figures/")
        or rel.startswith("results/processed/")
        or rel.startswith("results/tables/")
        or rel.startswith("analysis/")
    ):
        return True

    return False


def preferred_legacy(row: dict[str, str]) -> bool:
    return PREFERRED_LEGACY_MARKER.lower() in row.get("root", "").lower()


def selection_group(row: dict[str, str]) -> str | None:
    if is_noise(row):
        return None

    category = row.get("category", "")
    rel = normalized_rel(row).lower()

    # Prefer AutoMLClustering_full as paper source.
    if category in {
        "paper_summary_xlsx",
        "paper_figure_output",
        "paper_figure_script",
        "paper_table_script",
        "paper_tex",
        "raw_pipeline_result_json",
        "analysis_script",
        "analysis_or_result_csv",
    } and not preferred_legacy(row):
        return None

    if category == "raw_pipeline_result_json":
        if rel.startswith("results/"):
            return "raw_results"
        return None

    if category == "paper_summary_xlsx":
        if rel.startswith("results/analysis_results/"):
            return "analysis_summaries"
        if "task_progress/tables/" in rel:
            return "paper_tables"
        if "task_progress/figures/" in rel:
            return "paper_support_workbooks"
        return "summary_workbooks"

    if category == "analysis_or_result_csv":
        if rel.startswith("results/analysis_results/") or "task_progress/tables/" in rel:
            return "analysis_csv"
        return None

    if category == "paper_figure_output":
        if "task_progress/latex/figures/" in rel:
            return "paper_figures_latex"
        if "task_progress/figures/" in rel:
            return "paper_figures_reference"
        if "task_progress/word_screenshot/figures/" in rel:
            return "paper_figures_word_screenshot"
        return None

    if category == "paper_tex":
        if "task_progress/latex/" in rel:
            return "paper_tex"
        return None

    if category == "paper_table_script":
        if "src/pipeline/utils/" in rel and (
            "tab." in rel
            or "6.1." in rel
            or "summary_patch" in rel
        ):
            return "paper_table_scripts"
        return None

    if category == "analysis_script":
        if "src/pipeline/utils/" in rel and any(
            token in rel for token in ["analyze_cleaning", "analyze_cluster", "merge_form", "summary"]
        ):
            return "analysis_scripts"
        return None

    if category == "paper_figure_script":
        if "src/pipeline/utils/" in rel and (
            "fig_" in rel
            or "flowchart" in rel
            or "graph" in rel
        ):
            return "paper_figure_scripts"
        if "src/graph/" in rel:
            return "paper_figure_scripts"
        return None

    return None


def target_subdir(group: str) -> str:
    mapping = {
        "raw_results": "raw_results",
        "analysis_summaries": "analysis_summaries",
        "paper_tables": "paper_tables",
        "paper_support_workbooks": "paper_support_workbooks",
        "summary_workbooks": "summary_workbooks",
        "analysis_csv": "analysis_csv",
        "paper_figures_latex": "paper_figures/latex",
        "paper_figures_reference": "paper_figures/reference",
        "paper_figures_word_screenshot": "paper_figures/word_screenshot",
        "paper_tex": "paper_tex",
        "paper_table_scripts": "scripts/table",
        "analysis_scripts": "scripts/analysis",
        "paper_figure_scripts": "scripts/figure",
    }
    return mapping[group]


def main() -> None:
    args = parse_args()
    rows = read_rows(args.audit_csv)

    selected = []
    for row in rows:
        group = selection_group(row)
        if group is None:
            continue

        out = dict(row)
        out["selection_group"] = group
        out["target_subdir"] = target_subdir(group)
        selected.append(out)

    selected = sorted(
        selected,
        key=lambda r: (
            r["selection_group"],
            r.get("relative_path", ""),
            r.get("file_name", ""),
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = args.output_dir / "paper_exact_source_selection.csv"
    columns = [
        "root",
        "relative_path",
        "file_name",
        "extension",
        "category",
        "selection_group",
        "target_subdir",
        "size_bytes",
        "sha256",
        "latex_references",
        "python_inputs",
        "python_outputs",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.audit_csv),
        "selected_count": len(selected),
        "selection_group_counts": dict(Counter(row["selection_group"] for row in selected)),
        "category_counts": dict(Counter(row["category"] for row in selected)),
    }

    out_json = args.output_dir / "paper_exact_source_selection_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_md = args.output_dir / "paper_exact_source_selection.md"
    lines = [
        "# Paper-Exact Source Selection",
        "",
        "This file is generated by `scripts/47_select_paper_exact_sources.py`.",
        "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Selected files: {summary['selected_count']}",
        "",
        "## Selection group counts",
        "",
        "| Group | Count |",
        "|---|---:|",
    ]

    for key, value in sorted(summary["selection_group_counts"].items()):
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Selected files",
        "",
        "| Group | Category | File |",
        "|---|---|---|",
    ]

    for row in selected:
        lines.append(
            f"| {row['selection_group']} | {row['category']} | {row['relative_path']} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[TRACE] Wrote: {out_csv}")
    print(f"[TRACE] Wrote: {out_json}")
    print(f"[TRACE] Wrote: {out_md}")


if __name__ == "__main__":
    main()

