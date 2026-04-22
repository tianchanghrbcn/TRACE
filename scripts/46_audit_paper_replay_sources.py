#!/usr/bin/env python3
"""Audit paper-exact replay sources from legacy repositories and TRACE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCAN_EXTENSIONS = {
    ".xlsx", ".xls", ".csv", ".json", ".png", ".pdf", ".tex", ".py", ".md"
}


DEFAULT_ROOTS = [
    r"E:\algorithm paper\AutoMLClustering",
    r"E:\algorithm paper\AutoMLClustering_full",
    r"E:\TRACE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paper-exact replay sources.")
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/paper_replay_audit"))
    parser.add_argument("--max-files", type=int, default=0, help="0 means no limit.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return ""


def classify(path: Path, root: Path) -> str:
    rel = str(path.relative_to(root)).replace("\\", "/").lower()
    name = path.name.lower()
    suffix = path.suffix.lower()

    if ".git/" in rel or "/.git/" in rel:
        return "skip_git"

    if suffix in {".xlsx", ".xls"}:
        if any(k in name for k in ["summary", "tab", "table", "result", "analysis"]):
            return "paper_summary_xlsx"
        return "xlsx_candidate"

    if suffix == ".json":
        if any(k in name for k in ["cleaned_results", "clustered_results", "analyzed_results", "eigenvectors", "pipeline_run_manifest"]):
            return "raw_pipeline_result_json"
        if "manifest" in name:
            return "manifest_json"
        return "json_candidate"

    if suffix == ".csv":
        if any(k in rel for k in ["analysis", "summary", "result", "processed", "tables"]):
            return "analysis_or_result_csv"
        if any(k in name for k in ["summary", "metrics", "analysis", "score"]):
            return "analysis_or_result_csv"
        return "csv_candidate"

    if suffix in {".png", ".pdf"}:
        if any(k in rel for k in ["task_progress/latex/figures", "task_progress/figures", "figures"]):
            return "paper_figure_output"
        if any(k in name for k in ["fig", "figure", "graph", "heat", "radar", "line", "point"]):
            return "figure_output_candidate"
        return "pdf_or_image_candidate"

    if suffix == ".tex":
        if "latex_paper" in name or "paper" in name:
            return "paper_tex"
        return "tex_candidate"

    if suffix == ".py":
        if any(k in name for k in ["tab.", "tab_", "cal_", "_cal", "summary_patch"]):
            return "paper_table_script"
        if any(k in name for k in ["fig_", "plot", "graph", "flowchart"]):
            return "paper_figure_script"
        if any(k in name for k in ["analyze", "merge_form", "summary"]):
            return "analysis_script"
        return "python_reference"

    if suffix == ".md":
        if "migration" in name or "result" in name or "figure" in name:
            return "documentation_reference"
        return "markdown_reference"

    return "other"


def extract_latex_graphics(path: Path) -> list[str]:
    if path.suffix.lower() != ".tex":
        return []

    text = safe_read_text(path)
    graphics = []

    for pattern in [
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
        r"\\input\{([^}]+)\}",
    ]:
        graphics.extend(re.findall(pattern, text))

    return sorted(set(graphics))


def inspect_python_io(path: Path) -> dict[str, str]:
    if path.suffix.lower() != ".py":
        return {
            "python_inputs": "",
            "python_outputs": "",
        }

    text = safe_read_text(path)

    input_hits = []
    output_hits = []

    for pattern in [
        r"read_csv\([^\n]+",
        r"read_excel\([^\n]+",
        r"json\.load\([^\n]+",
        r"open\([^\n]+",
        r"load_workbook\([^\n]+",
    ]:
        input_hits.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    for pattern in [
        r"to_csv\([^\n]+",
        r"to_excel\([^\n]+",
        r"savefig\([^\n]+",
        r"json\.dump\([^\n]+",
        r"Workbook\(",
    ]:
        output_hits.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    return {
        "python_inputs": " || ".join(sorted(set(input_hits))[:20]),
        "python_outputs": " || ".join(sorted(set(output_hits))[:20]),
    }


def iter_candidate_files(root: Path, max_files: int):
    count = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if ".git" in path.parts:
            continue

        if path.suffix.lower() not in SCAN_EXTENSIONS:
            continue

        count += 1
        if max_files and count > max_files:
            break

        yield path


def main() -> None:
    args = parse_args()
    rows = []

    for root_str in args.roots:
        root = Path(root_str)

        if not root.exists():
            rows.append({
                "root": str(root),
                "relative_path": "",
                "file_name": "",
                "extension": "",
                "category": "missing_root",
                "size_bytes": "",
                "sha256": "",
                "latex_references": "",
                "python_inputs": "",
                "python_outputs": "",
            })
            continue

        for path in iter_candidate_files(root, args.max_files):
            category = classify(path, root)
            if category in {"skip_git", "python_reference", "markdown_reference", "other"}:
                continue

            latex_refs = extract_latex_graphics(path)
            py_io = inspect_python_io(path)

            rows.append({
                "root": str(root),
                "relative_path": str(path.relative_to(root)),
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "category": category,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "latex_references": " || ".join(latex_refs),
                "python_inputs": py_io["python_inputs"],
                "python_outputs": py_io["python_outputs"],
            })

    rows = sorted(rows, key=lambda r: (r["category"], r["root"], r["relative_path"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "paper_replay_source_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        columns = [
            "root",
            "relative_path",
            "file_name",
            "extension",
            "category",
            "size_bytes",
            "sha256",
            "latex_references",
            "python_inputs",
            "python_outputs",
        ]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    by_category = Counter(row["category"] for row in rows)
    by_extension = Counter(row["extension"] for row in rows)

    duplicate_hashes = defaultdict(list)
    for row in rows:
        if row["sha256"]:
            duplicate_hashes[row["sha256"]].append(row["relative_path"])

    duplicates = {
        digest: paths
        for digest, paths in duplicate_hashes.items()
        if len(paths) > 1
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "roots": args.roots,
        "total_candidates": len(rows),
        "category_counts": dict(sorted(by_category.items())),
        "extension_counts": dict(sorted(by_extension.items())),
        "duplicate_sha256_count": len(duplicates),
        "duplicate_groups_sample": dict(list(duplicates.items())[:20]),
    }

    summary_path = args.output_dir / "paper_replay_source_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = args.output_dir / "paper_replay_source_audit.md"
    lines = [
        "# Paper-Exact Replay Source Audit",
        "",
        "This audit identifies source files for Stage 3R paper-exact replay.",
        "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Total candidates: {summary['total_candidates']}",
        "",
        "## Category counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]

    for key, value in summary["category_counts"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Important candidates",
        "",
        "| Category | Root | File | Size bytes |",
        "|---|---|---|---:|",
    ]

    priority = {
        "paper_summary_xlsx",
        "paper_table_script",
        "paper_figure_script",
        "paper_figure_output",
        "paper_tex",
        "raw_pipeline_result_json",
        "analysis_script",
        "analysis_or_result_csv",
    }

    for row in rows:
        if row["category"] not in priority:
            continue
        lines.append(
            f"| {row['category']} | {row['root']} | {row['relative_path']} | {row['size_bytes']} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[TRACE] Wrote: {csv_path}")
    print(f"[TRACE] Wrote: {summary_path}")
    print(f"[TRACE] Wrote: {md_path}")


if __name__ == "__main__":
    main()

