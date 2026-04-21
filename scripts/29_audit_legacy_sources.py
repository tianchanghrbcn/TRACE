#!/usr/bin/env python3
"""Audit legacy AutoMLClustering sources before Stage 3 migration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "build",
    "dist",
    "bdist.linux-x86_64",
}

IGNORE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".egg",
    ".so",
    ".dll",
    ".exe",
    ".log",
}

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".sh",
    ".cfg",
    ".ini",
    ".tex",
}

RESULT_EXTENSIONS = {
    ".json",
    ".csv",
    ".xlsx",
    ".xls",
    ".txt",
}

FIGURE_EXTENSIONS = {
    ".png",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".svg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit legacy AutoMLClustering repositories for TRACE Stage 3 migration."
    )
    parser.add_argument(
        "--legacy-roots",
        nargs="+",
        required=True,
        help="Legacy repository roots to scan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/processed"),
        help="Directory for audit outputs.",
    )
    parser.add_argument(
        "--docs-output",
        type=Path,
        default=Path("docs/legacy_migration_plan.md"),
        help="Markdown migration-plan output.",
    )
    return parser.parse_args()


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if path.suffix.lower() in IGNORE_EXTENSIONS:
        return True
    return False


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        yield path


def sha1_prefix(path: Path, limit_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    try:
        with path.open("rb") as f:
            h.update(f.read(limit_bytes))
        return h.hexdigest()[:12]
    except Exception:
        return ""


def classify_file(path: Path, rel: str) -> tuple[str, str]:
    lower = rel.lower()
    suffix = path.suffix.lower()
    name = path.name.lower()

    if "pre_experiment" in lower or "alpha" in name or "weight_scan" in name:
        return "pre_experiment_candidate", "Review and migrate into src/pre_experiment."

    if "visual_demo" in lower or "demo" in name or "fig_1" in name or "fig_2" in name:
        return "visual_demo_candidate", "Review and migrate into src/visual_demo or src/figures."

    if "/utils/" in lower.replace("\\", "/"):
        if name.startswith("fig_") or "flowchart" in name:
            return "figure_candidate", "Migrate plotting logic into src/figures."
        if name.startswith("tab") or "_cal" in name or name in {
            "analyze_cleaning.py",
            "analyze_cluster.py",
            "merge_form.py",
            "summary_patch.py",
        }:
            return "analysis_candidate", "Migrate analysis logic into src/results_processing."
        return "utils_reference", "Review manually; likely analysis or figure support."

    if "/results/" in lower.replace("\\", "/") or "\\results\\" in lower:
        if suffix in RESULT_EXTENSIONS:
            return "result_artifact", "Use as archived Mode A raw result if needed."
        if suffix in FIGURE_EXTENSIONS:
            return "legacy_figure_output", "Do not commit generated figure outputs unless selected."

    if suffix in FIGURE_EXTENSIONS:
        return "legacy_figure_output", "Do not commit generated figure outputs unless selected."

    if suffix == ".py":
        if "pipeline" in lower or "train" in lower:
            return "legacy_pipeline_reference", "Keep only as reference; do not expose as reviewer entry point."
        if "cleaning" in lower or "clustering" in lower:
            return "legacy_method_reference", "Method logic already handled by Stage 2; review only if needed."
        return "python_reference", "Review manually."

    if suffix in {".yml", ".yaml", ".sh", ".cfg"}:
        return "legacy_environment_or_config", "Use only to document old environment assumptions."

    if suffix in {".md", ".txt"}:
        return "documentation_reference", "Use as documentation reference."

    return "other_reference", "No automatic migration."


def audit_root(root: Path) -> list[dict]:
    rows = []

    for path in iter_files(root):
        rel = str(path.relative_to(root))
        category, action = classify_file(path, rel)

        try:
            size_bytes = path.stat().st_size
        except Exception:
            size_bytes = ""

        rows.append(
            {
                "legacy_root": str(root),
                "relative_path": rel,
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": size_bytes,
                "category": category,
                "recommended_action": action,
                "sha1_prefix": sha1_prefix(path),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "legacy_root",
        "relative_path",
        "file_name",
        "extension",
        "size_bytes",
        "category",
        "recommended_action",
        "sha1_prefix",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_plan(path: Path, summary: dict, rows: list[dict]) -> None:
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    def top_rows(category: str, limit: int = 30) -> list[dict]:
        return sorted(
            by_category.get(category, []),
            key=lambda item: (str(item["legacy_root"]), str(item["relative_path"])),
        )[:limit]

    lines = []
    lines.append("# Legacy Migration Plan")
    lines.append("")
    lines.append("This document is generated by `scripts/29_audit_legacy_sources.py`.")
    lines.append("")
    lines.append("Stage 3 migrates useful legacy outputs and analysis code into TRACE without exposing old pipeline entry points.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Generated at UTC: {summary['generated_at_utc']}")
    lines.append(f"- Total files scanned: {summary['total_files']}")
    lines.append("")

    lines.append("### Category counts")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for category, count in summary["category_counts"].items():
        lines.append(f"| {category} | {count} |")
    lines.append("")

    lines.append("## Migration rules")
    lines.append("")
    lines.append("1. Do not copy legacy repositories wholesale into TRACE.")
    lines.append("2. Use `src/results_processing` for metric extraction, table construction, and result normalization.")
    lines.append("3. Use `src/figures` for plotting code only. Generated images belong under `figures/`.")
    lines.append("4. Use `src/pre_experiment` for alpha and weight-selection experiments.")
    lines.append("5. Use `src/visual_demo` for explanatory demo data and demo figures.")
    lines.append("6. Keep old pipeline/train scripts as historical references only; they are not reviewer entry points.")
    lines.append("")

    sections = [
        ("Analysis candidates", "analysis_candidate"),
        ("Figure candidates", "figure_candidate"),
        ("Pre-experiment candidates", "pre_experiment_candidate"),
        ("Visual-demo candidates", "visual_demo_candidate"),
        ("Result artifacts", "result_artifact"),
        ("Legacy pipeline references", "legacy_pipeline_reference"),
    ]

    for title, category in sections:
        rows_for_category = top_rows(category)
        lines.append(f"## {title}")
        lines.append("")
        if not rows_for_category:
            lines.append("No files found.")
            lines.append("")
            continue

        lines.append("| Legacy root | Relative path | Recommended action |")
        lines.append("|---|---|---|")
        for row in rows_for_category:
            lines.append(
                f"| {row['legacy_root']} | {row['relative_path']} | {row['recommended_action']} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    roots = [Path(root).resolve() for root in args.legacy_roots]

    all_rows: list[dict] = []
    missing_roots = []

    for root in roots:
        if not root.exists():
            missing_roots.append(str(root))
            continue
        all_rows.extend(audit_root(root))

    category_counts = Counter(row["category"] for row in all_rows)
    extension_counts = Counter(row["extension"] for row in all_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_roots": [str(root) for root in roots],
        "missing_roots": missing_roots,
        "total_files": len(all_rows),
        "category_counts": dict(sorted(category_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files_csv = args.output_dir / "legacy_audit_files.csv"
    summary_json = args.output_dir / "legacy_audit_summary.json"

    write_csv(files_csv, all_rows)
    write_json(summary_json, summary)
    write_plan(args.docs_output, summary, all_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[TRACE] Legacy audit file list written to: {files_csv}")
    print(f"[TRACE] Legacy audit summary written to: {summary_json}")
    print(f"[TRACE] Legacy migration plan written to: {args.docs_output}")


if __name__ == "__main__":
    main()

