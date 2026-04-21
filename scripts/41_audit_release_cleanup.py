#!/usr/bin/env python3
"""Audit release source cleanup status for TRACE."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")

DEFAULT_ROOTS = [
    "README.md",
    "configs",
    "docs",
    "scripts",
    "src/cleaning",
    "src/clustering",
    "src/pipeline",
    "src/results_processing",
    "src/figures",
    "src/pre_experiment",
    "src/visual_demo",
]

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    "results",
    "figures",
    "data",
    "legacy",
    "build",
    "dist",
}

SCAN_EXTENSIONS = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".txt",
    ".sh",
    ".ps1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TRACE release cleanup status.")
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--output-dir", type=Path, default=Path("results/logs/release_cleanup"))
    return parser.parse_args()


def iter_files(root: Path):
    if not root.exists():
        return

    if root.is_file():
        if root.suffix.lower() in SCAN_EXTENSIONS:
            yield root
        return

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        yield path


def scan_cjk(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []

    for path in paths:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            if CJK_RE.search(line):
                rows.append({
                    "path": str(path),
                    "line": line_no,
                    "text": line.strip(),
                })

    return rows


def git_status_porcelain() -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.stdout.splitlines()
    except Exception:
        return []


def git_tracked(path: str) -> bool:
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    scan_files = []
    for root_str in args.roots:
        scan_files.extend(list(iter_files(Path(root_str))))

    scan_files = sorted(set(scan_files))
    cjk_rows = scan_cjk(scan_files)
    status_lines = git_status_porcelain()

    suspicious_paths = [
        "src/pipeline/train",
        "src/pipeline/test",
        "src/pipeline/__pycache__",
    ]

    suspicious_report = []
    for path in suspicious_paths:
        p = Path(path)
        suspicious_report.append({
            "path": path,
            "exists": p.exists(),
            "tracked": git_tracked(path) if p.exists() else False,
        })

    by_path = Counter(row["path"] for row in cjk_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scanned_files": len(scan_files),
        "cjk_line_count": len(cjk_rows),
        "cjk_file_count": len(by_path),
        "top_cjk_files": dict(by_path.most_common(20)),
        "git_status_short": status_lines,
        "suspicious_paths": suspicious_report,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "cjk_lines.csv",
        cjk_rows,
        ["path", "line", "text"],
    )
    write_json(args.output_dir / "release_cleanup_summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[TRACE] CJK detail report written to: {args.output_dir / 'cjk_lines.csv'}")
    print(f"[TRACE] Cleanup summary written to: {args.output_dir / 'release_cleanup_summary.json'}")


if __name__ == "__main__":
    main()

