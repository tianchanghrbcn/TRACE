#!/usr/bin/env python3
"""Build Mode A paper-exact archive from selected legacy sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Mode A paper-exact archive.")
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=Path("analysis/paper_replay_audit/paper_exact_source_selection.csv"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("artifacts/paper_exact"),
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("analysis/paper_exact"),
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=Path("figures/paper_exact"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/paper_exact"),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous paper-exact archive outputs before copying.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Selection CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_relative_path(rel: str) -> Path:
    rel_path = Path(rel.replace("\\", "/"))
    parts = [part for part in rel_path.parts if part not in {"..", ".", ""}]
    return Path(*parts)


def replay_destination(row: dict[str, str], args: argparse.Namespace) -> Path | None:
    group = row["selection_group"]
    rel = safe_relative_path(row["relative_path"])

    if group == "raw_results":
        return args.results_root / "raw_results" / rel

    if group in {"analysis_summaries", "paper_tables", "paper_support_workbooks", "summary_workbooks", "analysis_csv"}:
        return args.analysis_root / group / rel

    if group in {"paper_figures_latex", "paper_figures_reference", "paper_figures_word_screenshot"}:
        return args.figures_root / group / rel

    if group in {"paper_tex", "paper_table_scripts", "analysis_scripts", "paper_figure_scripts"}:
        return args.analysis_root / "source_scripts" / group / rel

    return None


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def clean_outputs(args: argparse.Namespace) -> None:
    for path in [args.archive_root, args.analysis_root, args.figures_root, args.results_root]:
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    args = parse_args()

    if args.clean:
        clean_outputs(args)

    rows = read_rows(args.selection_csv)

    copied = []
    missing = []

    for row in rows:
        src = Path(row["root"]) / row["relative_path"]
        if not src.exists():
            missing.append(dict(row))
            continue

        archive_dst = args.archive_root / row["target_subdir"] / safe_relative_path(row["relative_path"])
        replay_dst = replay_destination(row, args)

        copy_file(src, archive_dst)
        if replay_dst is not None:
            copy_file(src, replay_dst)

        copied.append({
            "source": str(src),
            "archive_path": str(archive_dst),
            "replay_path": str(replay_dst) if replay_dst else "",
            "selection_group": row["selection_group"],
            "category": row["category"],
            "size_bytes": archive_dst.stat().st_size,
            "sha256": sha256_file(archive_dst),
        })

    args.analysis_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_csv": str(args.selection_csv),
        "archive_root": str(args.archive_root),
        "analysis_root": str(args.analysis_root),
        "figures_root": str(args.figures_root),
        "results_root": str(args.results_root),
        "copied_count": len(copied),
        "missing_count": len(missing),
        "copied": copied,
        "missing": missing,
    }

    manifest_path = args.analysis_root / "mode_a_paper_exact_archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.analysis_root / "mode_a_paper_exact_archive_manifest.csv"
    columns = [
        "source",
        "archive_path",
        "replay_path",
        "selection_group",
        "category",
        "size_bytes",
        "sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(copied)

    print(json.dumps({
        "copied_count": len(copied),
        "missing_count": len(missing),
        "manifest": str(manifest_path),
    }, indent=2, ensure_ascii=False))
    print("[TRACE] Mode A paper-exact archive was built.")


if __name__ == "__main__":
    main()

