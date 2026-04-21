#!/usr/bin/env python3
"""Remove local generated artifacts that are reproducible from scripts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_TARGETS = [
    "results/processed",
    "results/tables",
    "results/pre_experiment",
    "results/visual_demo",
    "figures/png",
    "figures/pdf",
    "figures/pre_experiment",
    "figures/visual_demo",
    "figures/figure_manifest.json",
    "figures/layered_figure_manifest.json",
    "figures/migrated_figure_batch1_manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean reproducible generated artifacts.")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def keep_gitkeep(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".gitkeep").write_text("", encoding="utf-8")


def remove_target(path: Path, apply: bool) -> None:
    if not path.exists():
        return

    print(f"[TRACE] Remove generated artifact: {path}")

    if not apply:
        return

    if path.is_dir():
        shutil.rmtree(path)
        keep_gitkeep(path)
    else:
        path.unlink()


def main() -> None:
    args = parse_args()

    for target in args.targets:
        remove_target(Path(target), args.apply)

    if not args.apply:
        print("[TRACE] Dry run only. Re-run with --apply to remove artifacts.")


if __name__ == "__main__":
    main()

