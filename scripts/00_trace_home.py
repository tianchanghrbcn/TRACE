#!/usr/bin/env python3
"""TRACE terminal home and reviewer menu."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BANNER = r"""
 _______  ____      _      ____  _____
|__   __||  _ \    / \    / ___|| ____|
   | |   | |_) |  / _ \  | |    |  _|
   | |   |  _ <  / ___ \ | |___ | |___
   |_|   |_| \_\/_/   \_\ \____||_____|

Empirical cleaning-clustering artifact
"""


MENU = [
    ("0", "Show this home page", None),
    ("1", "Run release validation", "python scripts/98_validate_release_package.py"),
    ("2", "Run Mode B smoke pipeline", "python scripts/90_run_smoke_from_scratch.py --config configs/mode_b_smoke.yaml --clean"),
    ("3", "Rebuild canonical result tables", "python scripts/30_build_canonical_results.py --results-dir results --output-dir results/processed"),
    ("4", "Rebuild paper and analysis tables", "python scripts/31_build_paper_tables.py --processed-dir results/processed --output-dir results/tables && python scripts/32_build_analysis_tables.py --processed-dir results/processed --output-dir results/tables"),
    ("5", "Regenerate figures", "python scripts/33_make_paper_figures.py --tables-dir results/tables --output-root figures && python scripts/35_make_layered_figures.py --processed-dir results/processed --tables-dir results/tables --output-root figures && python scripts/36_make_migrated_figure_batch.py --processed-dir results/processed --output-root figures"),
    ("6", "Regenerate pre-experiment outputs", "python scripts/38_build_pre_experiment_outputs.py --source-csv data/pre_experiment/alpha_metrics.csv --output-dir results/pre_experiment --figure-dir figures/pre_experiment"),
    ("7", "Regenerate visual demo", "python scripts/40_make_visual_demo.py --output-data-dir results/visual_demo --output-figure-dir figures/visual_demo"),
    ("8", "Show estimated long-run progress", "python scripts/95_monitor_repro_progress.py --log-dir results/logs --reference configs/runtime_reference.yaml"),
    ("9", "Build local release assets", "python scripts/44_build_release_assets.py --version v0.1.1-advisor --source-ref v0.1.1-advisor"),
    ("10", "Stage 4 TRACE validation placeholder", None),
    ("11", "Stage 4 new algorithm extension placeholder", None),
    ("12", "Stage 4 new dataset onboarding placeholder", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRACE terminal home page.")
    parser.add_argument("--interactive", action="store_true", help="Open numbered menu.")
    return parser.parse_args()


def run_git(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def exists(path: str) -> str:
    return "yes" if (ROOT / path).exists() else "no"


def print_home() -> None:
    branch = run_git(["branch", "--show-current"]) or "unknown"
    tags = run_git(["tag", "--points-at", "HEAD"]) or "none"
    short = run_git(["rev-parse", "--short", "HEAD"]) or "unknown"

    print(BANNER)
    print("You are in the TRACE advisor-review artifact.")
    print("")
    print(f"Project root : {ROOT}")
    print(f"Git branch   : {branch}")
    print(f"Git commit   : {short}")
    print(f"Git tag      : {tags}")
    print(f"Python       : {sys.version.split()[0]} ({platform.system()})")
    print("")
    print("Release checks:")
    print(f"  README.md                                  {exists('README.md')}")
    print(f"  data policy                                {exists('docs/data_policy.md')}")
    print(f"  release validation script                  {exists('scripts/98_validate_release_package.py')}")
    print(f"  runtime progress monitor                   {exists('scripts/95_monitor_repro_progress.py')}")
    print(f"  Stage 2 validation docs                    {exists('docs/stage2_validation.md')}")
    print(f"  Stage 3 replay docs                        {exists('docs/results_replay.md')}")
    print("")
    print("Recommended first command:")
    print("  python scripts/98_validate_release_package.py")
    print("")
    print("License:")
    print("  TRACE wrapper/orchestration code: MIT License")
    print("  Third-party method implementations: see THIRD_PARTY_NOTICES.md")
    print("")
    print("Note:")
    print("  This is a research artifact, not a web application.")
    print("  Use the CLI and generated reports as the reviewer-facing interface.")


def print_menu() -> None:
    print("")
    print("TRACE numbered menu")
    print("-------------------")
    for key, label, command in MENU:
        suffix = "" if command else " [planned/info]"
        print(f"  {key:>2}. {label}{suffix}")


def interactive_menu() -> None:
    print_home()
    print_menu()

    choice = input("\nSelect an option number: ").strip()
    selected = next((item for item in MENU if item[0] == choice), None)

    if selected is None:
        print(f"[TRACE] Unknown option: {choice}")
        return

    key, label, command = selected
    print(f"[TRACE] Selected: {key}. {label}")

    if command is None:
        print("[TRACE] This entry is informational or planned for Stage 4.")
        return

    print("[TRACE] Command:")
    print(f"  {command}")
    answer = input("Run this command now? [y/N]: ").strip().lower()
    if answer != "y":
        print("[TRACE] Command not executed.")
        return

    subprocess.run(command, cwd=ROOT, shell=True, check=False)


def main() -> None:
    args = parse_args()
    if args.interactive:
        interactive_menu()
    else:
        print_home()
        print_menu()


if __name__ == "__main__":
    main()

