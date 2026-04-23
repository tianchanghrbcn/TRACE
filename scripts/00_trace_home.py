#!/usr/bin/env python3
"""TRACE terminal home and reviewer menu."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


COMMANDS = {
    "1": {
        "title": "Run release validation",
        "cmd": ["scripts/98_validate_release_package.py"],
    },
    "2": {
        "title": "Mode A: validate paper replay",
        "cmd": ["scripts/62_validate_mode_a_paper_replay.py"],
    },
    "3": {
        "title": "Mode A: rebuild and validate paper replay",
        "cmd": ["scripts/62_validate_mode_a_paper_replay.py", "--rebuild"],
    },
    "4": {
        "title": "Mode B: run smoke pipeline",
        "cmd": ["scripts/90_run_smoke_from_scratch.py", "--config", "configs/mode_b_smoke.yaml", "--clean"],
    },
    "5": {
        "title": "Mode C: check strict execution proof",
        "cmd": ["scripts/63_validate_stage3_strict.py", "--skip-mode-b-rerun"],
    },
    "6": {
        "title": "Stage 3 strict validation",
        "cmd": ["scripts/63_validate_stage3_strict.py"],
    },
    "7": {
        "title": "Build combined paper-output traceability report",
        "cmd": ["scripts/61_build_paper_output_traceability_report.py"],
    },
    "8": {
        "title": "Check data availability",
        "cmd": ["scripts/45_validate_data_availability.py"],
    },
    "9": {
        "title": "Build local release assets",
        "cmd": ["scripts/44_build_release_assets.py", "--version", "v0.1.2-advisor", "--source-ref", "HEAD"],
    },
    "10": {
        "title": "Show estimated long-run progress",
        "cmd": ["scripts/95_monitor_repro_progress.py"],
    },
}


PLANNED = {}


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        value = proc.stdout.strip()
        return value or default
    except Exception:
        return default


def exists_label(path: str) -> str:
    return "yes" if (ROOT / path).exists() else "no"


def print_home() -> None:
    branch = git_value(["branch", "--show-current"])
    commit = git_value(["rev-parse", "--short", "HEAD"])
    tag = git_value(["describe", "--tags", "--exact-match"], default="no exact tag")

    print(
        r"""
 _______  ____      _      ____  _____
|__   __||  _ \    / \    / ___|| ____|
   | |   | |_) |  / _ \  | |    |  _|
   | |   |  _ <  / ___ \ | |___ | |___
   |_|   |_| \_\/_/   \_\ \____||_____|
"""
    )

    print("Empirical cleaning-clustering artifact")
    print()
    print("You are in the TRACE advisor-review artifact.")
    print()
    print(f"Project root : {ROOT}")
    print(f"Git branch   : {branch}")
    print(f"Git commit   : {commit}")
    print(f"Git tag      : {tag}")
    print(f"Python       : {platform.python_version()} ({platform.system()})")
    print()
    print("Release checks:")
    for label, path in [
        ("README.md", "README.md"),
        ("data policy", "docs/data_policy.md"),
        ("release validation script", "scripts/98_validate_release_package.py"),
        ("Mode A paper replay validation", "scripts/62_validate_mode_a_paper_replay.py"),
        ("Stage 3 strict validation", "scripts/63_validate_stage3_strict.py"),
        ("paper-output traceability docs", "docs/paper_output_traceability.md"),
        ("hardware/runtime docs", "docs/hardware_runtime.md"),
    ]:
        print(f"  {label:<40} {exists_label(path)}")

    print()
    print("Recommended first command:")
    print("  python scripts/98_validate_release_package.py")
    print()
    print("Mode definitions:")
    print("  Mode A: paper table/figure replay and traceability.")
    print("  Mode B: lightweight smoke pipeline from scratch.")
    print("  Mode C: strict cleaning-clustering proof checked from Linux validation evidence.")
    print()
    print("License:")
    print("  TRACE wrapper/orchestration code: MIT License")
    print("  Third-party method implementations: see THIRD_PARTY_NOTICES.md")
    print()
    print("Note:")
    print("  This is a research artifact, not a web application.")
    print("  Use the CLI and generated reports as the reviewer-facing interface.")


def print_menu() -> None:
    print()
    print("TRACE numbered menu")
    print("-------------------")
    print("   0. Show this home page")
    for key in sorted(COMMANDS, key=lambda value: int(value)):
        print(f"  {int(key):2d}. {COMMANDS[key]['title']}")


def run_command(cmd: list[str]) -> int:
    print()
    print("[TRACE] Running:", " ".join(cmd))
    return subprocess.call([sys.executable] + cmd, cwd=ROOT)


def interactive_loop() -> None:
    print_home()

    while True:
        print_menu()
        choice = input("\nSelect an entry, or press Enter to exit: ").strip()

        if choice == "":
            print("[TRACE] Exit.")
            return

        if choice == "0":
            print_home()
            continue

        if choice in PLANNED:
            print(f"[TRACE] {PLANNED[choice]} is planned for later work.")
            continue

        if choice not in COMMANDS:
            print(f"[TRACE] Unknown selection: {choice}")
            continue

        code = run_command(COMMANDS[choice]["cmd"])
        print(f"[TRACE] Command finished with return code {code}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRACE terminal home.")
    parser.add_argument("--interactive", action="store_true", help="Show numbered menu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.interactive:
        interactive_loop()
    else:
        print_home()
        print_menu()


if __name__ == "__main__":
    main()

