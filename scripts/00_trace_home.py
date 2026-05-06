#!/usr/bin/env python3
"""TRACE terminal home and reviewer menu.

This menu uses reviewer-facing workflow names instead of Mode A/B/C.
The old mode-a/mode-b/mode-c names are compatibility aliases only.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


COMMANDS = {
    "1": {
        "title": "Release check: validate artifact package",
        "cmd": ["scripts/trace.py", "release-check"],
    },
    "2": {
        "title": "Paper replay: reproduce paper tables/figures/traceability",
        "cmd": ["scripts/trace.py", "paper-replay"],
    },
    "3": {
        "title": "Pre-experiment and validity sensitivity",
        "cmd": ["scripts/trace.py", "preexp-validity"],
    },
    "4": {
        "title": "TRACE validation: paper-exact 1000 blind-random replay",
        "cmd": ["scripts/trace.py", "trace-validation", "--paper-exact"],
    },
    "5": {
        "title": "Benchmark smoke: quick from-scratch run",
        "cmd": ["scripts/trace.py", "benchmark-smoke", "--clean"],
    },
    "6": {
        "title": "Benchmark full audit: validate strict proof/logs",
        "cmd": ["scripts/trace.py", "benchmark-full-audit"],
    },
    "7": {
        "title": "Benchmark full audit: Linux full from-scratch rerun",
        "cmd": ["scripts/trace.py", "benchmark-full-audit", "--from-scratch"],
    },
    "8": {
        "title": "Build combined paper-output traceability report",
        "cmd": ["scripts/61_build_paper_output_traceability_report.py"],
    },
    "9": {
        "title": "Check data availability",
        "cmd": ["scripts/45_validate_data_availability.py"],
    },
    "10": {
        "title": "Show estimated long-run progress",
        "cmd": ["scripts/trace.py", "progress"],
    },
}


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
    print("Reviewer-facing workflow names:")
    print("  paper-replay           Reproduce paper tables, figures, and traceability.")
    print("  benchmark-smoke        Quick from-scratch benchmark smoke run.")
    print("  benchmark-full-audit   Long-running/full strict benchmark validation.")
    print("  trace-validation       Reproduce TRACE budget-guidance validation.")
    print("  preexp-validity        Replay alpha calibration and validity sensitivity.")
    print("  release-check          Validate final artifact package.")
    print()
    print("Deprecated compatibility aliases:")
    print("  mode-a -> paper-replay")
    print("  mode-b -> benchmark-smoke")
    print("  mode-c -> benchmark-full-audit")
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
        ("paper replay validation", "scripts/62_validate_mode_a_paper_replay.py"),
        ("strict benchmark validation", "scripts/63_validate_stage3_strict.py"),
        ("preexp/validity validation", "scripts/81_replay_pre_experiment_validity.py"),
        ("paper-output traceability docs", "docs/paper_output_traceability.md"),
        ("hardware/runtime docs", "docs/hardware_runtime.md"),
        ("paper replay config", "configs/paper_replay.yaml"),
        ("benchmark smoke config", "configs/benchmark_smoke.yaml"),
        ("benchmark full audit config", "configs/benchmark_full_audit.yaml"),
    ]:
        print(f"  {label:<40} {exists_label(path)}")

    print()
    print("Recommended first command:")
    print("  python scripts/trace.py release-check")
    print()
    print("Useful reviewer commands:")
    print("  python scripts/trace.py paper-replay")
    print("  python scripts/trace.py preexp-validity")
    print("  python scripts/trace.py trace-validation --paper-exact")
    print("  python scripts/trace.py benchmark-smoke --clean")
    print()
    print("Note:")
    print("  This is a research artifact, not a web application.")
    print("  Use the CLI and generated reports as the reviewer-facing interface.")


def print_menu() -> None:
    print()
    print("TRACE reviewer menu")
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
