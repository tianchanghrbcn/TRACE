#!/usr/bin/env python3
"""Unified TRACE command entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    print("[TRACE]", " ".join(cmd))
    return subprocess.call([sys.executable] + cmd, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRACE unified command entry.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("home", help="Show terminal home page.")
    sub.add_parser("progress", help="Show estimated long-run progress.")

    mode_a = sub.add_parser("mode-a", help="Mode A: paper-exact replay from archived results.")
    mode_a.add_argument("--audit", action="store_true", help="Run paper replay source audit first.")
    mode_a.add_argument("--clean", action="store_true", help="Clean previous Mode A paper-exact outputs.")
    mode_a.add_argument("--generated-summaries", action="store_true", help="Also generate and validate summary workbooks.")
    mode_a.add_argument("--paper-tables", action="store_true", help="Also run and validate selected paper table scripts.")
    mode_a.add_argument("--table-equivalence", action="store_true", help="Also validate generated paper-table equivalence against archived references.")
    mode_a.add_argument("--paper-figures", action="store_true", help="Also run selected paper figure scripts and validate captured outputs.")
    mode_a.add_argument("--figure-traceability", action="store_true", help="Also validate paper figure traceability against LaTeX references and archived outputs.")
    mode_a.add_argument("--paper-output-traceability", action="store_true", help="Build a combined paper table/figure traceability report.")

    sub.add_parser("mode-b", help="Mode B: smoke run from scratch.")
    sub.add_parser("mode-c", help="Mode C: strict execution-layer validation.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "home":
        raise SystemExit(run(["scripts/00_trace_home.py"]))

    if args.command == "progress":
        raise SystemExit(run([
            "scripts/95_monitor_repro_progress.py",
            "--log-dir", "results/logs",
            "--reference", "configs/runtime_reference.yaml",
        ]))

    if args.command == "mode-a":
        if args.audit:
            code = run(["scripts/46_audit_paper_replay_sources.py"])
            if code:
                raise SystemExit(code)

        code = run(["scripts/47_select_paper_exact_sources.py"])
        if code:
            raise SystemExit(code)

        build_cmd = ["scripts/48_build_mode_a_paper_exact_archive.py"]
        if args.clean:
            build_cmd.append("--clean")

        code = run(build_cmd)
        if code:
            raise SystemExit(code)

        code = run(["scripts/49_validate_mode_a_paper_exact.py"])
        if code:
            raise SystemExit(code)

        if args.generated_summaries:
            for cmd in [
                ["scripts/50_audit_paper_table_scripts.py"],
                ["scripts/51_build_paper_summary_workbooks.py"],
                ["scripts/52_validate_paper_summary_workbooks.py"],
            ]:
                code = run(cmd)
                if code:
                    raise SystemExit(code)


        if args.paper_tables:
            if not args.generated_summaries:
                print("[TRACE] --paper-tables requires generated summaries; running summary replay first.")
                for cmd in [
                    ["scripts/50_audit_paper_table_scripts.py"],
                    ["scripts/51_build_paper_summary_workbooks.py"],
                    ["scripts/52_validate_paper_summary_workbooks.py"],
                ]:
                    code = run(cmd)
                    if code:
                        raise SystemExit(code)

            for cmd in [
                ["scripts/53_run_paper_table_scripts.py", "--clean", "--timeout", "1200", "--include-analysis-scripts"],
                ["scripts/54_validate_paper_table_outputs.py"],
            ]:
                code = run(cmd)
                if code:
                    raise SystemExit(code)


        if args.table_equivalence:
            raw_code = run(["scripts/55_validate_paper_table_equivalence.py"])
            if raw_code:
                print("[TRACE] Raw table equivalence reported hard mismatches; running layered diagnostics.")
            code = run(["scripts/56_classify_table_equivalence_layers.py"])
            if code:
                raise SystemExit(code)


        if args.paper_figures:
            for cmd in [
                ["scripts/57_select_paper_figure_sources.py"],
                ["scripts/58_run_paper_figure_scripts.py", "--clean", "--timeout", "1200"],
                ["scripts/59_validate_paper_figure_outputs.py"],
            ]:
                code = run(cmd)
                if code:
                    raise SystemExit(code)


        if args.figure_traceability:
            code = run(["scripts/60_validate_paper_figure_traceability.py"])
            if code:
                raise SystemExit(code)


        if args.paper_output_traceability:
            code = run(["scripts/61_build_paper_output_traceability_report.py"])
            if code:
                raise SystemExit(code)

        raise SystemExit(0)

    if args.command == "mode-b":
        raise SystemExit(run([
            "scripts/90_run_smoke_from_scratch.py",
            "--config", "configs/mode_b_smoke.yaml",
            "--clean",
        ]))

    if args.command == "mode-c":
        print("[TRACE] Mode C strict validation is long-running.")
        print("[TRACE] Linux recommended command:")
        print("  bash scripts/97_validate_stage2_strict.sh")
        raise SystemExit(0)


if __name__ == "__main__":
    main()

