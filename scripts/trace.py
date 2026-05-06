#!/usr/bin/env python3
"""Unified TRACE reviewer-facing command entry point.

Reviewer-facing workflow names:
  paper-replay
  benchmark-smoke
  benchmark-full-audit
  trace-validation
  preexp-validity
  release-check

Deprecated compatibility aliases:
  mode-a -> paper-replay
  mode-b -> benchmark-smoke
  mode-c -> benchmark-full-audit

The alias names are retained only for compatibility. Reviewer-facing
documentation should use the new workflow names.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEPRECATED_COMMAND_ALIASES = {
    "mode-a": "paper-replay",
    "mode-b": "benchmark-smoke",
    "mode-c": "benchmark-full-audit",
}


def rewrite_deprecated_command_alias(argv: list[str]) -> tuple[list[str], str | None, str | None]:
    """Rewrite old workflow aliases before argparse sees them.

    This keeps old commands runnable while preventing them from appearing in
    the reviewer-facing top-level help.
    """
    if argv and argv[0] in DEPRECATED_COMMAND_ALIASES:
        old = argv[0]
        new = DEPRECATED_COMMAND_ALIASES[old]
        return [new] + argv[1:], old, new
    return argv, None, None


def run_python(cmd: list[str]) -> int:
    """Run a Python script from the project root."""
    full = [sys.executable] + cmd
    print()
    print("[TRACE] >>>", " ".join(full))
    return subprocess.call(full, cwd=ROOT)


def run_external(cmd: list[str]) -> int:
    """Run an external command from the project root."""
    print()
    print("[TRACE] >>>", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def run_many(commands: list[list[str]]) -> int:
    for cmd in commands:
        code = run_python(cmd)
        if code:
            return code
    return 0


def add_paper_replay_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "paper-replay",
        help="Reproduce paper-exact tables, figures, traceability, and validation.",
    )
    p.set_defaults(workflow="paper-replay")

    p.add_argument("--audit", action="store_true", help="Run paper replay source audit first.")
    p.add_argument("--clean", action="store_true", help="Clean previous paper-exact outputs.")
    p.add_argument(
        "--generated-summaries",
        action="store_true",
        help="Generate and validate paper summary workbooks.",
    )
    p.add_argument(
        "--paper-tables",
        action="store_true",
        help="Run and validate selected paper table scripts.",
    )
    p.add_argument(
        "--table-equivalence",
        action="store_true",
        help="Validate paper table equivalence against archived references.",
    )
    p.add_argument(
        "--paper-figures",
        action="store_true",
        help="Run selected paper figure scripts and validate outputs.",
    )
    p.add_argument(
        "--figure-traceability",
        action="store_true",
        help="Validate paper figure traceability.",
    )
    p.add_argument(
        "--paper-output-traceability",
        action="store_true",
        help="Build combined paper table/figure traceability report.",
    )
    p.add_argument(
        "--validate-paper-replay",
        action="store_true",
        help="Validate paper-output replay reports.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run the full reviewer-facing paper replay path.",
    )
    p.add_argument(
        "--minimal",
        action="store_true",
        help="Run only archive selection/build/validation. Useful for quick compatibility checks.",
    )


def add_benchmark_smoke_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "benchmark-smoke",
        help="Run a quick from-scratch cleaning-clustering smoke test.",
    )
    p.set_defaults(workflow="benchmark-smoke", clean=True)
    p.add_argument("--config", default="configs/benchmark_smoke.yaml", help="Smoke config path.")
    p.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        help="Clean previous smoke outputs before running.",
    )
    p.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Do not clean previous smoke outputs.",
    )


def add_full_audit_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "benchmark-full-audit",
        help="Validate strict/full benchmark execution proof, or run the long audit.",
    )
    p.set_defaults(workflow="benchmark-full-audit", skip_smoke_rerun=True)
    p.add_argument(
        "--proof-only",
        action="store_true",
        default=True,
        help="Validate existing strict proof/logs. Default and Windows-friendly.",
    )
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="Run the long full-from-scratch audit shell script. Linux/bash recommended.",
    )
    p.add_argument(
        "--skip-smoke-rerun",
        dest="skip_smoke_rerun",
        action="store_true",
        help="Skip rerunning benchmark-smoke during strict proof validation.",
    )
    p.add_argument(
        "--rerun-smoke",
        dest="skip_smoke_rerun",
        action="store_false",
        help="Rerun benchmark-smoke during strict proof validation.",
    )

    # Hidden legacy option. It still works for old commands/scripts but is not
    # shown in reviewer-facing help.
    p.add_argument(
        "--skip-mode-b-rerun",
        dest="skip_smoke_rerun",
        action="store_true",
        help=argparse.SUPPRESS,
    )


def add_preexp_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "preexp-validity",
        help="Replay pre-experiment calibration and validate alpha/seed sensitivity.",
    )
    p.set_defaults(workflow="preexp-validity")
    p.add_argument(
        "--generated-data-policy",
        choices=["warn", "fail", "ignore"],
        default="warn",
        help="How to handle local generated_seed_data/generated_error_model_data directories.",
    )
    p.add_argument(
        "--error-model-policy",
        choices=["warn", "fail", "ignore"],
        default="warn",
        help="How to handle error_model_sensitivity_* files.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Equivalent to --generated-data-policy fail --error-model-policy fail.",
    )


def add_trace_validation_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "trace-validation",
        help="Reproduce TRACE budget-guidance validation and Figure 7 inputs.",
    )
    p.set_defaults(workflow="trace-validation")
    p.add_argument(
        "--processed-dir",
        default="results/processed",
        help="Canonical processed results directory.",
    )
    p.add_argument(
        "--trace-output-dir",
        default="results/processed/trace",
        help="Directory containing TRACE replay CSVs.",
    )
    p.add_argument(
        "--static-output-dir",
        default="results/processed/trace_static",
        help="Output dir for static TRACE screening tables.",
    )
    p.add_argument(
        "--blind-output-dir",
        default=None,
        help="Output dir for blind-random validation. Default depends on random seeds.",
    )
    p.add_argument(
        "--figure-dir",
        default="figures/trace_validation",
        help="Output dir for TRACE validation figures.",
    )
    p.add_argument(
        "--random-seeds",
        type=int,
        default=1000,
        help="Blind-random replay count. Paper-exact value is 1000.",
    )
    p.add_argument("--seed", type=int, default=20260424, help="Base random seed.")
    p.add_argument("--path-granularity", choices=["path", "trial"], default="path")
    p.add_argument("--skip-static", action="store_true", help="Skip static TRACE screening build.")
    p.add_argument("--skip-blind", action="store_true", help="Skip blind-random validation.")
    p.add_argument("--skip-figures", action="store_true", help="Skip TRACE validation figures.")
    p.add_argument(
        "--paper-exact",
        action="store_true",
        help="Force the paper-exact 1000 blind-random replay.",
    )


def add_release_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("release-check", help="Run release package validation.")
    p.set_defaults(workflow="release-check")
    p.add_argument("--skip-stage3-strict", action="store_true", help="Skip strict workflow validation.")
    p.add_argument("--skip-strict-benchmark-proof", action="store_true", help="Skip strict workflow validation.")
    p.add_argument("--skip-benchmark-smoke", action="store_true", help="Skip benchmark-smoke rerun.")
    p.add_argument("--skip-preexp-validity", action="store_true", help="Skip pre-experiment/validity replay.")
    p.add_argument("--run-trace-validation", action="store_true", help="Also run TRACE paper-exact validation.")
    p.add_argument(
        "--rebuild-paper-replay",
        dest="rebuild_paper_replay",
        action="store_true",
        help="Ask release validation to rebuild paper replay.",
    )
    p.add_argument(
        "--rebuild-mode-a",
        dest="rebuild_paper_replay",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--allow-missing-full-audit-proof",
        action="store_true",
        help="Allow missing benchmark-full-audit proof as warning.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRACE unified reviewer command entry.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("home", help="Show terminal home page.")
    sub.add_parser("progress", help="Show estimated long-run progress.")

    add_paper_replay_parser(sub)
    add_benchmark_smoke_parser(sub)
    add_full_audit_parser(sub)
    add_trace_validation_parser(sub)
    add_preexp_parser(sub)
    add_release_parser(sub)

    return parser.parse_args(argv)


def print_deprecated_alias_notice(args: argparse.Namespace) -> None:
    old = getattr(args, "deprecated_alias", None)
    new = getattr(args, "canonical_command", None)
    if old and new:
        print()
        print(f"[TRACE] NOTE: '{old}' is a deprecated compatibility alias.")
        print(f"[TRACE] Please use '{new}' in reviewer-facing documentation.")


def paper_flags_enabled(args: argparse.Namespace) -> bool:
    names = [
        "audit",
        "generated_summaries",
        "paper_tables",
        "table_equivalence",
        "paper_figures",
        "figure_traceability",
        "paper_output_traceability",
        "validate_paper_replay",
    ]
    return any(bool(getattr(args, name, False)) for name in names)


def normalize_paper_replay_defaults(args: argparse.Namespace) -> None:
    """For paper-replay, no flags means full reviewer replay.

    For the deprecated mode-a alias, preserve the old minimal behavior unless
    the caller explicitly passes --all.
    """
    if getattr(args, "minimal", False):
        return

    original = getattr(args, "original_command", args.command)

    should_enable_all = bool(getattr(args, "all", False))
    if original == "paper-replay" and not paper_flags_enabled(args):
        should_enable_all = True

    if should_enable_all:
        args.audit = True
        args.generated_summaries = True
        args.paper_tables = True
        args.table_equivalence = True
        args.paper_figures = True
        args.figure_traceability = True
        args.paper_output_traceability = True
        args.validate_paper_replay = True


def run_paper_replay(args: argparse.Namespace) -> int:
    """Run reviewer-facing paper-level evidence replay.

    This intentionally delegates to scripts/62_validate_mode_a_paper_replay.py,
    whose semantics have been changed from old "Mode A archive validation" to
    paper-level evidence replay: tables, figures, traceability, and
    pre-experiment/validity checks.
    """
    cmd = ["scripts/62_validate_mode_a_paper_replay.py"]

    # Reviewer-facing default: rebuild paper-level evidence.
    # Validation-only behavior is available through --validate-paper-replay or --minimal.
    rebuild = True
    if getattr(args, "validate_paper_replay", False) or getattr(args, "minimal", False):
        rebuild = False
    if getattr(args, "all", False):
        rebuild = True

    if rebuild:
        cmd.append("--rebuild")

    return run_python(cmd)


def run_benchmark_smoke(args: argparse.Namespace) -> int:
    cmd = ["scripts/90_run_smoke_from_scratch.py", "--config", args.config]
    if args.clean:
        cmd.append("--clean")
    return run_python(cmd)


def run_benchmark_full_audit(args: argparse.Namespace) -> int:
    if args.from_scratch:
        print("[TRACE] Full from-scratch audit is long-running. Linux/bash is recommended.")
        return run_external(["bash", "scripts/97_validate_stage2_strict.sh"])

    cmd = ["scripts/63_validate_stage3_strict.py"]
    if args.skip_smoke_rerun:
        cmd.append("--skip-mode-b-rerun")
    return run_python(cmd)


def run_preexp_validity(args: argparse.Namespace) -> int:
    cmd = ["scripts/81_replay_pre_experiment_validity.py"]

    if args.strict:
        cmd += ["--generated-data-policy", "fail", "--error-model-policy", "fail"]
    else:
        cmd += [
            "--generated-data-policy",
            args.generated_data_policy,
            "--error-model-policy",
            args.error_model_policy,
        ]

    return run_python(cmd)


def assert_trace_replay_inputs(trace_dir: Path) -> None:
    required = [
        trace_dir / "trace_dataset_summary.csv",
        trace_dir / "trace_baseline_sequence.csv",
        trace_dir / "trace_replay_trials.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing TRACE replay inputs:\n  "
            + "\n  ".join(missing)
            + "\nRun or unpack the TRACE replay outputs before trace-validation."
        )


def run_trace_validation(args: argparse.Namespace) -> int:
    if args.paper_exact:
        args.random_seeds = 1000

    processed_dir = Path(args.processed_dir)
    trace_dir = Path(args.trace_output_dir)
    static_dir = Path(args.static_output_dir)
    blind_dir = Path(args.blind_output_dir) if args.blind_output_dir else trace_dir / f"blind_random_{args.random_seeds}"

    if not args.skip_static:
        code = run_python(
            [
                "scripts/34_build_trace_static_eval.py",
                "--processed-dir",
                str(processed_dir),
                "--output-dir",
                str(static_dir),
            ]
        )
        if code:
            return code

    if not args.skip_blind:
        trace_input_dir = trace_dir if trace_dir.is_absolute() else ROOT / trace_dir
        try:
            assert_trace_replay_inputs(trace_input_dir)
        except FileNotFoundError as exc:
            print(f"[TRACE] ERROR: {exc}", file=sys.stderr)
            return 2

        code = run_python(
            [
                "scripts/36_eval_trace_blind_random.py",
                "--trace-output-dir",
                str(trace_dir),
                "--output-dir",
                str(blind_dir),
                "--random-seeds",
                str(int(args.random_seeds)),
                "--seed",
                str(int(args.seed)),
                "--path-granularity",
                args.path_granularity,
                "--flush",
            ]
        )
        if code:
            return code

    if not args.skip_figures:
        code = run_python(
            [
                "scripts/37_plot_trace_validation.py",
                "--blind-dir",
                str(blind_dir),
                "--out-dir",
                args.figure_dir,
            ]
        )
        if code:
            return code

    return 0


def run_release_check(args: argparse.Namespace) -> int:
    cmd = ["scripts/98_validate_release_package.py"]

    if args.skip_stage3_strict or args.skip_strict_benchmark_proof:
        cmd.append("--skip-strict-benchmark-proof")

    if args.skip_benchmark_smoke:
        cmd.append("--skip-benchmark-smoke")

    if args.skip_preexp_validity:
        cmd.append("--skip-preexp-validity")

    if args.run_trace_validation:
        cmd.append("--run-trace-validation")

    if args.rebuild_paper_replay:
        cmd.append("--rebuild-paper-replay")

    if args.allow_missing_full_audit_proof:
        cmd.append("--allow-missing-full-audit-proof")

    return run_python(cmd)


def main() -> None:
    rewritten_argv, deprecated_alias, canonical_command = rewrite_deprecated_command_alias(sys.argv[1:])
    args = parse_args(rewritten_argv)

    args.deprecated_alias = deprecated_alias
    args.canonical_command = canonical_command
    args.original_command = deprecated_alias or args.command

    print_deprecated_alias_notice(args)

    if args.command == "home":
        raise SystemExit(run_python(["scripts/00_trace_home.py"]))

    if args.command == "progress":
        raise SystemExit(
            run_python(
                [
                    "scripts/95_monitor_repro_progress.py",
                    "--log-dir",
                    "results/logs",
                    "--reference",
                    "configs/runtime_reference.yaml",
                ]
            )
        )

    if args.workflow == "paper-replay":
        raise SystemExit(run_paper_replay(args))

    if args.workflow == "benchmark-smoke":
        raise SystemExit(run_benchmark_smoke(args))

    if args.workflow == "benchmark-full-audit":
        raise SystemExit(run_benchmark_full_audit(args))

    if args.workflow == "preexp-validity":
        raise SystemExit(run_preexp_validity(args))

    if args.workflow == "trace-validation":
        raise SystemExit(run_trace_validation(args))

    if args.workflow == "release-check":
        raise SystemExit(run_release_check(args))

    raise SystemExit(f"Unknown workflow: {args.workflow}")


if __name__ == "__main__":
    main()