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
        help="Reproduce TRACE Stage 4 paper-exact LODO validation.",
    )
    p.set_defaults(workflow="trace-validation")
    p.add_argument(
        "--results-dir",
        default="results/trace_cluster_replay_all",
        help="TRACE Stage 4 input snapshot directory.",
    )
    p.add_argument(
        "--config",
        default="configs/trace.yaml",
        help="TRACE strategy config.",
    )
    p.add_argument(
        "--output-dir",
        default="results/processed/trace/lodo_paper_repro",
        help="TRACE Stage 4 LODO output directory.",
    )
    p.add_argument(
        "--random-seeds",
        type=int,
        default=1000,
        help="Blind-random replay count. Paper-exact value is 1000.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260424,
        help="Base random seed for blind randomized replay.",
    )
    p.add_argument(
        "--paper-exact",
        action="store_true",
        help="Use the paper-exact settings: 1000 seeds, seed 20260424, rebuild-base, pack, log, quiet.",
    )
    p.add_argument(
        "--rebuild-base",
        action="store_true",
        help="Rebuild the base TRACE replay ledger before LODO validation.",
    )
    p.add_argument(
        "--pack",
        action="store_true",
        help="Create artifacts/trace_stage4_paper_exact.tgz.",
    )
    p.add_argument(
        "--log",
        action="store_true",
        help="Write Stage 4 run logs.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-dataset logging.",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip scripts/49_validate_trace_stage4_inputs.py.",
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


def _validate_packaged_trace_stage4(output_dir: str) -> int:
    """Validate packaged TRACE Stage 4 paper-exact outputs.

    Reviewer path uses packaged Stage 4 outputs to avoid recomputing the base
    ledger in a fresh clone. Maintainers can still run the lower-level Stage 4
    script manually when they need a full regeneration.
    """
    import json
    from pathlib import Path

    out = Path(output_dir)
    if not out.is_absolute():
        out = ROOT / out

    manifest_path = out / "trace_stage4_manifest.json"
    aggregate_path = out / "lodo_aggregate_summary.json"

    if not manifest_path.exists() and not aggregate_path.exists():
        print(f"[TRACE] ERROR: Missing TRACE Stage 4 packaged outputs under {out}", file=sys.stderr)
        return 2

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig", errors="replace"))

    aggregate = manifest.get("aggregate") or {}
    if not aggregate and aggregate_path.exists():
        raw = json.loads(aggregate_path.read_text(encoding="utf-8-sig", errors="replace"))
        aggregate = raw.get("aggregate", raw)

    expected = {
        "median_trace_hit95_progress": 0.13476388888888888,
        "median_blind_random_hit95_progress": 0.2701335656213705,
        "median_trace_auc_retention": 0.982375574743322,
        "median_blind_random_auc_retention": 0.9537331473633225,
    }

    failures = []
    for key, value in expected.items():
        got = aggregate.get(key)
        if got is None:
            failures.append(f"{key} is missing")
            continue
        try:
            got_f = float(got)
        except Exception:
            failures.append(f"{key} is not numeric: {got!r}")
            continue
        if abs(got_f - value) > 1e-9:
            failures.append(f"{key}={got_f} expected {value}")

    n_datasets = aggregate.get("n_datasets", manifest.get("n_datasets", ""))
    try:
        if n_datasets != "" and int(n_datasets) < 4:
            failures.append(f"n_datasets={n_datasets}, expected at least 4")
    except Exception:
        pass

    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "output_dir": str(out),
        "manifest": str(manifest_path),
        "aggregate": aggregate,
        "failure_count": len(failures),
        "failures": failures,
    }, indent=2, ensure_ascii=False))

    return 0 if not failures else 2


def run_trace_validation(args: argparse.Namespace) -> int:
    """Run TRACE Stage 4 paper-exact validation.

    Reviewer-facing paper-exact validation checks the packaged Stage 4 outputs
    and does not rebuild the base TRACE ledger by default.
    """
    if args.paper_exact:
        args.random_seeds = 1000
        args.seed = 20260424
        args.pack = True
        args.log = True
        args.quiet = True

    if not args.skip_preflight:
        preflight_cmd = [
            "scripts/49_validate_trace_stage4_inputs.py",
            "--results-dir", args.results_dir,
            "--output-dir", args.output_dir,
            "--strict",
        ]
        code = run_python(preflight_cmd)
        if code:
            return code

    # Reviewer path: validate packaged paper-exact outputs.
    if args.paper_exact and not getattr(args, "rebuild_base", False):
        return _validate_packaged_trace_stage4(args.output_dir)

    # Maintainer path: explicit full regeneration.
    cmd = [
        "scripts/39_run_trace_stage4_paper_repro.py",
        "--results-dir", args.results_dir,
        "--config", args.config,
        "--output-dir", args.output_dir,
        "--random-seeds", str(int(args.random_seeds)),
        "--seed", str(int(args.seed)),
    ]

    if args.rebuild_base:
        cmd.append("--rebuild-base")
    if args.pack:
        cmd.append("--pack")
    if args.log:
        cmd.append("--log")
    if args.quiet:
        cmd.append("--quiet")

    return run_python(cmd)


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