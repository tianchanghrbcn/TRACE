# Reproducibility Modes

TRACE exposes three reviewer-facing reproducibility modes.

## Mode A: result replay

Mode A starts from existing raw or archived result files and rebuilds canonical result tables and figures.

Commands:

    python scripts/30_validate_archived_results.py --results-dir results
    python scripts/30_build_canonical_results.py --results-dir results --output-dir results/processed
    python scripts/31_build_paper_tables.py --processed-dir results/processed --output-dir results/tables
    python scripts/32_build_analysis_tables.py --processed-dir results/processed --output-dir results/tables
    python scripts/33_make_paper_figures.py --tables-dir results/tables --output-root figures

## Mode B: smoke from scratch

Mode B runs a small from-scratch pipeline on one dirty instance and a small method subset.

Commands:

    python scripts/00_setup_check.py --config configs/mode_b_smoke.yaml --strict
    python scripts/90_run_smoke_from_scratch.py --config configs/mode_b_smoke.yaml --clean

## Mode C: full from scratch

Mode C is the full experiment mode. It may require specialized cleaner environments and long wall-clock time.

Command:

    python scripts/00_setup_check.py --config configs/mode_c_full.yaml --check-all-data --strict

Full execution should be planned separately. The quick reviewer path is Mode B plus Mode A replay.

