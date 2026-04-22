# TRACE

TRACE is an empirical artifact for studying how data cleaning changes unsupervised clustering.

The package contains:

- runnable cleaning-clustering pipeline entries,
- method registry for cleaners and clusterers,
- smoke and coverage scripts,
- canonical result processing,
- paper-table and figure replay scaffolds,
- pre-experiment replay,
- reviewer-facing visual demo,
- release validation utilities.

## Recommended reviewer path

Run the quick validation first:

    python scripts/98_validate_release_package.py

This command runs the smoke pipeline, rebuilds canonical result tables, regenerates table summaries, figures, pre-experiment outputs, and the visual demo.

## Reproducibility modes

Mode A: replay tables and figures from archived or already-generated results.

Mode B: run a lightweight smoke pipeline from scratch.

Mode C: full from-scratch experiment. This is long-running and is not recommended as the first reviewer action.

## Quick smoke test

    python scripts/00_setup_check.py --config configs/mode_b_smoke.yaml --strict
    python scripts/90_run_smoke_from_scratch.py --config configs/mode_b_smoke.yaml --clean

## Result replay

    python scripts/30_build_canonical_results.py --results-dir results --output-dir results/processed
    python scripts/31_build_paper_tables.py --processed-dir results/processed --output-dir results/tables
    python scripts/32_build_analysis_tables.py --processed-dir results/processed --output-dir results/tables
    python scripts/33_make_paper_figures.py --tables-dir results/tables --output-root figures

## Pre-experiment replay

    python scripts/38_build_pre_experiment_outputs.py --source-csv data/pre_experiment/alpha_metrics.csv --output-dir results/pre_experiment --figure-dir figures/pre_experiment

## Visual demo replay

    python scripts/40_make_visual_demo.py --output-data-dir results/visual_demo --output-figure-dir figures/visual_demo

## Long-running validation

The maintainer-side Stage 2 strict validation passed on Linux. It exercises setup checks, method registry, smoke run, clusterer coverage, dependency probes, HoloClean DB check, and individual cleaner coverage for mode, baran, holoclean, bigdansing, boostclean, horizon, scared, and unified.

The observed runtime was about six hours on the maintainer machine. Runtime on reviewer hardware may differ.

## Stage map

TRACE is organized into four stages:

- Stage 1: repository and baseline preparation.
- Stage 2: execution-layer validation for cleaners and clusterers.
- Stage 3: result replay, figures, pre-experiment, visual demo, and advisor-review package.
- Stage 4: planned TRACE validation, new algorithm extension, and new dataset onboarding.

Stage 1--3 are complete for advisor review. Stage 4 is planned.

See `docs/stage1_to_stage4_plan.md` for details.

## Terminal home

For orientation inside the artifact, run:

    python scripts/00_trace_home.py

For a numbered terminal menu:

    python scripts/00_trace_home.py --interactive

## Data availability

Reviewer-facing data are stored under:

- `data/raw/train/`
- `data/pre_experiment/`

Generated outputs are reproducible and ignored by default. See `docs/data_policy.md`.

## License

TRACE wrapper/orchestration code is released under the MIT License. Third-party method implementations retain their respective upstream licenses and notices. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
