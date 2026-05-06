# TRACE Stage 4 Paper-Exact Reproduction

## Status

TRACE Stage 4 is closed.

The maintained paper-exact reproduction path is the leave-one-dataset-out
TRACE validation used for the paper's budget-guidance result.

## Maintained commands

Preflight input validation:

    python scripts/49_validate_trace_stage4_inputs.py ^
      --results-dir results/trace_cluster_replay_all ^
      --output-dir results/processed/trace/lodo_paper_repro ^
      --strict

Paper-exact TRACE Stage 4 reproduction:

    python -u scripts/39_run_trace_stage4_paper_repro.py ^
      --results-dir results/trace_cluster_replay_all ^
      --config configs/trace.yaml ^
      --output-dir results/processed/trace/lodo_paper_repro ^
      --random-seeds 1000 ^
      --seed 20260424 ^
      --rebuild-base ^
      --pack ^
      --log ^
      --quiet

Unified reviewer-facing command:

    python scripts/trace.py trace-validation --paper-exact

## Required code

The TRACE Stage 4 path depends on:

    configs/trace.yaml
    src/analysis/trace_replay.py
    scripts/30_replay_trace.py
    scripts/36_eval_trace_blind_random.py
    scripts/38_lodo_trace_validation.py
    scripts/39_run_trace_stage4_paper_repro.py
    scripts/49_validate_trace_stage4_inputs.py

## Required input snapshot

The input snapshot is:

    results/trace_cluster_replay_all/

It must contain:

    eigenvectors.json
    cleaned_results.json
    clustered_results.json
    analyzed_results.json
    clustered_data/

This snapshot is the only Stage 4 data snapshot required for TRACE reproduction.

## Outputs

A successful paper-exact run creates:

    results/processed/trace/lodo_paper_repro/lodo_aggregate_summary.json
    results/processed/trace/lodo_paper_repro/lodo_folds.csv
    results/processed/trace/lodo_paper_repro/lodo_blind_random_dataset_summary.csv
    results/processed/trace/lodo_paper_repro/trace_stage4_manifest.json
    artifacts/trace_stage4_paper_exact.tgz

## Paper-aligned metrics

The paper-exact Linux run produced:

    TRACE T95 median            = 0.13476388888888888
    Blind random T95 median     = 0.2701335656213705
    TRACE AUC retention median  = 0.982375574743322
    Blind random AUC retention  = 0.9537331473633225

These correspond to the paper values:

    TRACE T95 median            = 13.5%
    Blind random T95 median     = 27.0%
    TRACE AUC retention median  = 0.982
    Blind random AUC retention  = 0.954

## Traceability

The successful Linux run recorded the following git commit in the Stage 4 manifest:

    730c25f4c9acf4330f0c3b5e9789f38bbfcf971c

## Non-dependencies

TRACE Stage 4 does not require:

    results/beers_summary.xlsx
    results/flights_summary.xlsx
    results/hospital_summary.xlsx
    results/rayyan_summary.xlsx
    results/paper_latest_process_snapshot/

Those files are used by paper-table, paper-figure, or Table 6 analysis paths,
not by TRACE Stage 4 reproduction.

