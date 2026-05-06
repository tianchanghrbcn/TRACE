# Reviewer Quickstart

This document describes the reviewer-facing validation path for the TRACE artifact.

## 1. Create and activate the conda environment

If the `trace-runner` environment is not already available, create it from the release asset:

    conda env create -f release_assets/trace_runner_environment.yml

Then activate it:

    conda activate trace-runner

If the environment already exists and you want to update it, use:

    conda env update -n trace-runner -f release_assets/trace_runner_environment.yml --prune

All commands below assume they are executed from the repository root after activating this environment.

## 2. Unpack release assets

Create the expected directories:

    mkdir -p release_assets results results/logs

Unpack the TRACE Stage 4 input snapshot:

    unzip -q release_assets/trace_cluster_replay_all.zip -d results/

Unpack the paper replay input snapshot:

    unzip -q release_assets/trace_paper_replay_inputs.zip -d .

Unpack the strict benchmark proof:

    tar -xzf release_assets/trace_benchmark_full_audit_proof*.tgz -C results/logs/

The TRACE input snapshot should contain:

    results/trace_cluster_replay_all/eigenvectors.json
    results/trace_cluster_replay_all/cleaned_results.json
    results/trace_cluster_replay_all/clustered_results.json
    results/trace_cluster_replay_all/analyzed_results.json
    results/trace_cluster_replay_all/clustered_data/

The paper replay input snapshot should contain:

    results/visual_demo/customer_segments/clean_withseg.csv
    results/visual_demo/customer_segments/demo_dirty/
    results/visual_demo/customer_segments/demo_results/eigenvectors.json
    results/analysis_results/
    results/processed/trace/cluster_replay_all/trace_dataset_summary.csv
    results/processed/trace/lodo_paper_repro/

The strict benchmark proof should contain:

    results/logs/stage2_strict_YYYYMMDD_HHMMSS/RESULT
    results/logs/stage2_strict_YYYYMMDD_HHMMSS/summary.tsv

The `RESULT` file should contain:

    PASSED

## 3. Recommended validation sequence

Run:

    conda activate trace-runner

    python scripts/trace.py preexp-validity --strict
    python scripts/trace.py trace-validation --paper-exact
    python scripts/trace.py paper-replay
    python scripts/trace.py benchmark-smoke --clean
    python scripts/trace.py benchmark-full-audit
    python scripts/trace.py release-check --run-trace-validation

Expected final status:

    PASS

with:

    warning_count = 0
    failure_count = 0

## 4. Paper table and figure replay

The maintained paper figure/table builders are:

    python scripts/50_build_all_paper_figures.py --input-root results --output-dir analysis/paper_generated/paper_artifact/figures --strict --clean-output
    python scripts/51_build_all_paper_tables.py --input-root results --output-dir analysis/paper_generated/paper_artifact/tables --strict --clean-output

The wrapper command is:

    python scripts/trace.py paper-replay

## 5. TRACE Stage 4

The paper-exact TRACE command is:

    python scripts/trace.py trace-validation --paper-exact

Expected paper-aligned metrics:

    TRACE T95 median            = 13.5%
    Blind random T95 median     = 27.0%
    TRACE AUC retention median  = 0.982
    Blind random AUC retention  = 0.954

Observed exact values in the final fresh-clone test:

    TRACE T95 median            = 0.13476388888888888
    Blind random T95 median     = 0.2701335656213705
    TRACE AUC retention median  = 0.982375574743322
    Blind random AUC retention  = 0.9537331473633225

## 6. Benchmark smoke and full audit

The old Mode B workflow is now named `benchmark-smoke`:

    python scripts/trace.py benchmark-smoke --clean

The old Mode C workflow is now named `benchmark-full-audit`:

    python scripts/trace.py benchmark-full-audit

`benchmark-full-audit` validates strict proof logs by default. A full from-scratch rerun is long-running and is not the default reviewer path.

## 7. UniClean

UniClean is included through paper-exact archived outputs and runtime evidence.

A full from-scratch UniClean deployment requires the external UniClean repository and is not part of the default smoke workflow.

UniClean runtime evidence is stored in:

    analysis/uniclean_external/
