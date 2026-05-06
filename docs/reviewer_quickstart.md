# Reviewer Quickstart

This document describes the reviewer-facing validation path for the TRACE artifact.

## 1. Activate the conda environment

Before running reviewer commands, activate the TRACE conda environment:

    conda activate trace-runner

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

or:

    PASS_WITH_WARNINGS with failure_count = 0

## 4. Maintainer-only TRACE base rebuild

The reviewer path uses the packaged base TRACE ledger.

Maintainers may explicitly rebuild the base ledger with:

    python scripts/trace.py trace-validation --paper-exact --rebuild-base

This is not the default reviewer path.
