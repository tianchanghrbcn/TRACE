# TRACE Environments

TRACE uses separate environments for different reproducibility modes.

## Mode A: archived-result reproduction

- File: `envs/mode_a_trace_runner.yml`
- Environment name: `trace-runner`
- Purpose: setup checks, archived-result validation, analysis tables, TRACE replay, and figures.
- This is the recommended reviewer-facing environment.

## Mode B: smoke from scratch

- File: `envs/mode_b_smoke.yml`
- Default environment name: `trace-runner`
- Purpose: run a small from-scratch pipeline subset.
- If the original pipeline requires additional packages, use the Mode C fallback environment.

## Mode C: full from scratch

- File: `envs/mode_c_pipeline_original.yml`
- Environment name: `torch110`
- Purpose: reproduce the original full cleaning-clustering pipeline.
- This file is intentionally kept close to the original working environment.

## Original setup script

- File: `envs/original_config.sh`
- This is the original setup script copied from the archived AutoMLClustering repository.
- It is kept for auditability.
- The normalized setup entry points are under `scripts/setup/`.