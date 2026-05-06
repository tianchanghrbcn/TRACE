# Stage 3 Plan

Stage 3 converts runnable TRACE experiment outputs into a reviewer-friendly result replay package.

## Stage 3.1

Build canonical result tables from raw pipeline outputs.

## Stage 3.2

Audit legacy result, analysis, table, and figure scripts from AutoMLClustering and AutoMLClustering_full.

The goal is not to copy the legacy repositories into TRACE. The goal is to classify files and decide what should be migrated into:

- `src/results_processing`
- `src/figures`
- `src/pre_experiment`
- `src/visual_demo`
- `docs`
- archived result inputs for Mode A replay

## Stage 3.3

Migrate analysis and table-generation logic into TRACE.

## Stage 3.4

Migrate paper-figure generation.

## Stage 3.5

Normalize pre-experiment code.

## Stage 3.6

Normalize visual-demo code.

## Stage 3.7

Prepare release/package v0 documentation.

