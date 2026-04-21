# Artifact Overview

TRACE is organized around a cleaning-clustering evaluation pipeline and a replay layer for tables and figures.

## Main components

- `src/pipeline`: pipeline execution, preprocessing, method registry, and runner.
- `src/cleaning`: wrapped cleaning methods.
- `src/clustering`: wrapped clustering methods.
- `src/results_processing`: canonical result tables and analysis summaries.
- `src/figures`: figure-generation code.
- `src/pre_experiment`: alpha/weight pre-experiment replay.
- `src/visual_demo`: reviewer-facing visual demo.
- `scripts`: command-line entry points.
- `configs`: experiment and method configuration files.
- `docs`: artifact documentation.

## Completed stages

- Stage 2 execution layer: strict validation passed on Linux.
- Stage 3.1 canonical result framework.
- Stage 3.2 legacy source audit.
- Stage 3.3 analysis and summary table builders.
- Stage 3.4 figure framework and first migrated figure batch.
- Stage 3.5 pre-experiment replay.
- Stage 3.6 reviewer-facing visual demo.

## Not included yet

Stage 4 will add TRACE validation code, real new-algorithm tests, and real new-dataset extension tests.

