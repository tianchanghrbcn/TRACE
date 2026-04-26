# TRACE Artifact Utilities

This folder is reserved for reviewer-facing TRACE validation utilities. The main validated command-line entry points are currently kept under `scripts/`:

- `scripts/34_rerun_clustering_from_cleaned_results.py`: cached-cleaning clustering replay with resume support.
- `scripts/30_replay_trace.py`: Stage 4 TRACE replay.
- `scripts/36_eval_trace_blind_random.py`: blind randomized path-order validation.
- `scripts/37_plot_trace_validation.py`: paper figures for TRACE validation.

The artifact protocol reuses cleaned tables from an exhaustive baseline, reruns clustering with trial-level logging, replays TRACE, and compares it against blind randomized path-order schedules.
