# Stage 3R: Paper-Exact Replay Plan

Stage 3R supplements the existing advisor pre-release with paper-exact replay.

The existing v0.1.1-advisor release validates the execution layer, result-processing scaffolds, pre-experiment replay, visual demo replay, terminal entry points, and release validation.

Stage 3R adds the missing paper-exact path:

    archived raw results
      -> analysis intermediate files
      -> summary workbooks
      -> paper-exact tables
      -> paper-exact figures

## Required outputs

- `analysis/` intermediate files
- `*_summary.xlsx` or their paper-equivalent workbooks
- paper-exact tables
- paper-exact figures
- raw/full results archive or download instructions
- unified Mode A / Mode B / Mode C entry points

## Mode definitions

Mode A:
    Replay from archived results to paper-exact tables and figures.

Mode B:
    Lightweight smoke run from scratch.

Mode C:
    Full or strict execution-layer validation.

## First task

Run:

    python scripts/46_audit_paper_replay_sources.py

Then inspect:

    analysis/paper_replay_audit/paper_replay_source_summary.json
    analysis/paper_replay_audit/paper_replay_source_candidates.csv
    analysis/paper_replay_audit/paper_replay_source_audit.md

