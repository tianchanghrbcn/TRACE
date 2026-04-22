# Mode A: Paper Figure Replay

Stage 3R.5 focuses on paper-exact figure replay.

The goal is to validate that figures in the paper are traceable to archived or generated outputs.

## Current scope

This stage prioritizes figure and table reproduction.

Narrative claims that rely directly on raw data or intermediate statistics will be handled later in claim-level traceability.

## Planned steps

1. Select paper figure sources.
2. Run legacy figure scripts in a controlled workspace.
3. Capture generated figure outputs.
4. Compare generated figures or source data against archived paper figures.
5. Build a paper-output subset map.

## First command

    python scripts/57_select_paper_figure_sources.py

