# Mode A: Paper-Exact Replay

Mode A replays paper-exact outputs from archived full results and legacy paper artifacts.

The current Mode A implementation has two layers:

1. Archived paper-exact replay.
2. Generated paper-exact replay.

This version implements the archived replay layer first. It selects and archives the final paper summaries, figures, source scripts, LaTeX references, and raw result JSON files from the legacy repositories.

## Run

To run the full Mode A archive replay:

    python scripts/trace.py mode-a --audit --clean

Or run the steps manually:

    python scripts/46_audit_paper_replay_sources.py
    python scripts/47_select_paper_exact_sources.py
    python scripts/48_build_mode_a_paper_exact_archive.py --clean
    python scripts/49_validate_mode_a_paper_exact.py

## Outputs

Generated outputs:

- `artifacts/paper_exact/`
- `analysis/paper_exact/`
- `figures/paper_exact/`
- `results/paper_exact/`

These outputs are generated artifacts and should not be committed directly. They should be packaged as a release asset when needed.

## Next step

The next Stage 3R step is generated paper-exact replay:

    archived raw results
      -> analysis intermediate files
      -> summary workbooks
      -> paper-exact figures

That step will migrate the selected table scripts and figure scripts into executable TRACE Mode A modules.

## Unified entry

Run archive replay and generated summary workbook replay together:

    python scripts/trace.py mode-a --audit --clean --generated-summaries
