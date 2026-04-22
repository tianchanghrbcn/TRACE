# Paper Figure Script Harness

Stage 3R.5.2 runs selected legacy paper figure scripts in a controlled workspace.

## Run

First select figure sources:

    python scripts/57_select_paper_figure_sources.py

Then run the figure harness:

    python scripts/58_run_paper_figure_scripts.py --clean --timeout 1200
    python scripts/59_validate_paper_figure_outputs.py

## Outputs

Generated outputs are placed under:

- `analysis/paper_generated/paper_figures/`
- `figures/paper_generated/`

Important files:

- `analysis/paper_generated/paper_figures/paper_figure_script_run_manifest.json`
- `analysis/paper_generated/paper_figures/paper_figure_script_run_report.md`
- `analysis/paper_generated/paper_figures/paper_figure_validation_report.json`
- `analysis/paper_generated/paper_figures/paper_figure_validation_report.md`

## Scope

This step validates that selected legacy figure scripts can be executed and that figure outputs can be captured.

It does not yet claim figure equivalence with the paper figures.

