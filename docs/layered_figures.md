# Layered Figures

Stage 3.4.2 creates semantic TRACE figure modules.

The modules are organized by what the figure explains rather than by paper section number:

- `src/figures/framework_figures.py`
- `src/figures/data_level_figures.py`
- `src/figures/process_level_figures.py`
- `src/figures/result_level_figures.py`

Run:

    python scripts/35_make_layered_figures.py --processed-dir results/processed --tables-dir results/tables --output-root figures

The first figures are scaffolds. Later Stage 3 work will migrate the old figure candidates into these modules.

