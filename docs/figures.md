# Figures

TRACE separates figure-generation code from generated figure files.

Code lives under:

- `src/figures/style.py`
- `src/figures/plot_utils.py`
- `src/figures/paper_figures.py`

Generated outputs live under:

- `figures/png`
- `figures/pdf`

Run:

    python scripts/33_make_paper_figures.py --tables-dir results/tables --output-root figures

The first Stage 3.4 figures are smoke-test scaffolds. They are generated from canonical tables and are intended to validate the figure pipeline before full Mode A archived results are connected.

