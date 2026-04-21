# Migrated Figure Batch 1

Stage 3.4.3 migrates the first batch of legacy figure ideas into stable TRACE figure modules.

The batch contains:

- `result_top_configuration_scores`
- `result_score_by_error_rate`
- `data_error_profile_heatmap`

These figures are generated from canonical Stage 3 tables rather than from legacy hard-coded paths.

Run:

    python scripts/36_make_migrated_figure_batch.py --processed-dir results/processed --output-root figures --top-k 10

Relationship to legacy scripts:

- `fig_4_plot_top10.py` inspired `result_top_configuration_scores`.
- `fig_7_CEGR_line_graph.py` inspired `result_score_by_error_rate`, but full CEGR and breakpoint analysis will be added after Mode A archived results expose EDR and full-search summary tables.
- `fig_5_heat_error_type.py` inspired `data_error_profile_heatmap`.

The migrated code uses English labels and semantic file names. It does not depend on paper section numbers.

