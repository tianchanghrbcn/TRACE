# Figure Migration Plan

Stage 3.4 does not copy old plotting scripts directly. It migrates their logic into stable TRACE figure modules.

## Legacy candidates

The legacy audit identified the following plotting candidates:

- `4.flowchart.py`
- `fig_4_plot_point.py`
- `fig_4_plot_score_eval.py`
- `fig_4_plot_top10.py`
- `fig_5_err_cluster_plot.py`
- `fig_5_err_grad_plot.py`
- `fig_5_heat_error_type.py`
- `fig_6_radar_graph_median.py`
- `fig_7_CEGR_line_graph.py`
- `fig_7_point_graph.py`
- `fig_10_make_hyper_heatmaps.py`

## Target structure

Use semantic figure names rather than paper-section numbers.

- Data-level visualizations: `src/figures/data_level_figures.py`
- Process-level visualizations: `src/figures/process_level_figures.py`
- Result-level visualizations: `src/figures/result_level_figures.py`
- Hyperparameter visualizations: `src/figures/hyperparameter_figures.py`
- Visual demo figures: `src/visual_demo`
- Alpha/pre-experiment figures: `src/pre_experiment`

## Migration rule

Each migrated figure should read from `results/processed` or `results/tables`, not directly from legacy paths.

