# TRACE Pre-experiment and Validity Validation Report

- Generated at UTC: 2026-05-06T09:46:10.338996+00:00
- Status: PASS

## Commands

| Step | Status | Return code |
|---|---|---:|
| pre_experiment_replay | PASS | 0 |
| alpha_sensitivity_replay | PASS | 0 |

## Checks

| Check | Status | Detail |
|---|---|---|
| project_root_exists | PASS | E:\TRACE |
| pre_source_csv_exists | PASS | data/pre_experiment/alpha_metrics.csv |
| pre_source_csv_has_rows | PASS | rows=20 |
| summary_workbook_dir_exists | PASS | analysis/validity_sensitivity/inputs/analysis_results |
| summary_workbook_exists:beers | PASS | analysis/validity_sensitivity/inputs/analysis_results/beers_summary.xlsx |
| summary_workbook_exists:flights | PASS | analysis/validity_sensitivity/inputs/analysis_results/flights_summary.xlsx |
| summary_workbook_exists:hospital | PASS | analysis/validity_sensitivity/inputs/analysis_results/hospital_summary.xlsx |
| summary_workbook_exists:rayyan | PASS | analysis/validity_sensitivity/inputs/analysis_results/rayyan_summary.xlsx |
| pre_experiment_alpha_metrics_exists | PASS | results/pre_experiment/alpha_metrics.csv |
| pre_experiment_alpha_metrics_has_rows | PASS | rows=20 |
| pre_experiment_manifest_exists | PASS | results/pre_experiment/pre_experiment_manifest.json |
| pre_experiment_figure_dir_exists | PASS | figures/pre_experiment |
| pre_experiment_figures_exist | PASS | figure_count=4 |
| alpha_file_exists:alpha_row_scores.csv | PASS | analysis/validity_sensitivity/alpha_row_scores.csv |
| alpha_csv_has_rows:alpha_row_scores.csv | PASS | rows=12192 |
| alpha_file_exists:alpha_cleaner_medians.csv | PASS | analysis/validity_sensitivity/alpha_cleaner_medians.csv |
| alpha_csv_has_rows:alpha_cleaner_medians.csv | PASS | rows=140 |
| alpha_file_exists:alpha_clusterer_medians.csv | PASS | analysis/validity_sensitivity/alpha_clusterer_medians.csv |
| alpha_csv_has_rows:alpha_clusterer_medians.csv | PASS | rows=96 |
| alpha_file_exists:alpha_combo_medians.csv | PASS | analysis/validity_sensitivity/alpha_combo_medians.csv |
| alpha_csv_has_rows:alpha_combo_medians.csv | PASS | rows=840 |
| alpha_file_exists:alpha_rank_correlation.csv | PASS | analysis/validity_sensitivity/alpha_rank_correlation.csv |
| alpha_csv_has_rows:alpha_rank_correlation.csv | PASS | rows=48 |
| alpha_file_exists:alpha_top_combo_stability.csv | PASS | analysis/validity_sensitivity/alpha_top_combo_stability.csv |
| alpha_csv_has_rows:alpha_top_combo_stability.csv | PASS | rows=16 |
| alpha_file_exists:alpha_sensitivity_summary.json | PASS | analysis/validity_sensitivity/alpha_sensitivity_summary.json |
| alpha_file_exists:alpha_sensitivity_report.md | PASS | analysis/validity_sensitivity/alpha_sensitivity_report.md |
| alpha_summary_status_PASS | PASS | status=PASS |
| alpha_base_value_matches_paper | PASS | base_alpha=0.47, expected=0.47 |
| alpha_tested_values_match_paper | PASS | observed=[0.25, 0.47, 0.5, 0.75], missing=[] |
| alpha_min_combo_spearman_ge_0.88 | PASS | min_combo_spearman=0.9012769201448446 |
| alpha_min_cleaner_spearman_ge_0.88 | PASS | min_cleaner_spearman=0.8833333333333333 |
| alpha_min_clusterer_spearman_ge_0.88 | PASS | min_clusterer_spearman=0.942857142857143 |
| alpha_top_combo_stability_ge_0.833 | PASS | top_combo_stability_rate=0.8333333333333334 |
| alpha_score_sanity_spearman_existing_vs_recomputed_ge_0.999 | PASS | spearman_existing_vs_recomputed=0.9999999999999999 |
| alpha_score_sanity_pearson_existing_vs_recomputed_ge_0.999 | PASS | pearson_existing_vs_recomputed=0.9999999999999999 |
| seed_file_exists:seed_sensitivity_runs.csv | PASS | analysis/validity_sensitivity/seed_sensitivity_runs.csv |
| seed_csv_has_rows:seed_sensitivity_runs.csv | PASS | rows=108 |
| seed_file_exists:seed_sensitivity_group_summary.csv | PASS | analysis/validity_sensitivity/seed_sensitivity_group_summary.csv |
| seed_csv_has_rows:seed_sensitivity_group_summary.csv | PASS | rows=24 |
| seed_file_exists:seed_sensitivity_summary.json | PASS | analysis/validity_sensitivity/seed_sensitivity_summary.json |
| seed_file_exists:seed_sensitivity_report.md | PASS | analysis/validity_sensitivity/seed_sensitivity_report.md |
| seed_summary_status_PASS_or_unspecified | PASS | status=PASS |
| seed_trend_preserved_rate_ge_0.708 | PASS | rate=0.7083333333333334, source=csv:seed_sensitivity_group_summary.csv:all_seeds_same_direction |
| error_model_sensitivity_files_not_present | PASS | none found |
| generated_sensitivity_data_dirs_not_present | PASS | none found |
| command_passed:pre_experiment_replay | PASS | C:\Users\chang\AppData\Local\Programs\Python\Python311\python.exe scripts/38_build_pre_experiment_outputs.py --source-csv data/pre_experiment/alpha_metrics.csv --output-dir results/pre_experiment --figure-dir figures/pre_experiment |
| command_passed:alpha_sensitivity_replay | PASS | C:\Users\chang\AppData\Local\Programs\Python\Python311\python.exe scripts/80_build_alpha_sensitivity.py --summary-dir analysis/validity_sensitivity/inputs/analysis_results --output-dir analysis/validity_sensitivity --alphas 0.25 0.47 0.5 0.75 --base-alpha 0.47 |

## Warnings

No warnings.

## Failures

No failures.
