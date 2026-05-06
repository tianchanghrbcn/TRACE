# Alpha Sensitivity Report

- Generated at UTC: 2026-05-06T07:30:33.165362+00:00
- Status: PASS
- Base alpha: 0.47
- Tested alphas: 0.25, 0.47, 0.5, 0.75

## Interpretation

The tested alpha values preserve high rank agreement with the base alpha. This supports the claim that the main outcome-level trends are not driven by a single scalarization weight.

## Key statistics

- Input row count: 3059
- Long row-score count: 12192
- Minimum combo-rank Spearman vs. base: 0.9012769201448446
- Minimum cleaner-rank Spearman vs. base: 0.8833333333333333
- Minimum clusterer-rank Spearman vs. base: 0.942857142857143
- Top-combo stability rate: 0.8333333333333334

## Files

- row_scores: `analysis\validity_sensitivity\alpha_row_scores.csv`
- combo_medians: `analysis\validity_sensitivity\alpha_combo_medians.csv`
- cleaner_medians: `analysis\validity_sensitivity\alpha_cleaner_medians.csv`
- clusterer_medians: `analysis\validity_sensitivity\alpha_clusterer_medians.csv`
- rank_correlation: `analysis\validity_sensitivity\alpha_rank_correlation.csv`
- top_combo_stability: `analysis\validity_sensitivity\alpha_top_combo_stability.csv`
- summary_json: `analysis\validity_sensitivity\alpha_sensitivity_summary.json`
- report_md: `analysis\validity_sensitivity\alpha_sensitivity_report.md`
