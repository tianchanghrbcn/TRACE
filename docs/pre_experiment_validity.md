# TRACE Pre-experiment and Validity/Sensitivity Replay

This page is the reviewer-facing entry point for the pre-experimental calibration and validity checks reported in the paper.

## What this validates

The script validates only the checks that are part of the submitted paper:

1. Pre-experimental calibration of the outcome-weight parameter, using `data/pre_experiment/alpha_metrics.csv`.
2. Alpha sensitivity over `0.25, 0.47, 0.50, 0.75`, with `0.47` as the fixed paper value.
3. Error-injection seed sensitivity, using archived seed-sensitivity tables.

It does **not** rerun the full cleaning or clustering benchmark, and it does **not** require the generated seed or error-model raw directories.

## Reviewer command

```bash
python scripts/81_replay_pre_experiment_validity.py
```

Expected output:

```text
results/pre_experiment/pre_experiment_validity_report.json
results/pre_experiment/pre_experiment_validity_report.md
analysis/validity_sensitivity/validity_sensitivity_summary.json
analysis/validity_sensitivity/validity_sensitivity_summary.md
```

A successful run checks the paper-facing values:

| Check | Expected paper-facing value |
|---|---:|
| Base alpha | 0.47 |
| Alpha sensitivity | minimum Spearman > 0.88 |
| Top-combo stability | 83.3% |
| Seed trend-direction preservation | 70.8% |

## Required archived inputs

```text
data/pre_experiment/alpha_metrics.csv
analysis/validity_sensitivity/inputs/analysis_results/beers_summary.xlsx
analysis/validity_sensitivity/inputs/analysis_results/flights_summary.xlsx
analysis/validity_sensitivity/inputs/analysis_results/hospital_summary.xlsx
analysis/validity_sensitivity/inputs/analysis_results/rayyan_summary.xlsx
analysis/validity_sensitivity/seed_sensitivity_summary.json
analysis/validity_sensitivity/seed_sensitivity_report.md
analysis/validity_sensitivity/seed_sensitivity_runs.csv
analysis/validity_sensitivity/seed_sensitivity_group_summary.csv
```

## Excluded from the reviewer path

The following are not needed for the submitted paper's pre-experiment/validity claims and should not be part of the required release package:

```text
analysis/validity_sensitivity/generated_seed_data/
analysis/validity_sensitivity/generated_error_model_data/
analysis/validity_sensitivity/error_model_sensitivity_*
```

Seed sensitivity is represented by the compact archived CSV/JSON/MD summary files listed above.
