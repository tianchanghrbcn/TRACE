# TRACE Validity Sensitivity Summary

- Generated at UTC: 2026-05-06T07:47:36.805454+00:00
- Status: PASS
- Scope: pre-experiment calibration, alpha sensitivity, and error-injection seed sensitivity.
- Error-model/type sensitivity is intentionally excluded from this reviewer-facing summary.

## Paper-aligned checks

| Claim support item | Value | Source |
|---|---:|---|
| Base alpha | 0.47 | `data/pre_experiment/alpha_metrics.csv` + `alpha_sensitivity_summary.json` |
| Min combo Spearman | 0.9012769201448446 | `alpha_sensitivity_summary.json` |
| Min cleaner Spearman | 0.8833333333333333 | `alpha_sensitivity_summary.json` |
| Min clusterer Spearman | 0.942857142857143 | `alpha_sensitivity_summary.json` |
| Top-combo stability rate | 0.8333333333333334 | `alpha_sensitivity_summary.json` |
| Seed trend-direction preserved rate | 0.7083333333333334 | csv:seed_sensitivity_group_summary.csv:all_seeds_same_direction |

## Warnings

No warnings.

## Failures

No failures.
