# Pre-Experiment

TRACE includes a small pre-experiment for inspecting the alpha weight used in the combined clustering objective.

Stage 3.5 migrates the legacy alpha/weight-search artifacts into a stable reviewer-facing format.

## Build outputs

Run:

    python scripts/38_build_pre_experiment_outputs.py --audit-csv results/processed/legacy_audit_files.csv --output-dir results/pre_experiment --figure-dir figures/pre_experiment

Generated outputs:

- `results/pre_experiment/alpha_metrics.csv`
- `results/pre_experiment/pre_experiment_manifest.json`
- `figures/pre_experiment/alpha_vs_median.png`
- `figures/pre_experiment/alpha_vs_median.pdf`
- `figures/pre_experiment/alpha_vs_variance.png`
- `figures/pre_experiment/alpha_vs_variance.pdf`

The reviewer-facing figures are regenerated with English labels. Legacy Chinese PDFs are not used as release outputs.

