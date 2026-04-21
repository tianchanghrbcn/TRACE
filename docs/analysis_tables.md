# Analysis Tables

Stage 3 analysis tables are built from canonical result tables under `results/processed`.

Run:

    python scripts/32_build_analysis_tables.py --processed-dir results/processed --output-dir results/tables

Generated tables:

- `cleaning_analysis_summary.csv`
- `clustering_analysis_summary.csv`
- `cleaning_clustering_summary.csv`

These tables are the bridge between raw pipeline outputs and later figure generation.

