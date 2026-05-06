# Result Replay

Stage 3 result replay starts from raw pipeline outputs or archived Mode A result files.

Validate inputs:

    python scripts/30_validate_archived_results.py --results-dir results

Build canonical result tables:

    python scripts/30_build_canonical_results.py --results-dir results --output-dir results/processed

Build initial paper-table summaries:

    python scripts/31_build_paper_tables.py --processed-dir results/processed --output-dir results/tables

The generated canonical tables are the input to later paper-table and paper-figure scripts.

