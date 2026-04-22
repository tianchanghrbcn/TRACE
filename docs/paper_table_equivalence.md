# Paper Table Equivalence Validation

Stage 3R.4.2 validates whether generated or available Mode A paper-table outputs match archived paper-exact references.

## Run

First run the paper-table harness:

    python scripts/trace.py mode-a --audit --clean --generated-summaries --paper-tables

Then run equivalence validation:

    python scripts/55_validate_paper_table_equivalence.py

## Outputs

Generated reports:

- `analysis/paper_generated/paper_tables/table_equivalence_report.json`
- `analysis/paper_generated/paper_tables/table_equivalence_report.csv`
- `analysis/paper_generated/paper_tables/table_equivalence_report.md`

## Validation levels

- `EXACT`: SHA-256 match.
- `SEMANTIC`: content-equivalent under numeric tolerance.
- `WARN_NO_REFERENCE`: generated fallback exists, but no archived reference with the same filename was found.
- `FAIL`: archived references exist, but none are equivalent.

## Scope

This step validates paper-table output equivalence.

It does not yet validate final paper figure equivalence or all narrative claims in the paper.

