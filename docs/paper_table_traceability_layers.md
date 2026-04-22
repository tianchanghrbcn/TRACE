
# Layered Paper Table Traceability

Stage 3R.4.2a separates paper-table equivalence into reviewer-facing and diagnostic layers.

## Layers

- `paper_facing`: outputs directly used by the paper table traceability layer.
- `supporting_analysis`: generated support statistics and helper workbooks.
- `upstream_intermediate`: regenerated upstream cleaning, clustering, and merged summary files.
- `unmapped`: outputs not yet assigned to a traceability layer.

## Run

First run Mode A with paper-table outputs:

    python scripts/trace.py mode-a --audit --clean --generated-summaries --paper-tables

Then run raw equivalence and layered classification:

    python scripts/55_validate_paper_table_equivalence.py
    python scripts/56_classify_table_equivalence_layers.py

## Interpretation

`PASS_WITH_DIAGNOSTIC_WARNINGS` is acceptable at this stage when paper-facing outputs have no hard mismatches but upstream intermediate files still show differences.

The next cleanup step should investigate diagnostic mismatches, then build the final paper-output subset map.
