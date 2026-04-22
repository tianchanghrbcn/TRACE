# Paper Output Traceability

Stage 3R.6 combines paper-table and paper-figure evidence into a single reviewer-facing traceability report.

## Run

First ensure that table and figure reports exist:

    python scripts/trace.py mode-a --audit --clean --generated-summaries --paper-tables --table-equivalence
    python scripts/trace.py mode-a --paper-figures --figure-traceability

Then build the combined report:

    python scripts/61_build_paper_output_traceability_report.py

## Output

Generated reports:

- `analysis/paper_generated/paper_output_traceability_report.json`
- `analysis/paper_generated/paper_output_traceability_report.md`

## Interpretation

`PASS_WITH_WARNINGS` is acceptable at this stage when:

- paper-facing tables have no hard mismatches;
- all LaTeX figure references have archived traceability;
- warnings only indicate archived-only figures or upstream intermediate diagnostics.

Narrative claim traceability is deferred until after table and figure traceability are stable.

