# Paper Output Traceability

Paper-level evidence is validated by:

    python scripts/trace.py paper-replay

The workflow covers:

- paper source audit and source selection;
- summary workbook generation and validation;
- paper table script execution;
- paper table output validation;
- layered table equivalence;
- paper figure script execution;
- paper figure output validation;
- figure traceability;
- combined paper-output traceability;
- pre-experiment and validity-sensitivity checks.

## Accepted warnings

Raw table equivalence may contain diagnostic failures for upstream intermediate
or supporting files. The reviewer-facing decision is based on the layered report.

The accepted condition is:

    paper-facing table hard_failure_count = 0
    figure traceability status = PASS

The combined report is:

    analysis/paper_generated/paper_output_traceability_report.json


