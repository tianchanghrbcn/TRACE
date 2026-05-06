# TRACE Strict Reviewer Workflow Validation

- Generated at UTC: 2026-05-06T13:09:48.591230+00:00
- Status: PASS_WITH_WARNINGS

## Workflow summary

| Workflow | Status | Accepted | Description |
|---|---|---:|---|
| paper-replay | PASS_WITH_WARNINGS | True | Paper-level evidence replay: tables, figures, traceability, and validity checks. |
| benchmark-smoke | PASS | True | Lightweight cleaning-clustering smoke pipeline from scratch. |
| benchmark-full-audit | PASS | True | Strict cleaning-clustering execution proof. |

## Failures

No hard failures.

## Warnings

- paper-replay accepted with status: PASS_WITH_WARNINGS
- paper-replay: Report has accepted warning status: paper_table_layers=PASS_WITH_DIAGNOSTIC_WARNINGS
- paper-replay: Report has accepted warning status: paper_output_traceability=PASS_WITH_WARNINGS

## Interpretation

PASS means paper-replay, benchmark-smoke, and benchmark-full-audit all passed.

PASS_WITH_WARNINGS is acceptable only when warnings are documented and do not affect paper-table, paper-figure, TRACE, or benchmark proof validity.

