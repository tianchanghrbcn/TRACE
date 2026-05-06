# Mode A Paper Replay Validation

Stage 3R.6.1 provides a single validation wrapper for Mode A paper replay.

## Run existing-report validation

    python scripts/62_validate_mode_a_paper_replay.py

## Rebuild and validate

    python scripts/62_validate_mode_a_paper_replay.py --rebuild

## Outputs

Generated reports:

- `analysis/paper_generated/mode_a_paper_replay_validation_report.json`
- `analysis/paper_generated/mode_a_paper_replay_validation_report.md`

## Interpretation

`PASS_WITH_WARNINGS` is acceptable at the current Stage 3R state when:

- paper-facing tables have no hard mismatches;
- every LaTeX-referenced figure has archived traceability;
- the figure harness has no failed scripts;
- remaining warnings are diagnostic or archived-only cases.

Narrative claim traceability is intentionally deferred.

