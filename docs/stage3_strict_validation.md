# Strict Reviewer Workflow Validation

The strict workflow validator checks three reviewer-facing components:

- `paper-replay`
- `benchmark-smoke`
- `benchmark-full-audit`

## Command

    python scripts/63_validate_stage3_strict.py --skip-smoke-rerun

To use a specific full-audit proof directory:

    python scripts/63_validate_stage3_strict.py ^
      --skip-smoke-rerun ^
      --full-audit-proof-dir results/logs/stage2_strict_YYYYMMDD_HHMMSS

## Accepted status

`PASS` means all three workflows passed.

`PASS_WITH_WARNINGS` is acceptable when:

- paper-replay has accepted diagnostic warnings only;
- benchmark-smoke passed;
- benchmark-full-audit proof is present and passed, or is explicitly allowed as
  missing for a local packaging check.

The final artifact proof should include the full-audit proof directory.


