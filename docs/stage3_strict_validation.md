# Stage 3 Strict Validation

This document describes the TRACE Stage 3 strict validation wrapper.

## Modes

Stage 3 strict validation checks:

- Mode A: paper table and figure replay validation.
- Mode B: lightweight smoke pipeline from scratch.
- Mode C: strict cleaning-clustering execution proof.

## Run

If the Linux Stage 2 strict proof is available under `results/logs/stage2_strict_*`:

    python scripts/63_validate_stage3_strict.py

If the proof is stored elsewhere:

    python scripts/63_validate_stage3_strict.py --mode-c-proof-dir path/to/stage2_strict_YYYYMMDD_HHMMSS

To rebuild Mode A before validation:

    python scripts/63_validate_stage3_strict.py --rebuild-mode-a

## Outputs

- `results/logs/stage3_strict_validation_report.json`
- `results/logs/stage3_strict_validation_report.md`

## Interpretation

`PASS_WITH_WARNINGS` is acceptable for the current Stage 3 state when:

- Mode A paper replay has accepted traceability warnings only;
- Mode B smoke rerun passes;
- Mode C strict Linux proof is present and passed;
- claim-level narrative traceability is deferred.

Mode C is checked from Linux strict proof by default rather than rerunning the long validation.

