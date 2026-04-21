# Release Cleanup

Stage 3.7.2 cleans the advisor-review package without changing the experimental logic.

## Goals

- Remove or ignore generated local artifacts.
- Remove CJK text from active release source files.
- Ensure no reviewer-facing entry point points to legacy `src/pipeline/train`.
- Keep legacy code as reference only when it is not needed by active release scripts.
- Re-run package validation after cleanup.

## Commands

Audit cleanup status:

    python scripts/41_audit_release_cleanup.py

Normalize CJK text:

    python scripts/42_normalize_release_source_text.py --apply

Remove generated artifacts:

    python scripts/43_clean_generated_artifacts.py --apply

Rebuild and validate:

    python scripts/98_validate_release_package.py

## Rule

Do not delete active cleaner, clusterer, pipeline, result-processing, figure, pre-experiment, or visual-demo code during this cleanup step.

