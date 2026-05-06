# Release Checklist

This checklist defines the advisor-review package v0.

## Required checks

- Mode B setup check passes.
- Mode C data/config setup check passes.
- Mode B smoke pipeline passes.
- Canonical result tables can be regenerated.
- Paper-table summaries can be regenerated.
- Analysis tables can be regenerated.
- Paper figure scaffolds can be regenerated.
- Layered figure scaffolds can be regenerated.
- First migrated figure batch can be regenerated.
- Pre-experiment outputs can be regenerated.
- Visual demo outputs can be regenerated.
- Stage 2 strict validation evidence is documented.
- Generated outputs are ignored unless explicitly selected as release artifacts.

## One-command package validation

    python scripts/98_validate_release_package.py

## Optional long-running validation

    bash scripts/97_validate_stage2_strict.sh

## Before sending to advisor

- Confirm `git status --short` is clean except ignored generated outputs.
- Confirm README and docs describe what is complete and what remains for Stage 4.
- Confirm no old `src/pipeline/train` entry point is exposed in README.
- Confirm reviewer-facing figures and docs use English labels.

