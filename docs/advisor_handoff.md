# Advisor Handoff: TRACE Package v0.1

This document summarizes the advisor-review package after Stage 1--3.

## Current status

TRACE now includes:

- runnable Mode B smoke pipeline,
- Stage 2 strict validation evidence,
- cleaner and clusterer method registry,
- canonical result replay framework,
- analysis table builders,
- paper-figure scaffolds,
- first migrated figure batch,
- pre-experiment replay,
- reviewer-facing visual demo,
- release validation script,
- estimated runtime progress monitor.

## Validation status

Stage 2 strict validation passed on Linux.

The strict validation covered:

- setup checks,
- method registry checks,
- static checks,
- Mode B smoke run,
- clusterer coverage,
- dependency probes,
- HoloClean DB check,
- cleaner coverage for mode, baran, holoclean, bigdansing, boostclean, horizon, scared, and unified.

Observed maintainer runtime was approximately six hours.

## Reviewer quick path

Run:

    python scripts/98_validate_release_package.py

This rebuilds the smoke pipeline outputs, canonical result tables, table summaries, figures, pre-experiment outputs, and visual demo.

## What remains for Stage 4

Stage 4 is not included in this advisor-review package.

Remaining work:

- add TRACE validation code,
- add actual new algorithm extension test,
- add actual new dataset extension test,
- finalize paper-specific figure/table selection,
- prepare final submission artifact.

