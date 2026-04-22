# Release Packaging

TRACE uses Git tags and GitHub Releases for advisor-review and submission artifact packages.

## Current advisor-review release

Recommended next advisor-review version:

    v0.1.2-advisor

This version is intended to add paper-output traceability on top of v0.1.1.

## What v0.1.2 should demonstrate

- Mode A: paper table and figure replay validation.
- Mode B: lightweight smoke pipeline from scratch.
- Mode C: strict execution-layer proof from Linux validation evidence.
- Combined Stage 3 strict validation.
- Release validation through `scripts/98_validate_release_package.py`.

## Why GitHub Release instead of a package registry

TRACE is a research artifact rather than an installable software product. Reviewers need source code, data, scripts, documentation, validation evidence, and release assets.

## Build local assets

Run:

    python scripts/44_build_release_assets.py --version v0.1.2-advisor --source-ref v0.1.2-advisor

Generated files are placed under `release/`.

## Suggested GitHub Release assets

- `TRACE-v0.1.2-advisor-source.zip`
- `TRACE-v0.1.2-advisor-handoff.zip`
- `TRACE-v0.1.2-advisor-stage2-strict-proof.zip`
- `TRACE-v0.1.2-advisor-paper-replay-proof.zip`

The Chinese advisor report is intended for internal advisor communication and does not need to be attached to the public reviewer-facing release.

