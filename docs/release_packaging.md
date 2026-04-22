# Release Packaging

TRACE uses Git tags and GitHub Releases for advisor-review and submission artifact packages.

## Current advisor-review release

Recommended final advisor-review version:

    v0.1.1-advisor

## Why GitHub Release instead of a package registry

TRACE is a research artifact rather than an installable software product. Reviewers need source code, data, scripts, documentation, and validation evidence.

The release should provide:

- source archive;
- handoff documentation archive;
- Stage 2 strict validation proof archive;
- optional advisor-only Chinese report.

## Build local assets

Run:

    python scripts/44_build_release_assets.py --version v0.1.1-advisor --source-ref v0.1.1-advisor

Generated files are placed under `release/`.

## Suggested GitHub Release assets

- `TRACE-v0.1.1-advisor-source.zip`
- `TRACE-v0.1.1-advisor-handoff.zip`
- `TRACE-v0.1.1-advisor-stage2-strict-proof.zip`

The Chinese advisor report is intended for internal advisor communication and does not need to be attached to the public reviewer-facing release.

