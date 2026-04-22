# Terminal Interface

TRACE uses a lightweight terminal interface rather than a web UI.

Run:

    python scripts/00_trace_home.py

For a numbered terminal menu:

    python scripts/00_trace_home.py --interactive

## Mode entries

The v0.1.2 advisor interface exposes the three Stage 3 modes:

- Mode A: paper table/figure replay and traceability.
- Mode B: lightweight smoke pipeline from scratch.
- Mode C: strict cleaning-clustering proof checked from Linux validation evidence.

## Recommended commands

Run release validation:

    python scripts/98_validate_release_package.py

Validate Mode A only:

    python scripts/62_validate_mode_a_paper_replay.py

Validate Stage 3 strict completion:

    python scripts/63_validate_stage3_strict.py

## Why not a web UI

TRACE is a research artifact. A command-line interface is easier to run in headless servers, easier to log, and less likely to introduce browser or deployment problems.

## Stage 4 placeholders

The menu includes planned Stage 4 entries, but does not implement them yet. This keeps the reviewer interface stable while making clear what remains future work.

