# Terminal Interface

TRACE uses a lightweight terminal interface rather than a web UI.

Run:

    python scripts/00_trace_home.py

For a numbered menu:

    python scripts/00_trace_home.py --interactive

## Why not a web UI

TRACE is a research artifact. A command-line interface is easier to run in headless servers, easier to log, and less likely to introduce browser or deployment problems.

The terminal interface provides:

- current repository context;
- recommended commands;
- release validation entry point;
- runtime progress monitor entry point;
- Stage 4 placeholder entries.

## Stage 4 placeholders

The menu includes planned Stage 4 entries, but does not implement them yet. This keeps the reviewer interface stable while making clear what remains future work.

