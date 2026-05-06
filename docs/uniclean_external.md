# UniClean External / Archived Support

UniClean is included in the paper as a candidate cleaner.

In the reviewer-facing TRACE artifact, UniClean is handled as an external/archived
cleaner:

- paper-replay includes UniClean through archived paper-exact outputs and
  downstream analysis reports;
- benchmark-smoke does not run UniClean by default;
- benchmark-full-audit records UniClean-related evidence through the external
  Linux proof logs when available;
- a full from-scratch UniClean rerun requires the upstream UniClean repository
  and its own environment.

This design keeps the default reviewer path lightweight while preserving the
paper-level evidence for UniClean.

## Relevant paper role

UniClean belongs to the semantic/contextual cleaner group and is treated as a
candidate cleaner, not as an oracle reference.

GroundTruth remains the oracle reference only and is excluded from ordinary
method rankings.

## Reviewer commands

The default reviewer commands are:

    python scripts/trace.py paper-replay
    python scripts/trace.py benchmark-smoke --clean
    python scripts/trace.py trace-validation --paper-exact
    python scripts/trace.py release-check --run-trace-validation

None of these commands require the reviewer to install UniClean.

## Full rerun

A full UniClean rerun is external-only. It requires the upstream UniClean
repository and should be documented by setting an external path such as:

    UNICLEAN_HOME=/path/to/UniClean

The TRACE artifact does not make this external rerun part of benchmark-smoke.
