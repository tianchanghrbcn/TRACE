# UniClean External Runtime Evidence

This folder contains runtime evidence for UniClean in the TRACE artifact.

## Why this exists

UniClean appears in the TRACE paper as a candidate cleaning method. The TRACE
repository does not vendor the full UniClean deployment because UniClean is an
external system with its own repository, environment, and dependencies.

Instead, TRACE provides:

- paper-exact archived outputs for downstream paper replay;
- runtime logs/evidence for UniClean runs;
- a registry entry that marks UniClean as external/archived.

## Contents

- `run_logs/`: extracted UniClean runtime logs from the release-side `train_log.zip`.
- `run_manifest.json`: file sizes and SHA256 checksums.
- `SHA256SUMS.txt`: checksum list for the extracted runtime files.

## Reviewer interpretation

The default reviewer smoke path does not run UniClean from scratch:

    python scripts/trace.py benchmark-smoke --clean

The paper-level replay includes UniClean through archived outputs and downstream
analysis evidence:

    python scripts/trace.py paper-replay

A full from-scratch UniClean rerun requires the external UniClean repository and
is not part of the default TRACE smoke workflow.
