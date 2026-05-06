# TRACE Release Packaging

TRACE uses git tags plus GitHub Release assets or archival storage for large
generated outputs.

## Source repository

The git repository contains code, configs, benchmark data, validation scripts,
paper replay reports, and lightweight evidence.

## Release assets

Large generated assets should be distributed outside the git tree:

- `trace_cluster_replay_all.zip`: TRACE Stage 4 input snapshot.
- `trace_stage4_paper_exact.tgz`: TRACE Stage 4 paper-exact output pack.
- `trace_benchmark_full_audit_proof_*.tgz`: strict benchmark proof logs.
- `SHA256SUMS.txt`: checksums for release assets.
- `release_manifest.json`: asset names, hashes, and intended unpack locations.

## Full reviewer validation

After unpacking release assets, the main command is:

    python scripts/trace.py release-check --run-trace-validation

If the strict benchmark proof asset is not unpacked locally, use:

    python scripts/trace.py release-check --run-trace-validation --allow-missing-full-audit-proof

The latter is acceptable only for local packaging checks, not for the final
artifact proof run.

## Do not commit

The following generated paths should not be committed directly:

    results/
    figures/
    artifacts/trace_stage4_paper_exact.tgz

Evidence reports copied into `analysis/` may be committed.


