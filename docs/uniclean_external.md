# UniClean External / Archived Support

UniClean appears in the paper as one of the candidate cleaning methods.

## Role in the paper

UniClean is included in the paper-level analysis as a semantic/contextual cleaner.
It appears in the paper's cleaning-method table and downstream repair/process
summaries.

## Reviewer-facing artifact behavior

UniClean is supported through paper-exact archived outputs and downstream
analysis artifacts.

The default reviewer workflows do not run UniClean from scratch:

    python scripts/trace.py benchmark-smoke --clean

The smoke workflow intentionally uses a lightweight method subset so that
reviewers can quickly confirm the pipeline mechanics.

## Full rerun behavior

A full from-scratch UniClean rerun requires the external UniClean repository and
its own environment/dependencies. TRACE therefore marks UniClean as:

    runner: external_archived
    paper_replay_supported: true
    benchmark_smoke_supported: false
    full_rerun_supported: external_only

## Evidence retained for release assets

The Linux full-audit proof package should retain UniClean run logs when present,
for example:

    uniclean_trace_all_clusterers_20260430_125709.log
    uniclean_trace_hc_smoke_20260430_124228.log

These logs are release-asset evidence and do not need to be committed as source
files in the git repository.

## Interpretation

UniClean is part of the paper-exact benchmark evidence. It is not part of the
default smoke rerun. This avoids requiring reviewers to install an additional
external cleaner before they can validate the main artifact paths.
