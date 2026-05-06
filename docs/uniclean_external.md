# UniClean External / Archived Support

UniClean is included in the TRACE paper as one of the candidate cleaning methods.

## Artifact policy

TRACE does not vendor the full UniClean deployment. UniClean is treated as an
external cleaner whose paper-level contribution is supported by archived outputs
and runtime evidence.

The method registry marks UniClean as:

    runner: external_archived
    paper_replay_supported: true
    benchmark_smoke_supported: false
    full_rerun_supported: external_only

## Reviewer-facing behavior

The default smoke workflow does not run UniClean:

    python scripts/trace.py benchmark-smoke --clean

The paper replay workflow includes UniClean through archived results and
downstream paper evidence:

    python scripts/trace.py paper-replay

The UniClean runtime evidence is stored in:

    analysis/uniclean_external/

## Why this is acceptable

The TRACE artifact evaluates the cleaning-clustering benchmark and paper-level
evidence chain. Requiring reviewers to deploy an additional external cleaner
before running the smoke workflow would make the artifact unnecessarily brittle.

For full audit evidence, the release package retains UniClean runtime logs and
checksums, while documenting that full UniClean deployment is external.

