# Known Limitations

This package is an advisor-review v0 release, not the final submission artifact.

## Current limitations

1. Stage 4 is not complete yet.
   - TRACE validation code is still pending.
   - New cleaner or clusterer extension tests are pending.
   - New dataset extension tests are pending.

2. Some legacy source files still exist as references.
   - They are not reviewer entry points.
   - The active entry points are documented in README and `scripts/`.

3. Some generated artifacts are intentionally ignored.
   - `results/processed/*.csv`
   - `results/tables/*.csv`
   - generated figures
   - generated visual-demo outputs
   - generated pre-experiment outputs

4. Full from-scratch validation is long-running.
   - Maintainer-observed Stage 2 strict validation took about six hours.
   - Runtime can differ substantially across machines.

5. Cleaner environments are heterogeneous.
   - Some methods require special conda environments.
   - The smoke path is intentionally lighter than the full validation path.

