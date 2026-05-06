# TRACE Artifact

This repository accompanies the PVLDB EA&B paper:

**TRACE: An Empirical Study of How Data Cleaning Affects Unsupervised Clustering**

TRACE studies how data cleaning affects unsupervised clustering through four
observable layers: data rewriting, process signals, outcome gains, and
hyperparameter shifts. The artifact provides paper-level replay, smoke
execution, TRACE budget-guidance validation, and strict audit evidence.

## Current status

Stage 1--4 are complete.

- Stage 1: data assets and corrupted-instance manifest.
- Stage 2: benchmark execution pipeline and strict audit evidence.
- Stage 3: paper table/figure replay and paper-output traceability.
- Stage 4: TRACE paper-exact leave-one-dataset-out validation.

Final packaging consists of release assets, checksums, and fresh-clone testing.

## Reviewer workflows

| Goal | Command |
|---|---|
| Release check | `python scripts/trace.py release-check --run-trace-validation` |
| Paper-level table/figure replay | `python scripts/trace.py paper-replay` |
| Pre-experiment and validity checks | `python scripts/trace.py preexp-validity --strict` |
| TRACE paper-exact validation | `python scripts/trace.py trace-validation --paper-exact` |
| Lightweight benchmark smoke run | `python scripts/trace.py benchmark-smoke --clean` |
| Strict benchmark proof validation | `python scripts/trace.py benchmark-full-audit` |

The release check may report `PASS_WITH_WARNINGS` when diagnostic-only table
equivalence warnings are present. The paper-facing table layer has no hard
failures, and paper figure validation/traceability pass.

## TRACE Stage 4

The maintained TRACE paper-exact command is:

    python scripts/trace.py trace-validation --paper-exact

Expected paper-aligned metrics:

    TRACE T95 median            = 13.5%
    Blind random T95 median     = 27.0%
    TRACE AUC retention median  = 0.982
    Blind random AUC retention  = 0.954

See `docs/trace_stage4_repro.md`.

## UniClean

UniClean is included in the paper-exact archived outputs and downstream
analysis. A full from-scratch UniClean rerun requires the external UniClean
repository and is not part of the default benchmark-smoke workflow.

UniClean runtime evidence is stored in:

    analysis/uniclean_external/

See `docs/uniclean_external.md`.

## Release assets

Large generated assets are not committed directly to git. They are distributed
through release assets or archival storage:

- TRACE Stage 4 input snapshot: `results/trace_cluster_replay_all/`
- TRACE Stage 4 paper-exact output pack: `artifacts/trace_stage4_paper_exact.tgz`
- Benchmark full-audit proof logs: `results/logs/stage2_strict_*/`
- Optional source/output archives and checksums.

## Directory map

| Path | Purpose |
|---|---|
| `configs/` | Workflow and method configuration. |
| `data/raw/train/` | Clean and dirty benchmark data. |
| `src/` | Pipeline, cleaning, clustering, and TRACE implementation. |
| `scripts/` | Reviewer-facing and validation scripts. |
| `analysis/paper_generated/` | Paper table/figure replay reports. |
| `analysis/validity_sensitivity/` | Pre-experiment and sensitivity evidence. |
| `analysis/uniclean_external/` | UniClean external runtime evidence. |
| `analysis/release_validation/` | Final validation reports copied from local runs. |
| `docs/` | Reviewer documentation. |

## Deprecated aliases

The old command aliases are retained only for compatibility:

| Deprecated alias | Use instead |
|---|---|
| `paper-replay` | `paper-replay` |
| `benchmark-smoke` | `benchmark-smoke` |
| `benchmark-full-audit` | `benchmark-full-audit` |

Reviewer-facing documentation uses the new workflow names to avoid confusion
with the `Mode impute` baseline cleaner.

