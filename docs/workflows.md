# TRACE Reviewer Workflows

TRACE uses reviewer-facing workflow names to avoid confusion with the `Mode impute`
baseline cleaner in the paper.

## Main commands

| Workflow | Command | Purpose |
|---|---|---|
| release-check | `python scripts/trace.py release-check` | Validate the release package. |
| paper-replay | `python scripts/trace.py paper-replay` | Reproduce paper-exact tables, figures, and traceability reports. |
| preexp-validity | `python scripts/trace.py preexp-validity` | Replay alpha calibration and validate alpha/seed sensitivity. |
| trace-validation | `python scripts/trace.py trace-validation --paper-exact` | Reproduce TRACE budget-guidance validation with 1000 blind-random replays. |
| benchmark-smoke | `python scripts/trace.py benchmark-smoke --clean` | Run a lightweight from-scratch benchmark smoke test. |
| benchmark-full-audit | `python scripts/trace.py benchmark-full-audit` | Validate strict benchmark proof/logs. |
| benchmark-full-audit from scratch | `python scripts/trace.py benchmark-full-audit --from-scratch` | Long-running Linux/bash full benchmark audit. |

## Deprecated aliases

The following names are retained only for compatibility:

| Deprecated alias | Use instead |
|---|---|
| mode-a | paper-replay |
| mode-b | benchmark-smoke |
| mode-c | benchmark-full-audit |

Reviewer-facing documentation should not use Mode A/B/C.


## TRACE Stage 4 paper-exact validation

TRACE Stage 4 is closed.

Reviewer-facing command:

    python scripts/trace.py trace-validation --paper-exact

Maintained preflight command:

    python scripts/49_validate_trace_stage4_inputs.py ^
      --results-dir results/trace_cluster_replay_all ^
      --output-dir results/processed/trace/lodo_paper_repro ^
      --strict

Maintained paper-exact command:

    python -u scripts/39_run_trace_stage4_paper_repro.py ^
      --results-dir results/trace_cluster_replay_all ^
      --config configs/trace.yaml ^
      --output-dir results/processed/trace/lodo_paper_repro ^
      --random-seeds 1000 ^
      --seed 20260424 ^
      --rebuild-base ^
      --pack ^
      --log ^
      --quiet

Expected paper-aligned metrics:

    TRACE T95 median            = 13.5%
    Blind random T95 median     = 27.0%
    TRACE AUC retention median  = 0.982
    Blind random AUC retention  = 0.954

See docs/trace_stage4_repro.md for required inputs and outputs.

## UniClean

UniClean is included in the paper-exact archived outputs and downstream paper
analysis. A full from-scratch UniClean rerun requires the external UniClean
repository and is not part of the default benchmark-smoke workflow.

See `docs/uniclean_external.md`.
