# TRACE Stage 4 reproduction

This document defines the maintained paper-level TRACE reproduction path.

## Maintained entry point

```bash
python scripts/39_run_trace_stage4_paper_repro.py \
  --results-dir results/trace_cluster_replay_all \
  --config configs/trace.yaml \
  --output-dir results/processed/trace/lodo_paper_repro \
  --random-seeds 1000 \
  --seed 20260424 \
  --pack
```

The command does **not** rerun cleaning or clustering by default. It expects the cached-cleaning clustering replay directory supplied through `--results-dir`.

## What is reproduced

The script reproduces the paper's leave-one-dataset-out TRACE validation:

- learn `q_tot` thresholds and cleaner priority order from three original tables;
- hold out the fourth original table for testing;
- repeat for all four held-out folds;
- compare TRACE with blind randomized path-order replay;
- regenerate the two validation figures.

## No-leakage protocol

Only the following TRACE components are learned from training folds:

- the `q_tot` regime thresholds;
- cleaner priority order within each regime.

The following remain fixed and are not tuned on held-out datasets:

- candidate cleaner--clusterer search space;
- score function `H`;
- process-signal rules;
- state transition rules;
- hit-to-95% threshold;
- blind-random replay protocol.

The held-out table does not participate in threshold selection, cleaner priority learning, or stopping-rule design. Full search is used only to define `H_full*` for evaluation and to provide the offline trial ledger.

## Main expected values

The paper-level run should produce values close to:

- TRACE median hit-to-95% progress: `0.1348`;
- Blind-random median hit-to-95% progress: `0.2701`;
- TRACE median AUC retention: `0.9824`;
- Blind-random median AUC retention: `0.9537`.

Small differences can occur if package versions or trial ledgers differ. Use `--strict-metrics` only for paper-exact artifact checks.
