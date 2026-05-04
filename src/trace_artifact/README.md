# TRACE Artifact Utilities

This directory is the reviewer-facing map for TRACE Stage 4 validation.

The maintained paper-level entry point is:

```bash
python -u scripts/39_run_trace_stage4_paper_repro.py \
  --results-dir results/trace_cluster_replay_all \
  --config configs/trace.yaml \
  --output-dir results/processed/trace/lodo_paper_repro \
  --random-seeds 1000 \
  --seed 20260424 \
  --skip-existing \
  --pack \
  --log \
  --quiet
```

This command reproduces the paper-level leave-one-dataset-out TRACE validation and writes a compact final JSON summary. The terminal output is intentionally reviewer-facing: it reports high-level progress and the final metric summary, but suppresses fold-level and trial-level intermediate logs. Compact logs are saved under the output directory and included in the packed artifact.

## What TRACE Stage 4 validates

TRACE is validated as a budget-guidance rule rather than as a full replacement for exhaustive search. The validation asks whether empirical rules learned from the cleaning--clustering study can guide the search toward high-quality cleaner--clusterer paths earlier than unguided enumeration.

The paper-level validation uses leave-one-dataset-out replay. In each fold:

1. Three source datasets are used as training folds.
2. Only the input-dependent TRACE components are learned from the training folds:
   - the `q_tot` thresholds,
   - the cleaner priority order within each error regime.
3. The held-out dataset is used only for replay evaluation.
4. The held-out dataset does not participate in threshold selection, priority selection, or stopping-rule selection.

The remaining TRACE rules are fixed by the paper design:

- candidate cleaner--clusterer search space,
- scoring function `H`,
- process-signal gates,
- state-transition rules,
- the 95% hit threshold,
- the blind randomized path-order replay protocol.

This design is intended to reduce hindsight bias compared with validating TRACE only on the same datasets from which its rules were summarized.

## Main TRACE files

### Configuration

- `configs/trace.yaml`

  TRACE replay configuration. It contains the validation threshold, output paths, cleaner grouping, replay behavior, and trial-ledger loading options.

### Core replay engine

- `src/analysis/trace_replay.py`

  Core TRACE replay engine. It loads saved pipeline outputs, constructs the full-search reference, builds trial-level path ledgers, replays the TRACE policy, computes dataset-level metrics, and writes replay artifacts.

### Paper-level entry point

- `scripts/39_run_trace_stage4_paper_repro.py`

  Reviewer-facing Stage 4 reproduction entry point. This is the recommended command for reproducing the TRACE validation reported in the paper. It orchestrates leave-one-dataset-out validation, writes the manifest and report, and optionally packs a paper-exact artifact bundle.

### Backend scripts

- `scripts/38_lodo_trace_validation.py`

  Leave-one-dataset-out validation backend. It learns input-dependent TRACE thresholds and cleaner priorities from training folds and tests on the held-out fold.

- `scripts/36_eval_trace_blind_random.py`

  Blind randomized path-order baseline. It randomly permutes cleaner--clusterer paths while preserving trial order within each path.

- `scripts/37_plot_trace_validation.py`

  Figure generation for TRACE validation. This is kept separate from the paper-level replay command so that running the validation and regenerating figures remain cleanly separated.

- `scripts/30_replay_trace.py`

  Low-level TRACE replay backend. It can replay TRACE directly from saved pipeline logs.

- `scripts/34_rerun_clustering_from_cleaned_results.py`

  Maintainer-side cached-cleaning clustering replay. It reuses cleaned outputs from exhaustive baseline runs and reruns clustering with trial-level logging. This script is not the default reviewer entry point because it is substantially more expensive.

## Paper-level outputs

The paper-level command writes outputs under:

```text
results/processed/trace/lodo_paper_repro/
```

Key files include:

```text
trace_stage4_manifest.json
trace_stage4_repro_report.md
lodo_aggregate_summary.json
lodo_folds.csv
lodo_blind_random_dataset_summary.csv
lodo_trace_dataset_summary.csv
logs/
```

When `--pack` is used, the command also writes:

```text
artifacts/trace_stage4_paper_exact.tgz
```

This packed artifact contains the compact run logs, manifest, report, and paper-exact summary outputs. Large clustering logs and raw `clustered_data/` outputs are not included in the repository.

## Expected paper-level metrics

The expected paper-level leave-one-dataset-out metrics are:

```text
n_datasets = 60

TRACE median hit-to-95 progress        = 0.13476388888888888
Blind random median hit-to-95 progress = 0.2701335656213705

TRACE median AUC retention             = 0.982375574743322
Blind random median AUC retention      = 0.9537331473633225
```

The paper reports these as:

```text
TRACE T95:        13.5%
Blind random T95: 27.0%

TRACE AUC retention:        0.982
Blind random AUC retention: 0.954
```

The final JSON summary printed by `scripts/39_run_trace_stage4_paper_repro.py` should include:

```json
{
  "metric_checks_all_within_tolerance": true,
  "aggregate": {
    "n_datasets": 60,
    "median_trace_hit95_progress": 0.13476388888888888,
    "median_blind_random_hit95_progress": 0.2701335656213705,
    "median_trace_auc_retention": 0.982375574743322,
    "median_blind_random_auc_retention": 0.9537331473633225
  }
}
```

## Regenerating figures

Figure generation is intentionally separated from the paper-level replay command.

After the LODO outputs exist, figures can be regenerated with:

```bash
python scripts/37_plot_trace_validation.py \
  --lodo-dir results/processed/trace/lodo_paper_repro
```

The two paper-facing TRACE validation figures are:

```text
lodo_hit95_progress_ecdf.pdf
lodo_auc_retention_ecdf.pdf
```

Depending on the selected output directory, the script may also write alias filenames:

```text
hit95_progress_ecdf.pdf
auc_retention_ecdf.pdf
```

## Artifact and logging policy

Large intermediate outputs are not committed to the repository. In particular, the following are maintainer-side artifacts:

```text
results/trace_cluster_replay_all/
results/processed/trace/*/trace_replay_trials.csv
clustered_data/
cleaned_data/
raw Optuna trial logs
```

The reviewer-facing artifact should instead include:

```text
trace_stage4_manifest.json
trace_stage4_repro_report.md
lodo_aggregate_summary.json
lodo_folds.csv
lodo_blind_random_dataset_summary.csv
compact logs
paper figures
```

This keeps the repository lightweight while preserving enough information to audit the paper-level TRACE validation.

## Recommended reviewer command

For a compact paper-level reproduction:

```bash
python -u scripts/39_run_trace_stage4_paper_repro.py \
  --results-dir results/trace_cluster_replay_all \
  --config configs/trace.yaml \
  --output-dir results/processed/trace/lodo_paper_repro \
  --random-seeds 1000 \
  --seed 20260424 \
  --skip-existing \
  --pack \
  --log \
  --quiet
```

The command prints high-level progress and the final summary only. The corresponding compact log is saved under:

```text
results/processed/trace/lodo_paper_repro/logs/
```

The packed artifact is saved under:

```text
artifacts/trace_stage4_paper_exact.tgz
```

## Notes for maintainers

`TRACE` Stage 4 is now sealed around the paper-level LODO validation. Development-only subset tests, exploratory baselines, and old plotting variants should be kept outside the main reviewer path or moved to `legacy/`.

The maintained path is:

```text
scripts/39_run_trace_stage4_paper_repro.py
        ↓
scripts/38_lodo_trace_validation.py
        ↓
scripts/36_eval_trace_blind_random.py
        ↓
scripts/37_plot_trace_validation.py
```

The low-level replay engine remains:

```text
src/analysis/trace_replay.py
```
