# TRACE Result Schema

Stage 3 converts raw pipeline outputs into canonical tables used by paper-table and figure replay.

The schema follows the four-level TRACE analysis framework:

1. Data-level change.
2. Process-level change.
3. Result-level change.
4. Hyperparameter-level change.

The first Stage 3 implementation builds these canonical tables:

- `trials.csv`
- `cleaning_metrics.csv`
- `result_metrics.csv`
- `best_configs.csv`
- `process_metrics.csv`
- `parameter_shifts.csv`
- `run_manifest.json`

The schema is intentionally conservative. It accepts partial smoke-test outputs and can later ingest full archived Mode A results without changing downstream table and figure scripts.

