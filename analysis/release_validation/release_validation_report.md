# TRACE Release Validation Report

- Generated at UTC: 2026-05-06T15:05:32.394564+00:00
- Status: PASS

## Static path checks

| Path | Exists | Size bytes |
|---|---:|---:|
| README.md | True | 3723 |
| LICENSE | True | 1262 |
| THIRD_PARTY_NOTICES.md | True | 1301 |
| configs/paper_replay.yaml | True | 610 |
| configs/benchmark_smoke.yaml | True | 487 |
| configs/benchmark_full_audit.yaml | True | 733 |
| configs/trace.yaml | True | 3117 |
| data/README.md | True | 994 |
| data/raw/train/beers/clean.csv | True | 238636 |
| data/raw/train/flights/clean.csv | True | 173153 |
| data/raw/train/hospital/clean.csv | True | 304322 |
| data/raw/train/rayyan/clean.csv | True | 257641 |
| data/pre_experiment/alpha_metrics.csv | True | 1196 |
| analysis/validity_sensitivity/inputs/validity_sensitivity_input_manifest.json | True | 2121 |
| analysis/validity_sensitivity/inputs/analysis_results/beers_summary.xlsx | True | 145622 |
| analysis/validity_sensitivity/inputs/analysis_results/flights_summary.xlsx | True | 148247 |
| analysis/validity_sensitivity/inputs/analysis_results/hospital_summary.xlsx | True | 135112 |
| analysis/validity_sensitivity/inputs/analysis_results/rayyan_summary.xlsx | True | 132236 |
| analysis/validity_sensitivity/seed_sensitivity_summary.json | True | 815 |
| analysis/validity_sensitivity/seed_sensitivity_report.md | True | 1098 |
| src/analysis/trace_replay.py | True | 86944 |
| docs/data_policy.md | True | 1448 |
| docs/hardware_runtime.md | True | 1578 |
| docs/release_packaging.md | True | 1333 |
| docs/terminal_interface.md | True | 992 |
| docs/workflows.md | True | 2171 |
| docs/pre_experiment_validity.md | True | 2390 |
| docs/trace_stage4_repro.md | True | 2908 |
| docs/uniclean_external.md | True | 1282 |
| docs/stage3_strict_validation.md | True | 835 |
| docs/paper_output_traceability.md | True | 892 |
| scripts/trace.py | True | 18366 |
| scripts/00_trace_home.py | True | 6659 |
| scripts/30_replay_trace.py | True | 2712 |
| scripts/36_eval_trace_blind_random.py | True | 26947 |
| scripts/38_lodo_trace_validation.py | True | 19587 |
| scripts/39_run_trace_stage4_paper_repro.py | True | 21830 |
| scripts/50_build_all_paper_figures.py | True | 7990 |
| scripts/51_build_all_paper_tables.py | True | 7202 |
| scripts/45_validate_data_availability.py | True | 2265 |
| scripts/49_validate_trace_stage4_inputs.py | True | 5634 |
| scripts/62_validate_mode_a_paper_replay.py | True | 9746 |
| scripts/63_validate_stage3_strict.py | True | 14838 |
| scripts/81_replay_pre_experiment_validity.py | True | 32003 |
| scripts/90_run_smoke_from_scratch.py | True | 7641 |

## Command checks

| Check | Status | Return code |
|---|---|---:|
| data_availability | PASS | 0 |
| setup_benchmark_smoke | PASS | 0 |
| setup_benchmark_full_audit | PASS | 0 |
| preexp_validity | PASS | 0 |
| paper_replay | PASS | 0 |
| benchmark_smoke | PASS | 0 |
| trace_validation_paper_exact | PASS | 0 |
| strict_reviewer_workflow_validation | PASS | 0 |

## Failures

No hard failures.

## Warnings

No warnings.

## Interpretation

PASS means the selected reviewer-facing release checks passed.
PASS_WITH_WARNINGS is acceptable only when skipped or warning-level checks are explicitly documented.

