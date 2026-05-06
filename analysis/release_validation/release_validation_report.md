# TRACE Release Validation Report

- Generated at UTC: 2026-05-06T18:02:20.253741+00:00
- Status: PASS

## Static path checks

| Path | Exists | Size bytes |
|---|---:|---:|
| README.md | True | 4028 |
| LICENSE | True | 1236 |
| THIRD_PARTY_NOTICES.md | True | 1301 |
| configs/paper_replay.yaml | True | 609 |
| configs/benchmark_smoke.yaml | True | 486 |
| configs/benchmark_full_audit.yaml | True | 732 |
| configs/trace.yaml | True | 3117 |
| data/README.md | True | 994 |
| data/raw/train/beers/clean.csv | True | 236225 |
| data/raw/train/flights/clean.csv | True | 170776 |
| data/raw/train/hospital/clean.csv | True | 303321 |
| data/raw/train/rayyan/clean.csv | True | 256640 |
| data/pre_experiment/alpha_metrics.csv | True | 1175 |
| analysis/validity_sensitivity/inputs/validity_sensitivity_input_manifest.json | True | 2121 |
| analysis/validity_sensitivity/inputs/analysis_results/beers_summary.xlsx | True | 145622 |
| analysis/validity_sensitivity/inputs/analysis_results/flights_summary.xlsx | True | 148247 |
| analysis/validity_sensitivity/inputs/analysis_results/hospital_summary.xlsx | True | 135112 |
| analysis/validity_sensitivity/inputs/analysis_results/rayyan_summary.xlsx | True | 132236 |
| analysis/validity_sensitivity/seed_sensitivity_summary.json | True | 787 |
| analysis/validity_sensitivity/seed_sensitivity_report.md | True | 1071 |
| src/analysis/trace_replay.py | True | 86944 |
| docs/data_policy.md | True | 1446 |
| docs/hardware_runtime.md | True | 1576 |
| docs/release_packaging.md | True | 1330 |
| docs/terminal_interface.md | True | 989 |
| docs/workflows.md | True | 2168 |
| docs/pre_experiment_validity.md | True | 2389 |
| docs/trace_stage4_repro.md | True | 2906 |
| docs/uniclean_external.md | True | 1280 |
| docs/stage3_strict_validation.md | True | 832 |
| docs/paper_output_traceability.md | True | 889 |
| scripts/trace.py | True | 20318 |
| scripts/00_trace_home.py | True | 6658 |
| scripts/30_replay_trace.py | True | 2712 |
| scripts/36_eval_trace_blind_random.py | True | 26947 |
| scripts/38_lodo_trace_validation.py | True | 19587 |
| scripts/39_run_trace_stage4_paper_repro.py | True | 21289 |
| scripts/50_build_all_paper_figures.py | True | 7799 |
| scripts/51_build_all_paper_tables.py | True | 7202 |
| scripts/45_validate_data_availability.py | True | 2265 |
| scripts/49_validate_trace_stage4_inputs.py | True | 5633 |
| scripts/62_validate_mode_a_paper_replay.py | True | 9745 |
| scripts/63_validate_stage3_strict.py | True | 9064 |
| scripts/81_replay_pre_experiment_validity.py | True | 32001 |
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

