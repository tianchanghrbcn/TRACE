# Paper claim traceability

| Paper item | Source file | Script | Expected value |
|---|---|---|---|
| TRACE median `T95` | `results/processed/trace/lodo_paper_repro/lodo_aggregate_summary.json` | `scripts/39_run_trace_stage4_paper_repro.py` | `0.1348` |
| Blind-random median `T95` | same | same | `0.2701` |
| TRACE median AUC retention | same | same | `0.9824` |
| Blind-random median AUC retention | same | same | `0.9537` |
| TRACE validation figures | `results/processed/trace/lodo_paper_repro/figures/` | `scripts/37_plot_trace_validation.py` | two PDF figures |

The full clustering trial ledger is not stored in the repository due to size. The paper-exact bundle contains the summaries, figures, manifest, and logs needed to audit the reported values.
