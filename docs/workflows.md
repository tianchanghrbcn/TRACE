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
