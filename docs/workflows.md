# TRACE Reviewer Workflows

TRACE uses reviewer-facing workflow names to avoid confusion with the `Mode impute`
baseline cleaner in the paper.

## Main commands

| Workflow | Command | Purpose |
|---|---|---|
| release-check | `python scripts/trace.py release-check --run-trace-validation` | Validate the reviewer-facing artifact package. |
| paper-replay | `python scripts/trace.py paper-replay` | Rebuild and validate paper tables, figures, and traceability reports. |
| preexp-validity | `python scripts/trace.py preexp-validity --strict` | Replay alpha calibration and validate alpha/seed sensitivity. |
| trace-validation | `python scripts/trace.py trace-validation --paper-exact` | Reproduce TRACE Stage 4 LODO validation with 1000 blind-random replays. |
| benchmark-smoke | `python scripts/trace.py benchmark-smoke --clean` | Run a lightweight from-scratch cleaning-clustering pipeline check. |
| benchmark-full-audit | `python scripts/trace.py benchmark-full-audit` | Validate strict benchmark proof/logs. |

## Expected validation status

The main release check may report `PASS_WITH_WARNINGS` when warnings are
diagnostic-only and have no paper-facing hard failures.

Accepted warning examples:

- raw table equivalence has diagnostic mismatches, but the paper-facing layer has
  zero hard failures;
- paper-output traceability has accepted warnings while figure traceability passes;
- release assets are not committed to git and must be unpacked before full audit.

## TRACE Stage 4

Maintained command:

    python scripts/trace.py trace-validation --paper-exact

Expected metrics:

    TRACE T95 median            = 13.5%
    Blind random T95 median     = 27.0%
    TRACE AUC retention median  = 0.982
    Blind random AUC retention  = 0.954

## UniClean

UniClean is supported through paper-exact archived outputs and runtime evidence:

    analysis/uniclean_external/

Full UniClean deployment is external and is not part of the default smoke run.

## Deprecated aliases

| Deprecated alias | Use instead |
|---|---|
| `paper-replay` | `paper-replay` |
| `benchmark-smoke` | `benchmark-smoke` |
| `benchmark-full-audit` | `benchmark-full-audit` |


