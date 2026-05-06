# TRACE Terminal Interface

The reviewer-facing interface is `scripts/trace.py`.

## Recommended commands

    python scripts/trace.py release-check --run-trace-validation
    python scripts/trace.py paper-replay
    python scripts/trace.py preexp-validity --strict
    python scripts/trace.py trace-validation --paper-exact
    python scripts/trace.py benchmark-smoke --clean
    python scripts/trace.py benchmark-full-audit

## Interactive menu

    python scripts/00_trace_home.py
    python scripts/00_trace_home.py --interactive

## Workflow names

- `paper-replay`: paper table/figure replay and traceability.
- `benchmark-smoke`: lightweight from-scratch benchmark smoke run.
- `benchmark-full-audit`: strict benchmark proof validation.
- `trace-validation`: TRACE Stage 4 paper-exact validation.
- `preexp-validity`: alpha calibration and validity sensitivity checks.
- `release-check`: final artifact validation gate.

Deprecated aliases are retained for compatibility only.


