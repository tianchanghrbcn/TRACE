# TRACE Setup Scripts

The setup scripts are compatibility helpers. Reviewer-facing workflows should use:

    python scripts/trace.py release-check --run-trace-validation
    python scripts/trace.py benchmark-smoke --clean
    python scripts/trace.py benchmark-full-audit

## Workflow setup

- paper-replay: standard Python analysis environment.
- benchmark-smoke: lightweight smoke runner environment.
- benchmark-full-audit: full benchmark environment with method-specific
  dependencies.
- trace-validation: standard Python environment plus TRACE Stage 4 snapshot.

HoloClean and UniClean may require external services or repositories for full
from-scratch reruns. The reviewer smoke path does not require deploying them.


