# TRACE Environments

TRACE uses different environments for different reviewer workflows.

## paper-replay

Paper replay uses the standard Python environment for table/figure scripts and
analysis reports.

## benchmark-smoke

The smoke workflow is lightweight and uses the default TRACE runner environment:

    python scripts/trace.py benchmark-smoke --clean

## benchmark-full-audit

The full benchmark audit may require additional method-specific environments,
including HoloClean services and torch-based cleaner dependencies. The final
artifact validates this path through strict Linux proof logs.

## trace-validation

TRACE Stage 4 uses the standard Python environment and the TRACE replay
snapshot.

## UniClean

UniClean deployment is external. TRACE provides archived outputs and runtime
evidence rather than vendoring the full UniClean environment.


