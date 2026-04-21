# Estimated Runtime Progress

TRACE includes an optional estimated progress monitor for long-running validation.

The monitor is not an exact algorithmic progress bar. It estimates progress using maintainer-observed reference runtimes and the current validation summary.

Run once:

    python scripts/95_monitor_repro_progress.py --log-dir results/logs --reference configs/runtime_reference.yaml

Watch mode:

    python scripts/95_monitor_repro_progress.py --log-dir results/logs --reference configs/runtime_reference.yaml --watch

Interpretation:

- Estimated progress is approximate.
- Long-running cleaners such as Baran and Unified may exceed the reference runtime.
- Low log activity does not necessarily mean failure.
- The authoritative validation result is still `RESULT` and `summary.tsv`.

