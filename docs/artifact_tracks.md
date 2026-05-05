# Artifact tracks

To avoid confusion with the cleaner named `mode`, the artifact uses **tracks** rather than "Mode A/B/C".

## Replay Track

Purpose: reproduce paper tables and figures from archived or generated result summaries. This is the recommended reviewer path.

Examples:

```bash
python scripts/39_run_trace_stage4_paper_repro.py --results-dir results/trace_cluster_replay_all --pack
```

## Smoke Track

Purpose: run a small end-to-end check that the environment, method registry, cleaners, clusterers, and result processors are wired correctly. It is not intended to reproduce the paper numbers.

## Full Track

Purpose: maintainer-side full rerun, including long-running clustering search and optional extension examples. This track can be expensive and is not the first reviewer path.

## Extension Track

Purpose: demonstrate how to onboard a new algorithm or dataset. UniClean and the tax dataset are planned as concrete examples of this track.
