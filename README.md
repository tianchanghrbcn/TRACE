# TRACE

Reproducibility package for:

**TRACE: An Empirical Study of Data Cleaning Effects on Unsupervised Clustering**

This repository supports three reproducibility modes:

- **Mode A**: reproduce the paper tables, TRACE replay results, and figures from archived raw results.
- **Mode B**: run a small smoke test from scratch.
- **Mode C**: run the full experiment from scratch.

The recommended review path is Mode A. The full from-scratch experiment is provided for auditability and can be computationally expensive. Wall-clock runtime is not a claim of the paper.

## Setup check

```bash
python scripts/00_setup_check.py --config configs/mode_b_smoke.yaml --strict
python scripts/00_setup_check.py --config configs/mode_c_full.yaml --check-all-data --strict
python scripts/00_setup_check.py --config configs/mode_a_release.yaml
