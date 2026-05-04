# Stage 3 Paper Artifact Rebuild

This document describes the reviewer-facing Stage 3 rebuild layer for paper figures and tables.

## Entry points

```bash
python scripts/50_build_all_paper_figures.py --list
python scripts/51_build_all_paper_tables.py --list
```

To build all registered figures:

```bash
python scripts/50_build_all_paper_figures.py \
  --input-root results \
  --output-dir results/paper_artifact/figures
```

To build all registered tables:

```bash
python scripts/51_build_all_paper_tables.py \
  --input-root results \
  --output-dir results/paper_artifact/tables
```

Optional copy targets can be supplied with `--paper-fig-dir` or `--paper-table-dir`.

## Builder layout

Individual figure builders live in:

```text
src/paper_artifact/figures/
```

Individual table builders live in:

```text
src/paper_artifact/tables/
```

Each builder should expose:

```python
ARTIFACT = {
    "id": "figXX_short_name",
    "paper_id": "Figure X",
    "label": "Short label",
    "description": "What this builder generates.",
}

def build(ctx):
    ...
```

The `build(ctx)` function should return `ArtifactResult` or a compatible dictionary:

```python
{
  "artifact_id": "figXX_short_name",
  "status": "success",
  "outputs": ["..."],
  "inputs": ["..."],
  "message": "...",
  "metadata": {...}
}
```

## Manifests

The aggregate scripts write:

```text
results/paper_artifact/figures/figures_manifest.json
results/paper_artifact/tables/tables_manifest.json
```

The manifests map paper outputs back to builder scripts and source inputs. They are intended to support claim traceability.

## Current status

This scaffold intentionally contains no paper figure/table builders yet. Builders will be added one by one as the paper-specific plotting and table-generation code is refactored into this structure.
