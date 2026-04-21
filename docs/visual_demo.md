# Visual Demo

The TRACE visual demo is a small synthetic example used to explain how cleaning can change the data geometry and the resulting clustering behavior.

The demo is explanatory. It is not used as an experimental result.

Run:

    python scripts/40_make_visual_demo.py --output-data-dir results/visual_demo --output-figure-dir figures/visual_demo

Generated data:

- `results/visual_demo/clean_points.csv`
- `results/visual_demo/dirty_points.csv`
- `results/visual_demo/statistical_impute_points.csv`
- `results/visual_demo/constraint_repair_points.csv`
- `results/visual_demo/context_repair_points.csv`

Generated figures:

- `figures/visual_demo/data_rewrite_demo.png`
- `figures/visual_demo/data_rewrite_demo.pdf`
- `figures/visual_demo/clustering_demo.png`
- `figures/visual_demo/clustering_demo.pdf`

The figure labels are in English and do not depend on legacy hard-coded paths.

