from __future__ import annotations

"""
Stage 3 builder for paper Table 6.

This module is intended to live at:
    src/paper_artifact/tables/table06_process_abs.py

It adapts the original 6.4.2 process-level table script.

Input
-----
Process-level clustering logs under one of these locations:

    <ctx.input_root>/clustered_data/
    <ctx.input_root>/trace_cluster_replay_all/clustered_data/
    <project_root>/results/trace_cluster_replay_all/clustered_data/
    <project_root>/results/clustered_data/

The expected directory structure is:

    clustered_data/<ALGORITHM>/<cleaner>/clustered_<dataset_id>/
    clustered_data/<ALGORITHM>/<cleaner>/cluster_<dataset_id>/

where each directory contains:
    *_summary.json          for KMEANS/GMM/HC
    *_core_stats.json       for DBSCAN

Output
------
Exactly one CSV file:

    <ctx.output_dir>/table_6/table6_process_abs.csv

No delta table, TeX, detail table, XLSX, or auxiliary outputs are generated.

Core logic preserved from the original script:
1. Mode is the baseline.
2. Absolute process values are collected for Mode and each cleaner.
3. Reported values are medians over dataset ids, algorithms, and cleaners
   according to the original family-specific loops.
4. W_shape is the log-CDF-Wasserstein distance between DBSCAN neighbor
   histograms, with Mode fixed to 0.
"""

import glob
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from src.paper_artifact.io import ArtifactResult, BuildContext


ARTIFACT = {
    "id": "table06_process_abs",
    "paper_id": "Table 6",
    "label": "Table 6: Process-level absolute signatures",
    "description": "Build Table 6 from clustered_data process JSON logs.",
    "enabled": True,
}


# --------------------------------------------------
# 1. Constants
# --------------------------------------------------
CLEANINGS = [
    "baran",
    "holoclean",
    "bigdansing",
    "boostclean",
    "horizon",
    "scared",
    "unified",
    "uniclean",
]
BASELINE = "mode"
ALL_TAGS = [BASELINE] + CLEANINGS

DISPLAY_COLUMNS = [
    "Mode",
    "Baran",
    "HoloClean",
    "BigDansing",
    "BoostClean",
    "Horizon",
    "SCAReD",
    "Unified",
    "UniClean",
]

TAG_TO_DISPLAY = {
    "mode": "Mode",
    "baran": "Baran",
    "holoclean": "HoloClean",
    "bigdansing": "BigDansing",
    "boostclean": "BoostClean",
    "horizon": "Horizon",
    "scared": "SCAReD",
    "unified": "Unified",
    "uniclean": "UniClean",
}

ALG_FAMILIES = {
    "centroid": ["KMEANS", "KMEANSNF", "KMEANSPPS", "GMM"],
    "density": ["DBSCAN"],
    "hierarch": ["HC"],
}

METRIC_ALIASES = {
    "GeoDecay": "Γ_Δ ↓",
    "AUC_Δ": "AUC_Δ ↓",
    "ΔSSE/NLL": "J_obj^(T) ↓",
    "Δn_avg": "E[n_nbr] ↑",
    "ΔW_cdf": "W_shape ↓",
    "Δn_core": "n_core ↑",
    "Δρ_noise": "ρ_noise ↓",
    "Δn_merge": "n_merge ↓",
    "Δh_max": "h_max ↓",
    "ΔR_intra/inter": "R_intra/inter ↓",
}

ROW_ORDER = [
    ("Centroid/model", "ΔSSE/NLL"),
    ("", "AUC_Δ"),
    ("", "GeoDecay"),
    ("Density", "ΔW_cdf"),
    ("", "Δn_avg"),
    ("", "Δn_core"),
    ("", "Δρ_noise"),
    ("Hierarchy", "Δn_merge"),
    ("", "Δh_max"),
    ("", "ΔR_intra/inter"),
]

VALUE_DECIMALS = {
    "ΔSSE/NLL": 3,
    "AUC_Δ": 3,
    "GeoDecay": 3,
    "ΔW_cdf": 3,
    "Δn_avg": 3,
    "Δn_core": 1,
    "Δρ_noise": 3,
    "Δn_merge": 1,
    "Δh_max": 3,
    "ΔR_intra/inter": 3,
}


# --------------------------------------------------
# 2. Metric functions
# --------------------------------------------------
def _geo_decay(js: dict):
    if js is None:
        return None
    for k in ("avg_geo_decay", "ll_geo_decay", "geo_decay"):
        if k in js and js[k] is not None:
            return js[k]
    return None


METRICS: dict[str, dict[str, tuple[Callable[[dict], Any], bool]]] = {
    "centroid": {
        "GeoDecay": (_geo_decay, True),
        "AUC_Δ": (lambda js: js.get("avg_auc_delta") or js.get("auc_ll"), True),
        "ΔSSE/NLL": (lambda js: js.get("best_sse") or js.get("best_nll"), True),
    },
    "density": {
        "Δn_avg": (
            lambda js: sum(i * c for i, c in enumerate(js["neighbor_hist"]))
            / max(sum(js["neighbor_hist"]), 1),
            False,
        ),
        "ΔW_cdf": (lambda js: js["neighbor_hist"], True),
        "Δn_core": (lambda js: js["core_count"], False),
        "Δρ_noise": (lambda js: js["noise_ratio"], True),
    },
    "hierarch": {
        "Δn_merge": (lambda js: js["n_merge_steps"], True),
        "Δh_max": (lambda js: js.get("h_max") or js.get("max_dist"), True),
        "ΔR_intra/inter": (lambda js: js["ratio_intra_inter"], True),
    },
}


# --------------------------------------------------
# 3. IO helpers
# --------------------------------------------------
def _resolve_clustered_roots(ctx: BuildContext) -> list[Path]:
    """Resolve process-level clustered_data roots for paper tables.

    IMPORTANT:
    Table 6 uses the latest paper process snapshot, not the TRACE Stage 4
    replay snapshot. This prevents mixing the frozen TRACE validation input
    with the latest process logs that include UniClean.
    """

    def as_clustered_data_dir(base: Path) -> Path | None:
        base = Path(base).resolve()
        if base.name == "clustered_data" and base.exists() and base.is_dir():
            return base
        nested = base / "clustered_data"
        if nested.exists() and nested.is_dir():
            return nested
        return None

    candidates = [
        # Preferred reviewer/release layout.
        ctx.input_root / "paper_latest_process_snapshot",
        ctx.input_root / "paper_latest_process_snapshot" / "clustered_data",

        # Alternative explicit names, kept for clarity if the release asset is renamed.
        ctx.input_root / "paper_process_snapshot_latest",
        ctx.input_root / "paper_process_snapshot_latest" / "clustered_data",
        ctx.input_root / "latest_process_snapshot",
        ctx.input_root / "latest_process_snapshot" / "clustered_data",

        # Project-root fallbacks.
        ctx.project_root / "results" / "paper_latest_process_snapshot",
        ctx.project_root / "results" / "paper_latest_process_snapshot" / "clustered_data",
        ctx.project_root / "results" / "paper_process_snapshot_latest",
        ctx.project_root / "results" / "paper_process_snapshot_latest" / "clustered_data",
        ctx.project_root / "results" / "latest_process_snapshot",
        ctx.project_root / "results" / "latest_process_snapshot" / "clustered_data",
    ]

    roots: list[Path] = []
    seen: set[Path] = set()

    for candidate in candidates:
        root = as_clustered_data_dir(candidate)
        if root is not None and root not in seen:
            roots.append(root)
            seen.add(root)

    if not roots:
        raise FileNotFoundError(
            "Table 6 requires the latest process snapshot, but no "
            "paper_latest_process_snapshot/clustered_data directory was found. "
            "Do not use results/trace_cluster_replay_all for Table 6, because "
            "that directory is reserved for TRACE Stage 4 replay validation."
        )

    return roots


def first_json(pattern: str):
    files = glob.glob(pattern)
    return files[0] if files else None


def read_json(p: str):
    with open(p, encoding="utf-8") as fp:
        return json.load(fp)


def _base(root: Path, alg: str, cleaning: str, cid: int) -> Path:
    p = root / alg / cleaning / f"clustered_{cid}"
    return p if p.exists() else root / alg / cleaning / f"cluster_{cid}"


def _find_json_for(root: Path, alg: str, cleaning: str, cid: int):
    base = _base(root, alg, cleaning, cid)
    pat = "*_core_stats.json" if alg == "DBSCAN" else "*_summary.json"
    return first_json(str(base / pat))


def collect_pair(roots: list[Path], alg: str, cleaning: str, cid: int):
    """Return (json_clean, json_mode) or (None, None).

    If multiple clustered_data roots are available, prefer a pair from the
    same root. This keeps the original single-root behavior when one root is
    provided, while allowing UniClean-only replay outputs to coexist with the
    original clustered_data root.
    """
    for root in roots:
        p_c = _find_json_for(root, alg, cleaning, cid)
        if not p_c:
            continue

        p_m = _find_json_for(root, alg, BASELINE, cid)
        if not p_m:
            # fallback: mode may live in another root
            for root2 in roots:
                p_m = _find_json_for(root2, alg, BASELINE, cid)
                if p_m:
                    break

        if p_c and p_m:
            return read_json(p_c), read_json(p_m)

    return None, None


def cdf_wasserstein(hist_c, hist_m):
    """CDF–Wasserstein distance, then transformed later as original script."""
    m = max(len(hist_c), len(hist_m))
    h_c = np.pad(hist_c, (0, m - len(hist_c)))
    h_m = np.pad(hist_m, (0, m - len(hist_m)))
    cdf_c = np.cumsum(h_c) / (h_c.sum() + 1e-12)
    cdf_m = np.cumsum(h_m) / (h_m.sum() + 1e-12)
    return wasserstein_distance(range(m), range(m), cdf_c, cdf_m)


def _safe_is_nan(v):
    try:
        return np.isnan(v)
    except TypeError:
        return False


# --------------------------------------------------
# 4. Main aggregation
# --------------------------------------------------
def build_abs_records(roots: list[Path]) -> pd.DataFrame:
    abs_records = []

    for family, algs in ALG_FAMILIES.items():
        for cleaning in CLEANINGS:
            buf_abs = {m: {tag: [] for tag in ALL_TAGS} for m in METRICS[family]}

            for alg in algs:
                for cid in range(60):
                    j_c, j_m = collect_pair(roots, alg, cleaning, cid)
                    if j_c is None:
                        continue

                    for mkey, (getter, _) in METRICS[family].items():
                        v_c, v_m = getter(j_c), getter(j_m)
                        if v_c is None or v_m is None:
                            continue

                        if mkey == "ΔW_cdf":
                            raw = cdf_wasserstein(v_c, v_m)
                            delta = math.log10(1.0 + raw) * 1000.0
                            buf_abs[mkey][cleaning].append(delta)
                            buf_abs[mkey][BASELINE].append(0.0)
                            continue

                        try:
                            if np.isnan(v_c) or np.isnan(v_m):
                                continue
                        except TypeError:
                            pass

                        buf_abs[mkey][cleaning].append(v_c)
                        buf_abs[mkey][BASELINE].append(v_m)

            for mkey, tag2lst in buf_abs.items():
                for tag, lst in tag2lst.items():
                    if lst:
                        abs_records.append(
                            (
                                family,
                                mkey,
                                tag,
                                float(statistics.median(lst)),
                            )
                        )

    if not abs_records:
        raise RuntimeError("No process metrics were collected from clustered_data.")

    abs_df = pd.DataFrame(
        abs_records,
        columns=["family", "metric", "tag", "ABS"],
    )

    pivot = (
        abs_df.pivot_table(
            index=["family", "metric"],
            columns="tag",
            values="ABS",
            aggfunc="first",
        )
        .reindex(columns=ALL_TAGS)
        .sort_index()
    )
    return pivot


def _format_value(metric: str, value) -> str:
    if value is None or pd.isna(value):
        return "–"

    nd = VALUE_DECIMALS.get(metric, 3)
    try:
        v = float(value)
    except Exception:
        return "–"

    if nd == 1:
        return f"{v:.1f}"
    return f"{v:.{nd}f}"


def make_paper_table(abs_pivot: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category, metric in ROW_ORDER:
        row = {
            "Category": category,
            "Metric": METRIC_ALIASES[metric],
        }

        for tag in ALL_TAGS:
            display = TAG_TO_DISPLAY[tag]
            value = np.nan
            try:
                # metric names are unique across families in this table, so search all families.
                matches = abs_pivot.xs(metric, level="metric", drop_level=False)
                if not matches.empty and tag in matches.columns:
                    value = matches.iloc[0][tag]
            except Exception:
                value = np.nan

            row[display] = _format_value(metric, value)

        rows.append(row)

    return pd.DataFrame(rows, columns=["Category", "Metric", *DISPLAY_COLUMNS])


# --------------------------------------------------
# 5. Stage 3 entry point
# --------------------------------------------------
def build(ctx: BuildContext) -> ArtifactResult:
    out_dir = ctx.output_dir / "table_6"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()

    roots = _resolve_clustered_roots(ctx)
    abs_pivot = build_abs_records(roots)
    table = make_paper_table(abs_pivot)

    out_path = out_dir / "table6_process_abs.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Reviewer-facing terminal summary: print the full Table 6 with all columns.
    print("[TABLE6] Generated Table 6 process_abs.csv:")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(table.to_string(index=False))


    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="success",
        outputs=[out_path],
        inputs=roots,
        message=f"Built Table 6 CSV under {out_dir}.",
        metadata={
            "output_subdir": "table_6",
            "clustered_data_roots": [str(p) for p in roots],
            "expected_output_count": 1,
            "actual_output_count": 1,
        },
    )
