#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate EDR--gain coupling curves for Finding 4.2-3.

This script keeps the original IO and figure format of the CEGR script:

Input:
    <project_root>/results/analysis_results/<task>_summary.xlsx

Output:
    <project_root>/task_progress/figures/6.4.3graph/

For compatibility with existing LaTeX paths, output filenames are kept as:
    CEGR_5pct_<task>.pdf / .png
    CEGR_5pct_legend.pdf / .png
    CEGR_5pct_<task>.xlsx
    EDR_CEGR_stats.xlsx
    CEGR_turning_region_stats.xlsx

However, the plotted quantity is no longer CEGR. The new quantity measures
how strongly the data-level EDR scale remains aligned with downstream gain:

    DeltaH(c) = H*(c) - H*(Mode)

Within each task_name x dataset_id x error_rate_bin x cluster_method group,
we rank deployable non-baseline, non-oracle cleaners by EDR, split them into
high-EDR and low-EDR halves, and compute:

    EDR-GainGap =
        median DeltaH(high-EDR cleaners) - median DeltaH(low-EDR cleaners)

A positive value means that higher-EDR cleaners also tend to yield higher
downstream gains in that bin. Values near zero or below zero mean that the
EDR-to-gain relation becomes weak, unstable, or reversed.

The figure still uses one line per clustering algorithm and one panel per task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.paper_artifact.io import ArtifactResult, BuildContext


# ----------------------------------------------------------------------
# Figure style.
# ----------------------------------------------------------------------
matplotlib.rcdefaults()
matplotlib.rcParams.update(
    {
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.2,
        "font.size": 7.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    }
)


TASK_ORDER = ["beers", "flights", "hospital", "rayyan"]

CLUSTER_METHOD_ORDER = [
    "DBSCAN",
    "GMM",
    "HC",
    "KMEANS",
    "KMEANSNF",
    "KMEANSPPS",
]

CLUSTER_METHOD_LABELS = {
    "DBSCAN": "DBSCAN",
    "GMM": "GMM",
    "HC": "HC",
    "KMEANS": "KMEANS",
    "KMEANSNF": "KMEANSNF",
    "KMEANSPPS": "KMEANSPPS",
}

CLUSTER_METHOD_MARKERS = {
    "DBSCAN": "o",
    "GMM": "X",
    "HC": "s",
    "KMEANS": "P",
    "KMEANSNF": "D",
    "KMEANSPPS": "*",
}

# Keep the same panel geometry as the original script.
PANEL_FIGSIZE = (1.785, 2.405)
LEGEND_FIGSIZE = (5.8, 0.32)

# Main plotting metric.
# Options:
#   "gain_gap"  : median DeltaH(high-EDR) - median DeltaH(low-EDR)
#   "spearman"  : Spearman correlation between EDR and DeltaH
#   "win_rate"  : fraction of groups where high-EDR cleaners outperform low-EDR cleaners
#
# Recommended default for the paper: "gain_gap".
PLOT_METRIC = "gain_gap"

METRIC_CONFIG = {
    "gain_gap": {
        "agg_col": "edr_gain_gap_median",
        "y_min": -0.06,
        "y_max": 0.12,
        "y_step": 0.03,
        "y_ticks": [-0.06, -0.03, 0.00, 0.03, 0.06, 0.09, 0.12],
    },
    "spearman": {
        "agg_col": "edr_gain_spearman_median",
        "y_min": -1.00,
        "y_max": 1.00,
        "y_step": 0.50,
    },
    "win_rate": {
        "agg_col": "high_edr_win_rate",
        "y_min": 0.00,
        "y_max": 1.00,
        "y_step": 0.25,
    },
}

BASELINE_METHOD_NAMES = {
    "mode",
    "mode impute",
    "mode_impute",
    "mode-impute",
}

ORACLE_METHOD_NAMES = {
    "groundtruth",
    "ground_truth",
    "gt",
    "oracle",
}


# Stage 3 IO overrides. They are set by build(ctx).
_STAGE3_INPUT_DIR: Path | None = None
_STAGE3_OUTPUT_DIR: Path | None = None



ARTIFACT = {
    "id": "fig03_edr_gain_coupling",
    "paper_id": "Figure 3",
    "label": "Figure 3: EDR--gain coupling curves",
    "description": "Build Figure 3 from the four *_summary.xlsx workbooks under results/.",
    "enabled": True,
}



# ----------------------------------------------------------------------
# Numeric helpers.
# ----------------------------------------------------------------------
def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division with zero-denominator protection."""
    den2 = den.replace(0, np.nan)
    return num / den2


def _fit_line(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    """Fit a simple linear trend."""
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)

    if len(x_arr) == 0:
        return np.array([])

    if len(np.unique(x_arr)) < 2:
        return np.full_like(y_arr, np.nanmean(y_arr), dtype=float)

    coef = np.polyfit(x_arr, y_arr, 1)
    return np.polyval(coef, x_arr)


def _find_simple_turn_bin(
    curve_df: pd.DataFrame,
    x_col: str = "error_rate_bin",
    y_col: str = "metric_median",
) -> dict[str, float]:
    """Find a descriptive peak and first-drop bin.

    This is not a statistical breakpoint estimator. It is used only to
    summarize whether the plotted EDR--gain coupling begins to weaken after
    its strongest point.
    """
    data = (
        curve_df[[x_col, y_col]]
        .dropna()
        .sort_values(x_col)
        .drop_duplicates(subset=[x_col], keep="last")
    )

    if data.empty:
        return {
            "peak_bin": np.nan,
            "peak_value": np.nan,
            "first_drop_after_peak_bin": np.nan,
            "n_points": 0,
        }

    xs = data[x_col].to_numpy(dtype=float)
    ys = data[y_col].to_numpy(dtype=float)

    peak_idx = int(np.nanargmax(ys))
    peak_bin = float(xs[peak_idx])
    peak_value = float(ys[peak_idx])

    first_drop = np.nan
    for i in range(peak_idx + 1, len(xs)):
        if ys[i] < ys[i - 1]:
            first_drop = float(xs[i])
            break

    return {
        "peak_bin": peak_bin,
        "peak_value": peak_value,
        "first_drop_after_peak_bin": first_drop,
        "n_points": int(len(xs)),
    }


# ----------------------------------------------------------------------
# Path and input helpers.
# ----------------------------------------------------------------------
def _project_root() -> Path:
    """Return project root from this script path."""
    return Path(__file__).resolve().parents[3]


def _data_dir(root: Path) -> Path:
    """Return the summary-workbook input directory."""
    if _STAGE3_INPUT_DIR is not None:
        return _STAGE3_INPUT_DIR

    candidates = [
        root / "results" / "analysis_results",
        Path.cwd() / "results" / "analysis_results",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Cannot find results/analysis_results.")


def _output_dir(root: Path) -> Path:
    """Return the figure/statistics output directory.

    Keep the original output folder unchanged unless Stage 3 overrides it.
    """
    if _STAGE3_OUTPUT_DIR is not None:
        out = _STAGE3_OUTPUT_DIR
    else:
        out = root / "task_progress" / "figures" / "6.4.3graph"

    out.mkdir(parents=True, exist_ok=True)
    return out


def _cluster_family(cluster_method: str) -> str:
    """Map clustering method to a coarse family."""
    method = str(cluster_method).upper()

    if method in {"KMEANS", "KMEANSNF", "KMEANSPPS", "GMM"}:
        return "centroid"
    if method == "DBSCAN":
        return "density"
    if method == "HC":
        return "hierarch"
    return "unknown"


def _normalize_error_rate_to_percent(error_rate: pd.Series) -> pd.Series:
    """Normalize error_rate to percentage scale if needed.

    If a workbook stores 0.05, 0.10, ..., 0.30, convert to 5, 10, ..., 30.
    If it already stores 5, 10, ..., 30, keep it unchanged.
    """
    er = _to_numeric(error_rate)
    max_val = er.max(skipna=True)

    if pd.notna(max_val) and max_val <= 1.0:
        return er * 100.0

    return er


def _load_summaries(data_dir: Path) -> pd.DataFrame:
    """Load all dataset summary workbooks."""
    frames = []

    for task in TASK_ORDER:
        path = data_dir / f"{task}_summary.xlsx"

        if not path.exists():
            print(f"[WARN] Missing {path}; skip {task}.")
            continue

        tmp = pd.read_excel(path)
        tmp["task_name"] = task
        frames.append(tmp)

    if not frames:
        raise RuntimeError("No summary workbook was loaded.")

    df = pd.concat(frames, ignore_index=True)

    required = [
        "dataset_id",
        "cleaning_method",
        "cluster_method",
        "error_rate",
        "EDR",
        "Combined Score",
        "Silhouette Score",
        "Davies-Bouldin Score",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "dataset_id",
        "error_rate",
        "EDR",
        "Combined Score",
        "Silhouette Score",
        "Davies-Bouldin Score",
        "Comb_relative",
        "Sil_relative",
        "DB_relative",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    df["dataset_id"] = df["dataset_id"].astype("Int64")
    df["cleaning_method"] = df["cleaning_method"].astype(str).str.lower().str.strip()
    df["cluster_method"] = df["cluster_method"].astype(str).str.upper().str.strip()
    df["family"] = df["cluster_method"].map(_cluster_family)

    df["error_rate_pct"] = _normalize_error_rate_to_percent(df["error_rate"])
    df["error_rate_bin"] = ((df["error_rate_pct"] / 5).round() * 5).astype(int)

    return df.sort_values(
        ["task_name", "dataset_id", "error_rate_bin", "cluster_method", "cleaning_method"]
    )


# ----------------------------------------------------------------------
# Baseline and group construction.
# ----------------------------------------------------------------------
def _is_baseline_method(name: str) -> bool:
    return str(name).lower().strip() in BASELINE_METHOD_NAMES


def _is_oracle_method(name: str) -> bool:
    return str(name).lower().strip() in ORACLE_METHOD_NAMES


def _attach_mode_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the Mode baseline score under the same task/dataset/error/cluster key.

    Primary route:
        merge exact Mode rows.

    Fallback route:
        if Mode rows are absent but Comb_relative is available,
        estimate Mode score by Combined Score / Comb_relative.
    """
    keys = ["task_name", "dataset_id", "error_rate_bin", "cluster_method"]

    mode_mask = df["cleaning_method"].map(_is_baseline_method)
    mode_rows = df.loc[mode_mask, keys + ["Combined Score"]].copy()

    out = df.copy()

    if not mode_rows.empty:
        mode_base = (
            mode_rows.groupby(keys, as_index=False, observed=False)["Combined Score"]
            .median()
            .rename(columns={"Combined Score": "mode_combined"})
        )
        out = out.merge(mode_base, on=keys, how="left")
    else:
        out["mode_combined"] = np.nan

    if "Comb_relative" in out.columns:
        estimated = _safe_divide(out["Combined Score"], out["Comb_relative"])
        out["mode_combined"] = out["mode_combined"].fillna(estimated)

    return out


def _edr_gain_relation_for_group(group: pd.DataFrame) -> dict | None:
    """Compute EDR--gain relation within one dirty instance and one clusterer.

    Input group key:
        task_name x dataset_id x error_rate_bin x cluster_method

    Rows:
        deployable non-baseline, non-oracle cleaners.

    Main output:
        edr_gain_gap = median DeltaH(high-EDR cleaners)
                       - median DeltaH(low-EDR cleaners)

    We use a rank split rather than an absolute threshold to avoid depending on
    dataset-specific EDR scales.
    """
    g = group.dropna(subset=["EDR", "delta_H"]).copy()

    if len(g) < 4:
        return None

    g = g.sort_values(["EDR", "cleaning_method"], ascending=[True, True]).reset_index(drop=True)
    n = len(g)
    half = n // 2

    low = g.iloc[:half].copy()
    high = g.iloc[-half:].copy()

    if low.empty or high.empty:
        return None

    low_delta = float(low["delta_H"].median())
    high_delta = float(high["delta_H"].median())
    low_edr = float(low["EDR"].median())
    high_edr = float(high["EDR"].median())

    if g["EDR"].nunique(dropna=True) > 1 and g["delta_H"].nunique(dropna=True) > 1:
        spearman = float(g["EDR"].corr(g["delta_H"], method="spearman"))
    else:
        spearman = np.nan

    return {
        "n_cleaners": int(n),
        "low_edr_median": low_edr,
        "high_edr_median": high_edr,
        "edr_span": high_edr - low_edr,
        "low_deltaH_median": low_delta,
        "high_deltaH_median": high_delta,
        "edr_gain_gap": high_delta - low_delta,
        "high_edr_wins": float(high_delta > low_delta),
        "edr_gain_spearman": spearman,
    }


def _compute_task_edr_gain_relation(
    task: str,
    sub: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute raw and aggregated EDR--gain relation curves for one task."""
    with_base = _attach_mode_baseline(sub)

    candidate_mask = (
        ~with_base["cleaning_method"].map(_is_baseline_method)
        & ~with_base["cleaning_method"].map(_is_oracle_method)
    )

    candidates = with_base.loc[candidate_mask].copy()
    candidates = candidates.dropna(subset=["Combined Score", "mode_combined", "EDR"])
    candidates["delta_H"] = candidates["Combined Score"] - candidates["mode_combined"]

    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    keys = ["task_name", "dataset_id", "error_rate_bin", "cluster_method"]
    rows = []

    for key_vals, group in candidates.groupby(keys, observed=False):
        stats = _edr_gain_relation_for_group(group)

        if stats is None:
            continue

        row = dict(zip(keys, key_vals))
        row.update(stats)
        rows.append(row)

    raw = pd.DataFrame(rows)

    if raw.empty:
        return raw, pd.DataFrame()

    agg = (
        raw.groupby(["error_rate_bin", "cluster_method"], observed=False, as_index=False)
        .agg(
            edr_gain_gap_median=("edr_gain_gap", "median"),
            edr_gain_gap_q25=("edr_gain_gap", lambda s: float(np.nanquantile(s, 0.25))),
            edr_gain_gap_q75=("edr_gain_gap", lambda s: float(np.nanquantile(s, 0.75))),
            high_edr_win_rate=("high_edr_wins", "mean"),
            edr_gain_spearman_median=("edr_gain_spearman", "median"),
            edr_span_median=("edr_span", "median"),
            high_deltaH_median=("high_deltaH_median", "median"),
            low_deltaH_median=("low_deltaH_median", "median"),
            n_groups=("edr_gain_gap", "size"),
        )
        .sort_values(["error_rate_bin", "cluster_method"])
    )

    agg["task_name"] = task
    return raw, agg


# ----------------------------------------------------------------------
# Descriptive statistics.
# ----------------------------------------------------------------------
def _compute_task_relation_stats(task: str, raw: pd.DataFrame, agg: pd.DataFrame) -> dict:
    """Compute lightweight descriptive statistics for one task."""
    if raw.empty or agg.empty:
        return {
            "task": task,
            "low_positive_num": 0,
            "low_positive_den": 0,
            "low_positive_pct": np.nan,
            "post_nonpositive_num": 0,
            "post_nonpositive_den": 0,
            "post_nonpositive_pct": np.nan,
            "post_below_low_baseline_num": 0,
            "post_below_low_baseline_den": 0,
            "post_below_low_baseline_pct": np.nan,
            "pooled_peak_bin": np.nan,
            "pooled_peak_value": np.nan,
            "pooled_first_drop_after_peak_bin": np.nan,
        }

    metric_col = METRIC_CONFIG[PLOT_METRIC]["agg_col"]

    low_curve = agg[agg["error_rate_bin"] <= 15].copy()
    post_curve = agg[agg["error_rate_bin"] >= 20].copy()

    low_positive_num = int((low_curve[metric_col] > 0).sum())
    low_positive_den = int(low_curve[metric_col].notna().sum())
    low_positive_pct = (
        100.0 * low_positive_num / low_positive_den if low_positive_den else np.nan
    )

    post_nonpositive_num = int((post_curve[metric_col] <= 0).sum())
    post_nonpositive_den = int(post_curve[metric_col].notna().sum())
    post_nonpositive_pct = (
        100.0 * post_nonpositive_num / post_nonpositive_den if post_nonpositive_den else np.nan
    )

    low_baseline = (
        low_curve.groupby("cluster_method", as_index=False, observed=False)[metric_col]
        .median()
        .rename(columns={metric_col: "low_noise_relation_baseline"})
    )

    post_compare = post_curve.merge(low_baseline, on="cluster_method", how="left")
    post_compare["below_low_noise_baseline"] = (
        post_compare[metric_col] < post_compare["low_noise_relation_baseline"]
    )

    post_below_num = int(post_compare["below_low_noise_baseline"].sum())
    post_below_den = int(post_compare["below_low_noise_baseline"].notna().sum())
    post_below_pct = 100.0 * post_below_num / post_below_den if post_below_den else np.nan

    pooled_curve = (
        agg.groupby("error_rate_bin", as_index=False, observed=False)[metric_col]
        .median()
        .rename(columns={metric_col: "metric_median"})
        .sort_values("error_rate_bin")
    )
    turn = _find_simple_turn_bin(pooled_curve, y_col="metric_median")

    return {
        "task": task,
        "metric": PLOT_METRIC,
        "low_positive_num": low_positive_num,
        "low_positive_den": low_positive_den,
        "low_positive_pct": low_positive_pct,
        "post_nonpositive_num": post_nonpositive_num,
        "post_nonpositive_den": post_nonpositive_den,
        "post_nonpositive_pct": post_nonpositive_pct,
        "post_below_low_baseline_num": post_below_num,
        "post_below_low_baseline_den": post_below_den,
        "post_below_low_baseline_pct": post_below_pct,
        "pooled_peak_bin": turn["peak_bin"],
        "pooled_peak_value": turn["peak_value"],
        "pooled_first_drop_after_peak_bin": turn["first_drop_after_peak_bin"],
    }


def _save_relation_region_stats(
    raw_all_list: list[pd.DataFrame],
    curve_all_list: list[pd.DataFrame],
    out_dir: Path,
) -> dict[str, float]:
    """Save lightweight EDR--gain coupling statistics.

    The output filename is kept compatible with the old CEGR script.
    """
    if not raw_all_list or not curve_all_list:
        return {
            "turn_low": 10.0,
            "turn_high": 20.0,
            "turn_median": 15.0,
        }

    raw_all = pd.concat(raw_all_list, ignore_index=True)
    curve_all = pd.concat(curve_all_list, ignore_index=True)

    metric_col = METRIC_CONFIG[PLOT_METRIC]["agg_col"]

    task_curve = (
        curve_all.groupby(["task_name", "error_rate_bin"], as_index=False, observed=False)[
            metric_col
        ]
        .median()
        .rename(columns={metric_col: "metric_median"})
        .sort_values(["task_name", "error_rate_bin"])
    )

    overall_curve = (
        task_curve.groupby("error_rate_bin", as_index=False, observed=False)["metric_median"]
        .median()
        .sort_values("error_rate_bin")
    )

    turn = _find_simple_turn_bin(overall_curve, y_col="metric_median")

    low_curve = task_curve[task_curve["error_rate_bin"] <= 15].copy()
    post_curve = task_curve[task_curve["error_rate_bin"] >= 20].copy()

    low_positive_num = int((low_curve["metric_median"] > 0).sum())
    low_positive_den = int(low_curve["metric_median"].notna().sum())
    low_positive_pct = (
        100.0 * low_positive_num / low_positive_den if low_positive_den else np.nan
    )

    post_nonpositive_num = int((post_curve["metric_median"] <= 0).sum())
    post_nonpositive_den = int(post_curve["metric_median"].notna().sum())
    post_nonpositive_pct = (
        100.0 * post_nonpositive_num / post_nonpositive_den
        if post_nonpositive_den
        else np.nan
    )

    low_baseline = (
        low_curve.groupby("task_name", as_index=False, observed=False)["metric_median"]
        .median()
        .rename(columns={"metric_median": "low_noise_relation_baseline"})
    )
    post_compare = post_curve.merge(low_baseline, on="task_name", how="left")
    post_compare["below_low_noise_baseline"] = (
        post_compare["metric_median"] < post_compare["low_noise_relation_baseline"]
    )

    post_below_num = int(post_compare["below_low_noise_baseline"].sum())
    post_below_den = int(post_compare["below_low_noise_baseline"].notna().sum())
    post_below_pct = (
        100.0 * post_below_num / post_below_den if post_below_den else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "quantity": "EDR_gain_coupling",
                "plot_metric": PLOT_METRIC,
                "low_noise_positive_num": low_positive_num,
                "low_noise_positive_den": low_positive_den,
                "low_noise_positive_pct": low_positive_pct,
                "post_turn_nonpositive_num": post_nonpositive_num,
                "post_turn_nonpositive_den": post_nonpositive_den,
                "post_turn_nonpositive_pct": post_nonpositive_pct,
                "post_turn_below_low_baseline_num": post_below_num,
                "post_turn_below_low_baseline_den": post_below_den,
                "post_turn_below_low_baseline_pct": post_below_pct,
                "overall_peak_bin": turn["peak_bin"],
                "overall_peak_value": turn["peak_value"],
                "overall_first_drop_after_peak_bin": turn["first_drop_after_peak_bin"],
                "recommended_turn_region_low": 10.0,
                "recommended_turn_region_high": 20.0,
                "recommended_turn_median": 15.0,
            }
        ]
    )

    output_path = out_dir / "CEGR_turning_region_stats.xlsx"

    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="key_stats", index=False)
        task_curve.to_excel(writer, sheet_name="task_bin_curve", index=False)
        overall_curve.to_excel(writer, sheet_name="overall_curve", index=False)
        curve_all.to_excel(writer, sheet_name="cluster_curve", index=False)
        raw_all.to_excel(writer, sheet_name="raw_group_rows", index=False)
        post_compare.to_excel(writer, sheet_name="post_vs_low", index=False)

    print("[INFO] EDR--gain coupling descriptive statistics:")
    print(summary.to_string(index=False, float_format="%.4f"))
    print(f"[INFO] Saved {output_path}")

    return {
        "turn_low": 10.0,
        "turn_high": 20.0,
        "turn_median": 15.0,
    }


# ----------------------------------------------------------------------
# Plotting helpers.
# ----------------------------------------------------------------------
def _color_map() -> dict[str, str]:
    """Use Matplotlib default color cycle with stable method ordering."""
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = [f"C{i}" for i in range(len(CLUSTER_METHOD_ORDER))]

    return {
        method: colors[i % len(colors)]
        for i, method in enumerate(CLUSTER_METHOD_ORDER)
    }


def _style_axis(ax: plt.Axes) -> None:
    """Apply compact paper-figure styling."""
    cfg = METRIC_CONFIG[PLOT_METRIC]

    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Zero reference: positive means higher EDR cleaners outperform lower EDR cleaners.
    # This is intentionally subtle and does not change the original panel format.
    ax.axhline(0.0, color="0.45", linewidth=0.7, linestyle="--", zorder=0)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=3.0,
        width=0.8,
        pad=1.5,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=1.8,
        width=0.6,
        pad=1.5,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.margins(x=0.03, y=0.06)

    ax.set_ylim(cfg["y_min"], cfg["y_max"])
    if "y_ticks" in cfg:
        ax.set_yticks(cfg["y_ticks"])
    else:
        ax.set_yticks(
            np.arange(
                cfg["y_min"],
                cfg["y_max"] + cfg["y_step"] / 2.0,
                cfg["y_step"],
            )
        )
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def _plot_relation_lines(ax: plt.Axes, agg: pd.DataFrame) -> tuple[list, list[str]]:
    """Plot EDR--gain relation lines and return legend handles."""
    colors = _color_map()
    handles = []
    labels = []

    cfg = METRIC_CONFIG[PLOT_METRIC]
    metric_col = cfg["agg_col"]

    x_ticks = sorted(agg["error_rate_bin"].dropna().unique().tolist())
    if x_ticks:
        ax.set_xticks(x_ticks)

    for method in CLUSTER_METHOD_ORDER:
        sub = agg[agg["cluster_method"].astype(str).str.upper() == method].copy()

        if sub.empty or metric_col not in sub.columns:
            continue

        sub = sub.sort_values("error_rate_bin").copy()

        # Plot-level clipping only. Raw values remain in the exported tables.
        sub["metric_plot"] = sub[metric_col].clip(
            lower=cfg["y_min"],
            upper=cfg["y_max"],
        )

        label = CLUSTER_METHOD_LABELS.get(method, method)

        line, = ax.plot(
            sub["error_rate_bin"],
            sub["metric_plot"],
            marker=CLUSTER_METHOD_MARKERS.get(method, "o"),
            color=colors.get(method),
            markerfacecolor=colors.get(method),
            markeredgecolor=colors.get(method),
            linewidth=1.2,
            markersize=4.2,
            label=label,
        )

        handles.append(line)
        labels.append(label)

    _style_axis(ax)
    return handles, labels


def _save_one_relation_figure(task: str, agg: pd.DataFrame, out_dir: Path) -> tuple[list, list[str]]:
    """Save one dataset-specific panel with old CEGR-compatible filename."""
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    handles, labels = _plot_relation_lines(ax, agg)

    fig.subplots_adjust(left=0.145, right=0.995, bottom=0.135, top=0.995)

    # Keep old filenames so existing LaTeX includegraphics paths remain unchanged.
    pdf_path = out_dir / f"CEGR_5pct_{task}.pdf"

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    print(f"[INFO] Saved {pdf_path}")

    return handles, labels


def _save_legend_only(handles: list, labels: list[str], out_dir: Path) -> None:
    """Save a separate horizontal legend for LaTeX composition."""
    if not handles or not labels:
        return

    fig = plt.figure(figsize=LEGEND_FIGSIZE)
    legend = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

    for text in legend.get_texts():
        text.set_fontsize(7.0)

    # Keep old filenames.
    pdf_path = out_dir / "CEGR_5pct_legend.pdf"

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    print(f"[INFO] Saved {pdf_path}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main() -> None:
    if PLOT_METRIC not in METRIC_CONFIG:
        raise ValueError(
            f"Unknown PLOT_METRIC={PLOT_METRIC!r}. "
            f"Choose from {list(METRIC_CONFIG)}."
        )

    root = _project_root()
    input_dir = _data_dir(root)
    out_dir = _output_dir(root)

    df = _load_summaries(input_dir)

    stats_rows = []
    raw_all_list = []
    curve_all_list = []

    legend_handles = []
    legend_labels = []

    for task in TASK_ORDER:
        sub = df[df["task_name"] == task].copy()

        if sub.empty:
            print(f"[WARN] No rows for {task}.")
            continue

        raw_relation, agg = _compute_task_edr_gain_relation(task, sub)

        if raw_relation.empty or agg.empty:
            print(f"[WARN] No EDR--gain relation rows for {task}.")
            continue

        raw_all_list.append(raw_relation.copy())
        curve_all_list.append(agg.copy())

        stats_rows.append(_compute_task_relation_stats(task, raw_relation, agg))

        handles, labels = _save_one_relation_figure(task, agg, out_dir)
        if not legend_handles:
            legend_handles = handles
            legend_labels = labels

    _save_legend_only(legend_handles, legend_labels, out_dir)

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)



def _clean_stage3_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.is_file():
            child.unlink()


def build(ctx: BuildContext) -> ArtifactResult:
    """Stage 3 wrapper.

    It only overrides input/output directories and restricts outputs to five PDFs.
    The computation and plotting code above is otherwise the original script.
    """
    global _STAGE3_INPUT_DIR, _STAGE3_OUTPUT_DIR

    out_dir = ctx.output_dir / "figure_3"
    if not ctx.dry_run:
        _clean_stage3_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    _STAGE3_INPUT_DIR = ctx.input_root
    _STAGE3_OUTPUT_DIR = out_dir

    try:
        main()
    finally:
        _STAGE3_INPUT_DIR = None
        _STAGE3_OUTPUT_DIR = None

    outputs = [
        out_dir / "CEGR_5pct_beers.pdf",
        out_dir / "CEGR_5pct_flights.pdf",
        out_dir / "CEGR_5pct_hospital.pdf",
        out_dir / "CEGR_5pct_rayyan.pdf",
        out_dir / "CEGR_5pct_legend.pdf",
    ]

    missing = [p for p in outputs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 3 outputs: " + ", ".join(str(p) for p in missing))

    inputs = [ctx.input_root / f"{task}_summary.xlsx" for task in TASK_ORDER]

    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="success",
        outputs=outputs,
        inputs=inputs,
        message=f"Built Figure 3 with {len(outputs)} PDF files under {out_dir}.",
        metadata={
            "output_subdir": "figure_3",
            "expected_output_count": 5,
            "actual_output_count": len(outputs),
            "tasks": TASK_ORDER,
            "plot_metric": PLOT_METRIC,
        },
    )



if __name__ == "__main__":
    main()
    
    