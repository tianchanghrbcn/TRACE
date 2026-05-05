from __future__ import annotations

"""
Stage 3 builder for paper Figure 5.

This module is intended to live at:
    src/paper_artifact/figures/fig05_score_eval_all.py

It reads the four task summary workbooks from ctx.input_root:
    beers_summary.xlsx
    flights_summary.xlsx
    hospital_summary.xlsx
    rayyan_summary.xlsx

and writes exactly 10 PDF files to:
    <ctx.output_dir>/figure_5/

Outputs:
    top10_bar_error_beers.pdf
    top10_bar_error_flights.pdf
    top10_bar_error_hospital.pdf
    top10_bar_error_rayyan.pdf

    mean_sd_scatter_beers.pdf
    mean_sd_scatter_flights.pdf
    mean_sd_scatter_hospital.pdf
    mean_sd_scatter_rayyan.pdf

    top10_bar_error_legend_cluster.pdf
    top10_bar_error_legend_cleaning.pdf

No EPS / CSV / PNG / auxiliary outputs are produced.
"""

import math
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.paper_artifact.io import ArtifactResult, BuildContext


ARTIFACT = {
    "id": "fig05_score_eval_all",
    "paper_id": "Figure 5",
    "label": "Figure 5: Top-10 score and stability panels",
    "description": "Build Figure 5 from the four *_summary.xlsx workbooks under results/.",
    "enabled": True,
}


# ---------------------------------------------------------------------
# 0. Figure settings
# ---------------------------------------------------------------------

TASK_ORDER = ["beers", "flights", "hospital", "rayyan"]
EXPECTED_INPUT_FILES = [f"{task}_summary.xlsx" for task in TASK_ORDER]

FIGSIZE_SINGLE = (1.75, 1.10)

SCORE_Y_MIN = 0.5
SCORE_Y_MAX = 0.9

SD_Y_MIN = 0.0
SD_Y_MAX_MANUAL = None

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "hatch.linewidth": 0.55,
})


# ---------------------------------------------------------------------
# 1. Name normalization
# ---------------------------------------------------------------------

CLEANER_MAP = {
    "mode": "Mode",
    "mode impute": "Mode",
    "mode imputation": "Mode",
    "baran": "Baran",
    "holoclean": "HoloClean",
    "bigdansing": "BigDansing",
    "boostclean": "BoostClean",
    "horizon": "Horizon",
    "scared": "SCAReD",
    "unified": "Unified",
    "uniclean": "UniClean",
    "groundtruth": "GroundTruth",
    "ground truth": "GroundTruth",
}

CLUSTER_MAP = {
    "hc": "HC",
    "hierarchical": "HC",
    "agglomerative": "HC",
    "kmeans": "k-means",
    "k-means": "k-means",
    "k_means": "k-means",
    "kmeanspps": "k-means-PPS",
    "k-meanspps": "k-means-PPS",
    "k-means-pps": "k-means-PPS",
    "kmeans-pps": "k-means-PPS",
    "kmeans_pps": "k-means-PPS",
    "kmeansnf": "k-means-NF",
    "k-meansnf": "k-means-NF",
    "k-means-nf": "k-means-NF",
    "kmeans-nf": "k-means-NF",
    "kmeans_nf": "k-means-NF",
    "gmm": "GMM",
    "gmm-em": "GMM",
    "dbscan": "DBSCAN",
}


def canonical_cleaner(x: object) -> str:
    s = str(x).strip()
    return CLEANER_MAP.get(s.lower(), s)


def canonical_cluster(x: object) -> str:
    s = str(x).strip()
    return CLUSTER_MAP.get(s.lower(), s)


def safe_filename(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s.strip("_")


def task_sort_key(task: str) -> Tuple[int, str]:
    low = str(task).lower()
    if low in TASK_ORDER:
        return (TASK_ORDER.index(low), low)
    return (len(TASK_ORDER), low)


# ---------------------------------------------------------------------
# 2. Visual encoding
# ---------------------------------------------------------------------

CLEANER_COLORS = {
    "Mode":       "#74A9CF",
    "Baran":      "#F6B75E",
    "HoloClean":  "#7BC87C",
    "BigDansing": "#E88D8D",
    "BoostClean": "#B6A0D8",
    "Horizon":    "#C49A7B",
    "SCAReD":     "#E7A3C5",
    "Unified":    "#9E9E9E",
}
DEFAULT_CLEANER_COLOR = "#C8C8C8"

CLUSTER_HATCHES = {
    "HC": "",
    "k-means": "////",
    "k-means-PPS": "\\\\\\\\",
    "k-means-NF": "++",
    "GMM": "xx",
    "DBSCAN": "....",
}
DEFAULT_HATCH = "--"

CLEANER_ORDER = [
    "Mode", "Baran", "HoloClean", "BigDansing",
    "BoostClean", "Horizon", "SCAReD", "Unified", "UniClean"
]

CLUSTER_ORDER = [
    "HC", "k-means", "k-means-PPS", "k-means-NF", "GMM", "DBSCAN"
]


def cleaner_color(cleaner: str) -> str:
    return CLEANER_COLORS.get(cleaner, DEFAULT_CLEANER_COLOR)


def cluster_hatch(cluster: str) -> str:
    return CLUSTER_HATCHES.get(cluster, DEFAULT_HATCH)


# ---------------------------------------------------------------------
# 3. IO and input loading
# ---------------------------------------------------------------------

def _read_one_workbook(path: Path) -> pd.DataFrame:
    data = pd.read_excel(path, sheet_name=None)
    if isinstance(data, dict):
        frames = []
        for sheet_name, df in data.items():
            if df is not None and not df.empty:
                tmp = df.copy()
                tmp["_sheet_name"] = sheet_name
                frames.append(tmp)
        if not frames:
            raise ValueError(f"Workbook contains no non-empty sheets: {path}")
        return pd.concat(frames, ignore_index=True)
    return data


def load_summary_workbooks(input_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    required_cols = {"task_name", "cleaning_method", "cluster_method", "Combined Score"}
    inputs: list[Path] = []
    frames: list[pd.DataFrame] = []

    for name in EXPECTED_INPUT_FILES:
        path = input_root / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required workbook: {path}. "
                f"Expected these files under {input_root}: {EXPECTED_INPUT_FILES}"
            )

        df = _read_one_workbook(path)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

        df = df.copy()
        df["Combined Score"] = (
            df["Combined Score"]
            .astype(str)
            .str.replace(r"[^\d.\-eE+]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )
        df = df.dropna(subset=["Combined Score"])
        if df.empty:
            raise ValueError(f"{path.name} has no valid Combined Score values.")

        df["task_name"] = df["task_name"].astype(str).str.strip().str.lower()
        df["cleaner_label"] = df["cleaning_method"].map(canonical_cleaner)
        df["cluster_label"] = df["cluster_method"].map(canonical_cluster)

        frames.append(df)
        inputs.append(path)

    return pd.concat(frames, ignore_index=True), inputs


def clean_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()


# ---------------------------------------------------------------------
# 4. Core statistics
# ---------------------------------------------------------------------

def build_score_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    score_stats = (
        df_all
        .groupby(["task_name", "cleaner_label", "cluster_label"], dropna=False)
        .agg(
            abs_mean=("Combined Score", "mean"),
            abs_sd=("Combined Score", "std"),
            n=("Combined Score", "count"),
        )
        .reset_index()
    )
    score_stats["abs_sd"] = score_stats["abs_sd"].fillna(0.0)
    score_stats["rank_label"] = score_stats["cleaner_label"] + " + " + score_stats["cluster_label"]
    return score_stats


def build_gt_best_mean(score_stats: pd.DataFrame) -> pd.Series:
    gt_rows = score_stats[score_stats["cleaner_label"] == "GroundTruth"].copy()
    if gt_rows.empty:
        return pd.Series(dtype=float)
    return (
        gt_rows
        .loc[gt_rows.groupby("task_name")["abs_mean"].idxmax()]
        .set_index("task_name")["abs_mean"]
    )


def build_ref_sd_by_task(df_all: pd.DataFrame) -> pd.Series:
    gt_score_by_task_cluster = (
        df_all[df_all["cleaner_label"] == "GroundTruth"]
        .groupby(["task_name", "cluster_label"], dropna=False)["Combined Score"]
        .mean()
        .rename("GT_score")
    )

    if gt_score_by_task_cluster.empty:
        return pd.Series(dtype=float)

    df_rel = (
        df_all
        .merge(gt_score_by_task_cluster, on=["task_name", "cluster_label"], how="inner")
        .assign(rel_score=lambda d: 100.0 * d["Combined Score"] / d["GT_score"])
    )

    ref_stats = (
        df_rel
        .groupby(["task_name", "cleaner_label", "cluster_label"], dropna=False)
        .agg(rel_mean=("rel_score", "mean"), sd=("Combined Score", "std"))
        .reset_index()
    )
    return ref_stats.groupby("task_name")["sd"].median()


def select_top10_by_task(score_stats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    top10_by_task: dict[str, pd.DataFrame] = {}
    for task in TASK_ORDER:
        sub = score_stats[score_stats["task_name"] == task].copy()
        top10 = (
            sub[sub["cleaner_label"] != "GroundTruth"]
            .sort_values("abs_mean", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        if top10.empty:
            raise ValueError(f"No non-GroundTruth Top-10 combinations found for task={task}")
        top10_by_task[task] = top10
    return top10_by_task


def nice_ceiling(x: float, step: float = 0.03, min_value: float = 0.12) -> float:
    if not np.isfinite(x) or x <= 0:
        return min_value
    return max(min_value, math.ceil(x / step) * step)


def compute_shared_sd_ymax(top10_by_task: dict[str, pd.DataFrame], ref_sd_by_task: pd.Series) -> float:
    if SD_Y_MAX_MANUAL is not None:
        return float(SD_Y_MAX_MANUAL)

    all_sd_values = []
    for task, top10 in top10_by_task.items():
        all_sd_values.extend(top10["abs_sd"].dropna().astype(float).tolist())
        if task in ref_sd_by_task.index and np.isfinite(ref_sd_by_task.loc[task]):
            all_sd_values.append(float(ref_sd_by_task.loc[task]))

    global_max_sd = max(all_sd_values) if all_sd_values else 0.12
    return nice_ceiling(global_max_sd * 1.10, step=0.03, min_value=0.12)


# ---------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------

def style_common_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.88", linewidth=0.38)

    for spine in ax.spines.values():
        spine.set_linewidth(0.58)
        spine.set_color("0.18")

    ax.tick_params(axis="y", width=0.50, length=2.0, pad=1.0)


def plot_top10_mean_chart(top10: pd.DataFrame, gt_ref: Optional[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    x = np.arange(len(top10))
    scores = top10["abs_mean"].to_numpy(dtype=float)
    bar_heights = np.maximum(scores - SCORE_Y_MIN, 0.0)

    for i, row in top10.iterrows():
        ax.bar(
            x[i],
            bar_heights[i],
            bottom=SCORE_Y_MIN,
            width=1.0,
            align="center",
            color=cleaner_color(row["cleaner_label"]),
            edgecolor="0.16",
            linewidth=0.30,
            hatch=cluster_hatch(row["cluster_label"]),
            zorder=3,
        )

    if gt_ref is not None and np.isfinite(gt_ref):
        gt_ref = float(gt_ref)
        if SCORE_Y_MIN <= gt_ref <= SCORE_Y_MAX:
            ax.axhline(gt_ref, linestyle=(0, (4, 2)), linewidth=1.05, color="0.05", zorder=4)
            gt_text = f"GT={gt_ref:.3f}"
        elif gt_ref > SCORE_Y_MAX:
            gt_text = f"GT={gt_ref:.3f}>"
        else:
            gt_text = f"GT={gt_ref:.3f}<"

        ax.text(
            0.97,
            0.94,
            gt_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.9,
            color="0.05",
        )

    ax.set_ylim(SCORE_Y_MIN, SCORE_Y_MAX)
    yticks = np.linspace(SCORE_Y_MIN, SCORE_Y_MAX, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.1f}" for v in yticks], fontsize=7.1)
    ax.set_xlim(-0.5, len(top10) - 0.5)
    ax.margins(x=0)

    style_common_axis(ax)
    fig.subplots_adjust(left=0.225, right=0.995, bottom=0.03, top=0.985)
    return fig


def plot_sd_stability_chart(top10: pd.DataFrame, ref_sd: Optional[float], sd_y_max: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    x = np.arange(len(top10))
    sd_values = top10["abs_sd"].to_numpy(dtype=float)

    for i, row in top10.iterrows():
        ax.bar(
            x[i],
            sd_values[i],
            bottom=0.0,
            width=1.0,
            align="center",
            color=cleaner_color(row["cleaner_label"]),
            edgecolor="0.16",
            linewidth=0.30,
            hatch=cluster_hatch(row["cluster_label"]),
            zorder=3,
        )

    if ref_sd is not None and np.isfinite(ref_sd):
        ref_sd = float(ref_sd)
        if SD_Y_MIN <= ref_sd <= sd_y_max:
            ax.axhline(ref_sd, linestyle=(0, (4, 2)), linewidth=1.00, color="0.05", zorder=4)
            ref_text = f"Ref SD={ref_sd:.3f}"
        elif ref_sd > sd_y_max:
            ref_text = f"Ref SD={ref_sd:.3f}>"
        else:
            ref_text = f"Ref SD={ref_sd:.3f}<"

        ax.text(
            0.97,
            0.06,
            ref_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.9,
            color="0.05",
        )

    ax.set_ylim(SD_Y_MIN, sd_y_max)
    ax.invert_yaxis()

    yticks = np.linspace(SD_Y_MIN, sd_y_max, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.2f}" for v in yticks], fontsize=7.1)
    ax.set_xlim(-0.5, len(top10) - 0.5)
    ax.margins(x=0)

    style_common_axis(ax)
    fig.subplots_adjust(left=0.245, right=0.995, bottom=0.03, top=0.985)
    return fig


def save_pdf(fig: plt.Figure, path: Path, pad_inches: float = 0.005) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return path


def save_cluster_legend(out_dir: Path, used_clusters: set[str]) -> Path:
    cluster_order = [c for c in CLUSTER_ORDER if c in used_clusters]
    extras = sorted([c for c in used_clusters if c not in cluster_order])
    cluster_order += extras

    handles = [
        Patch(
            facecolor="white",
            edgecolor="0.16",
            linewidth=0.45,
            hatch=cluster_hatch(name),
            label=name,
        )
        for name in cluster_order
    ]
    handles.append(
        Line2D([0], [0], color="0.05", linewidth=1.00, linestyle=(0, (4, 2)), label="GT/ref.")
    )

    fig, ax = plt.subplots(figsize=(5.8, 0.32))
    ax.axis("off")
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=7.8,
        handlelength=1.45,
        handletextpad=0.35,
        columnspacing=0.80,
        borderaxespad=0.0,
    )
    return save_pdf(fig, out_dir / "top10_bar_error_legend_cluster.pdf", pad_inches=0.002)


def save_cleaning_legend(out_dir: Path) -> Path:
    handles = [
        Patch(
            facecolor=cleaner_color(name),
            edgecolor="0.16",
            linewidth=0.35,
            label=name,
        )
        for name in CLEANER_ORDER
    ]

    fig, ax = plt.subplots(figsize=(7.2, 0.32))
    ax.axis("off")
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=7.8,
        handlelength=1.15,
        handletextpad=0.32,
        columnspacing=0.72,
        borderaxespad=0.0,
    )
    return save_pdf(fig, out_dir / "top10_bar_error_legend_cleaning.pdf", pad_inches=0.002)


# ---------------------------------------------------------------------
# 6. Stage 3 entry point
# ---------------------------------------------------------------------

def build(ctx: BuildContext) -> ArtifactResult:
    input_root = Path(ctx.input_root)
    out_dir = Path(ctx.output_dir) / "figure_5"

    if not ctx.dry_run:
        clean_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    df_all, inputs = load_summary_workbooks(input_root)
    score_stats = build_score_stats(df_all)
    gt_best_mean = build_gt_best_mean(score_stats)
    ref_sd_by_task = build_ref_sd_by_task(df_all)
    top10_by_task = select_top10_by_task(score_stats)
    sd_y_max = compute_shared_sd_ymax(top10_by_task, ref_sd_by_task)

    outputs: list[Path] = []
    used_clusters: set[str] = set()

    for task in TASK_ORDER:
        top10 = top10_by_task[task]
        used_clusters.update(top10["cluster_label"].dropna().tolist())

        gt_ref = float(gt_best_mean.loc[task]) if task in gt_best_mean.index else None
        fig_top = plot_top10_mean_chart(top10=top10, gt_ref=gt_ref)
        outputs.append(save_pdf(fig_top, out_dir / f"top10_bar_error_{safe_filename(task)}.pdf"))

        ref_sd = float(ref_sd_by_task.loc[task]) if task in ref_sd_by_task.index else None
        fig_sd = plot_sd_stability_chart(top10=top10, ref_sd=ref_sd, sd_y_max=sd_y_max)
        outputs.append(save_pdf(fig_sd, out_dir / f"mean_sd_scatter_{safe_filename(task)}.pdf"))

    outputs.append(save_cluster_legend(out_dir, used_clusters))
    outputs.append(save_cleaning_legend(out_dir))

    expected_names = {
        *(f"top10_bar_error_{task}.pdf" for task in TASK_ORDER),
        *(f"mean_sd_scatter_{task}.pdf" for task in TASK_ORDER),
        "top10_bar_error_legend_cluster.pdf",
        "top10_bar_error_legend_cleaning.pdf",
    }
    actual_names = {p.name for p in outputs}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise RuntimeError(f"Output file mismatch. Missing={missing}; extra={extra}")

    metadata = {
        "output_subdir": "figure_5",
        "expected_output_count": 10,
        "actual_output_count": len(outputs),
        "tasks": TASK_ORDER,
    }

    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="success",
        outputs=outputs,
        inputs=inputs,
        message=f"Built Figure 5 with {len(outputs)} PDF files under {out_dir}.",
        metadata=metadata,
    )
