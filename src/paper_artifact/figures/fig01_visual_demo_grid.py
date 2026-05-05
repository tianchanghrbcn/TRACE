from __future__ import annotations

"""
Stage 3 builder for paper Figure 1.

This module is intended to live at:
    src/paper_artifact/figures/fig01_visual_demo_grid.py

It adapts the visual-demo grid plotting script into the Stage 3 paper-artifact
framework.

Canonical input layout
----------------------
The builder resolves a visual-demo root from the following candidates:

    <ctx.input_root>/visual_demo/customer_segments/
    <ctx.input_root>/visual_demo/
    <project_root>/visual_demo/
    <project_root>/                         # legacy fallback

The demo root must contain:

    clean_withseg.csv
    demo_dirty/
    demo_results/
        eigenvectors.json
        cleaned_data/{mode,holoclean,baran}/repaired_2.csv
        clustered_data/HC/{mode,holoclean,baran}/clustered_2/...

Output
------
Exactly one PDF file:

    <ctx.output_dir>/figure_1/figure1_grid.pdf

No PNG or auxiliary outputs are generated.
"""

from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, AutoMinorLocator
from matplotlib.patches import Ellipse, Polygon
from matplotlib.lines import Line2D

from src.paper_artifact.io import ArtifactResult, BuildContext


ARTIFACT = {
    "id": "fig01_visual_demo_grid",
    "paper_id": "Figure 1",
    "label": "Figure 1: Visual cleaning-clustering demo grid",
    "description": "Build the visual demo grid from the visual_demo customer-segment demo inputs.",
    "enabled": True,
}


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Glyph")

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# ============================== Constants ==============================
BASE_DIR_NAME = "demo_results"
DIRTY_DIR_NAME = "demo_dirty"
EIGENV_NAME = "eigenvectors.json"
CLEAN_FILE_NAME = "clean_withseg.csv"
TARGET_ID = 2

COLUMNS = [
    ("Clean", "clean"),
    ("Dirty", "dirty"),
    ("Statistical", "mode"),
    ("Constraint-based", "holoclean"),
    ("Context-aware", "baran"),
]
SUBLABELS = ["(a)", "(b)", "(c)", "(d)", "(e)"]

SEG_COLOR = {"A": "#1f77b4", "B": "#ffbf00", "C": "#d62728",
             "D": "#17becf", "E": "#7f7f7f"}
SEG_ORDER = ["A", "B", "C", "D", "E"]

YSCALE_K = 1000.0
POINT_SIZE = 70
TICK_SIZE = 22
SUBLABEL_FONTSIZE = 26
LEGEND_FONTSIZE = 26

RING_MISSING_COLOR = "#d62728"
RING_LW = 1.8
RING_S_DELTA = 200

ANOM_MARKER_COLOR = "#555555"
ANOM_MARKER_SIZE = 130
ANOM_MARKER_LW = 2.2

ERR_BOX_COLOR = "black"
ERR_BOX_LW = 1.8
ERR_BOX_LINESTYLE = (0, (5, 3))

ELLIPSE_LW = 2.0
ELLIPSE_NSTD = 2.0


# ============================== Path helpers ==============================
def resolve_demo_root(ctx: BuildContext) -> Path:
    candidates = [
        ctx.input_root / "visual_demo" / "customer_segments",
        ctx.input_root / "visual_demo",
        ctx.project_root / "visual_demo",
        ctx.project_root,  # legacy fallback for the original script layout
    ]

    for cand in candidates:
        if (
            (cand / CLEAN_FILE_NAME).exists()
            and (cand / BASE_DIR_NAME / EIGENV_NAME).exists()
            and (cand / DIRTY_DIR_NAME).exists()
        ):
            return cand

    raise FileNotFoundError(
        "Cannot find visual-demo inputs. Expected clean_withseg.csv, "
        "demo_dirty/, and demo_results/eigenvectors.json under one of:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def clean_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()


# ============================== Utilities ==============================
def tag_err(df: pd.DataFrame, ref: pd.DataFrame):
    miss = df.age.isna() | df.income.isna()
    anom = (~miss) & ((df.age != ref.age) | (df.income != ref.income))
    return np.select([miss, anom], ["missing", "anomaly"], "normal")


def read_eigen(path: Path) -> dict[int, dict[str, str]]:
    mp: dict[int, dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for x in data:
        dataset_id = int(x["dataset_id"])
        csv_file = x.get("csv_file") or x.get("csv") or x.get("dirty_csv")
        if not csv_file:
            raise KeyError(f"Missing csv_file/csv/dirty_csv for dataset_id={dataset_id} in {path}")
        mp[dataset_id] = {"csv": str(csv_file)}
    return mp


def safe_load_cleaned(demo_root: Path, algo: str, seg_map: pd.Series) -> pd.DataFrame:
    fp = demo_root / BASE_DIR_NAME / "cleaned_data" / algo / f"repaired_{TARGET_ID}.csv"
    if not fp.is_file():
        return pd.DataFrame(columns=["ID", "age", "income", "segment"])
    df = pd.read_csv(fp)[["ID", "age", "income"]].merge(seg_map, on="ID", how="left")
    for c in ("age", "income"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_cluster_labels(demo_root: Path, algo: str) -> pd.DataFrame:
    root_clustered = demo_root / BASE_DIR_NAME / "clustered_data" / "HC"
    folder = root_clustered / algo / f"clustered_{TARGET_ID}"
    stem = f"repaired_{TARGET_ID}"
    for state in ("cleaned", "raw"):
        csv_path = folder / f"{stem}_{state}_clusters.csv"
        if csv_path.exists():
            dfc = pd.read_csv(csv_path, usecols=lambda c: c.lower() in ("orig_index", "cluster"))
            if {"orig_index", "cluster"}.issubset(dfc.columns):
                return dfc[["orig_index", "cluster"]].copy()
    return pd.DataFrame(columns=["orig_index", "cluster"])


def hc_cluster_dirty(df: pd.DataFrame, k: int = 5) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    x = df[["age", "income"]].to_numpy(dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        m = np.isnan(col)
        if m.any():
            col[m] = np.nanmedian(col)
    x_std = (x - x.mean(0)) / (x.std(0) + 1e-9)
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(x_std)


def add_cluster_hull(ax, x, y, color, linewidth=2.0, pad_frac=0.04,
                     linestyle="-", alpha=0.9):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return

    if n == 1:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        r = 0.02 * max(abs(x1 - x0), abs(y1 - y0))
        ax.add_patch(Ellipse((float(x[0]), float(y[0])),
                             width=2 * r, height=2 * r, angle=0,
                             fill=False, edgecolor=color,
                             linewidth=linewidth, linestyle=linestyle,
                             alpha=alpha, zorder=3))
        return

    if n == 2:
        p0 = np.array([x[0], y[0]])
        p1 = np.array([x[1], y[1]])
        v = p1 - p0
        length = np.linalg.norm(v)
        if length < 1e-12:
            return
        u = v / length
        n_perp = np.array([-u[1], u[0]])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ref = max(abs(x1 - x0), abs(y1 - y0))
        t = max(0.08 * length, 0.015 * ref)
        verts = np.array([p0 - t * n_perp, p1 - t * n_perp,
                          p1 + t * n_perp, p0 + t * n_perp])
        ax.add_patch(Polygon(verts, closed=True, fill=False,
                             edgecolor=color, linewidth=linewidth,
                             linestyle=linestyle, alpha=alpha, zorder=3))
        return

    try:
        from scipy.spatial import ConvexHull
        pts = np.column_stack([x, y])
        if len(np.unique(pts, axis=0)) < 3:
            return
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
    except Exception:
        return

    if pad_frac > 0:
        cx, cy = hull_pts.mean(axis=0)
        directions = hull_pts - np.array([cx, cy])
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        unit = directions / norms
        eq_r = float(np.mean(norms))
        hull_pts = hull_pts + unit * (pad_frac * eq_r)

    ax.add_patch(Polygon(hull_pts, closed=True, fill=False,
                         edgecolor=color, linewidth=linewidth,
                         linestyle=linestyle, alpha=alpha, zorder=3))


# ============================== Top row ==============================
def draw_top_panel(ax, df_plot, x0, y0,
                   x_left_abs, x_right_abs, y_bottom_abs, y_top_abs,
                   x_out_rel, y_out_rel, x_thr_rel, y_thr_rel):
    if df_plot.empty:
        ax.text(0.5, 0.5, "missing", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        return

    s = POINT_SIZE
    norm = df_plot.err_type == "normal"
    for seg in SEG_ORDER:
        m = norm & df_plot.segment.astype(str).str.startswith(seg, na=False)
        if m.any():
            ax.scatter(df_plot.loc[m, "age"].to_numpy() - x0,
                       (df_plot.loc[m, "income"].to_numpy() - y0) / YSCALE_K,
                       c=SEG_COLOR[seg], s=s, lw=0, alpha=0.85, zorder=2)

    miss_mask = df_plot.err_type == "missing"
    miss_pts = []
    if miss_mask.any():
        mxs, mys = [], []
        for _, r in df_plot.loc[miss_mask].iterrows():
            if pd.isna(r.age) and pd.notna(r.income):
                xr, yr, d = x_out_rel, (r.income - y0) / YSCALE_K, "x"
            elif pd.isna(r.income) and pd.notna(r.age):
                xr, yr, d = (r.age - x0), y_out_rel / YSCALE_K, "y"
            elif pd.isna(r.age) and pd.isna(r.income):
                xr, yr, d = x_out_rel, y_out_rel / YSCALE_K, "both"
            else:
                continue
            mxs.append(xr)
            mys.append(yr)
            miss_pts.append((xr, yr, d))
        if mxs:
            ax.scatter(np.asarray(mxs), np.asarray(mys), marker="o",
                       facecolors="none", edgecolors=RING_MISSING_COLOR,
                       s=s + RING_S_DELTA, linewidths=RING_LW, zorder=4)

    anom_mask = df_plot.err_type == "anomaly"
    anom_pts = []
    if anom_mask.any():
        axs, ays = [], []
        for _, r in df_plot.loc[anom_mask].iterrows():
            xr = (r.age - x0) if pd.notna(r.age) else x_out_rel
            yr = ((r.income - y0) / YSCALE_K) if pd.notna(r.income) else (y_out_rel / YSCALE_K)
            x_over = pd.notna(r.age) and (r.age > (x0 + x_thr_rel))
            y_over = pd.notna(r.income) and (r.income > (y0 + y_thr_rel))
            if x_over:
                xr = min(xr, x_thr_rel * 0.99)
            if y_over:
                yr = min(yr, (y_thr_rel / YSCALE_K) * 0.99)
            d = "both" if (x_over and y_over) else ("x" if x_over else ("y" if y_over else "none"))
            axs.append(xr)
            ays.append(yr)
            anom_pts.append((xr, yr, d))
        if axs:
            ax.scatter(np.asarray(axs), np.asarray(ays), marker="x",
                       c=ANOM_MARKER_COLOR, s=ANOM_MARKER_SIZE,
                       linewidths=ANOM_MARKER_LW, alpha=0.95, zorder=4)

    x_min_rel = x_left_abs - x0
    x_max_rel = x_right_abs - x0
    y_min_rel = (y_bottom_abs - y0) / YSCALE_K
    y_max_rel = (y_top_abs - y0) / YSCALE_K
    span_x = x_max_rel - x_min_rel
    span_y = y_max_rel - y_min_rel
    min_pts_for_box = 3

    def add_error_box(cx, cy, w, h):
        ell = Ellipse((cx, cy), width=w, height=h, angle=0,
                      fill=False, edgecolor=ERR_BOX_COLOR,
                      linestyle=ERR_BOX_LINESTYLE, linewidth=ERR_BOX_LW,
                      alpha=0.85, zorder=5)
        ax.add_patch(ell)

    def line_box(values, span, min_dim_frac=0.10, padding_frac=1.15):
        arr = np.asarray(values, dtype=float)
        center = float(arr.mean())
        if len(arr) >= 2:
            rng = float(arr.max() - arr.min())
            length = max(rng * padding_frac, span * min_dim_frac)
        else:
            length = span * min_dim_frac
        return center, length

    mx_y = [p[1] for p in miss_pts if p[2] == "x"]
    if len(mx_y) >= min_pts_for_box:
        cy, h = line_box(mx_y, span_y, min_dim_frac=0.12, padding_frac=1.10)
        add_error_box(x_out_rel, cy, span_x * 0.05, h)

    my_x = [p[0] for p in miss_pts if p[2] == "y"]
    if len(my_x) >= min_pts_for_box:
        cx, w = line_box(my_x, span_x, min_dim_frac=0.12, padding_frac=1.10)
        add_error_box(cx, y_out_rel / YSCALE_K, w, span_y * 0.05)

    ax_y = [p[1] for p in anom_pts if p[2] in ("x", "both")]
    if len(ax_y) >= min_pts_for_box:
        cy, h = line_box(ax_y, span_y, min_dim_frac=0.12, padding_frac=1.10)
        add_error_box(x_thr_rel * 0.99, cy, span_x * 0.05, h)

    ay_x = [p[0] for p in anom_pts if p[2] in ("y", "both")]
    if len(ay_x) >= min_pts_for_box:
        cx, w = line_box(ay_x, span_x, min_dim_frac=0.12, padding_frac=1.10)
        add_error_box(cx, (y_thr_rel / YSCALE_K) * 0.99, w, span_y * 0.05)

    ax.set_xlim(x_min_rel, x_max_rel)
    ax.set_ylim(y_min_rel, y_max_rel)


# ============================== Bottom row ==============================
def draw_bottom_panel(ax, x_arr, y_arr, labels, color_map, xlim=None, ylim=None):
    if len(x_arr) == 0:
        ax.text(0.5, 0.5, "missing", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        return

    sorted_clusters = sorted(set(int(c) for c in labels))
    for c in sorted_clusters:
        m = labels == c
        color = color_map[int(c)]
        ax.scatter(x_arr[m], y_arr[m], s=POINT_SIZE, lw=0,
                   color=color, alpha=0.85, zorder=2)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    for c in sorted_clusters:
        m = labels == c
        color = color_map[int(c)]
        add_cluster_hull(ax, x_arr[m], y_arr[m], color,
                         linewidth=ELLIPSE_LW, pad_frac=0.05)


# ============================== Data preparation ==============================
def prepare_top_data(demo_root: Path):
    clean_file = demo_root / CLEAN_FILE_NAME
    if not clean_file.is_file():
        raise FileNotFoundError(f"Missing visual-demo clean file: {clean_file}")

    base = pd.read_csv(clean_file)[["ID", "age", "income", "segment"]]
    for c in ("age", "income"):
        base[c] = pd.to_numeric(base[c], errors="coerce")

    xmin, xmax = base.age.min(), base.age.max()
    ymin, ymax = base.income.min(), base.income.max()
    x_thr = np.percentile(base.age, 75) + 1.5 * np.subtract(*np.percentile(base.age, [75, 25]))
    y_thr = np.percentile(base.income, 75) + 1.5 * np.subtract(*np.percentile(base.income, [75, 25]))
    dx = 0.02 * (xmax - xmin)
    dy = 0.02 * (ymax - ymin)
    x_left_abs = xmin - 1.5 * dx
    y_bottom_abs = ymin - 1.5 * dy
    x0 = 0.5 * (x_left_abs + x_thr)
    y0 = 0.5 * (y_bottom_abs + y_thr)
    x_out_rel = (xmin - dx) - x0
    y_out_rel = (ymin - dy) - y0
    x_thr_rel = x_thr - x0
    y_thr_rel = y_thr - y0

    seg_map = base.set_index("ID")["segment"]
    base["err_type"] = "normal"
    base_ref = base.set_index("ID")

    eigen = read_eigen(demo_root / BASE_DIR_NAME / EIGENV_NAME)
    if TARGET_ID not in eigen:
        raise KeyError(f"{EIGENV_NAME} does not contain dataset_id={TARGET_ID}")

    dirty_path = demo_root / DIRTY_DIR_NAME / eigen[TARGET_ID]["csv"]
    if not dirty_path.exists():
        raise FileNotFoundError(f"Missing dirty demo CSV: {dirty_path}")

    dirty = pd.read_csv(dirty_path)[["ID", "age", "income"]].merge(seg_map, on="ID", how="left")
    for c in ("age", "income"):
        dirty[c] = pd.to_numeric(dirty[c], errors="coerce")
    dirty.set_index("ID", inplace=True)
    dirty["err_type"] = tag_err(dirty, base_ref)
    dirty.reset_index(inplace=True)

    cleaned_frames = {}
    for algo in ("mode", "holoclean", "baran"):
        dfc = safe_load_cleaned(demo_root, algo, seg_map)
        if dfc.empty:
            cleaned_frames[algo] = dfc
            continue
        dfc.set_index("ID", inplace=True)
        dfc["err_type"] = tag_err(dfc, base_ref)
        dfc.reset_index(inplace=True)
        cleaned_frames[algo] = dfc

    panels = {
        "clean": base,
        "dirty": dirty,
        "mode": cleaned_frames["mode"],
        "holoclean": cleaned_frames["holoclean"],
        "baran": cleaned_frames["baran"],
    }

    frame = dict(
        x0=x0, y0=y0,
        x_left_abs=x_left_abs, x_right_abs=x_thr,
        y_bottom_abs=y_bottom_abs, y_top_abs=y_thr,
        x_out_rel=x_out_rel, y_out_rel=y_out_rel,
        x_thr_rel=x_thr_rel, y_thr_rel=y_thr_rel,
    )
    return panels, frame, base


def prepare_bottom_data(demo_root: Path, panels_top: dict, base: pd.DataFrame):
    bottom = {}

    seg_to_id = {s: i for i, s in enumerate(SEG_ORDER)}
    clean_labels = (
        base["segment"].astype(str).str[:1].map(seg_to_id).fillna(-1).astype(int).to_numpy()
    )
    bottom["clean"] = (
        base.age.to_numpy(dtype=float),
        base.income.to_numpy(dtype=float),
        clean_labels,
    )

    dirty = panels_top["dirty"][["age", "income"]].copy()
    if not dirty.empty:
        d_labels = hc_cluster_dirty(dirty, k=5)
        bottom["dirty"] = (
            dirty.age.to_numpy(dtype=float),
            dirty.income.to_numpy(dtype=float),
            d_labels,
        )
    else:
        bottom["dirty"] = (np.array([]), np.array([]), np.array([]))

    for algo in ("mode", "holoclean", "baran"):
        cleaned_csv = demo_root / BASE_DIR_NAME / "cleaned_data" / algo / f"repaired_{TARGET_ID}.csv"
        labels_df = load_cluster_labels(demo_root, algo)
        if not cleaned_csv.is_file() or labels_df.empty:
            bottom[algo] = (np.array([]), np.array([]), np.array([]))
            continue
        dfc = pd.read_csv(cleaned_csv).reset_index(drop=False).rename(columns={"index": "orig_index"})
        if "orig_index" not in dfc.columns:
            dfc.insert(0, "orig_index", np.arange(len(dfc), dtype=int))
        dfm = dfc.merge(labels_df, on="orig_index", how="inner")
        if dfm.empty:
            bottom[algo] = (np.array([]), np.array([]), np.array([]))
            continue
        bottom[algo] = (
            dfm.age.to_numpy(dtype=float),
            dfm.income.to_numpy(dtype=float),
            dfm["cluster"].to_numpy(dtype=int),
        )

    return bottom


def build_palette(labels):
    uniq = sorted(set(int(x) for x in labels)) if len(labels) else []
    cmap = list(plt.get_cmap("tab10").colors)
    return {lab: cmap[i % len(cmap)] for i, lab in enumerate(uniq)}


# ============================== Stage 3 entry point ==============================
def build(ctx: BuildContext) -> ArtifactResult:
    demo_root = resolve_demo_root(ctx)
    out_dir = ctx.output_dir / "figure_1"
    clean_output_dir(out_dir)

    panels_top, frame, base = prepare_top_data(demo_root)
    bottom = prepare_bottom_data(demo_root, panels_top, base)

    x0, y0 = frame["x0"], frame["y0"]
    top_xlim = (frame["x_left_abs"] - x0, frame["x_right_abs"] - x0)
    top_ylim = (
        (frame["y_bottom_abs"] - y0) / YSCALE_K,
        (frame["y_top_abs"] - y0) / YSCALE_K,
    )

    other_x_all = []
    other_y_all = []
    for k in ("dirty", "mode", "holoclean", "baran"):
        x, y, _ = bottom[k]
        if len(x) == 0:
            continue
        other_x_all.append(x - x0)
        other_y_all.append((y - y0) / YSCALE_K)

    if other_x_all:
        all_x = np.concatenate(other_x_all)
        all_y = np.concatenate(other_y_all)
        all_x = all_x[np.isfinite(all_x)]
        all_y = all_y[np.isfinite(all_y)]
        if len(all_x) > 0 and len(all_y) > 0:
            x_lo, x_hi = np.percentile(all_x, [0.5, 99.5])
            y_lo, y_hi = np.percentile(all_y, [0.5, 99.5])
            pad_x = 0.05 * max(1e-9, x_hi - x_lo)
            pad_y = 0.05 * max(1e-9, y_hi - y_lo)
            bottom_xlim = (x_lo - pad_x, x_hi + pad_x)
            bottom_ylim = (y_lo - pad_y, y_hi + pad_y)
        else:
            bottom_xlim, bottom_ylim = top_xlim, top_ylim
    else:
        bottom_xlim, bottom_ylim = top_xlim, top_ylim

    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(24, 9.0),
        gridspec_kw={"wspace": 0.20, "hspace": 0.22},
    )

    for j, (_, key) in enumerate(COLUMNS):
        ax = axes[0, j]
        df_plot = panels_top[key]
        draw_top_panel(
            ax, df_plot,
            frame["x0"], frame["y0"],
            frame["x_left_abs"], frame["x_right_abs"],
            frame["y_bottom_abs"], frame["y_top_abs"],
            frame["x_out_rel"], frame["y_out_rel"],
            frame["x_thr_rel"], frame["y_thr_rel"],
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="major", direction="in", labelsize=TICK_SIZE,
                       length=5, width=1.1)
        ax.tick_params(which="minor", direction="in", length=2.5, width=0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(np.round(v))}"))

    for j, (title, key) in enumerate(COLUMNS):
        ax = axes[1, j]
        x, y, labels = bottom[key]
        if len(x) > 0:
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            labels = np.asarray(labels)[mask]
        if len(x) > 0:
            x_rel = x - x0
            y_rel = (y - y0) / YSCALE_K
            if key == "clean":
                cmap = {i: SEG_COLOR[s] for i, s in enumerate(SEG_ORDER)}
            else:
                cmap = build_palette(labels)
            xl = top_xlim if key == "clean" else bottom_xlim
            yl = top_ylim if key == "clean" else bottom_ylim
            draw_bottom_panel(ax, x_rel, y_rel, labels, cmap, xlim=xl, ylim=yl)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="major", direction="in", labelsize=TICK_SIZE,
                       length=5, width=1.1)
        ax.tick_params(which="minor", direction="in", length=2.5, width=0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(np.round(v))}"))
        ax.set_xlabel(
            f"{SUBLABELS[j]} {title}",
            fontsize=SUBLABEL_FONTSIZE,
            fontweight="bold",
            labelpad=10,
        )

    err_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=SEG_COLOR[s], markeredgecolor=SEG_COLOR[s],
               markersize=12, label=f"Segment {s}")
        for s in SEG_ORDER
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=RING_MISSING_COLOR, markeredgewidth=RING_LW,
               markersize=14, label="Missing Value"),
        Line2D([0], [0], marker="x", color=ANOM_MARKER_COLOR,
               markersize=14, markeredgewidth=ANOM_MARKER_LW,
               linestyle="", label="Anomaly / Noise"),
        Line2D([0], [0], color=ERR_BOX_COLOR, linestyle="--",
               linewidth=ERR_BOX_LW, label="Error Region"),
        Line2D([0], [0], color="dimgray", linestyle="-",
               linewidth=ELLIPSE_LW, label="Cluster Hull"),
    ]

    fig.subplots_adjust(bottom=0.12, top=0.82, left=0.045, right=0.985)
    fig.legend(handles=err_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99),
               ncol=5, prop={"size": LEGEND_FONTSIZE, "weight": "bold"},
               frameon=True, fancybox=False, edgecolor="black",
               framealpha=0.95,
               borderpad=0.3, columnspacing=1.4,
               handletextpad=0.4, labelspacing=0.4)

    out_path = out_dir / "figure1_grid.pdf"
    fig.savefig(out_path, dpi=600, facecolor="white")
    plt.close(fig)

    inputs = [
        demo_root / CLEAN_FILE_NAME,
        demo_root / BASE_DIR_NAME / EIGENV_NAME,
        demo_root / DIRTY_DIR_NAME,
        demo_root / BASE_DIR_NAME / "cleaned_data",
        demo_root / BASE_DIR_NAME / "clustered_data" / "HC",
    ]

    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="success",
        outputs=[out_path],
        inputs=inputs,
        message=f"Built Figure 1 under {out_dir}.",
        metadata={
            "output_subdir": "figure_1",
            "demo_root": str(demo_root),
            "target_id": TARGET_ID,
            "expected_output_count": 1,
            "actual_output_count": 1,
        },
    )
