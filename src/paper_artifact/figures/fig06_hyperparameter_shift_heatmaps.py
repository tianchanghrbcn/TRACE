from __future__ import annotations

"""
Stage 3 builder for paper Figure 6.

This module is intended to live at:
    src/paper_artifact/figures/fig06_hyperparameter_shift_heatmaps.py

It reads the four task summary workbooks from ctx.input_root:
    beers_summary.xlsx
    flights_summary.xlsx
    hospital_summary.xlsx
    rayyan_summary.xlsx

and directly computes the Mode-relative hyperparameter-shift matrices that
the original heatmap script consumed from table10_*_hyper_shift.xlsx.

It writes exactly five PDF files:
    <ctx.output_dir>/figure_6/hyper_heat_kmeans_dk.pdf
    <ctx.output_dir>/figure_6/hyper_heat_gmm_dncomp.pdf
    <ctx.output_dir>/figure_6/hyper_heat_dbscan_deps.pdf
    <ctx.output_dir>/figure_6/hyper_heat_dbscan_dminpts.pdf
    <ctx.output_dir>/figure_6/hyper_heat_cleaner_codes.pdf

No intermediate XLSX / CSV / PNG / EPS files are generated.
"""

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FixedFormatter

from src.paper_artifact.io import ArtifactResult, BuildContext


ARTIFACT = {
    "id": "fig06_hyperparameter_shift_heatmaps",
    "paper_id": "Figure 6",
    "label": "Figure 6: Mode-relative hyperparameter-shift heatmaps",
    "description": "Build Figure 6 directly from the four *_summary.xlsx workbooks under results/.",
    "enabled": True,
}


DATASETS = ["beers", "flights", "hospital", "rayyan"]
ERROR_BINS = [5, 10, 15, 20, 25, 30]

BASELINE_CLEANER = "Mode"
DROP_BASELINE_ROW = True

CLEAN_ORDER_FULL = [
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

DISPLAY_CLEAN_ORDER = [
    c for c in CLEAN_ORDER_FULL
    if not (DROP_BASELINE_ROW and c == BASELINE_CLEANER)
]

ROMAN_CODES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
CLEAN_TO_CODE = dict(zip(DISPLAY_CLEAN_ORDER, ROMAN_CODES[:len(DISPLAY_CLEAN_ORDER)]))

ORACLE_CLEANERS = {"GroundTruth", "Oracle", "GT"}

METHOD_ALIASES = {
    "mode": "Mode",
    "modeimpute": "Mode",
    "modeimputation": "Mode",
    "modeimputer": "Mode",
    "none": "Mode",
    "baran": "Baran",
    "holoclean": "HoloClean",
    "holo": "HoloClean",
    "bigdansing": "BigDansing",
    "bigdans": "BigDansing",
    "boostclean": "BoostClean",
    "horizon": "Horizon",
    "scared": "SCAReD",
    "unified": "Unified",
    "uniclean": "UniClean",
    "groundtruth": "GroundTruth",
    "gt": "GroundTruth",
    "oracle": "GroundTruth",
}

COLUMN_ALIASES: dict[str, list[str]] = {
    "dataset_id": ["dataset_id", "dirty_id", "instance_id", "id"],
    "error_rate": ["error_rate", "q_tot", "qtot", "total_error_rate"],
    "cleaning_method": ["cleaning_method", "cleaner", "method", "cleaning"],
    "cluster_method": ["cluster_method", "clusterer", "clustering_method", "algorithm"],
    "parameters": ["parameters", "params", "hyperparameters", "best_params", "best_parameters"],
}

COLORBAR_LIMITS = {
    "dk": (-24.0, 24.0),
    "dncomp": (-36.0, 36.0),
    "deps": (-1.0, 1.0),
    "dminpts": (-4.0, 4.0),
}

PLOT_SPECS = {
    "dk": {"kind": "count"},
    "dncomp": {"kind": "count"},
    "deps": {"kind": "eps"},
    "dminpts": {"kind": "count"},
}

N_DISPLAY_ROWS = len(DISPLAY_CLEAN_ORDER)
N_DISPLAY_COLS = len(ERROR_BINS)

HEATMAP_WIDTH_IN = 2.40
HEATMAP_HEIGHT_IN = HEATMAP_WIDTH_IN * N_DISPLAY_ROWS / N_DISPLAY_COLS
HEATMAP_FIGSIZE = (HEATMAP_WIDTH_IN, HEATMAP_HEIGHT_IN)

HEATMAP_AX_RECT = [0.16, 0.22, 0.76, 0.76]
CBAR_AX_RECT = [0.16, 0.090, 0.76, 0.035]

XTICK_FONTSIZE = 9.5
YTICK_FONTSIZE = 9.5
CBAR_TICK_FONTSIZE = 9.5
MISSING_CELL_COLOR = "0.88"

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def norm_key(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def normalize_method(x: Any) -> str:
    return METHOD_ALIASES.get(norm_key(x), str(x).strip())


def normalize_cluster(x: Any) -> str:
    return norm_key(x).upper()


def find_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    norm_to_col = {norm_key(c): c for c in df.columns}
    for alias in COLUMN_ALIASES[canonical]:
        key = norm_key(alias)
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def require_column(df: pd.DataFrame, canonical: str, path: Path) -> str:
    col = find_column(df, canonical)
    if col is None:
        raise ValueError(f"Missing required column {canonical!r} in {path}. Available columns: {list(df.columns)}")
    return col


def to_numeric(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def normalize_error_rate_to_percent(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    max_val = vals.max(skipna=True)
    if pd.notna(max_val) and max_val <= 1.0:
        return vals * 100.0
    return vals


def error_rate_bin(s: pd.Series) -> pd.Series:
    pct = normalize_error_rate_to_percent(s)
    return ((pct / 5).round() * 5).astype("Int64")


def read_one_workbook(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0)


def load_summary_workbooks(input_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frames = []
    inputs: list[Path] = []

    for task in DATASETS:
        path = input_root / f"{task}_summary.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"Missing required workbook: {path}")

        raw = read_one_workbook(path)
        c_dataset = require_column(raw, "dataset_id", path)
        c_error = require_column(raw, "error_rate", path)
        c_clean = require_column(raw, "cleaning_method", path)
        c_cluster = require_column(raw, "cluster_method", path)
        c_params = require_column(raw, "parameters", path)

        df = raw.copy()
        df["task_name"] = task
        df["dataset_id"] = pd.to_numeric(df[c_dataset], errors="coerce").astype("Int64")
        df["error_rate_bin"] = error_rate_bin(df[c_error])
        df["cleaner"] = df[c_clean].map(normalize_method)
        df["clusterer"] = df[c_cluster].map(normalize_cluster)
        df["parameters_raw"] = df[c_params]

        frames.append(df[["task_name", "dataset_id", "error_rate_bin", "cleaner", "clusterer", "parameters_raw"]])
        inputs.append(path)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["dataset_id", "error_rate_bin", "clusterer", "cleaner"])
    out["dataset_id"] = out["dataset_id"].astype(int)
    out["error_rate_bin"] = out["error_rate_bin"].astype(int)
    return out, inputs


def parse_params(raw: Any) -> dict[str, Any]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    if isinstance(raw, dict):
        return {norm_key(k): v for k, v in raw.items()}

    s = str(raw).strip()
    if not s or s in {"{}", "nan", "None"}:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
            if isinstance(obj, dict):
                return {norm_key(k): v for k, v in obj.items()}
        except Exception:
            pass

    params: dict[str, Any] = {}
    for key, val in re.findall(r"([^=,{}]+)\s*=\s*([^,{}]+)", s):
        params[norm_key(key)] = val.strip()
    return params


def get_first(params: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        nk = norm_key(key)
        if nk in params:
            return params[nk]
    return np.nan


def get_numeric(params: dict[str, Any], keys: list[str]) -> float:
    return to_numeric(get_first(params, keys))


def extract_hyperparameters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        params = parse_params(row["parameters_raw"])
        clusterer = str(row["clusterer"]).upper()
        rec = row.to_dict()
        rec.update({"k": np.nan, "eps": np.nan, "minpts": np.nan, "ncomp": np.nan})

        if clusterer in {"KMEANS", "KMEANSNF", "KMEANSPPS"}:
            rec["k"] = get_numeric(params, ["k", "n_clusters", "ncomponents", "n_components"])

        elif clusterer == "DBSCAN":
            rec["eps"] = get_numeric(params, ["eps", "epsilon", "covariancetype", "covariance_type", "covariance"])
            rec["minpts"] = get_numeric(params, ["min_samples", "minsamples", "minpts", "min_pts", "ncomponents", "n_components"])

        elif clusterer == "GMM":
            rec["ncomp"] = get_numeric(params, ["n_components", "ncomponents", "n_clusters", "k"])

        rows.append(rec)

    return pd.DataFrame(rows)


def attach_mode_baseline(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    keys = ["task_name", "dataset_id", "error_rate_bin", "clusterer"]
    base = (
        df[df["cleaner"] == BASELINE_CLEANER][keys + [value_col]]
        .dropna(subset=[value_col])
        .groupby(keys, as_index=False, observed=False)[value_col]
        .median()
        .rename(columns={value_col: f"mode_{value_col}"})
    )
    return df.merge(base, on=keys, how="left")


def make_shift_rows(df: pd.DataFrame, clusterers: set[str], value_col: str, param_key: str) -> pd.DataFrame:
    sub = df[df["clusterer"].isin(clusterers)].copy()
    sub = attach_mode_baseline(sub, value_col)
    sub[param_key] = sub[value_col] - sub[f"mode_{value_col}"]
    sub = sub.dropna(subset=[param_key])
    sub = sub[~sub["cleaner"].isin(ORACLE_CLEANERS)]
    return sub


def build_matrix(rows: pd.DataFrame, param_key: str) -> pd.DataFrame:
    # Aggregate exactly at the heatmap level:
    # error-rate bins x cleaner, with median over tasks / dirty ids / clusterer variants.
    mat = (
        rows
        .groupby(["error_rate_bin", "cleaner"], as_index=False, observed=False)[param_key]
        .median()
        .pivot(index="error_rate_bin", columns="cleaner", values=param_key)
    )
    mat = mat.reindex(index=ERROR_BINS, columns=CLEAN_ORDER_FULL)
    return mat


def select_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=DISPLAY_CLEAN_ORDER)


def _format_bin_label(x) -> str:
    try:
        val = float(x)
        if abs(val - round(val)) < 1e-9:
            return str(int(round(val)))
        return f"{val:g}"
    except Exception:
        return str(x)


def _colorbar_ticks(vmin: float, vmax: float) -> np.ndarray:
    return np.linspace(vmin, vmax, 5)


def _format_cbar_tick(x: float, kind: str) -> str:
    if abs(x) < 1e-12:
        return "0"
    if kind == "eps":
        if abs(x) >= 1:
            return f"{x:.1f}"
        return f"{x:.2f}".rstrip("0").rstrip(".")
    if abs(x - round(x)) < 1e-9:
        return f"{x:.0f}"
    return f"{x:.1f}"


def _clip_for_display(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.asarray(data, dtype=float).copy()
    finite = np.isfinite(arr)
    arr[finite] = np.clip(arr[finite], vmin, vmax)
    return arr


def plot_heatmap(df: pd.DataFrame, param_key: str, outfile: Path) -> None:
    if param_key not in COLORBAR_LIMITS:
        raise ValueError(f"No colorbar range is defined for {param_key!r}.")

    spec = PLOT_SPECS[param_key]
    df = df.copy().reindex(index=ERROR_BINS)
    df = select_analysis_columns(df)

    data_df = df.T
    raw_data = data_df.to_numpy(dtype=float)

    vmin, vmax = COLORBAR_LIMITS[param_key]
    display_data = _clip_for_display(raw_data, vmin, vmax)

    row_codes = [CLEAN_TO_CODE.get(name, name) for name in data_df.index]
    col_bins = [_format_bin_label(x) for x in data_df.columns]

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color=MISSING_CELL_COLOR)

    fig = plt.figure(figsize=(HEATMAP_WIDTH_IN, HEATMAP_HEIGHT_IN))
    ax = fig.add_axes(HEATMAP_AX_RECT)
    cax = fig.add_axes(CBAR_AX_RECT)

    im = ax.imshow(
        np.ma.masked_invalid(display_data),
        aspect="equal",
        cmap=cmap,
        norm=norm,
        origin="upper",
    )

    ax.set_xticks(np.arange(data_df.shape[1]))
    ax.set_xticklabels(col_bins, fontsize=XTICK_FONTSIZE)

    ax.set_yticks(np.arange(data_df.shape[0]))
    ax.set_yticklabels(row_codes, fontsize=YTICK_FONTSIZE)

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticks(np.arange(-0.5, data_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data_df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0, pad=1.5)

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    ticks = np.asarray(_colorbar_ticks(vmin, vmax), dtype=float)
    tick_labels = [_format_cbar_tick(t, spec["kind"]) for t in ticks]
    cbar.ax.xaxis.set_major_locator(FixedLocator(ticks))
    cbar.ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE, length=2, pad=1)
    cbar.set_label("")

    tick_texts = cbar.ax.get_xticklabels()
    if tick_texts:
        tick_texts[0].set_ha("left")
        tick_texts[-1].set_ha("right")

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.25")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, format="pdf", bbox_inches="tight", pad_inches=0.003)
    plt.close(fig)


N_DISPLAY_ROWS = len(DISPLAY_CLEAN_ORDER)
N_DISPLAY_COLS = len(ERROR_BINS)
HEATMAP_WIDTH_IN = 2.40
HEATMAP_HEIGHT_IN = HEATMAP_WIDTH_IN * N_DISPLAY_ROWS / N_DISPLAY_COLS
HEATMAP_AX_RECT = [0.16, 0.22, 0.76, 0.76]
CBAR_AX_RECT = [0.16, 0.090, 0.76, 0.035]
XTICK_FONTSIZE = 9.5
YTICK_FONTSIZE = 9.5
CBAR_TICK_FONTSIZE = 9.5
MISSING_CELL_COLOR = "0.88"


def save_cleaner_codebook(out_dir: Path) -> Path:
    entries = [f"{CLEAN_TO_CODE[name]}: {name}" for name in DISPLAY_CLEAN_ORDER if name in CLEAN_TO_CODE]
    row1 = "   ".join(entries[:4])
    row2 = "   ".join(entries[4:])

    fig, ax = plt.subplots(figsize=(6.4, 0.46))
    ax.axis("off")
    ax.text(0.5, 0.68, row1, ha="center", va="center", fontsize=8.6)
    ax.text(0.5, 0.25, row2, ha="center", va="center", fontsize=8.6)

    out_path = out_dir / "hyper_heat_cleaner_codes.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.002)
    plt.close(fig)
    return out_path


def _clean_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()


def build(ctx: BuildContext) -> ArtifactResult:
    out_dir = Path(ctx.output_dir) / "figure_6"
    if not ctx.dry_run:
        _clean_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    raw, inputs = load_summary_workbooks(Path(ctx.input_root))
    parsed = extract_hyperparameters(raw)

    dk_rows = make_shift_rows(parsed, {"KMEANS", "KMEANSNF", "KMEANSPPS"}, "k", "dk")
    dn_rows = make_shift_rows(parsed, {"GMM"}, "ncomp", "dncomp")
    deps_rows = make_shift_rows(parsed, {"DBSCAN"}, "eps", "deps")
    dmin_rows = make_shift_rows(parsed, {"DBSCAN"}, "minpts", "dminpts")

    dk = build_matrix(dk_rows, "dk")
    dn = build_matrix(dn_rows, "dncomp")
    deps = build_matrix(deps_rows, "deps")
    dmin = build_matrix(dmin_rows, "dminpts")

    outputs: list[Path] = []

    p = out_dir / "hyper_heat_kmeans_dk.pdf"
    plot_heatmap(dk, "dk", p)
    outputs.append(p)

    p = out_dir / "hyper_heat_gmm_dncomp.pdf"
    plot_heatmap(dn, "dncomp", p)
    outputs.append(p)

    p = out_dir / "hyper_heat_dbscan_deps.pdf"
    plot_heatmap(deps, "deps", p)
    outputs.append(p)

    p = out_dir / "hyper_heat_dbscan_dminpts.pdf"
    plot_heatmap(dmin, "dminpts", p)
    outputs.append(p)

    outputs.append(save_cleaner_codebook(out_dir))

    expected_names = {
        "hyper_heat_kmeans_dk.pdf",
        "hyper_heat_gmm_dncomp.pdf",
        "hyper_heat_dbscan_deps.pdf",
        "hyper_heat_dbscan_dminpts.pdf",
        "hyper_heat_cleaner_codes.pdf",
    }
    actual_names = {p.name for p in outputs}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise RuntimeError(f"Output file mismatch. Missing={missing}; extra={extra}")

    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="success",
        outputs=outputs,
        inputs=inputs,
        message=f"Built Figure 6 with {len(outputs)} PDF files under {out_dir}.",
        metadata={
            "output_subdir": "figure_6",
            "expected_output_count": 5,
            "actual_output_count": len(outputs),
            "tasks": DATASETS,
        },
    )
