from __future__ import annotations

"""
Stage 3 builder for paper Figure 7.

This module is intended to live at:
    src/paper_artifact/figures/fig07_trace_validation.py

It adapts the previous TRACE validation plotting script into the Stage 3
paper-artifact framework.

Input
-----
The builder expects TRACE LODO paper reproduction outputs under:

    <ctx.input_root>/processed/trace/lodo_paper_repro/

Required files:
    lodo_aggregate_summary.json
    lodo_blind_random_dataset_summary.csv

Output
------
Exactly two PDF files:

    <ctx.output_dir>/figure_7/lodo_hit95_progress_ecdf.pdf
    <ctx.output_dir>/figure_7/lodo_auc_retention_ecdf.pdf

No PNG, TXT, CSV, or auxiliary outputs are generated.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from src.paper_artifact.io import ArtifactResult, BuildContext


ARTIFACT = {
    "id": "fig07_trace_validation",
    "paper_id": "Figure 7",
    "label": "Figure 7: TRACE leave-one-dataset-out validation",
    "description": "Build TRACE LODO validation ECDF panels from Stage 4 outputs.",
    "enabled": True,
}


plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Keep exported panels physically identical.
FIG_W = 5.35
FIG_H = 3.70
LEFT = 0.16
RIGHT = 0.94
BOTTOM = 0.20
TOP = 0.96


def ecdf(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.sort(values)
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y


def pct(x):
    return f"{100.0 * float(x):.1f}%"


def style_axes(ax):
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        length=4.5,
        width=0.9,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def make_figure():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)
    return fig, ax


def save_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not use bbox_inches="tight"; paired panels should keep identical size.
    plt.savefig(path)
    plt.close()
    return path


def get_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")


def resolve_lodo_dir(ctx: BuildContext) -> Path:
    """Resolve the TRACE LODO output directory from the Stage 3 input root."""
    candidates = [
        ctx.input_root / "processed" / "trace" / "lodo_paper_repro",
        ctx.input_root / "processed" / "trace" / "lodo",
        ctx.project_root / "results" / "processed" / "trace" / "lodo_paper_repro",
        ctx.project_root / "results" / "processed" / "trace" / "lodo",
    ]

    for cand in candidates:
        aggregate = cand / "lodo_aggregate_summary.json"
        dataset = cand / "lodo_blind_random_dataset_summary.csv"
        if aggregate.exists() and dataset.exists():
            return cand

    raise FileNotFoundError(
        "Cannot find TRACE LODO outputs. Expected lodo_aggregate_summary.json "
        "and lodo_blind_random_dataset_summary.csv under one of:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nRun scripts/39_run_trace_stage4_paper_repro.py first."
    )


def clean_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()


def build(ctx: BuildContext) -> ArtifactResult:
    lodo_dir = resolve_lodo_dir(ctx)
    out_dir = ctx.output_dir / "figure_7"

    if not ctx.dry_run:
        clean_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = lodo_dir / "lodo_aggregate_summary.json"
    dataset_path = lodo_dir / "lodo_blind_random_dataset_summary.csv"

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    ds = pd.read_csv(dataset_path)

    trace_hit_col = get_col(ds, ["trace_hit95_progress"])
    blind_hit_col = get_col(ds, ["blind_random_hit95_progress_median"])
    trace_auc_col = get_col(ds, ["trace_auc_retention"])
    blind_auc_col = get_col(ds, ["blind_random_auc_retention_median"])

    median_trace_hit = float(aggregate["median_trace_hit95_progress"])
    median_blind_hit = float(aggregate["median_blind_random_hit95_progress"])
    median_trace_auc = float(aggregate["median_trace_auc_retention"])
    median_blind_auc = float(aggregate["median_blind_random_auc_retention"])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    trace_color = colors[0]
    blind_color = colors[1]

    outputs: list[Path] = []

    # --------------------------------------------------
    # Panel 1: ECDF of hit-to-95% progress
    # --------------------------------------------------
    trace_hit = (
        pd.to_numeric(ds[trace_hit_col], errors="coerce")
        .fillna(1.0)
        .clip(0.0, 1.0)
    )
    blind_hit = (
        pd.to_numeric(ds[blind_hit_col], errors="coerce")
        .fillna(1.0)
        .clip(0.0, 1.0)
    )

    x_trace, y_trace = ecdf(trace_hit)
    x_blind, y_blind = ecdf(blind_hit)

    fig, ax = make_figure()
    ax.step(
        x_trace,
        y_trace,
        where="post",
        linewidth=2.0,
        color=trace_color,
        label=f"TRACE, median={pct(median_trace_hit)}",
    )
    ax.step(
        x_blind,
        y_blind,
        where="post",
        linewidth=2.0,
        color=blind_color,
        label=f"Blind random, median={pct(median_blind_hit)}",
    )
    ax.axvline(median_trace_hit, linestyle="--", linewidth=1.2, color=trace_color)
    ax.axvline(median_blind_hit, linestyle="--", linewidth=1.2, color=blind_color)

    ax.set_xlabel("Budget fraction to 95% optimum")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)

    outputs.append(save_pdf(out_dir / "lodo_hit95_progress_ecdf.pdf"))

    # --------------------------------------------------
    # Panel 2: ECDF of AUC retention
    # --------------------------------------------------
    trace_auc = (
        pd.to_numeric(ds[trace_auc_col], errors="coerce")
        .dropna()
        .clip(0.0, 1.0)
    )
    blind_auc = (
        pd.to_numeric(ds[blind_auc_col], errors="coerce")
        .dropna()
        .clip(0.0, 1.0)
    )

    x_trace_auc, y_trace_auc = ecdf(trace_auc)
    x_blind_auc, y_blind_auc = ecdf(blind_auc)

    fig, ax = make_figure()
    ax.step(
        x_trace_auc,
        y_trace_auc,
        where="post",
        linewidth=2.0,
        color=trace_color,
        label=f"TRACE, median={median_trace_auc:.3f}",
    )
    ax.step(
        x_blind_auc,
        y_blind_auc,
        where="post",
        linewidth=2.0,
        color=blind_color,
        label=f"Blind random, median={median_blind_auc:.3f}",
    )
    ax.axvline(median_trace_auc, linestyle="--", linewidth=1.2, color=trace_color)
    ax.axvline(median_blind_auc, linestyle="--", linewidth=1.2, color=blind_color)

    xmin = max(0.0, min(float(trace_auc.min()), float(blind_auc.min())) - 0.015)
    ax.set_xlabel("AUC retention")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(xmin, 1.005)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)

    outputs.append(save_pdf(out_dir / "lodo_auc_retention_ecdf.pdf"))

    expected = {
        "lodo_hit95_progress_ecdf.pdf",
        "lodo_auc_retention_ecdf.pdf",
    }
    actual = {p.name for p in outputs}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"Output file mismatch. Missing={missing}; extra={extra}")

    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="success",
        outputs=outputs,
        inputs=[aggregate_path, dataset_path],
        message=f"Built Figure 7 with {len(outputs)} PDF files under {out_dir}.",
        metadata={
            "output_subdir": "figure_7",
            "expected_output_count": 2,
            "actual_output_count": len(outputs),
            "lodo_dir": str(lodo_dir),
            "median_trace_hit95_progress": median_trace_hit,
            "median_blind_random_hit95_progress": median_blind_hit,
            "median_trace_auc_retention": median_trace_auc,
            "median_blind_random_auc_retention": median_blind_auc,
        },
    )
