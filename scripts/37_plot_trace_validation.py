from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Keep the two exported panels physically identical.
# Slightly wider than before; height unchanged.
FIG_W = 5.25
FIG_H = 3.70

# Fixed margins. Do not use bbox_inches="tight", otherwise the two panels
# may be cropped differently after export.
LEFT = 0.16
RIGHT = 0.94
BOTTOM = 0.18
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


def savefig(path_base: Path, aliases=None):
    aliases = aliases or []
    all_paths = [path_base] + [Path(p) for p in aliases]
    for base in all_paths:
        # Do not use bbox_inches="tight"; it may crop panels differently.
        plt.savefig(base.with_suffix(".pdf"))
        plt.savefig(base.with_suffix(".png"), dpi=300)
    plt.close()


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blind-dir",
        required=True,
        help="Directory containing trace_blind_random_* outputs."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for figures. Default: <blind-dir>/figures"
    )
    args = parser.parse_args()

    blind_dir = Path(args.blind_dir)
    out_dir = Path(args.out_dir) if args.out_dir else blind_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = blind_dir / "trace_blind_random_aggregate_summary.json"
    dataset_path = blind_dir / "trace_blind_random_dataset_summary.csv"

    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing aggregate file: {aggregate_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset summary file: {dataset_path}")

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

    # -----------------------------
    # Figure 1: ECDF of hit-to-95% progress
    # -----------------------------
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
        x_trace, y_trace,
        where="post",
        linewidth=2.0,
        color=trace_color,
        label=f"TRACE, median={pct(median_trace_hit)}"
    )
    ax.step(
        x_blind, y_blind,
        where="post",
        linewidth=2.0,
        color=blind_color,
        label=f"Blind random, median={pct(median_blind_hit)}"
    )
    ax.axvline(median_trace_hit, linestyle="--", linewidth=1.2, color=trace_color)
    ax.axvline(median_blind_hit, linestyle="--", linewidth=1.2, color=blind_color)

    ax.set_xlabel("Budget fraction to 95% optimum")
    ax.set_ylabel("Cumulative fraction")

    # Keep the 1.0 tick inside the plotting region so it is not clipped in LaTeX.
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))

    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)

    savefig(
        out_dir / "fig1_hit95_progress_ecdf",
        aliases=[out_dir / "hit95_progress_ecdf"]
    )

    # -----------------------------
    # Figure 2: ECDF of AUC retention
    # -----------------------------
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
        x_trace_auc, y_trace_auc,
        where="post",
        linewidth=2.0,
        color=trace_color,
        label=f"TRACE, median={median_trace_auc:.3f}"
    )
    ax.step(
        x_blind_auc, y_blind_auc,
        where="post",
        linewidth=2.0,
        color=blind_color,
        label=f"Blind random, median={median_blind_auc:.3f}"
    )
    ax.axvline(median_trace_auc, linestyle="--", linewidth=1.2, color=trace_color)
    ax.axvline(median_blind_auc, linestyle="--", linewidth=1.2, color=blind_color)

    xmin = max(0.0, min(float(trace_auc.min()), float(blind_auc.min())) - 0.015)
    ax.set_xlabel("AUC retention")
    ax.set_ylabel("Cumulative fraction")

    # Keep 1.00 safely inside the axis. Format AUC ticks with two decimals.
    ax.set_xlim(xmin, 1.005)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))

    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)

    savefig(
        out_dir / "fig3_auc_retention_ecdf",
        aliases=[out_dir / "auc_retention_ecdf"]
    )

    # -----------------------------
    # Summary text
    # -----------------------------
    auc_abs_gain = median_trace_auc - median_blind_auc
    blind_gap = 1.0 - median_blind_auc
    trace_gap = 1.0 - median_trace_auc
    gap_reduction = (blind_gap - trace_gap) / blind_gap if blind_gap > 0 else float("nan")
    hit_reduction = (median_blind_hit - median_trace_hit) / median_blind_hit

    summary_lines = [
        f"n_datasets: {aggregate.get('n_datasets')}",
        f"random_seeds: {aggregate.get('random_seeds')}",
        f"median_trace_hit95_progress: {median_trace_hit}",
        f"median_blind_random_hit95_progress: {median_blind_hit}",
        f"relative_hit95_budget_reduction: {hit_reduction}",
        f"median_trace_auc_retention: {median_trace_auc}",
        f"median_blind_random_auc_retention: {median_blind_auc}",
        f"auc_absolute_gain: {auc_abs_gain}",
        f"auc_gap_to_ideal_trace: {trace_gap}",
        f"auc_gap_to_ideal_blind_random: {blind_gap}",
        f"auc_gap_reduction_ratio: {gap_reduction}",
    ]
    (out_dir / "figure_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Figures written to: {out_dir}")
    for p in sorted(out_dir.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
