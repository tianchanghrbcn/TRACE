from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ecdf(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.sort(values)
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y


def pct(x):
    return f"{100.0 * float(x):.1f}%"


def savefig(path_base: Path):
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), dpi=300)
    plt.savefig(path_base.with_suffix(".pdf"))
    plt.close()


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")


def main():
    parser = argparse.ArgumentParser(description="Plot TRACE validation figures from blind-random outputs.")
    parser.add_argument("--blind-dir", required=True, help="Directory containing trace_blind_random_* outputs.")
    parser.add_argument("--out-dir", default=None, help="Output directory for figures. Default: <blind-dir>/figures")
    args = parser.parse_args()

    blind_dir = Path(args.blind_dir)
    out_dir = Path(args.out_dir) if args.out_dir else blind_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = blind_dir / "trace_blind_random_aggregate_summary.json"
    dataset_path = blind_dir / "trace_blind_random_dataset_summary.csv"
    fixed_budget_path = blind_dir / "trace_blind_random_fixed_budget.csv"

    for path in [aggregate_path, dataset_path, fixed_budget_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    ds = pd.read_csv(dataset_path)
    fb = pd.read_csv(fixed_budget_path)

    trace_hit_col = get_col(ds, ["trace_hit95_progress"])
    blind_hit_col = get_col(ds, ["blind_random_hit95_progress_median"])
    trace_auc_col = get_col(ds, ["trace_auc_retention"])
    blind_auc_col = get_col(ds, ["blind_random_auc_retention_median"])
    budget_col = get_col(fb, ["budget_fraction"])
    trace_ret_col = get_col(fb, ["trace_retention"])
    blind_ret_col = get_col(fb, ["blind_random_retention_median"])

    median_trace_hit = float(aggregate["median_trace_hit95_progress"])
    median_blind_hit = float(aggregate["median_blind_random_hit95_progress"])
    median_trace_auc = float(aggregate["median_trace_auc_retention"])
    median_blind_auc = float(aggregate["median_blind_random_auc_retention"])

    auc_abs_gain = median_trace_auc - median_blind_auc
    blind_gap = 1.0 - median_blind_auc
    trace_gap = 1.0 - median_trace_auc
    gap_reduction = (blind_gap - trace_gap) / blind_gap if blind_gap > 0 else float("nan")

    # Figure 1: ECDF of hit-to-95% progress.
    x_trace, y_trace = ecdf(ds[trace_hit_col].astype(float))
    x_blind, y_blind = ecdf(ds[blind_hit_col].astype(float))

    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    ax.step(x_trace, y_trace, where="post", label=f"TRACE, median={pct(median_trace_hit)}")
    ax.step(x_blind, y_blind, where="post", label=f"Blind random, median={pct(median_blind_hit)}")
    ax.axvline(median_trace_hit, linestyle="--", linewidth=1)
    ax.axvline(median_blind_hit, linestyle="--", linewidth=1)
    ax.set_xlabel("Budget progress to reach 95% of exhaustive-search optimum")
    ax.set_ylabel("Cumulative fraction of datasets")
    ax.set_title("Hit-to-95% budget progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0.0, y=0.0)
    savefig(out_dir / "fig1_hit95_progress_ecdf")

    # Figure 2a: median retention versus budget, full y-scale.
    ret = (
        fb.groupby(budget_col)[[trace_ret_col, blind_ret_col]]
        .median()
        .reset_index()
        .sort_values(budget_col)
    )

    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    ax.plot(ret[budget_col], ret[trace_ret_col], marker="o", label=f"TRACE, AUC={median_trace_auc:.3f}")
    ax.plot(ret[budget_col], ret[blind_ret_col], marker="o", label=f"Blind random, AUC={median_blind_auc:.3f}")
    ax.set_xlabel("Budget fraction of full trial budget")
    ax.set_ylabel("Median score retention")
    ax.set_title("Median retention versus budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(float(ret[budget_col].min()), float(ret[budget_col].max()))
    ax.set_ylim(0.0, 1.02)
    savefig(out_dir / "fig2a_median_retention_vs_budget_fullscale")

    # Figure 2b: median retention versus budget, zoomed y-scale.
    ymin = float(min(ret[trace_ret_col].min(), ret[blind_ret_col].min()))
    ymin = max(0.0, ymin - 0.03)
    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    ax.plot(ret[budget_col], ret[trace_ret_col], marker="o", label=f"TRACE, AUC={median_trace_auc:.3f}")
    ax.plot(ret[budget_col], ret[blind_ret_col], marker="o", label=f"Blind random, AUC={median_blind_auc:.3f}")
    ax.set_xlabel("Budget fraction of full trial budget")
    ax.set_ylabel("Median score retention")
    ax.set_title("Median retention versus budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(float(ret[budget_col].min()), float(ret[budget_col].max()))
    ax.set_ylim(ymin, 1.005)
    savefig(out_dir / "fig2b_median_retention_vs_budget_zoomed")

    # Figure 3: ECDF of AUC retention, useful for appendix.
    x_trace_auc, y_trace_auc = ecdf(ds[trace_auc_col].astype(float))
    x_blind_auc, y_blind_auc = ecdf(ds[blind_auc_col].astype(float))

    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    ax.step(x_trace_auc, y_trace_auc, where="post", label=f"TRACE, median={median_trace_auc:.3f}")
    ax.step(x_blind_auc, y_blind_auc, where="post", label=f"Blind random, median={median_blind_auc:.3f}")
    ax.axvline(median_trace_auc, linestyle="--", linewidth=1)
    ax.axvline(median_blind_auc, linestyle="--", linewidth=1)
    ax.set_xlabel("AUC retention")
    ax.set_ylabel("Cumulative fraction of datasets")
    ax.set_title("AUC retention distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    xmin = max(0.0, min(x_trace_auc.min(), x_blind_auc.min()) - 0.02)
    ax.set_xlim(xmin, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0.0, y=0.0)
    savefig(out_dir / "fig3_auc_retention_ecdf")

    summary_lines = [
        f"n_datasets: {aggregate.get('n_datasets')}",
        f"random_seeds: {aggregate.get('random_seeds')}",
        f"median_trace_hit95_progress: {median_trace_hit}",
        f"median_blind_random_hit95_progress: {median_blind_hit}",
        f"relative_hit95_budget_reduction: {(median_blind_hit - median_trace_hit) / median_blind_hit}",
        f"median_trace_auc_retention: {median_trace_auc}",
        f"median_blind_random_auc_retention: {median_blind_auc}",
        f"auc_absolute_gain: {auc_abs_gain}",
        f"auc_gap_to_ideal_trace: {trace_gap}",
        f"auc_gap_to_ideal_blind_random: {blind_gap}",
        f"auc_gap_reduction_ratio: {gap_reduction}",
        "",
        "Paper-ready wording:",
        (
            f"TRACE reaches 95% of the exhaustive-search optimum after a median "
            f"{pct(median_trace_hit)} of the full trial budget, compared with "
            f"{pct(median_blind_hit)} for blind randomized path-order schedules."
        ),
        (
            f"TRACE also improves median AUC retention from {median_blind_auc:.3f} "
            f"to {median_trace_auc:.3f}, an absolute gain of {auc_abs_gain:.3f}."
        ),
        (
            f"Because AUC retention is bounded above by 1.0 and both methods are near "
            f"the ceiling, this corresponds to reducing the remaining area deficit "
            f"to the ideal curve by {100.0 * gap_reduction:.1f}%."
        ),
    ]
    (out_dir / "figure_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Figures written to: {out_dir}")
    for p in sorted(out_dir.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
