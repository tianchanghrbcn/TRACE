#!/usr/bin/env python3
"""Build alpha-sensitivity evidence from saved paper summary workbooks.

This script does not rerun cleaning or clustering. It recomputes the outcome
score under several alpha values using saved Silhouette and Davies-Bouldin
scores, then checks whether method rankings remain stable.

Outputs:
  analysis/validity_sensitivity/alpha_row_scores.csv
  analysis/validity_sensitivity/alpha_combo_medians.csv
  analysis/validity_sensitivity/alpha_cleaner_medians.csv
  analysis/validity_sensitivity/alpha_clusterer_medians.csv
  analysis/validity_sensitivity/alpha_rank_correlation.csv
  analysis/validity_sensitivity/alpha_top_combo_stability.csv
  analysis/validity_sensitivity/alpha_sensitivity_summary.json
  analysis/validity_sensitivity/alpha_sensitivity_report.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASKS = ["beers", "flights", "hospital", "rayyan"]
REQUIRED_COLUMNS = [
    "cleaning_method",
    "cluster_method",
    "Silhouette Score",
    "Davies-Bouldin Score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build alpha sensitivity from saved summary workbooks.")
    parser.add_argument("--summary-dir", type=Path, default=Path("results/analysis_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/validity_sensitivity"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.47, 0.50, 0.75])
    parser.add_argument("--base-alpha", type=float, default=0.47)
    return parser.parse_args()


def read_summary_workbook(path: Path) -> pd.DataFrame:
    """Read the sheet that contains the required paper-summary columns."""
    obj = pd.read_excel(path, sheet_name=None)

    for sheet_name, df in obj.items():
        columns = set(str(c) for c in df.columns)
        if all(col in columns for col in REQUIRED_COLUMNS):
            out = df.copy()
            out["source_sheet"] = sheet_name
            return out

    available = {
        sheet: list(df.columns)
        for sheet, df in obj.items()
    }
    raise ValueError(f"No sheet in {path} contains required columns. Available: {available}")


def load_all_summaries(summary_dir: Path) -> pd.DataFrame:
    frames = []

    for task in TASKS:
        path = summary_dir / f"{task}_summary.xlsx"
        if not path.exists():
            print(f"[WARN] Missing summary workbook: {path}")
            continue

        df = read_summary_workbook(path)
        df["task_name"] = task
        df["source_workbook"] = str(path)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No summary workbooks found under {summary_dir}")

    merged = pd.concat(frames, ignore_index=True)

    for column in ["Silhouette Score", "Davies-Bouldin Score", "Combined Score", "error_rate", "dataset_id"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    merged["cleaning_method"] = merged["cleaning_method"].astype(str).str.strip()
    merged["cluster_method"] = merged["cluster_method"].astype(str).str.strip()

    if "dataset_id" in merged.columns:
        merged["dataset_id"] = merged["dataset_id"].astype("Int64")
    else:
        merged["dataset_id"] = pd.NA

    if "error_rate" in merged.columns:
        merged["error_rate_bin"] = ((merged["error_rate"] / 5).round() * 5).astype("Int64")
    else:
        merged["error_rate_bin"] = pd.NA

    return merged


def compute_h_alpha(df: pd.DataFrame, alpha: float) -> pd.Series:
    sil = pd.to_numeric(df["Silhouette Score"], errors="coerce")
    db = pd.to_numeric(df["Davies-Bouldin Score"], errors="coerce")

    s = (sil + 1.0) / 2.0
    d = 1.0 / (1.0 + db)

    eps = 1.0e-12
    s = np.maximum(s, eps)
    d = np.maximum(d, eps)

    return 1.0 / (alpha / s + (1.0 - alpha) / d)


def build_long_scores(df: pd.DataFrame, alphas: list[float]) -> pd.DataFrame:
    rows = []

    keep_cols = [
        "task_name",
        "dataset_id",
        "error_rate_bin",
        "cleaning_method",
        "cluster_method",
        "Silhouette Score",
        "Davies-Bouldin Score",
    ]

    if "Combined Score" in df.columns:
        keep_cols.append("Combined Score")

    base = df[keep_cols].copy()

    for alpha in alphas:
        tmp = base.copy()
        tmp["alpha"] = alpha
        tmp["H_alpha"] = compute_h_alpha(df, alpha)
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["H_alpha"])
    return out


def median_table(long_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        long_df.groupby(["alpha"] + group_cols, observed=False, as_index=False)
        .agg(
            median_H=("H_alpha", "median"),
            mean_H=("H_alpha", "mean"),
            n_rows=("H_alpha", "size"),
        )
        .sort_values(["alpha"] + group_cols)
    )


def spearman_rank_corr(a: pd.Series, b: pd.Series) -> float | None:
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return None

    aa = a.loc[common].rank(ascending=False, method="average")
    bb = b.loc[common].rank(ascending=False, method="average")

    corr = aa.corr(bb, method="pearson")
    if pd.isna(corr):
        return None
    return float(corr)


def rank_correlations(
    table: pd.DataFrame,
    base_alpha: float,
    scope_name: str,
    item_cols: list[str],
) -> list[dict[str, Any]]:
    rows = []

    for task in sorted(table["task_name"].dropna().unique()):
        task_df = table[table["task_name"] == task].copy()
        base_df = task_df[np.isclose(task_df["alpha"], base_alpha)].copy()

        if base_df.empty:
            continue

        base_index = pd.MultiIndex.from_frame(base_df[item_cols])
        base_scores = pd.Series(base_df["median_H"].to_numpy(), index=base_index)

        for alpha in sorted(task_df["alpha"].unique()):
            current_df = task_df[np.isclose(task_df["alpha"], alpha)].copy()
            current_index = pd.MultiIndex.from_frame(current_df[item_cols])
            current_scores = pd.Series(current_df["median_H"].to_numpy(), index=current_index)

            corr = spearman_rank_corr(base_scores, current_scores)

            rows.append(
                {
                    "scope": scope_name,
                    "task_name": task,
                    "alpha": float(alpha),
                    "base_alpha": float(base_alpha),
                    "common_item_count": int(len(base_scores.index.intersection(current_scores.index))),
                    "spearman_with_base": corr,
                }
            )

    return rows


def top_combo_stability(combo_table: pd.DataFrame, base_alpha: float) -> pd.DataFrame:
    rows = []

    for task in sorted(combo_table["task_name"].dropna().unique()):
        task_df = combo_table[combo_table["task_name"] == task].copy()
        base_df = task_df[np.isclose(task_df["alpha"], base_alpha)].copy()

        if base_df.empty:
            continue

        base_top = base_df.sort_values("median_H", ascending=False).iloc[0]
        base_combo = (base_top["cleaning_method"], base_top["cluster_method"])

        for alpha in sorted(task_df["alpha"].unique()):
            current_df = task_df[np.isclose(task_df["alpha"], alpha)].copy()
            if current_df.empty:
                continue

            top = current_df.sort_values("median_H", ascending=False).iloc[0]
            combo = (top["cleaning_method"], top["cluster_method"])

            rows.append(
                {
                    "task_name": task,
                    "alpha": float(alpha),
                    "base_alpha": float(base_alpha),
                    "base_top_cleaner": base_combo[0],
                    "base_top_clusterer": base_combo[1],
                    "top_cleaner": combo[0],
                    "top_clusterer": combo[1],
                    "same_as_base_top": combo == base_combo,
                    "top_median_H": float(top["median_H"]),
                    "base_top_median_H": float(base_top["median_H"]),
                }
            )

    return pd.DataFrame(rows)


def score_sanity(df: pd.DataFrame, base_alpha: float) -> dict[str, Any]:
    """Compare recomputed H_alpha=base_alpha with existing Combined Score, if available."""
    if "Combined Score" not in df.columns:
        return {
            "available": False,
            "reason": "Combined Score column not found",
        }

    tmp = df.dropna(subset=["Combined Score", "Silhouette Score", "Davies-Bouldin Score"]).copy()
    if tmp.empty:
        return {
            "available": False,
            "reason": "No rows with Combined Score and internal metrics",
        }

    recomputed = compute_h_alpha(tmp, base_alpha)
    existing = pd.to_numeric(tmp["Combined Score"], errors="coerce")

    return {
        "available": True,
        "row_count": int(len(tmp)),
        "spearman_existing_vs_recomputed": float(existing.rank().corr(recomputed.rank(), method="pearson")),
        "pearson_existing_vs_recomputed": float(existing.corr(recomputed, method="pearson")),
        "median_abs_difference": float(np.nanmedian(np.abs(existing - recomputed))),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Alpha Sensitivity Report",
        "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Status: {summary['status']}",
        f"- Base alpha: {summary['base_alpha']}",
        f"- Tested alphas: {', '.join(str(a) for a in summary['alphas'])}",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Key statistics",
        "",
        f"- Input row count: {summary['input_row_count']}",
        f"- Long row-score count: {summary['row_score_count']}",
        f"- Minimum combo-rank Spearman vs. base: {summary['min_combo_spearman']}",
        f"- Minimum cleaner-rank Spearman vs. base: {summary['min_cleaner_spearman']}",
        f"- Minimum clusterer-rank Spearman vs. base: {summary['min_clusterer_spearman']}",
        f"- Top-combo stability rate: {summary['top_combo_stability_rate']}",
        "",
        "## Files",
        "",
    ]

    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alphas = sorted(set(float(a) for a in args.alphas))
    if args.base_alpha not in alphas:
        alphas.append(args.base_alpha)
        alphas = sorted(alphas)

    df = load_all_summaries(args.summary_dir)
    long_df = build_long_scores(df, alphas)

    combo_table = median_table(
        long_df,
        ["task_name", "cleaning_method", "cluster_method"],
    )
    cleaner_table = median_table(
        long_df,
        ["task_name", "cleaning_method"],
    )
    clusterer_table = median_table(
        long_df,
        ["task_name", "cluster_method"],
    )

    correlations = []
    correlations.extend(
        rank_correlations(
            combo_table,
            args.base_alpha,
            "cleaner_clusterer_combo",
            ["cleaning_method", "cluster_method"],
        )
    )
    correlations.extend(
        rank_correlations(
            cleaner_table,
            args.base_alpha,
            "cleaner",
            ["cleaning_method"],
        )
    )
    correlations.extend(
        rank_correlations(
            clusterer_table,
            args.base_alpha,
            "clusterer",
            ["cluster_method"],
        )
    )
    corr_df = pd.DataFrame(correlations)

    top_df = top_combo_stability(combo_table, args.base_alpha)

    row_scores_path = args.output_dir / "alpha_row_scores.csv"
    combo_path = args.output_dir / "alpha_combo_medians.csv"
    cleaner_path = args.output_dir / "alpha_cleaner_medians.csv"
    clusterer_path = args.output_dir / "alpha_clusterer_medians.csv"
    corr_path = args.output_dir / "alpha_rank_correlation.csv"
    top_path = args.output_dir / "alpha_top_combo_stability.csv"
    summary_path = args.output_dir / "alpha_sensitivity_summary.json"
    report_path = args.output_dir / "alpha_sensitivity_report.md"

    long_df.to_csv(row_scores_path, index=False, encoding="utf-8-sig")
    combo_table.to_csv(combo_path, index=False, encoding="utf-8-sig")
    cleaner_table.to_csv(cleaner_path, index=False, encoding="utf-8-sig")
    clusterer_table.to_csv(clusterer_path, index=False, encoding="utf-8-sig")
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    nonbase_corr = corr_df[~np.isclose(corr_df["alpha"], args.base_alpha)].copy()

    def min_corr(scope: str) -> float | None:
        vals = nonbase_corr.loc[
            nonbase_corr["scope"] == scope,
            "spearman_with_base",
        ].dropna()
        if vals.empty:
            return None
        return float(vals.min())

    min_combo = min_corr("cleaner_clusterer_combo")
    min_cleaner = min_corr("cleaner")
    min_clusterer = min_corr("clusterer")

    nonbase_top = top_df[~np.isclose(top_df["alpha"], args.base_alpha)].copy()
    top_rate = None
    if not nonbase_top.empty:
        top_rate = float(nonbase_top["same_as_base_top"].mean())

    # This is a practical diagnostic status, not a claim about theorem-level robustness.
    accepted = (
        (min_combo is None or min_combo >= 0.75)
        and (min_cleaner is None or min_cleaner >= 0.75)
        and (min_clusterer is None or min_clusterer >= 0.75)
    )

    status = "PASS" if accepted else "CHECK"

    if status == "PASS":
        interpretation = (
            "The tested alpha values preserve high rank agreement with the base alpha. "
            "This supports the claim that the main outcome-level trends are not driven "
            "by a single scalarization weight."
        )
    else:
        interpretation = (
            "Some alpha values show lower rank agreement with the base alpha. "
            "Inspect alpha_rank_correlation.csv before making a paper-level robustness claim."
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary_dir": str(args.summary_dir),
        "alphas": alphas,
        "base_alpha": args.base_alpha,
        "input_row_count": int(len(df)),
        "row_score_count": int(len(long_df)),
        "min_combo_spearman": min_combo,
        "min_cleaner_spearman": min_cleaner,
        "min_clusterer_spearman": min_clusterer,
        "top_combo_stability_rate": top_rate,
        "score_sanity": score_sanity(df, args.base_alpha),
        "interpretation": interpretation,
        "outputs": {
            "row_scores": str(row_scores_path),
            "combo_medians": str(combo_path),
            "cleaner_medians": str(cleaner_path),
            "clusterer_medians": str(clusterer_path),
            "rank_correlation": str(corr_path),
            "top_combo_stability": str(top_path),
            "summary_json": str(summary_path),
            "report_md": str(report_path),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, summary)

    print(json.dumps(
        {
            "status": status,
            "input_row_count": summary["input_row_count"],
            "row_score_count": summary["row_score_count"],
            "min_combo_spearman": min_combo,
            "min_cleaner_spearman": min_cleaner,
            "min_clusterer_spearman": min_clusterer,
            "top_combo_stability_rate": top_rate,
            "output": str(summary_path),
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"[TRACE] Alpha sensitivity report written to: {report_path}")


if __name__ == "__main__":
    main()
