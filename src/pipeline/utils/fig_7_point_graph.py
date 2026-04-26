#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_6.4.3_scatter.py
───────────────────────────────────────────────────────────
• 每张图 X 轴列顺序：不同 dataset_id 按其最小 error_rate 升序排列
• 其它绘图 / 输出逻辑保持不变
"""
import os, sys, warnings
import matplotlib
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# ── 字体设置 ───────────────────────────────────────────────
for fp in (
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\Times New Roman.ttf",
    r"/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
):
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        break
else:
    warnings.warn("Times New Roman 未找到，使用系统 serif。")

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "TimesNewRomanPSMT", "Times New Roman PS"],
    "pdf.fonttype": 42,
})
sns.set_style("ticks")

# ── 工具函数 ───────────────────────────────────────────────
def save_pdf(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", format="pdf")

# ── 主程序 ─────────────────────────────────────────────────
def main():
    tasks    = ["beers", "rayyan", "flights", "hospital"]
    data_dir = Path("..") / ".." / ".." / "results" / "analysis_results"
    out_dir  = Path("..") / ".." / ".." / "task_progress" / "figures" / "6.4.3graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for t in tasks:
        f = data_dir / f"{t}_summary.xlsx"
        if not f.exists():
            warnings.warn(f"{f} missing, skip")
            continue
        df = pd.read_excel(f)
        df["task_name"] = t
        frames.append(df)

    if not frames:
        sys.exit("No data loaded – check paths.")
    df_all = pd.concat(frames, ignore_index=True)

    need_cols = ["dataset_id", "cluster_method", "task_name", "error_rate",
                 "EDR", "F1", "Combined Score", "Silhouette Score", "DB_relative"]
    for c in need_cols:
        if c not in df_all.columns:
            sys.exit(f"[ERR] missing column: {c}")

    df_all["dataset_id"]       = pd.to_numeric(df_all["dataset_id"])
    df_all["Silhouette Score"] = (df_all["Silhouette Score"] + 1) / 2

    metric_pairs = [("EDR", "Combined Score"),
                    ("F1",  "Silhouette Score")]

    # ── 逐数据集绘制 ───────────────────────────────────────
    for task in tasks:
        sub = df_all[df_all["task_name"] == task].copy()
        if sub.empty:
            continue

        # ── 关键：根据 dataset_id 的 **最小** error_rate 排序 ──
        order_df = (sub.groupby("dataset_id", as_index=False)
                        .agg(min_err=("error_rate", "min"))
                        .sort_values("min_err"))
        id_order = order_df["dataset_id"].tolist()          # 升序 id
        pos_map  = {ds: idx for idx, ds in enumerate(id_order)}

        # 一条竖线隔开最后一列，仅作视觉参考
        boundaries = [len(id_order) - 0.5]

        for xm, ym in metric_pairs:
            # Pearson r 先按 (dataset_id, cluster_method) 聚合
            rows = []
            for (ds, cm), g in sub.groupby(["dataset_id", "cluster_method"]):
                rows.append({
                    "dataset_id": ds,
                    "cluster_method": cm,
                    "r": np.nan if len(g) < 2 else pearsonr(g[xm], g[ym])[0]
                })
            long = (pd.DataFrame(rows)
                    .fillna({"r": 0})
                    .assign(xpos=lambda d: d["dataset_id"].map(pos_map))
                    .assign(cluster_method=lambda d:
                            pd.Categorical(d["cluster_method"],
                                           sorted(d["cluster_method"].unique()),
                                           ordered=True)))

            # ── 绘图 ────────────────────────────────────
            fig = plt.figure(figsize=(6.5, 4.5))
            ax  = sns.scatterplot(
                    data=long,
                    x="xpos", y="cluster_method",
                    size=long["r"].abs(), hue="r",
                    palette="RdBu", sizes=(60, 600),
                    edgecolor="black", alpha=.85, legend=False
                 )
            ax.axis("tight")
            for bx in boundaries:
                ax.axvline(bx, ls="--", lw=1, c="gray")

            ax.set_xticks([]); ax.set_xlabel("")
            ax.set_ylabel(""); ax.tick_params(axis="y", labelsize=16)
            for tick in ax.get_yticklabels():
                tick.set_rotation(0); tick.set_ha("right"); tick.set_va("center")

            sm   = matplotlib.cm.ScalarMappable(
                        norm=plt.Normalize(-1, 1), cmap="RdBu")
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                                pad=0.12, fraction=0.035, aspect=30)
            cbar.set_label("Pearson r", fontsize=16)
            cbar.ax.tick_params(labelsize=16)

            plt.tight_layout()
            tag = task[:2].upper()
            pdf = out_dir / f"{tag}_{xm}_vs_{ym}.pdf"
            save_pdf(fig, pdf)
            plt.close()
            print("saved →", pdf)

    print("✅ Figures regenerated with columns ordered by min(error_rate).")

# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    main()
