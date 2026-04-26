#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_top10_per_dataset_abs_cn.py
-------------------------------------------------------------
为 beers / flights / hospital / rayyan 各绘制
  • Top‑10 组合的 (均值 ± 标准差) 绝对 Combined Score 柱状图
  • 参考虚线 = GroundTruth 最优组合平均得分
  • 纵轴统一 0‑1，等分为 10 份
—— 中文界面：中文 SimSun，必要英文保留 Times New Roman
-------------------------------------------------------------
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties   # ← 新增

# ---- 0. 字体设置 ---------------------------------------------------------
matplotlib.rcParams["font.family"] = ["Times New Roman", "sans-serif"]  # 默认英文字体
matplotlib.rcParams["axes.unicode_minus"] = False                       # 负号正常显示
cn_font = FontProperties(family='SimSun')                               # 仅中文用宋体
cn_font_title = FontProperties(family='SimSun', size=16)
cn_font_legend = FontProperties(family='SimSun', size=14)

UTIL_DIR = Path(__file__).resolve().parent
RES_DIR  = UTIL_DIR / ".." / ".." / ".." / "results" / "analysis_results"
FIG_DIR  = UTIL_DIR / ".." / ".." / ".." / "task_progress" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. 读取 CSV 并清洗 ------------------------------------------
dfs = []
for p in RES_DIR.glob("*.csv"):
    df = pd.read_csv(
        p,
        na_values=["", "NA", "N/A", "-", "null"],
        keep_default_na=True,
    )
    if "Combined Score" in df.columns:
        df["Combined Score"] = (
            df["Combined Score"].astype(str)
                                .str.replace(r"[^\d.\-eE+]", "", regex=True)
                                .pipe(pd.to_numeric, errors="coerce")
        )
    dfs.append(df)

if not dfs:
    raise SystemExit(f"❌ 未找到 csv 于 {RES_DIR}")
df_all = pd.concat(dfs, ignore_index=True)

# ---------- 2. 组合均值 ± 方差 ------------------------------------------
stats = (
    df_all
    .groupby(["task_name", "cleaning_method", "cluster_method"])
    .agg(abs_mean=("Combined Score", "mean"),
         var      =("Combined Score", "var"))
    .reset_index()
    .assign(label=lambda d: d["cleaning_method"] + " + " + d["cluster_method"])
)

# ---------- 3. 每个数据集 GroundTruth 最佳组合 ---------------------------
gt_best = (
    stats.query("cleaning_method == 'GroundTruth'")
         .loc[lambda d: d.groupby("task_name")["abs_mean"].idxmax()]
         .set_index("task_name")["abs_mean"]        # Series: task → best GT mean
)

# ---------- 4. 逐数据集绘制 Top‑10 --------------------------------------
for task, sub in stats.groupby("task_name"):
    top10 = (
        sub.query("cleaning_method != 'GroundTruth'")
           .nlargest(10, "abs_mean")
           .reset_index(drop=True)
    )

    x = np.arange(len(top10))
    colors = sns.color_palette("tab10", len(top10))

    plt.figure(figsize=(6, 5), tight_layout=True)

    plt.bar(
        x,
        top10["abs_mean"],
        yerr=np.sqrt(top10["var"]),
        width=0.6,
        capsize=4,
        color=colors,
        edgecolor="none",
    )

    # GroundTruth 参考虚线
    gt_ref = gt_best[task]
    plt.axhline(
        gt_ref, ls="--", lw=1, c="black",
        label=f"GroundTruth 最佳均值 = {gt_ref:.3f}"
    )

    # ---- 统一纵轴 0‑1 并等分 10 份 -----------------------------
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 11), fontsize=14)

    plt.xticks(x, top10["label"], rotation=18, ha="right", fontsize=12)
    plt.ylabel("平均综合分数", fontsize=15, fontproperties=cn_font)  # 中文宋体
    plt.title(f"“{task}” 数据集 Top-10 组合（均值 ± 标准差）", fontproperties=cn_font_title)                # ↑ 字号 16 → 20
    plt.legend(framealpha=0.35, prop=cn_font_legend)        # 中文宋体
    plt.savefig(FIG_DIR / "5.3.1graph" / f"top10_bar_error_{task}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "5.3.1graph" / f"top10_bar_error_{task}.eps",
                format="eps", bbox_inches="tight")
    plt.close()

print("Top‑10 柱状图已保存")
