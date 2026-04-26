#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制“错误率–综合得分”折线图（每条线 = cluster_method；直接 PDF）。

输入 : ../../../results/analysis_results/{task}_cluster.csv
输出 : ../../../task_progress/figures/5.3.2graph/{task}_combined_score.pdf
"""
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ── 中文字体（如无宋体请改为 SimHei / 微软雅黑） ────────────────────────────
cn_font        = FontProperties(family='SimSun', size=16)
cn_font_title  = FontProperties(family='SimSun', size=18)

# ---------- 目录 / 数据集 -----------------------------------------------------
TASK_NAMES = ["beers", "flights", "hospital", "rayyan"]
CSV_ROOT   = Path("../../../results/analysis_results")
FIG_ROOT   = Path("../../../task_progress/figures/5.3.2graph")
FIG_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- 颜色 & 标记池 -----------------------------------------------------
COLOR_LIST = (
    list(mpl.colormaps["tab10"].colors)
    + list(mpl.colormaps["Set1"].colors)
    + list(mpl.colormaps["Dark2"].colors)
    + list(mpl.colormaps["tab20"].colors)
    + list(mpl.colormaps["tab20b"].colors)
    + list(mpl.colormaps["tab20c"].colors)
)
MARKERS = ["o","s","D","^","v",">","<","h","p","X","8","*","P"]

# ---------- 构建 cluster_method → 颜色/标记 映射 -----------------------------
clusterers = set()
for t in TASK_NAMES:
    f = CSV_ROOT / f"{t}_cluster.csv"
    if f.exists():
        clusterers |= set(pd.read_csv(f, usecols=["cluster_method"])
                          ["cluster_method"].unique())

if len(clusterers) > len(COLOR_LIST):
    COLOR_LIST *= (len(clusterers) // len(COLOR_LIST) + 1)

STYLE_MAP = {clu: (COLOR_LIST[i], MARKERS[i % len(MARKERS)])
             for i, clu in enumerate(sorted(clusterers))}

# ---------- 主循环 -----------------------------------------------------------
for task in TASK_NAMES:
    csv = CSV_ROOT / f"{task}_cluster.csv"
    if not csv.exists():
        print(f"[WARN] {csv} 缺失，已跳过");  continue

    df = pd.read_csv(csv)
    need = {"error_rate", "Combined Score", "cleaning_method", "cluster_method"}
    if not need <= set(df.columns):
        print(f"[ERROR] {csv} 缺少列 {need - set(df.columns)}");  continue

    # — 误差率取最近 5 的倍数 —
    df["error_bin"] = ((df["error_rate"] / 5).round() * 5).astype(int)
    x_order = sorted(df["error_bin"].unique())

    # 每 cluster_method × error_bin 取最高得分
    best = (df.groupby(["cluster_method", "error_bin"])
              ["Combined Score"].max().reset_index())

    # ---------- 绘图 ---------------------------------------------------------
    plt.figure(figsize=(6.5, 4.5))
    for clu, sub in best.groupby("cluster_method"):
        y = (sub.set_index("error_bin")
                 .reindex(x_order)["Combined Score"].values)
        color, marker = STYLE_MAP[clu]
        plt.plot(x_order, y, label=clu,
                 color=color, marker=marker,
                 linewidth=1.8, markersize=8)

    # 中文标题与轴
    plt.title(f"{task.capitalize()}：综合得分随错误率变化",
              fontproperties=cn_font_title, pad=6)
    plt.xlabel("误差率 (%)", fontproperties=cn_font)
    plt.ylabel("综合得分",   fontproperties=cn_font)
    plt.xticks(fontsize=16);  plt.yticks(fontsize=16)
    plt.grid(alpha=0.25)

    # 图例
    leg = plt.legend(title="聚类方法", fontsize=12, loc="upper right",
                     framealpha=0.4)
    leg.get_title().set_fontproperties(cn_font)
    for txt in leg.get_texts():
        txt.set_fontproperties(cn_font)
    leg.get_frame().set_edgecolor("0.5"); leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    # 直接保存 PDF
    pdf_path = FIG_ROOT / f"{task}_combined_score_cluster.pdf"
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] saved {pdf_path}")

print("✅ PDF 图已生成:", FIG_ROOT.resolve())
