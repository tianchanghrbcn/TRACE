#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制各数据集“错误率–综合得分”折线图（直接 PDF）。

输入 : ../../../results/analysis_results/{task}_cluster.csv
输出 : ../../../task_progress/figures/5.3.2graph/{task}_combined_score.pdf
"""

import subprocess, shutil, importlib   # 仍保留，若想再扩展为 SVG→PDF 可用
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties    # 中文字体

# ── 中文字体（如无宋体请改成 SimHei / 微软雅黑） ────────────────────────────
cn_font        = FontProperties(family='SimSun', size=16)
cn_font_title  = FontProperties(family='SimSun', size=18)

# ------------ 1. constants ---------------------------------------------------
TASK_NAMES = ["beers", "flights", "hospital", "rayyan"]
CSV_ROOT   = Path("../../../results/analysis_results")
FIG_ROOT   = Path("../../../task_progress/figures/5.3.2graph")
FIG_ROOT.mkdir(parents=True, exist_ok=True)

# high-contrast colour池 & marker 池
COLOR_LIST = (
    list(mpl.colormaps["tab10"].colors)
    + list(mpl.colormaps["Set1"].colors)
    + list(mpl.colormaps["Dark2"].colors)
    + list(mpl.colormaps["tab20"].colors)
    + list(mpl.colormaps["tab20b"].colors)
    + list(mpl.colormaps["tab20c"].colors)
)
MARKERS = ["o","s","D","^","v",">","<","h","p","X","8","*","P"]

# ------------ 2. 全局 style map（清洗方法→颜色/形状） -------------------------
cleaners = set()
for t in TASK_NAMES:
    f = CSV_ROOT / f"{t}_cluster.csv"
    if f.exists():
        cleaners |= set(pd.read_csv(f, usecols=["cleaning_method"])
                        ["cleaning_method"].unique())

if len(cleaners) > len(COLOR_LIST):
    COLOR_LIST *= len(cleaners) // len(COLOR_LIST) + 1

STYLE_MAP = {cl: (COLOR_LIST[i], MARKERS[i % len(MARKERS)])
             for i, cl in enumerate(sorted(cleaners))}

# ------------ 3. 主循环 -------------------------------------------------------
for task in TASK_NAMES:
    csv = CSV_ROOT / f"{task}_cluster.csv"
    if not csv.exists():
        print(f"[WARN] {csv} 缺失，已跳过");  continue

    df = pd.read_csv(csv)
    needed = {"error_rate","Combined Score","cleaning_method"}
    if not needed <= set(df.columns):
        print(f"[ERROR] {csv} 缺少列 {needed - set(df.columns)}");  continue

    # —— ³ error_bin = 最近的 5 的倍数（整数） ————————————————
    df["error_bin"] = ((df["error_rate"] / 5).round() * 5).astype(int)

    # 同一 error_bin、cleaning_method 取最高分（可按需改成均值/中位数）
    best = (df.groupby(["cleaning_method", "error_bin"])
              ["Combined Score"].max().reset_index())

    # x 轴次序
    x_order = sorted(best["error_bin"].unique())

    plt.figure(figsize=(6.5, 4.5))
    for cln, sub in best.groupby("cleaning_method"):
        y = (sub.set_index("error_bin")
                 .reindex(x_order)["Combined Score"].values)
        color, marker = STYLE_MAP[cln]
        plt.plot(x_order, y, label=cln,
                 color=color, marker=marker,
                 linewidth=1.8, markersize=8)

    # —— 中文标题 & 轴标签 —————————————————————————————
    plt.title(f"{task.capitalize()}：综合得分随错误率变化",
              fontproperties=cn_font_title, pad=6)
    plt.xlabel("误差率 (%)", fontproperties=cn_font)
    plt.ylabel("综合得分",    fontproperties=cn_font)
    plt.xticks(fontsize=16);  plt.yticks(fontsize=16)

    # 图例
    leg = plt.legend(title="清洗方法",
                     fontsize=11,
                     loc="upper right",
                     framealpha=0.4)
    leg.get_title().set_fontproperties(cn_font)
    for text in leg.get_texts():
        text.set_fontproperties(cn_font)
    leg.get_frame().set_edgecolor("0.5"); leg.get_frame().set_linewidth(0.8)

    plt.grid(alpha=0.25)
    plt.tight_layout()

    pdf_path = FIG_ROOT / f"{task}_combined_score_cleaning.pdf"
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] saved {pdf_path}")

print("✅ PDF 图已生成:", FIG_ROOT.resolve())
