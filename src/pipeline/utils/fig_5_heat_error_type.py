#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_error_type_heatmap_max.py
-----------------------------------------------------------------
• (0,0) = GroundTruth 最大 Combined Score，并参与色阶计算
• 其余 15 格 = 各清洗算法下的最大 Combined Score
• 打印 4 × 4 分数矩阵；保存 EPS + PDF（tight 布局）
• 本版：标题与坐标轴标签改为中文，使用宋体
"""

from pathlib import Path
import re, subprocess
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties          # ← 中文字体

# ── 中文字体（如无宋体请改为 SimHei / 微软雅黑） ────────────────────────────
cn_font        = FontProperties(family='SimSun', size=16)
cn_font_title  = FontProperties(family='SimSun', size=18)
# ---------------------------------------------------------------------------

TASKS     = ["beers", "flights", "hospital", "rayyan"]
CSV_ROOT  = Path("../../../results/analysis_results")
FIG_ROOT  = Path("../../../task_progress/figures/5.3.2graph")
FIG_ROOT.mkdir(parents=True, exist_ok=True)

LEVELS    = [0, 5, 10, 15]                  # (%) 刻度
CMAP_BASE = plt.colormaps["coolwarm"]
NUM_RE    = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_numeric(series: pd.Series) -> pd.Series:
    def _extract(x):
        m = NUM_RE.search(str(x))
        return float(m.group()) if m else np.nan
    return series.apply(_extract).astype(float)

def best_gt_score(df: pd.DataFrame) -> Optional[float]:
    mask = df["cleaning_method"].str.lower().str.contains("ground")
    return df.loc[mask, "Combined Score"].max() if mask.any() else None

def print_matrix(name: str, mat: pd.DataFrame):
    print(f"\n=== {name} ===")
    hdr = "Anom\\Miss | " + " | ".join(f"{c:>9}%" for c in mat.columns)
    print(hdr); print("-" * len(hdr))
    for a in mat.index:
        row = " | ".join(f"{mat.loc[a,m]:>9.4f}"
                         if not np.isnan(mat.loc[a,m]) else "    NaN   "
                         for m in mat.columns)
        print(f"{a:>10}% | {row}")

def draw_one(task: str):
    csv = CSV_ROOT / f"{task}_cluster.csv"
    if not csv.is_file():
        print(f"[WARN] {csv} 不存在，跳过 {task}")
        return

    df = pd.read_csv(csv)

    for col in ["anomaly", "missing", "Combined Score"]:
        if col not in df.columns:
            raise KeyError(f"{csv} 缺少列 {col}")
        df[col] = parse_numeric(df[col])
    df = df.dropna(subset=["anomaly", "missing", "Combined Score"])

    gt_val = best_gt_score(df)

    non_gt = df[~df["cleaning_method"].str.lower().str.contains("ground")]
    pivot = (non_gt.groupby(["anomaly", "missing"])["Combined Score"]
                   .max()
                   .reset_index()
                   .pivot(index="anomaly", columns="missing",
                          values="Combined Score")
                   .reindex(index=LEVELS, columns=LEVELS))

    if gt_val is not None:
        pivot.loc[0, 0] = gt_val

    print_matrix(task, pivot)

    vals = pivot.values[~np.isnan(pivot.values)]
    vmin, vmax = vals.min(), vals.max()
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 1e-3, vmax + 1e-3

    cmap = CMAP_BASE.copy()
    cmap.set_bad("white")

    fig = plt.figure(figsize=(6.3, 4.3), dpi=300)
    ax = fig.add_subplot(111)
    im = ax.imshow(pivot, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

    labels = [f"{x}%" for x in LEVELS]
    ax.set_xticks(range(len(LEVELS)), labels, fontproperties=cn_font)
    ax.set_yticks(range(len(LEVELS)), labels, fontproperties=cn_font)

    # ── 中文坐标标题 ───────────────────────────────────────────────────────
    ax.set_xlabel("缺失率", fontproperties=cn_font)
    ax.set_ylabel("异常率", fontproperties=cn_font)

    # ── 中文总标题 ────────────────────────────────────────────────────────
    ax.set_title(f"{task.capitalize()}：错误类型热图",
                 fontproperties=cn_font_title, pad=10)

    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.set_xticks(np.arange(-.5, len(LEVELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(LEVELS), 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=0.4)

    for i, a in enumerate(LEVELS):
        for j, m in enumerate(LEVELS):
            v = pivot.loc[a, m]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=16, weight="bold", color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("综合得分", fontproperties=cn_font, fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()

    eps = FIG_ROOT / f"{task}_heatmap.eps"
    pdf = FIG_ROOT / f"{task}_heatmap.pdf"
    fig.savefig(eps, format="eps", bbox_inches="tight")
    try:
        subprocess.run(
            ["epstopdf", str(eps), "--outfile", str(pdf)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        fig.savefig(pdf, format="pdf", bbox_inches="tight")

    plt.close(fig)
    print(f"[OK] {task}: 图像保存为 {eps.name} / {pdf.name}")

if __name__ == "__main__":
    for task in TASKS:
        draw_one(task)
