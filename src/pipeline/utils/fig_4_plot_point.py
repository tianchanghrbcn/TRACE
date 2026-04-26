#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 生成各数据集的 mean–SD 散点图（tight 布局，直接 PDF）
#
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties          # ← 新增

# ── 中文字体（如无宋体可改为 SimHei / 微软雅黑） ─────────────────────────
cn_font        = FontProperties(family='SimSun')            # 轴标签
cn_font_title  = FontProperties(family='SimSun', size=18)   # 标题

# ---------- 0. 目录 ----------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]
CSV_DIR  = ROOT_DIR / "results" / "analysis_results"
PDF_DIR  = ROOT_DIR / "task_progress" / "figures" / "5.3.1graph"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. 读取 + 数值化 --------------------------------------------------
numeric_cols = ["Combined Score"]
dfs = []
for p in CSV_DIR.glob("*.csv"):
    df = pd.read_csv(
        p,
        na_values=["", "NA", "N/A", "-", "null"],
        keep_default_na=True
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[^\d.\-eE+]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )
    dfs.append(df)

if not dfs:
    raise SystemExit(f"❌ 找不到 csv 于 {CSV_DIR}")

df_all = pd.concat(dfs, ignore_index=True)

# ---------- 2. 计算相对均值 + SD ---------------------------------------------
gt = (
    df_all.query("cleaning_method == 'GroundTruth'")
          .groupby(["task_name", "cluster_method"])["Combined Score"]
          .mean()
          .rename("GT_score")
)
df = (
    df_all.merge(gt, on=["task_name", "cluster_method"])
          .assign(rel_score=lambda d: 100 * d["Combined Score"] / d["GT_score"])
)

stats = (
    df.groupby(["task_name", "cleaning_method", "cluster_method"])
      .agg(rel_mean=("rel_score", "mean"),
           sd=("Combined Score", "std"))
      .reset_index()
)

# ---------- 3. 样式映射 -------------------------------------------------------
cluster_markers = {
    "KMEANS": "o", "KMEANSNF": "s", "KMEANSPPS": "P",
    "GMM": "D", "DBSCAN": "^", "HC": "v",
}
cleaner_palette = sns.color_palette("tab10", n_colors=9)
cleaner_list = sorted(stats["cleaning_method"].unique())
cleaner_colors = {c: cleaner_palette[i % 10] for i, c in enumerate(cleaner_list)}

# ---------- 4. 逐 task 绘制 ---------------------------------------------------
for task, sub in stats.groupby("task_name"):
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # (1) 散点
    for _, row in sub.iterrows():
        ax.scatter(row["rel_mean"], row["sd"],
                   marker=cluster_markers.get(row["cluster_method"], "o"),
                   color=cleaner_colors[row["cleaning_method"]],
                   s=120, alpha=.85,
                   edgecolor="k", linewidth=.4)

    # (2) 参考线
    ax.axvline(100, color="grey", lw=.8, ls="--")
    ax.axhline(sub["sd"].median(), color="grey", lw=.8, ls="--")

    # (3) 中文标题 & 轴
    ax.set_title(f"均值–标准差散点图 · {task}", fontproperties=cn_font_title)        # ★
    ax.set_xlabel("相对平均得分（占 GT 的 %）", fontproperties=cn_font, fontsize=16)  # ★
    ax.set_ylabel("得分标准差",             fontproperties=cn_font, fontsize=16)  # ★
    ax.tick_params(axis='both', which='major', labelsize=14)

    # (4) 自定义图例
    handles_clean = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cleaner_colors[c], markersize=9,
                   label=c)
        for c in cleaner_list
    ]
    handles_cluster = [
        plt.Line2D([0], [0], marker=cluster_markers[k], color='k',
                   markersize=9, linestyle='', label=k)
        for k in cluster_markers
    ]

    leg1 = ax.legend(handles=handles_clean, title="Cleaning",
                     loc='upper left', bbox_to_anchor=(0.02, 0.98),
                     borderpad=0.5, frameon=True, framealpha=0.4,
                     fontsize=12, title_fontsize=12)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=handles_cluster, title="Cluster",
                     loc='upper left', bbox_to_anchor=(0.38, 0.98),
                     borderpad=0.5, frameon=True, framealpha=0.4,
                     fontsize=12, title_fontsize=12)

    # (5) 保存 PDF
    pdf_path = PDF_DIR / f"mean_sd_scatter_{task}.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved {pdf_path}")

print(f"✅ Figures saved to {PDF_DIR.resolve()}")
