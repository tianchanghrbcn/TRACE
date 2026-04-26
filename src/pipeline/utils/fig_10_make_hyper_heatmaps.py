
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_10_make_hyper_heatmaps.py
严格读取四个 table10_*_hyper_shift.xlsx，并按行(错误率)×列(清洗方法)画热力图。
- 仅用 pandas + matplotlib（不依赖 seaborn）
- 识别列名模式：'<delta-var>.<cleaning>'，如 'Δk.Unified', 'Δeps.bigdansing', 'ΔminPts.scared'
- 丢弃 GroundTruth 列；大小写与常见拼写统一映射
- 输出 PDF 矢量图

在原有绘图逻辑不变的基础上，额外输出 6.4.1 / 6.4.2 所需统计：
- 6.4.1:
    * Δk / Δn_comp 在全部非基线单元中的“负向偏移比例”
    * 在中等噪声段（默认 10%--20%）中的“正向偏移比例”
- 6.4.2:
    * Δε 在全部非基线单元中的“近零偏移比例”
    * 在高噪声段（默认 >=20%）中的 Δε<0、ΔminPts>0 及其联动比例
    * 同时输出高噪声段与非高噪声段的对照，便于判断“更常”是否成立

说明：
- 统计默认排除 Mode 列（若存在），因为本节讨论的是“相对 Mode 的偏移”。
- “Δε 近零”的阈值默认取观测矩阵中最小的非零绝对偏移值；如你已知搜索步长，可在
  DEPS_SMALL_THRESHOLD_OVERRIDE 中手工指定。
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================== 1) 路径配置 ===================
# 按你的说明设置默认路径；也可以改成绝对路径
ROOT = Path(__file__).resolve().parents[3]  # .../AutoMLClustering_full/
IN_DIR  = ROOT / "task_progress" / "tables" / "6.4.4tables"
OUT_DIR = ROOT / "task_progress" / "figures" / "new_6.4graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)

F_KMEANS = IN_DIR / "table10_kmeans_hyper_shift.xlsx"
F_DBSCAN = IN_DIR / "table10_dbscan_hyper_shift.xlsx"
F_GMM    = IN_DIR / "table10_gmm_hyper_shift.xlsx"
# （HC 没有在这四张图中用到；如需扩展可按相同方式添加）

# =================== 2) 名称与映射 ===================
# 统一的清洗方法顺序（有则显示、无则跳过）
CLEAN_ORDER = [
    "Unified", "Baran", "BigDansing", "BoostClean",
    "HoloClean", "Horizon", "Mode", "SCAReD"
]

# 清洗方法别名映射（大小写/拼写统一）
CLEAN_ALIASES: Dict[str, str] = {
    "unified":    "Unified",
    "baran":      "Baran",
    "bigdansing": "BigDansing",
    "bigdans":    "BigDansing",
    "boostclean": "BoostClean",
    "holoclean":  "HoloClean",
    "horizon":    "Horizon",
    "mode":       "Mode",
    "scared":     "SCAReD",   # 原表常见写法
    "scareded":   "SCAReD",   # 兜底
    "scared_":    "SCAReD",
    "scared.":    "SCAReD",
    "scared-":    "SCAReD",
    "scared ":    "SCAReD",
}

# 变量名归一化（前缀）——仅用于从 DBSCAN 宽表中拆分 Δeps / ΔminPts
def normalize_var_name(s: str) -> str:
    s_low = s.lower()
    if "k" in s_low and "eps" not in s_low and "min" not in s_low and "comp" not in s_low:
        return "dk"         # Δk
    if "eps" in s_low:
        return "deps"       # Δε
    if "min" in s_low:      # minPts / min_samples
        return "dminpts"    # ΔminPts
    if "comp" in s_low:     # n_components
        return "dncomp"     # Δn_comp
    return s_low

# =================== 3) 6.4 统计配置 ===================
# 与 6.1 的 turning region 对齐：中等噪声段默认取 10%--20%
MID_NOISE_LOW  = 10
MID_NOISE_HIGH = 20

# 高噪声段默认取 >=20%
HIGH_NOISE_MIN = 20

# 若已知 DBSCAN eps 的单步搜索步长，可直接填数值（例如 0.2）；
# 留空则自动用观测矩阵中最小的非零绝对偏移值作为“一个观测步长”
DEPS_SMALL_THRESHOLD_OVERRIDE: Optional[float] = None

# =================== 4) 通用读取器 ===================
def read_first_sheet(filepath: Path) -> pd.DataFrame:
    """读取 Excel 第一张表，首列作为行索引（存放错误率档位）。"""
    df = pd.read_excel(filepath, sheet_name=0, header=0, index_col=0, engine="openpyxl")
    # 行索引若是字符串，尽量转为数字（如 '5', '10'）
    try:
        df.index = pd.to_numeric(df.index, errors="ignore")
    except Exception:
        pass
    return df

def split_var_and_clean(col: str) -> Tuple[str, str]:
    """
    从形如 'Δeps.Unified' 或 'Δk.bigdansing' 拆出 (变量前缀, 清洗名称)。
    若没有点，则尝试用下划线等分隔；最后一段视为清洗方法。
    """
    c = str(col).strip()

    if "." in c:
        var, meth = c.split(".", 1)
    elif "_" in c:
        parts = c.split("_")
        var, meth = "_".join(parts[:-1]), parts[-1]
    else:
        # 没有分隔符：尝试正则从尾部提取方法（字母序列）
        m = re.search(r"([A-Za-z]+)$", c)
        if m:
            meth = m.group(1)
            var  = c[:m.start()]
        else:
            # 全列看成变量，方法未知
            var, meth = c, ""

    var_norm  = normalize_var_name(var.replace("Δ", "").strip())
    meth_norm = CLEAN_ALIASES.get(meth.strip().lower(), meth.strip())
    return var_norm, meth_norm

def clean_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗宽表：
    - 去掉 GroundTruth 列
    - 解析 'Δvar.method'；返回 MultiIndex 列：(var_norm, method_norm)
    """
    cols_info = []
    for col in df.columns:
        var_norm, meth_norm = split_var_and_clean(col)

        # 丢掉基准或未知方法
        if meth_norm == "" or meth_norm.lower() == "groundtruth":
            continue

        cols_info.append((var_norm, meth_norm, col))

    if not cols_info:
        raise ValueError(f"无法从列名解析出任何变量×方法：{list(df.columns)}")

    # 构建 MultiIndex 列
    arrays = [(v, m) for v, m, _ in cols_info]
    multi  = pd.MultiIndex.from_tuples(arrays, names=["delta", "clean"])
    df2    = pd.DataFrame({(v, m): df[orig] for v, m, orig in cols_info}, index=df.index)
    df2.columns = multi
    return df2

# =================== 5) 统计辅助函数 ===================
def get_nonbaseline_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).strip().lower() != "mode"]

def subset_rows_by_noise(df: pd.DataFrame,
                         low: Optional[float] = None,
                         high: Optional[float] = None) -> pd.DataFrame:
    idx_num = pd.to_numeric(pd.Index(df.index), errors="coerce")
    mask = np.ones(len(df), dtype=bool)
    if low is not None:
        mask &= (idx_num >= low)
    if high is not None:
        mask &= (idx_num <= high)
    return df.loc[df.index[mask]]

def flatten_values(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(dtype=float).ravel()
    return arr[~np.isnan(arr)]

def count_condition_from_df(df: pd.DataFrame, cond_func):
    arr = flatten_values(df)
    den = int(arr.size)
    if den == 0:
        return 0, 0, np.nan
    num = int(np.sum(cond_func(arr)))
    pct = 100.0 * num / den
    return num, den, pct

def infer_small_threshold(df: pd.DataFrame) -> float:
    arr = np.abs(flatten_values(df))
    arr = arr[arr > 1e-12]
    if arr.size == 0:
        return np.nan
    vals = np.unique(np.round(arr, 10))
    return float(vals.min())

def per_bin_direction_stats(df: pd.DataFrame, param_name: str,
                            near_zero_threshold: Optional[float] = None) -> pd.DataFrame:
    rows = []
    for idx in df.index:
        row_df = df.loc[[idx]]
        neg_num, neg_den, neg_pct = count_condition_from_df(row_df, lambda x: x < 0)
        pos_num, pos_den, pos_pct = count_condition_from_df(row_df, lambda x: x > 0)
        zero_num, zero_den, zero_pct = count_condition_from_df(row_df, lambda x: x == 0)
        rec = {
            "param": param_name,
            "error_rate_bin": idx,
            "negative_num": neg_num,
            "negative_den": neg_den,
            "negative_pct": neg_pct,
            "positive_num": pos_num,
            "positive_den": pos_den,
            "positive_pct": pos_pct,
            "zero_num": zero_num,
            "zero_den": zero_den,
            "zero_pct": zero_pct,
        }
        if near_zero_threshold is not None and not np.isnan(near_zero_threshold):
            nz_num, nz_den, nz_pct = count_condition_from_df(
                row_df, lambda x: np.abs(x) <= near_zero_threshold
            )
            rec["near_zero_num"] = nz_num
            rec["near_zero_den"] = nz_den
            rec["near_zero_pct"] = nz_pct
            rec["near_zero_threshold"] = near_zero_threshold
        rows.append(rec)
    return pd.DataFrame(rows)

# =================== 6) 矩阵 → 热力图 ===================
def plot_heatmap(df: pd.DataFrame, title: str, outfile: Path):
    """ df: 行=错误档位(数值/字符串)，列=清洗方法（普通列） """
    # 对齐列顺序
    cols = [c for c in CLEAN_ORDER if c in df.columns]
    df = df[cols]

    # === 原代码：去掉名为 'error_bin' 的空行（若存在） ===
    to_drop = [ix for ix in df.index if str(ix).strip().lower() == "error_bin"]
    if to_drop:
        df = df.drop(index=to_drop)

    # 对称色标（便于比较正负）
    vmin = float(np.nanmin(df.values))
    vmax = float(np.nanmax(df.values))
    m = max(abs(vmin), abs(vmax))
    vmin, vmax = -m, m

    # 构造矩阵（行从小到大排序）
    try:
        idx_sorted = sorted(df.index, key=lambda x: float(x))
        df = df.loc[idx_sorted]
    except Exception:
        pass
    data = df.values.astype(float)

    # 画图
    plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto", cmap="coolwarm",
                   vmin=vmin, vmax=vmax, origin="upper")

    # 坐标轴与刻度
    ax.set_title(title, fontsize=18, pad=10)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels([str(i) for i in df.index], fontsize=14)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=14)

    # 网格线（可选）
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # === 原代码：在热力图中填入具体数据 ===
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            txt = f"{val:.2f}"
            color = "white" if (m > 0 and abs(val) >= 0.6 * m) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12.5, color=color)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[OK] 保存：{outfile}")

# =================== 7) 主流程 ===================
def main():
    # ---- 7.1 读取三张宽表 ----
    dk_raw  = read_first_sheet(F_KMEANS)
    db_raw  = read_first_sheet(F_DBSCAN)
    gmm_raw = read_first_sheet(F_GMM)

    dk_wide  = clean_wide(dk_raw)
    db_wide  = clean_wide(db_raw)
    gmm_wide = clean_wide(gmm_raw)

    # ---- 7.2 提取各参数矩阵（原逻辑不变） ----
    if ("dk" not in dk_wide.columns.get_level_values(0)):
        raise RuntimeError(f"{F_KMEANS} 中未找到 Δk 列。")
    dk = dk_wide.xs("dk", axis=1, level="delta").apply(pd.to_numeric, errors="coerce")
    dk = dk.loc[:, [c for c in CLEAN_ORDER if c in dk.columns]]

    if ("deps" in db_wide.columns.get_level_values(0)):
        deps = db_wide.xs("deps", axis=1, level="delta").apply(pd.to_numeric, errors="coerce")
        deps = deps.loc[:, [c for c in CLEAN_ORDER if c in deps.columns]]
    else:
        deps = None
        print(f"[WARN] {F_DBSCAN} 中未找到 Δε（deps）列，跳过对应热力图与统计。")

    if ("dminpts" in db_wide.columns.get_level_values(0)):
        dmin = db_wide.xs("dminpts", axis=1, level="delta").apply(pd.to_numeric, errors="coerce")
        dmin = dmin.loc[:, [c for c in CLEAN_ORDER if c in dmin.columns]]
    else:
        dmin = None
        print(f"[WARN] {F_DBSCAN} 中未找到 ΔminPts（dminpts）列，跳过对应热力图与统计。")

    if ("dncomp" not in gmm_wide.columns.get_level_values(0)):
        raise RuntimeError(f"{F_GMM} 中未找到 Δn_components 列。")
    dn = gmm_wide.xs("dncomp", axis=1, level="delta").apply(pd.to_numeric, errors="coerce")
    dn = dn.loc[:, [c for c in CLEAN_ORDER if c in dn.columns]]

    # ---- 7.3 绘图（原逻辑不变） ----
    plot_heatmap(dk, "KMeans – Δk",
                 OUT_DIR / "hyper_heat_kmeans_dk.pdf")

    if deps is not None:
        plot_heatmap(deps, "DBSCAN – Δε",
                     OUT_DIR / "hyper_heat_dbscan_deps.pdf")

    if dmin is not None:
        plot_heatmap(dmin, "DBSCAN – ΔminPts",
                     OUT_DIR / "hyper_heat_dbscan_dminpts.pdf")

    plot_heatmap(dn, "GMM – Δn_components",
                 OUT_DIR / "hyper_heat_gmm_dncomp.pdf")

    # ---- 7.4 6.4.1：KMeans/GMM 模型规模偏移统计 ----
    dk_stat = dk.loc[:, get_nonbaseline_cols(dk)].copy()
    dn_stat = dn.loc[:, get_nonbaseline_cols(dn)].copy()

    dk_mid = subset_rows_by_noise(dk_stat, low=MID_NOISE_LOW, high=MID_NOISE_HIGH)
    dn_mid = subset_rows_by_noise(dn_stat, low=MID_NOISE_LOW, high=MID_NOISE_HIGH)

    dk_neg_num, dk_neg_den, dk_neg_pct = count_condition_from_df(dk_stat, lambda x: x < 0)
    dk_mid_pos_num, dk_mid_pos_den, dk_mid_pos_pct = count_condition_from_df(dk_mid, lambda x: x > 0)

    dn_neg_num, dn_neg_den, dn_neg_pct = count_condition_from_df(dn_stat, lambda x: x < 0)
    dn_mid_pos_num, dn_mid_pos_den, dn_mid_pos_pct = count_condition_from_df(dn_mid, lambda x: x > 0)

    summary_641 = pd.DataFrame([
        {
            "param": "Δk",
            "overall_negative_num": dk_neg_num,
            "overall_negative_den": dk_neg_den,
            "overall_negative_pct": dk_neg_pct,
            "mid_noise_positive_num": dk_mid_pos_num,
            "mid_noise_positive_den": dk_mid_pos_den,
            "mid_noise_positive_pct": dk_mid_pos_pct,
            "mid_noise_range": f"{MID_NOISE_LOW}-{MID_NOISE_HIGH}"
        },
        {
            "param": "Δn_comp",
            "overall_negative_num": dn_neg_num,
            "overall_negative_den": dn_neg_den,
            "overall_negative_pct": dn_neg_pct,
            "mid_noise_positive_num": dn_mid_pos_num,
            "mid_noise_positive_den": dn_mid_pos_den,
            "mid_noise_positive_pct": dn_mid_pos_pct,
            "mid_noise_range": f"{MID_NOISE_LOW}-{MID_NOISE_HIGH}"
        }
    ])

    bybin_641 = pd.concat([
        per_bin_direction_stats(dk_stat, "Δk"),
        per_bin_direction_stats(dn_stat, "Δn_comp")
    ], ignore_index=True)

    # ---- 7.5 6.4.2：DBSCAN 密度参数偏移统计 ----
    summary_642 = pd.DataFrame()
    bybin_642 = pd.DataFrame()

    if deps is not None and dmin is not None:
        deps_stat = deps.loc[:, get_nonbaseline_cols(deps)].copy()
        dmin_stat = dmin.loc[:, [c for c in get_nonbaseline_cols(dmin) if c in deps_stat.columns]].copy()
        deps_stat = deps_stat.loc[:, dmin_stat.columns]  # 对齐列顺序

        # Δε“近零偏移”阈值：优先用手工指定，否则用观测矩阵最小非零绝对值
        deps_small_thresh = (
            float(DEPS_SMALL_THRESHOLD_OVERRIDE)
            if DEPS_SMALL_THRESHOLD_OVERRIDE is not None
            else infer_small_threshold(deps_stat)
        )

        deps_small_num, deps_small_den, deps_small_pct = count_condition_from_df(
            deps_stat, lambda x: np.abs(x) <= deps_small_thresh
        )

        deps_high = subset_rows_by_noise(deps_stat, low=HIGH_NOISE_MIN, high=None)
        dmin_high = subset_rows_by_noise(dmin_stat, low=HIGH_NOISE_MIN, high=None)

        deps_nonhigh = subset_rows_by_noise(deps_stat, low=None, high=HIGH_NOISE_MIN - 1e-9)
        dmin_nonhigh = subset_rows_by_noise(dmin_stat, low=None, high=HIGH_NOISE_MIN - 1e-9)

        # 分别统计 Δε<0 / ΔminPts>0 的比例
        deps_high_neg_num, deps_high_neg_den, deps_high_neg_pct = count_condition_from_df(
            deps_high, lambda x: x < 0
        )
        dmin_high_pos_num, dmin_high_pos_den, dmin_high_pos_pct = count_condition_from_df(
            dmin_high, lambda x: x > 0
        )

        # 联动比例：在同一 error_bin × cleaning 单元上，同时满足 Δε<0 且 ΔminPts>0
        high_idx = deps_high.index.intersection(dmin_high.index)
        high_cols = [c for c in deps_high.columns if c in dmin_high.columns]
        deps_high_aligned = deps_high.loc[high_idx, high_cols]
        dmin_high_aligned = dmin_high.loc[high_idx, high_cols]
        high_joint_mask = (deps_high_aligned.to_numpy(dtype=float) < 0) & (dmin_high_aligned.to_numpy(dtype=float) > 0)
        high_valid = ~np.isnan(deps_high_aligned.to_numpy(dtype=float)) & ~np.isnan(dmin_high_aligned.to_numpy(dtype=float))
        joint_high_num = int(np.sum(high_joint_mask & high_valid))
        joint_high_den = int(np.sum(high_valid))
        joint_high_pct = (100.0 * joint_high_num / joint_high_den) if joint_high_den else np.nan

        nonhigh_idx = deps_nonhigh.index.intersection(dmin_nonhigh.index)
        nonhigh_cols = [c for c in deps_nonhigh.columns if c in dmin_nonhigh.columns]
        deps_nonhigh_aligned = deps_nonhigh.loc[nonhigh_idx, nonhigh_cols]
        dmin_nonhigh_aligned = dmin_nonhigh.loc[nonhigh_idx, nonhigh_cols]
        nonhigh_joint_mask = (deps_nonhigh_aligned.to_numpy(dtype=float) < 0) & (dmin_nonhigh_aligned.to_numpy(dtype=float) > 0)
        nonhigh_valid = ~np.isnan(deps_nonhigh_aligned.to_numpy(dtype=float)) & ~np.isnan(dmin_nonhigh_aligned.to_numpy(dtype=float))
        joint_nonhigh_num = int(np.sum(nonhigh_joint_mask & nonhigh_valid))
        joint_nonhigh_den = int(np.sum(nonhigh_valid))
        joint_nonhigh_pct = (100.0 * joint_nonhigh_num / joint_nonhigh_den) if joint_nonhigh_den else np.nan

        summary_642 = pd.DataFrame([{
            "deps_small_threshold_used": deps_small_thresh,
            "deps_small_num": deps_small_num,
            "deps_small_den": deps_small_den,
            "deps_small_pct": deps_small_pct,
            "high_noise_threshold": HIGH_NOISE_MIN,
            "deps_negative_high_num": deps_high_neg_num,
            "deps_negative_high_den": deps_high_neg_den,
            "deps_negative_high_pct": deps_high_neg_pct,
            "dminpts_positive_high_num": dmin_high_pos_num,
            "dminpts_positive_high_den": dmin_high_pos_den,
            "dminpts_positive_high_pct": dmin_high_pos_pct,
            "joint_conservative_high_num": joint_high_num,
            "joint_conservative_high_den": joint_high_den,
            "joint_conservative_high_pct": joint_high_pct,
            "joint_conservative_nonhigh_num": joint_nonhigh_num,
            "joint_conservative_nonhigh_den": joint_nonhigh_den,
            "joint_conservative_nonhigh_pct": joint_nonhigh_pct,
        }])

        bybin_642 = pd.concat([
            per_bin_direction_stats(deps_stat, "Δε", near_zero_threshold=deps_small_thresh),
            per_bin_direction_stats(dmin_stat, "ΔminPts")
        ], ignore_index=True)

    # ---- 7.6 导出统计 ----
    out_xlsx = OUT_DIR / "hyper_shift_direction_stats.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_641.to_excel(writer, sheet_name="summary_641", index=False)
        if not summary_642.empty:
            summary_642.to_excel(writer, sheet_name="summary_642", index=False)
        bybin_641.to_excel(writer, sheet_name="bybin_641", index=False)
        if not bybin_642.empty:
            bybin_642.to_excel(writer, sheet_name="bybin_642", index=False)
        # 方便人工核查的原始矩阵（排除 Mode 后统计用矩阵）
        dk_stat.to_excel(writer, sheet_name="mat_dk")
        dn_stat.to_excel(writer, sheet_name="mat_dncomp")
        if deps is not None:
            deps_stat.to_excel(writer, sheet_name="mat_deps")
        if dmin is not None:
            dmin_stat.to_excel(writer, sheet_name="mat_dminpts")

    # 同步导出 csv
    summary_641.to_csv(OUT_DIR / "hyper_shift_summary_641.csv", index=False)
    if not summary_642.empty:
        summary_642.to_csv(OUT_DIR / "hyper_shift_summary_642.csv", index=False)
    bybin_641.to_csv(OUT_DIR / "hyper_shift_bybin_641.csv", index=False)
    if not bybin_642.empty:
        bybin_642.to_csv(OUT_DIR / "hyper_shift_bybin_642.csv", index=False)

    # ---- 7.7 终端打印 ----
    print("\n===== 6.4.1（模型规模偏移：总体主导方向 + 中噪声例外） =====")
    print(summary_641.to_string(index=False, float_format="%.4f"))

    if not summary_642.empty:
        print("\n===== 6.4.2（DBSCAN：Δε 总体较稳 + 高噪声联动偏移） =====")
        print(summary_642.to_string(index=False, float_format="%.4f"))

    print(f"\n[OK] 6.4.1 / 6.4.2 统计已写出：{out_xlsx}")

if __name__ == "__main__":
    main()
