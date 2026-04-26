
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties   # ★

matplotlib.rc('font', family='Times New Roman')  # 英文默认
cn_font         = FontProperties(family='SimSun')
cn_font_title   = FontProperties(family='SimSun', size=16)
cn_font_legend  = FontProperties(family='SimSun', size=14)

# ============================== NEW HELPERS ==============================
def _fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return np.array([])
    if len(np.unique(x)) < 2:
        return np.full_like(y, np.nanmean(y), dtype=float)
    coef = np.polyfit(x, y, 1)
    return np.polyval(coef, x)


def find_piecewise_breakpoint(curve_df, x_col="error_rate_bin", y_col="CEGR_median"):
    """
    在离散 bin 上寻找双线性分段的最佳断点：
      左段: x <= c
      右段: x > c
    用 SSE 最小准则选 c。
    """
    d = (curve_df[[x_col, y_col]]
         .dropna()
         .sort_values(x_col)
         .drop_duplicates(subset=[x_col], keep="last"))

    xs = d[x_col].to_numpy(dtype=float)
    ys = d[y_col].to_numpy(dtype=float)
    uniq = np.unique(xs)

    best = {
        "break_bin": np.nan,
        "sse": np.inf,
        "n_points": len(xs)
    }
    if len(uniq) < 4:
        return best

    for c in uniq[1:-1]:
        left = xs <= c
        right = xs > c
        if left.sum() < 2 or right.sum() < 2:
            continue
        yhat_l = _fit_line(xs[left], ys[left])
        yhat_r = _fit_line(xs[right], ys[right])
        sse = np.sum((ys[left] - yhat_l) ** 2) + np.sum((ys[right] - yhat_r) ** 2)
        if sse < best["sse"]:
            best = {
                "break_bin": float(c),
                "sse": float(sse),
                "n_points": int(len(xs))
            }
    return best


def bootstrap_task_breakpoints(raw_ratio_df, n_boot=1000, seed=42):
    """
    对每个 task 的原始 CEGR 行（dataset_id × error_bin × cluster_method）做 bootstrap：
      - 在每个 error_bin 内重采样 CEGR
      - 重算该 task 在每个 error_bin 的中位 CEGR
      - 估计该 task 的分段断点
    返回：
      per_task_summary: 每个任务的原始断点和 bootstrap 区间
      overall_summary : 汇总所有任务 bootstrap 断点后的整体区间
      boot_samples_df : 每次 bootstrap 的断点样本
    """
    rng = np.random.default_rng(seed)
    per_task_rows = []
    boot_rows = []

    for task in sorted(raw_ratio_df["task_name"].unique()):
        sub = raw_ratio_df[raw_ratio_df["task_name"] == task].copy()
        if sub.empty:
            continue

        # 原始 task 曲线：对 dataset_id 与 cluster_method 再取中位数，得到 task × bin 曲线
        task_curve = (sub.groupby("error_rate_bin", as_index=False, observed=False)["CEGR"]
                        .median()
                        .rename(columns={"CEGR": "CEGR_median"})
                        .sort_values("error_rate_bin"))
        orig_bp = find_piecewise_breakpoint(task_curve)

        grouped = [(ebin, g["CEGR"].dropna().to_numpy(dtype=float))
                   for ebin, g in sub.groupby("error_rate_bin", observed=False)]
        samples = []
        for b in range(n_boot):
            rec = []
            for ebin, vals in grouped:
                if len(vals) == 0:
                    continue
                samp = rng.choice(vals, size=len(vals), replace=True)
                rec.append({"error_rate_bin": ebin, "CEGR_median": float(np.median(samp))})
            boot_curve = pd.DataFrame(rec)
            if len(boot_curve) < 4:
                continue
            bp = find_piecewise_breakpoint(boot_curve)
            if pd.notna(bp["break_bin"]):
                samples.append(bp["break_bin"])
                boot_rows.append({"task": task, "boot_id": b, "break_bin": bp["break_bin"]})

        if samples:
            q025, q25, q50, q75, q975 = np.quantile(samples, [0.025, 0.25, 0.50, 0.75, 0.975])
            per_task_rows.append({
                "task": task,
                "orig_break_bin": orig_bp["break_bin"],
                "boot_median_bin": q50,
                "turn_region_50_low": q25,
                "turn_region_50_high": q75,
                "turn_region_95_low": q025,
                "turn_region_95_high": q975,
                "n_boot_valid": len(samples)
            })
        else:
            per_task_rows.append({
                "task": task,
                "orig_break_bin": orig_bp["break_bin"],
                "boot_median_bin": np.nan,
                "turn_region_50_low": np.nan,
                "turn_region_50_high": np.nan,
                "turn_region_95_low": np.nan,
                "turn_region_95_high": np.nan,
                "n_boot_valid": 0
            })

    per_task_summary = pd.DataFrame(per_task_rows)
    boot_samples_df = pd.DataFrame(boot_rows)

    overall = {
        "overall_boot_median_bin": np.nan,
        "overall_turn_region_50_low": np.nan,
        "overall_turn_region_50_high": np.nan,
        "overall_turn_region_95_low": np.nan,
        "overall_turn_region_95_high": np.nan,
        "n_boot_samples_total": 0
    }
    if not boot_samples_df.empty:
        vals = boot_samples_df["break_bin"].to_numpy(dtype=float)
        q025, q25, q50, q75, q975 = np.quantile(vals, [0.025, 0.25, 0.50, 0.75, 0.975])
        overall = {
            "overall_boot_median_bin": float(q50),
            "overall_turn_region_50_low": float(q25),
            "overall_turn_region_50_high": float(q75),
            "overall_turn_region_95_low": float(q025),
            "overall_turn_region_95_high": float(q975),
            "n_boot_samples_total": int(len(vals))
        }
    overall_summary = pd.DataFrame([overall])
    return per_task_summary, overall_summary, boot_samples_df


def _bootstrap_median_diff(a, b, n_boot=5000, seed=42):
    """返回 median(a)-median(b) 的 bootstrap 区间"""
    a = np.asarray(pd.Series(a).dropna().to_numpy(dtype=float))
    b = np.asarray(pd.Series(b).dropna().to_numpy(dtype=float))
    if len(a) == 0 or len(b) == 0:
        return {
            "median_a": np.nan,
            "median_b": np.nan,
            "diff_median": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
        }

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        boots.append(float(np.median(sa) - np.median(sb)))
    q025, q975 = np.quantile(boots, [0.025, 0.975])
    return {
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "diff_median": float(np.median(a) - np.median(b)),
        "ci_low": float(q025),
        "ci_high": float(q975),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def _try_load_process_detail():
    """
    尝试读取 6.4.2table.py 新增输出的 process detail。
    读取成功则返回 DataFrame，否则返回 None。
    """
    candidates = [
        Path(r"D:\algorithm paper\AutoMLClustering\task_progress\tables\6.4.2tables\table6_process_detail.xlsx"),
        Path(r"D:\algorithm paper\AutoMLClustering_full\task_progress\tables\6.4.2tables\table6_process_detail.xlsx"),
        Path(__file__).resolve().parents[3] / "task_progress" / "tables" / "6.4.2tables" / "table6_process_detail.xlsx",
        Path(__file__).resolve().parents[3] / "task_progress" / "tables" / "6.4.2tables" / "table6_process_detail.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                if p.suffix.lower() == ".xlsx":
                    df = pd.read_excel(p)
                else:
                    df = pd.read_csv(p)
                print(f"[INFO] loaded process detail: {p}")
                return df
            except Exception as e:
                print(f"[WARN] failed to read process detail {p}: {e}")
    print("[WARN] process detail not found; 6.3 的过程-结果联动统计将跳过。")
    return None


def _cluster_family(cm: str):
    cm = str(cm).upper()
    if cm in {"KMEANS", "KMEANSNF", "KMEANSPPS", "GMM"}:
        return "centroid"
    if cm == "DBSCAN":
        return "density"
    if cm == "HC":
        return "hierarch"
    return "unknown"


def _estimate_mode_baseline_from_relative(df):
    """
    当 summary 中没有显式的 mode 行时，用相对值列反推 baseline。
    假定：
      Comb_relative = Combined / ModeCombined
      Sil_relative  = Sil / ModeSil
      DB_relative   = ModeDB / DB  （因为 DB 越小越好）
    """
    out = df.copy()
    out["mode_combined_est"] = np.where(
        out["Comb_relative"].replace(0, np.nan).notna(),
        out["Combined Score"] / out["Comb_relative"].replace(0, np.nan),
        np.nan
    )
    out["mode_sil_est"] = np.where(
        out["Sil_relative"].replace(0, np.nan).notna(),
        out["Silhouette Score"] / out["Sil_relative"].replace(0, np.nan),
        np.nan
    )
    out["mode_db_est"] = np.where(
        out["DB_relative"].replace(0, np.nan).notna(),
        out["Davies-Bouldin Score"] * out["DB_relative"].replace(0, np.nan),
        np.nan
    )
    return out


# =======================================================================


def main():
    #--------------------------------------------------------------------
    # A) 读取并合并
    #--------------------------------------------------------------------
    task_names = ["beers", "rayyan", "flights", "hospital"]
    data_dir   = os.path.join("..", "..", "..", "results", "analysis_results")
    out_dir    = os.path.join("..", "..", "..", "task_progress", "figures", "6.4.3graph")
    os.makedirs(out_dir, exist_ok=True)

    dfs = []
    for t in task_names:
        fp = os.path.join(data_dir, f"{t}_summary.xlsx")
        if not os.path.isfile(fp):
            print(f"[WARN] missing {fp}, skip {t}")
            continue
        tmp = pd.read_excel(fp)
        tmp["task_name"] = t
        dfs.append(tmp)

    if not dfs:
        print("[ERROR] No data loaded."); sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df["error_rate"] = df["error_rate"].astype(float)

    # 标准化 key 字段
    df["dataset_id"] = pd.to_numeric(df["dataset_id"], errors="coerce").astype("Int64")
    df["cleaning_method"] = df["cleaning_method"].astype(str).str.lower().str.strip()
    df["cluster_method"] = df["cluster_method"].astype(str).str.upper().str.strip()
    df["family"] = df["cluster_method"].map(_cluster_family)

    #--------------------------------------------------------------------
    # B) 最近 5 的倍数 error_bin
    #--------------------------------------------------------------------
    df["error_rate_bin"] = ((df["error_rate"] / 5).round() * 5).astype(int)
    df = df.sort_values("error_rate_bin")

    #--------------------------------------------------------------------
    # C) 主循环（原有 CEGR 逻辑）
    #--------------------------------------------------------------------
    stats_rows = []                     # 原有新增统计
    cegr_raw_all = []                   # ★ NEW: 收集 task 内原始 CEGR 行
    task_bin_curve_all = []             # ★ NEW: 收集 task × error_bin 的 pooled CEGR 曲线

    for task in sorted(df["task_name"].unique()):
        sub = df[df["task_name"] == task]
        if sub.empty:
            continue

        # ------ 1) 计算 CEGR（与你原来的逻辑相同） --------------------------
        rec = []
        for (ds, ebin, cm), g in sub.groupby(
                ["dataset_id", "error_rate_bin", "cluster_method"], observed=False):
            if len(g) < 2:
                continue
            best  = g.loc[g["EDR"].idxmax()]
            worst = g.loc[g["EDR"].idxmin()]
            d_edr = best["EDR"] - worst["EDR"]
            if abs(d_edr) < 1e-10:
                continue
            rec.append({
                "task_name": task,
                "dataset_id": ds,
                "error_rate_bin": ebin,
                "cluster_method": cm,
                "CEGR": (best["Combined Score"] - worst["Combined Score"]) / d_edr
            })

        ratio_df = pd.DataFrame(rec)
        if ratio_df.empty:
            print(f"[WARN] no CEGR for {task}"); continue

        cegr_raw_all.append(ratio_df.copy())  # ★ NEW

        agg = (ratio_df.groupby(["error_rate_bin", "cluster_method"], observed=False, as_index=False)
               .CEGR.median()
               .rename(columns={"CEGR": "CEGR_median"})
               .sort_values(["error_rate_bin", "cluster_method"]))

        # ★ NEW: task × error_bin 的 pooled 曲线（跨 dataset_id / cluster_method 取中位数）
        pooled_curve = (ratio_df.groupby(["task_name", "error_rate_bin"], observed=False, as_index=False)
                        .CEGR.median()
                        .rename(columns={"CEGR": "CEGR_median"})
                        .sort_values(["task_name", "error_rate_bin"]))
        task_bin_curve_all.append(pooled_curve.copy())

        # -------------------- ### NEW BLOCK ① ### ------------------------
        # 找 “CEGR 最大” 所在 bin  ⇒ 最佳 EDR
        idx_max = agg["CEGR_median"].idxmax()
        best_bin = agg.loc[idx_max, "error_rate_bin"]
        best_edr = best_bin / 100     # 把 5,10,... → 0.05,0.10… （如需整数请留用）

        # -------------------- ### NEW BLOCK ② ### ------------------------
        # 计算 ΔComb / ΔEDR ≈ 当 ΔEDR≈0.2 时的平均斜率
        sub_sorted = sub.sort_values("EDR")
        slopes = []
        for (ds, cm), g in sub_sorted.groupby(["dataset_id","cluster_method"]):
            g = g.dropna(subset=["EDR","Combined Score"]).sort_values("EDR")
            if len(g) < 2:  continue
            d_edr = g["EDR"].diff()
            d_comb= g["Combined Score"].diff()
            mask  = d_edr.between(0.19,0.21)       # ≈0.2
            slopes.extend((d_comb[mask] / d_edr[mask]).tolist())

        avg_slope = np.nan if not slopes else np.nanmean(slopes)
        comb_gain_02 = None if np.isnan(avg_slope) else avg_slope * 0.2

        stats_rows.append({
            "task": task,
            "best_EDR": best_edr,
            "avg_slope_ΔComb/ΔEDR≈0.2": avg_slope,
            "expected_ΔComb@ΔEDR=0.2": comb_gain_02
        })
        # -----------------------------------------------------------------

        # ---------------------- 绘图（原逻辑） ----------------------------
        table_path = os.path.join(out_dir, f"CEGR_5pct_{task}.xlsx")
        agg.to_excel(table_path, index=False)

        plt.figure(figsize=(6.5, 4.5))
        sns.lineplot(
            data=agg,
            x="error_rate_bin",
            y="CEGR_median",
            hue="cluster_method",
            style="cluster_method",
            markers=True,
            dashes=False,
            linewidth=2,
            markersize=13
        )
        plt.xlabel("错误率",         fontsize=18, fontproperties=cn_font)
        plt.ylabel("CEGR 中位数",   fontsize=18, fontproperties=cn_font)

        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=18)

        leg = plt.legend(title="聚类方法",
                         fontsize=12, title_fontsize=13,
                         loc="lower right", frameon=True, framealpha=0.5)
        for txt in leg.get_texts():
            txt.set_fontproperties(cn_font_legend)
        leg.get_title().set_fontproperties(cn_font_legend)

        plt.tight_layout()
        out_pdf = os.path.join(out_dir, f"CEGR_5pct_{task}.pdf")
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] saved {out_pdf}")

    # -------------------- ### NEW BLOCK ③ ### ----------------------------
    # 导出原有新增统计表
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(out_dir, "EDR_CEGR_stats.xlsx")
        stats_df.to_excel(stats_path, index=False)
        print("\n===== 关键统计 =====")
        print(stats_df.to_string(index=False, float_format="%.4f"))
        print(f"[OK] stats saved → {stats_path}")

    # ====================== NEW: 6.1-3 所需统计 ========================
    summary_613 = None
    per_task_turn = pd.DataFrame()
    overall_turn = pd.DataFrame()
    boot_samples_df = pd.DataFrame()
    turn_low = turn_high = turn_median = np.nan

    if cegr_raw_all and task_bin_curve_all:
        raw_all = pd.concat(cegr_raw_all, ignore_index=True)
        curve_all = pd.concat(task_bin_curve_all, ignore_index=True)

        # (1) 低噪声段 CEGR > 0 的比例（用于 [xx]%）
        low_mask = curve_all["error_rate_bin"] <= 15
        low_df = curve_all.loc[low_mask].copy()
        low_pos_num = int((low_df["CEGR_median"] > 0).sum())
        low_pos_den = int(len(low_df))
        low_pos_pct = (100.0 * low_pos_num / low_pos_den) if low_pos_den else np.nan

        # (2) 每个 task 的低噪声基准：该 task 在 0~15% 的中位 CEGR
        low_baseline = (low_df.groupby("task_name", as_index=False)["CEGR_median"]
                        .median()
                        .rename(columns={"CEGR_median": "low_noise_baseline"}))
        curve_all2 = curve_all.merge(low_baseline, on="task_name", how="left")

        # (3) bootstrap turning region: 每个 task 各自估计断点，再做总体汇总
        per_task_turn, overall_turn, boot_samples_df = bootstrap_task_breakpoints(
            raw_all, n_boot=1000, seed=42
        )

        # 取总体 50% 区间作为“集中在约 [a,b]% 附近”的经验性 turning region
        turn_low = overall_turn.loc[0, "overall_turn_region_50_low"]
        turn_high = overall_turn.loc[0, "overall_turn_region_50_high"]
        turn_median = overall_turn.loc[0, "overall_boot_median_bin"]

        # (4) 转折后低于各自低噪声基准的比例（用于 [yy]%）
        post_threshold = turn_high if pd.notna(turn_high) else 15
        post_mask = curve_all2["error_rate_bin"] > post_threshold
        post_df = curve_all2.loc[post_mask].copy()

        below_num = int((post_df["CEGR_median"] < post_df["low_noise_baseline"]).sum()) if len(post_df) else 0
        below_den = int(len(post_df))
        below_pct = (100.0 * below_num / below_den) if below_den else np.nan

        # 额外输出：转折后 <=0 的比例，供你决定是否写入正文
        nonpos_num = int((post_df["CEGR_median"] <= 0).sum()) if len(post_df) else 0
        nonpos_den = int(len(post_df))
        nonpos_pct = (100.0 * nonpos_num / nonpos_den) if nonpos_den else np.nan

        # 每个 task 的原始曲线与低噪声基准差，便于人工检查
        curve_all2["below_low_noise_baseline"] = curve_all2["CEGR_median"] < curve_all2["low_noise_baseline"]
        curve_all2["nonpositive"] = curve_all2["CEGR_median"] <= 0

        summary_613 = pd.DataFrame([{
            "low_noise_positive_num": low_pos_num,
            "low_noise_positive_den": low_pos_den,
            "low_noise_positive_pct": low_pos_pct,
            "turn_region_50_low": turn_low,
            "turn_region_50_high": turn_high,
            "turn_median_bin": turn_median,
            "turn_region_95_low": overall_turn.loc[0, "overall_turn_region_95_low"],
            "turn_region_95_high": overall_turn.loc[0, "overall_turn_region_95_high"],
            "post_turn_threshold_used": post_threshold,
            "post_turn_below_baseline_num": below_num,
            "post_turn_below_baseline_den": below_den,
            "post_turn_below_baseline_pct": below_pct,
            "post_turn_nonpositive_num": nonpos_num,
            "post_turn_nonpositive_den": nonpos_den,
            "post_turn_nonpositive_pct": nonpos_pct
        }])

        summary_path = os.path.join(out_dir, "CEGR_turning_region_stats.xlsx")
        with pd.ExcelWriter(summary_path) as writer:
            summary_613.to_excel(writer, sheet_name="key_stats", index=False)
            per_task_turn.to_excel(writer, sheet_name="per_task_turning", index=False)
            overall_turn.to_excel(writer, sheet_name="overall_turning", index=False)
            curve_all2.to_excel(writer, sheet_name="task_bin_curve", index=False)
            if not boot_samples_df.empty:
                boot_samples_df.to_excel(writer, sheet_name="boot_samples", index=False)

        print("\n===== 6.1-3（CEGR 分段/转折）统计 =====")
        print(summary_613.to_string(index=False, float_format="%.4f"))
        print("\n----- 每个任务的断点 bootstrap 摘要 -----")
        if not per_task_turn.empty:
            print(per_task_turn.to_string(index=False, float_format="%.4f"))
        print(f"[OK] turning-region stats saved → {summary_path}")
    # ==================================================================

    # ====================== NEW: 6.3 所需统计 ==========================
    # 1) 先构造相对 Mode 的结果层比较
    keys = ["task_name", "dataset_id", "error_rate_bin", "cluster_method"]
    score_cols = ["Combined Score", "Silhouette Score", "Davies-Bouldin Score"]

    mode_rows = df[df["cleaning_method"] == "mode"][keys + score_cols].copy()
    has_explicit_mode = not mode_rows.empty

    nonmode = df[df["cleaning_method"] != "mode"].copy()

    if has_explicit_mode:
        mode_rows = mode_rows.rename(columns={
            "Combined Score": "mode_combined",
            "Silhouette Score": "mode_sil",
            "Davies-Bouldin Score": "mode_db",
        })
        nonmode = nonmode.merge(mode_rows, on=keys, how="left")
    else:
        # 尝试用相对值反推 baseline
        est = _estimate_mode_baseline_from_relative(nonmode)
        nonmode["mode_combined"] = est["mode_combined_est"]
        nonmode["mode_sil"] = est["mode_sil_est"]
        nonmode["mode_db"] = est["mode_db_est"]

    # 结果层 delta / improvement flags
    nonmode["delta_H"] = nonmode["Combined Score"] - nonmode["mode_combined"]
    nonmode["H_positive"] = nonmode["delta_H"] > 0

    # 优先用显式 baseline；如果缺失，再退回到相对列
    nonmode["sil_improved"] = np.where(
        nonmode["mode_sil"].notna(),
        nonmode["Silhouette Score"] > nonmode["mode_sil"],
        nonmode.get("Sil_relative", pd.Series(np.nan, index=nonmode.index)) > 1.0
    )
    nonmode["db_improved"] = np.where(
        nonmode["mode_db"].notna(),
        nonmode["Davies-Bouldin Score"] < nonmode["mode_db"],
        nonmode.get("DB_relative", pd.Series(np.nan, index=nonmode.index)) > 1.0
    )
    nonmode["both_improved"] = nonmode["sil_improved"] & nonmode["db_improved"]
    nonmode["compact_only"] = nonmode["sil_improved"] & (~nonmode["db_improved"])
    nonmode["turn_regime"] = np.where(
        nonmode["error_rate_bin"] <= 15, "pre_turn",
        np.where(nonmode["error_rate_bin"] >= 20, "post_turn", "turning_band")
    )

    # 2) 组合层（task × cleaning × clustering）的均值与波动
    combo_cols = ["task_name", "cleaning_method", "cluster_method", "family"]
    combo_summary = (
        nonmode.groupby(combo_cols, as_index=False)
              .agg(
                  n_rows=("Combined Score", "size"),
                  mean_combined=("Combined Score", "mean"),
                  std_combined=("Combined Score", "std"),
                  mean_delta_H=("delta_H", "mean"),
                  H_positive_rate=("H_positive", "mean"),
                  sil_improved_rate=("sil_improved", "mean"),
                  db_improved_rate=("db_improved", "mean"),
                  both_improved_rate=("both_improved", "mean"),
                  compact_only_rate=("compact_only", "mean"),
                  mean_error_rate=("error_rate", "mean"),
              )
    )
    combo_summary["std_combined"] = combo_summary["std_combined"].fillna(0.0)

    task_std_cut = (combo_summary.groupby("task_name", as_index=False)["std_combined"]
                    .median()
                    .rename(columns={"std_combined": "task_std_median"}))
    combo_summary = combo_summary.merge(task_std_cut, on="task_name", how="left")
    combo_summary["low_variance_combo"] = combo_summary["std_combined"] <= combo_summary["task_std_median"]

    # 3) 合并过程层明细（如果有）
    process_detail = _try_load_process_detail()
    merged = None
    if process_detail is not None and not process_detail.empty:
        process_detail = process_detail.copy()
        process_detail["dataset_id"] = pd.to_numeric(process_detail["dataset_id"], errors="coerce").astype("Int64")
        process_detail["cleaning_method"] = process_detail["cleaning_method"].astype(str).str.lower().str.strip()
        process_detail["cluster_method"] = process_detail["cluster_method"].astype(str).str.upper().str.strip()

        merged = nonmode.merge(
            process_detail,
            on=["dataset_id", "cleaning_method", "cluster_method"],
            how="left",
            suffixes=("", "_proc")
        )

        proc_combo = (
            merged.groupby(combo_cols, as_index=False)
                  .agg(
                      process_signature_rate=("process_positive_signature", "mean"),
                      process_changed_rate=("process_changed_any", "mean"),
                  )
        )
        combo_summary = combo_summary.merge(proc_combo, on=combo_cols, how="left")
        combo_summary["process_signature_majority"] = combo_summary["process_signature_rate"] >= 0.5
    else:
        combo_summary["process_signature_rate"] = np.nan
        combo_summary["process_changed_rate"] = np.nan
        combo_summary["process_signature_majority"] = False

    # 4) 6.3-1：正向过程签名 -> 正收益 + 低波动
    sig_combo = combo_summary[combo_summary["process_signature_majority"] == True].copy()
    summary_631 = pd.DataFrame([{
        "signature_combo_num": int(len(sig_combo)),
        "positive_mean_deltaH_num": int((sig_combo["mean_delta_H"] > 0).sum()) if len(sig_combo) else 0,
        "positive_mean_deltaH_den": int(len(sig_combo)),
        "positive_mean_deltaH_pct": float(100.0 * (sig_combo["mean_delta_H"] > 0).mean()) if len(sig_combo) else np.nan,
        "low_variance_num": int(sig_combo["low_variance_combo"].sum()) if len(sig_combo) else 0,
        "low_variance_den": int(len(sig_combo)),
        "low_variance_pct": float(100.0 * sig_combo["low_variance_combo"].mean()) if len(sig_combo) else np.nan,
    }])

    # 5) 6.3-2：匹配组 vs 失配组 的波动差
    matched_combo = combo_summary[
        (combo_summary["process_signature_majority"] == True) &
        (combo_summary["mean_delta_H"] > 0) &
        (combo_summary["both_improved_rate"] >= 0.5)
    ].copy()

    mismatch_combo = combo_summary[
        (combo_summary["process_signature_majority"] == True) &
        (combo_summary["mean_delta_H"] > 0) &
        (combo_summary["both_improved_rate"] < 0.5)
    ].copy()

    diff_632 = _bootstrap_median_diff(
        mismatch_combo["std_combined"] if len(mismatch_combo) else [],
        matched_combo["std_combined"] if len(matched_combo) else [],
        n_boot=5000,
        seed=42
    )
    summary_632 = pd.DataFrame([{
        "matched_combo_num": int(len(matched_combo)),
        "mismatch_combo_num": int(len(mismatch_combo)),
        "matched_std_median": diff_632["median_b"],     # 注意：bootstrap_median_diff(a,b) 返回 a-b
        "mismatch_std_median": diff_632["median_a"],
        "std_median_diff_mismatch_minus_matched": diff_632["diff_median"],
        "std_diff_ci_low": diff_632["ci_low"],
        "std_diff_ci_high": diff_632["ci_high"],
    }])

    # 6) 6.3-3：转折前后，只改善紧凑度 vs 两者同时改善 的正收益率
    regime_df = nonmode[nonmode["turn_regime"].isin(["pre_turn", "post_turn"])].copy()
    regime_df["structure_group"] = np.select(
        [regime_df["both_improved"], regime_df["compact_only"]],
        ["both_improved", "compact_only"],
        default="other"
    )
    regime_focus = regime_df[regime_df["structure_group"].isin(["both_improved", "compact_only"])].copy()

    turning_stats = (
        regime_focus.groupby(["turn_regime", "structure_group"], as_index=False)
                    .agg(
                        positive_num=("H_positive", "sum"),
                        total=("H_positive", "size"),
                        positive_rate=("H_positive", "mean"),
                    )
    )
    turning_stats["positive_rate_pct"] = turning_stats["positive_rate"] * 100.0

    # 额外输出：pre/post 中 both - compact_only 的差
    diff_rows = []
    for regime in ["pre_turn", "post_turn"]:
        sub = regime_focus[regime_focus["turn_regime"] == regime]
        a = sub.loc[sub["structure_group"] == "both_improved", "H_positive"].astype(float).to_numpy()
        b = sub.loc[sub["structure_group"] == "compact_only", "H_positive"].astype(float).to_numpy()
        if len(a) and len(b):
            diff = _bootstrap_median_diff(a, b, n_boot=5000, seed=42)
            diff_rows.append({
                "turn_regime": regime,
                "both_positive_rate_pct": float(100.0 * np.mean(a)),
                "compact_only_positive_rate_pct": float(100.0 * np.mean(b)),
                "rate_diff_both_minus_compact_only": float(np.mean(a) - np.mean(b)),
                "rate_diff_ci_low": diff["ci_low"],
                "rate_diff_ci_high": diff["ci_high"],
                "n_both": int(len(a)),
                "n_compact_only": int(len(b)),
            })
        else:
            diff_rows.append({
                "turn_regime": regime,
                "both_positive_rate_pct": np.nan,
                "compact_only_positive_rate_pct": np.nan,
                "rate_diff_both_minus_compact_only": np.nan,
                "rate_diff_ci_low": np.nan,
                "rate_diff_ci_high": np.nan,
                "n_both": int(len(a)),
                "n_compact_only": int(len(b)),
            })
    turning_diff = pd.DataFrame(diff_rows)

    # 7) 输出 6.3 统计文件
    out_63 = os.path.join(out_dir, "result_stability_signature_stats.xlsx")
    with pd.ExcelWriter(out_63) as writer:
        combo_summary.to_excel(writer, sheet_name="combo_summary", index=False)
        summary_631.to_excel(writer, sheet_name="finding_631", index=False)
        summary_632.to_excel(writer, sheet_name="finding_632", index=False)
        turning_stats.to_excel(writer, sheet_name="finding_633_rates", index=False)
        turning_diff.to_excel(writer, sheet_name="finding_633_diff", index=False)
        if merged is not None and not merged.empty:
            merged.to_excel(writer, sheet_name="merged_row_level", index=False)

    print("\n===== 6.3-1（正向过程签名 -> 正收益 / 低波动）统计 =====")
    print(summary_631.to_string(index=False, float_format="%.4f"))

    print("\n===== 6.3-2（匹配组 vs 失配组）波动差统计 =====")
    print(summary_632.to_string(index=False, float_format="%.4f"))

    print("\n===== 6.3-3（拐点前后：只改善紧凑度 vs 同时改善两者）正收益率 =====")
    if not turning_stats.empty:
        print(turning_stats.to_string(index=False, float_format="%.4f"))
    print("\n----- 6.3-3 差值及区间 -----")
    if not turning_diff.empty:
        print(turning_diff.to_string(index=False, float_format="%.4f"))

    print(f"[OK] 6.3 stats saved → {out_63}")
    # ==================================================================

    print("[INFO] Done. Each task ⇒ PDF chart + data table + stats.")

if __name__ == "__main__":
    main()
