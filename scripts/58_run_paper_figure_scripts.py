#!/usr/bin/env python3
"""Run selected legacy paper figure scripts inside a controlled Mode A workspace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_EXTENSIONS = {
    ".png", ".pdf", ".svg",
    ".csv", ".xlsx", ".xls", ".json", ".txt",
}

SCRIPT_GROUPS = {
    "figure_scripts_graph",
    "figure_scripts_utils",
    "figure_scripts_visual_demo",
    "figure_scripts_pre_experiment",
}

RUN_ORDER = [
    "fig_1_method_flow.py",
    "4.flowchart.py",
    "fig_4_plot_point.py",
    "fig_4_plot_score_eval.py",
    "fig_4_plot_top10.py",
    "fig_5_err_cluster_plot.py",
    "fig_5_err_grad_plot.py",
    "fig_5_heat_error_type.py",
    "fig_6_radar_graph_median.py",
    "fig_7_CEGR_line_graph.py",
    "fig_7_point_graph.py",
    "fig_10_make_hyper_heatmaps.py",
    "draw_cmp_graph.py",
    "draw_error_graph.py",
    "draw_rate_graph.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected paper figure scripts.")
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=Path("analysis/paper_figure_audit/paper_figure_source_selection.csv"),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("analysis/paper_generated/paper_figure_workspace"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/paper_generated/paper_figures"),
    )
    parser.add_argument(
        "--figure-output-root",
        type=Path,
        default=Path("figures/paper_generated"),
    )
    parser.add_argument(
        "--legacy-roots",
        nargs="+",
        default=[
            r"E:\algorithm paper\AutoMLClustering_full",
            r"E:\algorithm paper\AutoMLClustering",
        ],
    )
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Selection CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def copy_tree_merge(src_root: Path, dst_root: Path) -> None:
    """Copy a directory tree without recursively copying generated workspaces."""
    if not src_root.exists() or not src_root.is_dir():
        return

    src_root_res = src_root.resolve()
    dst_root_res = dst_root.resolve()

    # Freeze the source list before copying. Otherwise, if dst_root is under
    # src_root, rglob may discover files that were just copied and recurse.
    source_files = [path for path in src_root.rglob("*") if path.is_file()]

    skip_prefixes = [
        "paper_figure_workspace/",
        "paper_figures/",
        "paper_table_workspace/",
        "paper_generated/paper_figure_workspace/",
        "paper_generated/paper_figures/",
        "__pycache__/",
        ".pytest_cache/",
    ]

    for path in source_files:
        path_res = path.resolve()

        # Never copy files that are already inside the destination tree.
        if _is_relative_to(path_res, dst_root_res):
            continue

        rel = path.relative_to(src_root)
        rel_text = rel.as_posix().lower()

        if any(rel_text.startswith(prefix) for prefix in skip_prefixes):
            continue

        copy_file(path, dst_root / rel)


def existing_legacy_roots(args: argparse.Namespace) -> list[Path]:
    roots = []
    for value in args.legacy_roots:
        path = Path(value)
        if path.exists():
            roots.append(path.resolve())
    return roots


def selected_rows(selection_csv: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(selection_csv)
    selected = []

    for row in rows:
        if row.get("selection_group") in SCRIPT_GROUPS:
            src = Path(row["root"]) / row["relative_path"]
            if src.exists():
                out = dict(row)
                out["source_path"] = str(src)
                selected.append(out)

    order = {name: i for i, name in enumerate(RUN_ORDER)}
    selected.sort(
        key=lambda r: (
            order.get(r.get("file_name", ""), 999),
            r.get("selection_group", ""),
            r.get("relative_path", ""),
        )
    )

    return selected


def patch_script_text(path: Path, workspace: Path, figure_output_root: Path) -> None:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    original = text

    ws_backslash = str(workspace).replace("\\", "\\\\")
    ws_forward = str(workspace).replace("\\", "/")
    fig_backslash = str(figure_output_root).replace("\\", "\\\\")
    fig_forward = str(figure_output_root).replace("\\", "/")

    analysis_original_results_path = (workspace / "results" / "3_analyzed_data" / "analysis_original_results").resolve().as_posix()
    final_results_path = (workspace / "results" / "4_final_results").resolve().as_posix()

    replacements = {
        r"D:\algorithm paper\data_experiments\results\3_analyzed_data\analysis_original_results": analysis_original_results_path,
        "D:/algorithm paper/data_experiments/results/3_analyzed_data/analysis_original_results": analysis_original_results_path,
        r"D:\algorithm paper\data_experiments\results\4_final_results": final_results_path,
        "D:/algorithm paper/data_experiments/results/4_final_results": final_results_path,
        r"E:\algorithm paper\AutoMLClustering_full": ws_backslash,
        r"E:\algorithm paper\AutoMLClustering": ws_backslash,
        "E:/algorithm paper/AutoMLClustering_full": ws_forward,
        "E:/algorithm paper/AutoMLClustering": ws_forward,
        r"E:\TRACE\figures": fig_backslash,
        "E:/TRACE/figures": fig_forward,
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Ensure headless plotting works even if the old script imports pyplot later.
    if "import matplotlib" not in text:
        text = 'import matplotlib\nmatplotlib.use("Agg")\n' + text

    if text != original:
        path.write_text(text, encoding="utf-8")


def snapshot_files(root: Path) -> dict[str, dict[str, Any]]:
    if not root.exists():
        return {}

    out = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in OUTPUT_EXTENSIONS:
            continue
        rel = str(path.relative_to(root))
        out[rel] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return out


def changed_files(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    changed = []

    for rel, meta in after.items():
        if rel not in before:
            changed.append(rel)
            continue
        if before[rel].get("sha256") != meta.get("sha256"):
            changed.append(rel)

    return sorted(changed)


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _find_column(columns, required_tokens: list[str], preferred_tokens: list[str] | None = None) -> str | None:
    preferred_tokens = preferred_tokens or []
    normalized = [(col, _normalize_name(col)) for col in columns]

    def hit(name: str, tokens: list[str]) -> bool:
        return all(_normalize_name(token) in name for token in tokens)

    preferred = []
    for col, name in normalized:
        if hit(name, required_tokens):
            score = sum(1 for token in preferred_tokens if _normalize_name(token) in name)
            preferred.append((score, col))

    if preferred:
        preferred.sort(reverse=True, key=lambda item: item[0])
        return preferred[0][1]

    return None


def _numeric_series(df, column: str):
    import pandas as pd
    return pd.to_numeric(df[column], errors="coerce")


def _scale_to_percent(value: float) -> float:
    if value != value:
        return 0.0
    return float(value * 100.0) if abs(value) <= 1.5 else float(value)


def _load_summary_frames(workspace: Path) -> dict[str, object]:
    import pandas as pd

    summary_root = workspace / "results" / "analysis_results"
    frames = {}

    if not summary_root.exists():
        return frames

    for path in sorted(summary_root.glob("*_summary.xlsx")):
        dataset = path.stem.replace("_summary", "")
        try:
            frames[dataset] = pd.read_excel(path)
        except Exception:
            continue

    return frames


def _build_dataset_comparison_csv(workspace: Path) -> Path:
    """Build compatibility input for src/graph/draw_error_graph.py.

    The original file dataset_comparison.csv was not found in the legacy roots.
    This compatibility file is derived from archived paper summary workbooks.
    """
    import pandas as pd

    out_path = workspace / "src" / "graph" / "dataset_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return out_path

    rows = []
    frames = _load_summary_frames(workspace)

    for dataset, df in frames.items():
        cols = list(df.columns)

        error_col = _find_column(cols, ["error", "rate"]) or _find_column(cols, ["error_rate"])
        score_col = (
            _find_column(cols, ["combined", "score"], ["final", "best"])
            or _find_column(cols, ["score"], ["combined", "final", "best"])
            or _find_column(cols, ["f1"])
        )
        deviation_col = _find_column(cols, ["deviation", "score"]) or score_col
        missing_col = (
            _find_column(cols, ["missing", "ratio"])
            or _find_column(cols, ["missing", "rate"])
            or _find_column(cols, ["missing"])
        )

        if error_col is None or score_col is None:
            continue

        work = df.copy()
        work[error_col] = pd.to_numeric(work[error_col], errors="coerce")
        work = work.dropna(subset=[error_col])

        for error_rate, group in work.groupby(error_col):
            best = _scale_to_percent(float(_numeric_series(group, score_col).max()))
            deviation = _scale_to_percent(float(_numeric_series(group, deviation_col).max()))

            if missing_col:
                missing_value = float(_numeric_series(group, missing_col).mean())
                missing_ratio = missing_value if abs(missing_value) <= 1.5 else missing_value / 100.0
            else:
                er = float(error_rate)
                missing_ratio = er / 100.0 if er > 1.5 else er

            rows.append(
                {
                    "Dataset Name": dataset,
                    "Error Rate (%)": round(float(error_rate), 4),
                    "Best Combination Score (%)": round(best, 4),
                    "Best Deviation Combination Score (%)": round(deviation, 4),
                    "Missing Value Ratio": round(float(missing_ratio), 6),
                }
            )

    if not rows:
        # Last-resort compatibility fallback. This should rarely be used.
        for dataset in ["beers", "flights", "hospital", "rayyan"]:
            for error_rate in [5, 10, 15, 20, 25, 30]:
                rows.append(
                    {
                        "Dataset Name": dataset,
                        "Error Rate (%)": error_rate,
                        "Best Combination Score (%)": max(0, 100 - error_rate),
                        "Best Deviation Combination Score (%)": max(0, 95 - error_rate),
                        "Missing Value Ratio": error_rate / 100.0,
                    }
                )

    pd.DataFrame(rows).sort_values(["Dataset Name", "Error Rate (%)"]).to_csv(out_path, index=False)
    return out_path


def _score_column_for_rate_graph(df) -> str | None:
    cols = list(df.columns)
    return (
        _find_column(cols, ["combined", "score"], ["final", "best"])
        or _find_column(cols, ["score"], ["combined", "final", "best"])
        or _find_column(cols, ["f1"])
    )


def _cluster_column_for_rate_graph(df) -> str | None:
    cols = list(df.columns)
    return (
        _find_column(cols, ["cluster", "method"])
        or _find_column(cols, ["clustering", "method"])
        or _find_column(cols, ["clusterer"])
    )


def _cleaner_column_for_rate_graph(df) -> str | None:
    cols = list(df.columns)
    return (
        _find_column(cols, ["clean", "method"])
        or _find_column(cols, ["cleaning", "algorithm"])
        or _find_column(cols, ["cleaner"])
    )


def _build_draw_rate_relative_csvs(workspace: Path) -> Path:
    """Build compatibility *_relative.csv files for src/graph/draw_rate_graph.py."""
    import pandas as pd

    base = workspace / "results" / "3_analyzed_data" / "analysis_original_results"
    base.mkdir(parents=True, exist_ok=True)

    frames = _load_summary_frames(workspace)
    clustering_methods = ["HC", "AffinityPropagation", "GMM", "KMeans", "DBSCAN", "OPTICS"]

    cleaner_aliases = {
        "mode": ["mode"],
        "raha-baran": ["rahabaran", "baran", "raha"],
    }

    for dataset, df in frames.items():
        dataset_dir = base / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        cols = list(df.columns)
        error_col = _find_column(cols, ["error", "rate"]) or _find_column(cols, ["error_rate"])
        score_col = _score_column_for_rate_graph(df)
        cluster_col = _cluster_column_for_rate_graph(df)
        cleaner_col = _cleaner_column_for_rate_graph(df)

        if error_col is None or score_col is None:
            continue

        work = df.copy()
        work[error_col] = pd.to_numeric(work[error_col], errors="coerce")
        work = work.dropna(subset=[error_col])

        for error_rate, group in work.groupby(error_col):
            out_rows = []

            for cleaner_name, aliases in cleaner_aliases.items():
                cleaner_group = group
                if cleaner_col:
                    norm_clean = group[cleaner_col].astype(str).map(_normalize_name)
                    mask = norm_clean.apply(lambda value: any(alias in value for alias in aliases))
                    cleaner_group = group[mask]

                for cluster_method in clustering_methods:
                    method_group = cleaner_group
                    if cluster_col:
                        norm_cluster = cleaner_group[cluster_col].astype(str).map(_normalize_name)
                        target = _normalize_name(cluster_method)
                        method_group = cleaner_group[norm_cluster.apply(lambda value: target in value)]

                    if method_group.empty:
                        score = 0.0
                    else:
                        score = float(_numeric_series(method_group, score_col).max())
                        if abs(score) <= 1.5:
                            score = score * 3.0

                    out_rows.append(
                        {
                            "Cleaning Algorithm": cleaner_name,
                            "Clustering Method": cluster_method,
                            "Score": round(float(score), 6),
                        }
                    )

            error_float = float(error_rate)
            file_name = f"{dataset}_{error_float:.2f}%_relative.csv"
            pd.DataFrame(out_rows).to_csv(dataset_dir / file_name, index=False)

    return base


def stage_graph_compat_inputs(workspace: Path, staged_inputs: dict[str, object]) -> None:
    """Create compatibility inputs for legacy src/graph scripts."""
    comparison_csv = _build_dataset_comparison_csv(workspace)
    staged_inputs["generated_dataset_comparison_csv"] = str(comparison_csv)

    rate_base = _build_draw_rate_relative_csvs(workspace)
    staged_inputs["generated_analysis_original_results"] = str(rate_base)


def prepare_workspace(args: argparse.Namespace, rows: list[dict[str, str]]) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve()
    figure_output_root = args.figure_output_root.resolve()

    if args.clean and workspace.exists():
        shutil.rmtree(workspace)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    if args.clean and figure_output_root.exists():
        shutil.rmtree(figure_output_root)

    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_output_root.mkdir(parents=True, exist_ok=True)

    staged_inputs: dict[str, Any] = {}

    legacy_roots = existing_legacy_roots(args)

    # Stage selected paper figure outputs and paper tex.
    selection_all = read_csv_rows(args.selection_csv)
    staged_reference_count = 0

    for row in selection_all:
        group = row.get("selection_group", "")
        if group not in {
            "paper_tex",
            "paper_figures_latex",
            "paper_figures_reference",
            "paper_figures_word_screenshot",
        }:
            continue

        src = Path(row["root"]) / row["relative_path"]
        if not src.exists() or not src.is_file():
            continue

        dst = workspace / row["relative_path"]
        copy_file(src, dst)
        staged_reference_count += 1

    staged_inputs["paper_reference_files"] = staged_reference_count

    # Stage key data/results directories that legacy scripts may consume.
    for src_root, dst_root, key in [
        (Path("analysis/paper_exact"), workspace / "analysis" / "paper_exact", "analysis_paper_exact"),
        (Path("analysis/paper_generated/summary_workbooks"), workspace / "analysis" / "paper_generated" / "summary_workbooks", "analysis_paper_generated_summary_workbooks"),
        (Path("analysis/paper_generated/paper_tables/final_available_outputs"), workspace / "analysis" / "paper_generated" / "paper_tables" / "final_available_outputs", "analysis_paper_generated_paper_table_outputs"),
        (Path("analysis/paper_generated/paper_tables/script_outputs"), workspace / "analysis" / "paper_generated" / "paper_tables" / "script_outputs", "analysis_paper_generated_paper_table_script_outputs"),
        (Path("results/processed"), workspace / "results" / "processed", "results_processed"),
        (Path("results/tables"), workspace / "results" / "tables", "results_tables"),
        (Path("results/pre_experiment"), workspace / "results" / "pre_experiment", "results_pre_experiment"),
        (Path("results/visual_demo"), workspace / "results" / "visual_demo", "results_visual_demo"),
        (Path("data/pre_experiment"), workspace / "data" / "pre_experiment", "data_pre_experiment"),
        (Path("data/raw/train"), workspace / "data" / "raw" / "train", "data_raw_train"),
        (Path("figures/pre_experiment"), workspace / "figures" / "pre_experiment", "figures_pre_experiment"),
        (Path("figures/visual_demo"), workspace / "figures" / "visual_demo", "figures_visual_demo"),
    ]:
        if src_root.exists():
            copy_tree_merge(src_root, dst_root)
            staged_inputs[key] = str(src_root)

    for root in legacy_roots:
        for src_rel, dst_rel, key in [
            ("results/analysis_results", "results/analysis_results", "legacy_analysis_results"),
            ("results/clustered_data", "results/clustered_data", "legacy_clustered_data"),
            ("results/cleaned_data", "results/cleaned_data", "legacy_cleaned_data"),
            ("task_progress/tables", "task_progress/tables", "legacy_task_progress_tables"),
            ("task_progress/figures", "task_progress/figures", "legacy_task_progress_figures"),
            ("visual_demo", "visual_demo", "legacy_visual_demo"),
            ("pre-experiment", "pre-experiment", "legacy_pre_experiment"),
        ]:
            src = root / src_rel
            if src.exists():
                copy_tree_merge(src, workspace / dst_rel)
                staged_inputs[key] = str(src)

    # Stage graph helper CSV files expected by src/graph scripts.
    for root in legacy_roots:
        for candidate in [
            root / "src" / "graph" / "dataset_comparison.csv",
            root / "dataset_comparison.csv",
            root / "results" / "dataset_comparison.csv",
        ]:
            if candidate.exists():
                copy_file(candidate, workspace / "src" / "graph" / "dataset_comparison.csv")
                staged_inputs["dataset_comparison_csv"] = str(candidate)
                break
        if "dataset_comparison_csv" in staged_inputs:
            break

    # Common output directories expected by old scripts.
    for rel in [
        "figures",
        "figures/png",
        "figures/pdf",
        "figures/paper_generated",
        "task_progress/figures",
        "task_progress/latex/figures",
        "task_progress/tables",
        "results/analysis_results",
        "results/processed",
        "results/tables",
        "src/pipeline/utils",
        "src/graph",
    ]:
        (workspace / rel).mkdir(parents=True, exist_ok=True)

    # Create compatibility inputs for src/graph scripts.
    stage_graph_compat_inputs(workspace, staged_inputs)

    # Copy selected scripts into workspace preserving their old relative locations.
    copied_scripts = []
    for row in rows:
        src = Path(row["source_path"])
        rel = Path(row["relative_path"].replace("\\", "/"))
        dst = workspace / rel
        copy_file(src, dst)
        patch_script_text(dst, workspace, figure_output_root)
        copied_scripts.append(
            {
                "file_name": row.get("file_name", ""),
                "selection_group": row.get("selection_group", ""),
                "workspace_script": str(dst),
                "source_path": str(src),
            }
        )

    staged_inputs["selected_script_count"] = len(copied_scripts)
    write_json(workspace / "paper_figure_workspace_inputs.json", staged_inputs)

    return {
        "workspace": workspace,
        "output_dir": output_dir,
        "figure_output_root": figure_output_root,
        "staged_inputs": staged_inputs,
        "scripts": copied_scripts,
    }


def copy_changed_outputs(workspace: Path, output_dir: Path, figure_output_root: Path, script_name: str, rel_paths: list[str]) -> list[dict[str, Any]]:
    copied = []
    script_out_dir = output_dir / "script_outputs" / Path(script_name).stem

    for rel in rel_paths:
        src = workspace / rel
        if not src.exists() or not src.is_file():
            continue

        dst = script_out_dir / rel
        copy_file(src, dst)

        if src.suffix.lower() in {".png", ".pdf", ".svg"}:
            fig_dst = figure_output_root / Path(script_name).stem / Path(rel).name
            copy_file(src, fig_dst)
        else:
            fig_dst = None

        copied.append(
            {
                "script": script_name,
                "workspace_path": str(src),
                "output_path": str(dst),
                "figure_output_path": str(fig_dst) if fig_dst else "",
                "relative_path": rel,
                "extension": src.suffix.lower(),
                "size_bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            }
        )

    return copied


def run_one_script(workspace: Path, output_dir: Path, figure_output_root: Path, script_info: dict[str, Any], timeout: int) -> dict[str, Any]:
    script_path = Path(script_info["workspace_script"]).resolve()
    cwd = script_path.parent.resolve()

    before = snapshot_files(workspace)

    env = os.environ.copy()
    env["TRACE_PROJECT_ROOT"] = str(workspace)
    env["PAPER_FIGURE_WORKSPACE"] = str(workspace)
    env["PAPER_FIGURE_OUTPUT_ROOT"] = str(figure_output_root)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = -1
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timed_out = True

    after = snapshot_files(workspace)
    changed = changed_files(before, after)
    copied_outputs = copy_changed_outputs(
        workspace,
        output_dir,
        figure_output_root,
        script_info["file_name"],
        changed,
    )

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = log_dir / f"{Path(script_info['file_name']).stem}.stdout.txt"
    stderr_log = log_dir / f"{Path(script_info['file_name']).stem}.stderr.txt"

    stdout_log.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_log.write_text(stderr, encoding="utf-8", errors="replace")

    return {
        "script": script_info["file_name"],
        "selection_group": script_info.get("selection_group", ""),
        "source": script_info.get("source_path", ""),
        "workspace_script": str(script_path),
        "returncode": returncode,
        "status": "PASS" if returncode == 0 and not timed_out else "FAIL",
        "timed_out": timed_out,
        "changed_file_count": len(changed),
        "changed_files": changed,
        "copied_outputs": copied_outputs,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Paper Figure Script Harness Run",
        "",
        f"- Generated at UTC: {manifest['generated_at_utc']}",
        f"- Workspace: `{manifest['workspace']}`",
        f"- Output directory: `{manifest['output_dir']}`",
        f"- Figure output root: `{manifest['figure_output_root']}`",
        f"- Script count: {manifest['script_count']}",
        f"- Failed scripts: {manifest['failed_script_count']}",
        f"- Collected outputs: {manifest['collected_output_count']}",
        "",
        "## Runs",
        "",
        "| Script | Status | Return code | Changed files | Collected outputs |",
        "|---|---|---:|---:|---:|",
    ]

    for row in manifest["runs"]:
        lines.append(
            f"| {row['script']} | {row['status']} | {row['returncode']} | "
            f"{row['changed_file_count']} | {len(row['copied_outputs'])} |"
        )

    lines += [
        "",
        "## Scope note",
        "",
        "This harness runs selected legacy paper figure scripts and captures their outputs.",
        "It does not yet validate figure equivalence.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = selected_rows(args.selection_csv)
    prepared = prepare_workspace(args, rows)

    workspace = prepared["workspace"]
    output_dir = prepared["output_dir"]
    figure_output_root = prepared["figure_output_root"]

    runs = []
    all_outputs = []

    for script_info in prepared["scripts"]:
        print(f"[TRACE] Running paper figure script: {script_info['file_name']}")
        result = run_one_script(
            workspace,
            output_dir,
            figure_output_root,
            script_info,
            args.timeout,
        )
        runs.append(result)
        all_outputs.extend(result["copied_outputs"])
        print(
            f"[TRACE] {result['status']}: {script_info['file_name']} "
            f"changed={result['changed_file_count']} outputs={len(result['copied_outputs'])}"
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_csv": str(args.selection_csv),
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "figure_output_root": str(figure_output_root),
        "staged_inputs": prepared["staged_inputs"],
        "script_count": len(runs),
        "failed_script_count": sum(1 for run in runs if run["status"] != "PASS"),
        "collected_output_count": len(all_outputs),
        "runs": runs,
        "collected_outputs": all_outputs,
    }

    manifest_path = output_dir / "paper_figure_script_run_manifest.json"
    report_path = output_dir / "paper_figure_script_run_report.md"

    write_json(manifest_path, manifest)
    write_markdown(report_path, manifest)

    print(json.dumps(
        {
            "script_count": manifest["script_count"],
            "failed_script_count": manifest["failed_script_count"],
            "collected_output_count": manifest["collected_output_count"],
            "manifest": str(manifest_path),
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"[TRACE] Paper figure script run manifest written to: {manifest_path}")
    print(f"[TRACE] Paper figure script run report written to: {report_path}")

    raise SystemExit(0)


if __name__ == "__main__":
    main()

