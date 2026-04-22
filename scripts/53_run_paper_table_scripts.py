#!/usr/bin/env python3
"""Run selected legacy paper table scripts inside a controlled Mode A workspace."""

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


RUN_ORDER = [
    "summary_patch.py",
    "6.1.1_cal_1.py",
    "6.1.1_cal_2.py",
    "6.1.1_cal_3.py",
    "6.1.2_cal_1.py",
    "6.1.2_cal_2.py",
    "Tab.8_cal.py",
    "Tab.9_cal.py",
    "Tab.11-12_cal.py",
]

OUTPUT_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".png", ".pdf", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper table scripts in a controlled workspace.")
    parser.add_argument(
        "--archive-manifest",
        type=Path,
        default=Path("analysis/paper_exact/mode_a_paper_exact_archive_manifest.json"),
    )
    parser.add_argument(
        "--generated-summary-root",
        type=Path,
        default=Path("analysis/paper_generated/summary_workbooks"),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("analysis/paper_generated/paper_table_workspace"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/paper_generated/paper_tables"),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--legacy-roots",
        nargs="+",
        default=[
            r"E:\algorithm paper\AutoMLClustering_full",
            r"E:\algorithm paper\AutoMLClustering",
        ],
        help="Legacy roots used to stage paper-exact inputs such as comparison.json and clustered_data.",
    )
    parser.add_argument(
        "--include-analysis-scripts",
        action="store_true",
        help="Also run archived analysis scripts. Default runs paper table scripts only.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree_merge(src_root: Path, dst_root: Path) -> None:
    if not src_root.exists():
        return
    for path in src_root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(src_root)
            copy_file(path, dst_root / rel)


def snapshot_files(root: Path) -> dict[str, dict[str, Any]]:
    if not root.exists():
        return {}

    out: dict[str, dict[str, Any]] = {}
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


def patch_script_text(path: Path, workspace: Path) -> None:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    original = text

    ws_backslash = str(workspace).replace("\\", "\\\\")
    ws_forward = str(workspace).replace("\\", "/")

    replacements = {
        r"E:\algorithm paper\AutoMLClustering_full": ws_backslash,
        r"E:\algorithm paper\AutoMLClustering": ws_backslash,
        "E:/algorithm paper/AutoMLClustering_full": ws_forward,
        "E:/algorithm paper/AutoMLClustering": ws_forward,
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    if text != original:
        path.write_text(text, encoding="utf-8")


def selected_scripts(manifest: dict[str, Any], include_analysis_scripts: bool) -> list[dict[str, Any]]:
    groups = {"paper_table_scripts"}
    if include_analysis_scripts:
        groups.add("analysis_scripts")

    rows = []
    for item in manifest.get("copied", []):
        if item.get("selection_group") not in groups:
            continue
        src = Path(item.get("replay_path") or item.get("archive_path"))
        if src.exists():
            rows.append(
                {
                    "file_name": src.name,
                    "selection_group": item.get("selection_group", ""),
                    "source": item.get("source", ""),
                    "archive_path": item.get("archive_path", ""),
                    "replay_path": item.get("replay_path", ""),
                    "path": src,
                }
            )

    order_index = {name: i for i, name in enumerate(RUN_ORDER)}
    rows.sort(key=lambda r: (order_index.get(r["file_name"], 999), r["file_name"]))
    return rows


def existing_legacy_roots(args: argparse.Namespace) -> list[Path]:
    roots = []
    for value in getattr(args, "legacy_roots", []):
        path = Path(value)
        if path.exists():
            roots.append(path.resolve())
    return roots


def copy_first_existing_file(candidates: list[Path], dst: Path) -> str:
    for src in candidates:
        if src.exists() and src.is_file():
            copy_file(src, dst)
            return str(src)
    return ""


def copy_first_existing_dir(candidates: list[Path], dst: Path) -> str:
    for src in candidates:
        if src.exists() and src.is_dir():
            copy_tree_merge(src, dst)
            return str(src)
    return ""


def prepare_workspace(args: argparse.Namespace, manifest: dict[str, Any]) -> Path:
    workspace = args.workspace

    if args.clean and workspace.exists():
        shutil.rmtree(workspace)
    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    workspace.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    staged_inputs: dict[str, str] = {}

    legacy_roots = existing_legacy_roots(args)

    # ------------------------------------------------------------------
    # 1. Stage archived paper summaries in the exact legacy location.
    #
    # Legacy table scripts expect files such as:
    #   results/analysis_results/beers_summary.xlsx
    # with the original paper schema, e.g. Sheet1 with columns such as
    # cluster_method, cleaning_method, F1, error_rate, parameters, etc.
    #
    # The generated summaries are kept only as auxiliary outputs because
    # their schema is different.
    # ------------------------------------------------------------------
    archived_summary_root = Path("analysis/paper_exact/analysis_summaries/results/analysis_results")
    if archived_summary_root.exists():
        copy_tree_merge(archived_summary_root, workspace / "results" / "analysis_results")
        staged_inputs["archived_summary_root"] = str(archived_summary_root)
    else:
        # Fallback only. This is less compatible with legacy scripts.
        for path in args.generated_summary_root.glob("*.xlsx"):
            copy_file(path, workspace / "results" / "analysis_results" / path.name)
        staged_inputs["archived_summary_root"] = "missing; used generated summaries as fallback"

    # Keep generated summary workbooks in a separate directory for traceability.
    if args.generated_summary_root.exists():
        copy_tree_merge(args.generated_summary_root, workspace / "results" / "generated_summary_workbooks")
        staged_inputs["generated_summary_root"] = str(args.generated_summary_root)

    # ------------------------------------------------------------------
    # 2. Stage archived analysis CSV inputs and task_progress table files.
    # ------------------------------------------------------------------
    analysis_csv_roots = [
        Path("analysis/paper_exact/analysis_csv/results/analysis_results"),
        Path("analysis/paper_exact/analysis_csv/task_progress/tables"),
    ]

    for src_root in analysis_csv_roots:
        if not src_root.exists():
            continue
        if src_root.as_posix().endswith("results/analysis_results"):
            copy_tree_merge(src_root, workspace / "results" / "analysis_results")
            staged_inputs["analysis_csv_results"] = str(src_root)
        else:
            copy_tree_merge(src_root, workspace / "task_progress" / "tables")
            staged_inputs["analysis_csv_task_progress_tables"] = str(src_root)

    # ------------------------------------------------------------------
    # 3. Stage raw result JSON files.
    # ------------------------------------------------------------------
    for item in manifest.get("copied", []):
        if item.get("selection_group") != "raw_results":
            continue
        replay_path = Path(item.get("replay_path", ""))
        archive_path = Path(item.get("archive_path", ""))
        src = replay_path if replay_path.exists() else archive_path
        if not src.exists():
            continue
        copy_file(src, workspace / "results" / src.name)

    # ------------------------------------------------------------------
    # 4. Stage full legacy result directories when available.
    # These are needed by Tab.8/Tab.9 and hyperparameter-shift scripts.
    # ------------------------------------------------------------------
    copied_clustered = copy_first_existing_dir(
        [root / "results" / "clustered_data" for root in legacy_roots],
        workspace / "results" / "clustered_data",
    )
    if copied_clustered:
        staged_inputs["clustered_data"] = copied_clustered

    copied_cleaned = copy_first_existing_dir(
        [root / "results" / "cleaned_data" for root in legacy_roots],
        workspace / "results" / "cleaned_data",
    )
    if copied_cleaned:
        staged_inputs["cleaned_data"] = copied_cleaned

    copied_analysis_results = copy_first_existing_dir(
        [root / "results" / "analysis_results" for root in legacy_roots],
        workspace / "results" / "analysis_results",
    )
    if copied_analysis_results:
        staged_inputs["legacy_analysis_results"] = copied_analysis_results

    # ------------------------------------------------------------------
    # 5. Stage comparison.json for Tab.8/analyze_cleaning-style scripts.
    # ------------------------------------------------------------------
    comparison_src = copy_first_existing_file(
        [
            root / "src" / "pipeline" / "train" / "comparison.json"
            for root in legacy_roots
        ] + [
            root / "src" / "pipeline" / "comparison.json"
            for root in legacy_roots
        ] + [
            root / "comparison.json"
            for root in legacy_roots
        ],
        workspace / "src" / "pipeline" / "train" / "comparison.json",
    )
    if comparison_src:
        staged_inputs["comparison_json"] = comparison_src

    # ------------------------------------------------------------------
    # 6. Stage dataset folders. Some scripts infer clean/dirty CSV paths.
    # ------------------------------------------------------------------
    copied_datasets = copy_first_existing_dir(
        [root / "datasets" / "train" for root in legacy_roots] +
        [root / "data" / "raw" / "train" for root in legacy_roots] +
        [Path("data/raw/train")],
        workspace / "datasets" / "train",
    )
    if copied_datasets:
        staged_inputs["datasets_train"] = copied_datasets

    # ------------------------------------------------------------------
    # 7. Mirror selected scripts into src/pipeline/utils.
    # ------------------------------------------------------------------
    for row in selected_scripts(manifest, include_analysis_scripts=True):
        dst = workspace / "src" / "pipeline" / "utils" / row["file_name"]
        copy_file(row["path"], dst)
        patch_script_text(dst, workspace)

    # ------------------------------------------------------------------
    # 8. Create common directories expected by old scripts.
    # ------------------------------------------------------------------
    for rel in [
        "results/analysis_results",
        "results/analysis_results/stats",
        "results/clustered_data",
        "results/cleaned_data",
        "task_progress/tables",
        "task_progress/tables/6.4.1tables",
        "task_progress/tables/6.4.2tables",
        "task_progress/tables/6.4.3tables",
        "task_progress/tables/6.4.4tables",
        "task_progress/tables/6.4.5tables",
        "task_progress/tables/6.4.6tables",
        "src/pipeline/train",
    ]:
        (workspace / rel).mkdir(parents=True, exist_ok=True)

    write_json(workspace / "paper_table_workspace_inputs.json", staged_inputs)

    return workspace


def copy_changed_outputs(
    workspace: Path,
    output_dir: Path,
    script_name: str,
    rel_paths: list[str],
) -> list[dict[str, Any]]:
    copied = []
    script_out_dir = output_dir / "script_outputs" / Path(script_name).stem

    for rel in rel_paths:
        src = workspace / rel
        if not src.exists() or not src.is_file():
            continue

        dst = script_out_dir / rel
        copy_file(src, dst)

        copied.append(
            {
                "script": script_name,
                "workspace_path": str(src),
                "output_path": str(dst),
                "relative_path": rel,
                "size_bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            }
        )

    return copied


def run_one_script(
    workspace: Path,
    output_dir: Path,
    script_info: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    script_name = script_info["file_name"]
    script_path = (workspace / "src" / "pipeline" / "utils" / script_name).resolve()
    cwd = (workspace / "src" / "pipeline" / "utils").resolve()

    before = snapshot_files(workspace)

    env = os.environ.copy()
    env["TRACE_PROJECT_ROOT"] = str(workspace)
    env["PAPER_REPLAY_WORKSPACE"] = str(workspace)

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
    copied_outputs = copy_changed_outputs(workspace, output_dir, script_name, changed)

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{Path(script_name).stem}.stdout.txt").write_text(stdout, encoding="utf-8", errors="replace")
    (log_dir / f"{Path(script_name).stem}.stderr.txt").write_text(stderr, encoding="utf-8", errors="replace")

    return {
        "script": script_name,
        "selection_group": script_info.get("selection_group", ""),
        "source": script_info.get("source", ""),
        "workspace_script": str(script_path),
        "returncode": returncode,
        "status": "PASS" if returncode == 0 and not timed_out else "FAIL",
        "timed_out": timed_out,
        "changed_file_count": len(changed),
        "changed_files": changed,
        "copied_outputs": copied_outputs,
        "stdout_log": str(log_dir / f"{Path(script_name).stem}.stdout.txt"),
        "stderr_log": str(log_dir / f"{Path(script_name).stem}.stderr.txt"),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Paper Table Script Harness Run",
        "",
        f"- Generated at UTC: {manifest['generated_at_utc']}",
        f"- Workspace: `{manifest['workspace']}`",
        f"- Output directory: `{manifest['output_dir']}`",
        f"- Script count: {manifest['script_count']}",
        f"- Failed scripts: {manifest['failed_script_count']}",
        "",
        "## Runs",
        "",
        "| Script | Status | Return code | Changed files |",
        "|---|---|---:|---:|",
    ]

    for row in manifest["runs"]:
        lines.append(
            f"| {row['script']} | {row['status']} | {row['returncode']} | {row['changed_file_count']} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "This harness executes legacy paper table scripts in a controlled workspace.",
        "A PASS here means the harness could run the scripts and capture their outputs.",
        "It does not yet imply byte-identical paper table reproduction.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.archive_manifest = args.archive_manifest.resolve()
    args.generated_summary_root = args.generated_summary_root.resolve()
    args.workspace = args.workspace.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = normalize_args(parse_args())
    manifest = read_json(args.archive_manifest)

    workspace = prepare_workspace(args, manifest)
    scripts = selected_scripts(manifest, include_analysis_scripts=args.include_analysis_scripts)

    runs = []
    all_outputs = []

    for script in scripts:
        print(f"[TRACE] Running paper table script: {script['file_name']}")
        result = run_one_script(workspace, args.output_dir, script, args.timeout)
        runs.append(result)
        all_outputs.extend(result["copied_outputs"])
        print(f"[TRACE] {result['status']}: {script['file_name']} changed={result['changed_file_count']}")

    out_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_manifest": str(args.archive_manifest),
        "workspace": str(workspace),
        "output_dir": str(args.output_dir),
        "script_count": len(scripts),
        "failed_script_count": sum(1 for run in runs if run["status"] != "PASS"),
        "runs": runs,
        "collected_output_count": len(all_outputs),
        "collected_outputs": all_outputs,
    }

    manifest_path = args.output_dir / "paper_table_script_run_manifest.json"
    report_path = args.output_dir / "paper_table_script_run_report.md"
    write_json(manifest_path, out_manifest)
    write_markdown(report_path, out_manifest)

    print(json.dumps(
        {
            "script_count": out_manifest["script_count"],
            "failed_script_count": out_manifest["failed_script_count"],
            "collected_output_count": out_manifest["collected_output_count"],
            "manifest": str(manifest_path),
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"[TRACE] Paper table script run manifest written to: {manifest_path}")
    print(f"[TRACE] Paper table script run report written to: {report_path}")

    raise SystemExit(0 if out_manifest["failed_script_count"] == 0 else 1)


if __name__ == "__main__":
    main()

