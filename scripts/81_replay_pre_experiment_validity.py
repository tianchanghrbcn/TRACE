#!/usr/bin/env python3
"""Replay and validate TRACE pre-experiment + validity-sensitivity evidence.

Reviewer-facing purpose
-----------------------
This script does NOT rerun the full cleaning or clustering benchmark.  It only
replays the lightweight pre-experiment evidence and validates the robustness
checks that are stated in the paper:

1. pre-experimental alpha calibration from data/pre_experiment/alpha_metrics.csv;
2. alpha-sensitivity replay from saved paper summary workbooks;
3. seed-sensitivity evidence from archived sensitivity CSV/JSON/Markdown files.

Default command from the repository root:

    python scripts/81_replay_pre_experiment_validity.py

For final release-package checking, use:

    python scripts/81_replay_pre_experiment_validity.py --error-model-policy fail --generated-data-policy fail

Outputs
-------
- results/pre_experiment/pre_experiment_manifest.json
- figures/pre_experiment/*
- analysis/validity_sensitivity/alpha_*.csv/json/md
- analysis/validity_sensitivity/validity_sensitivity_summary.json
- analysis/validity_sensitivity/validity_sensitivity_summary.md
- analysis/validity_sensitivity/pre_experiment_validity_report.json
- analysis/validity_sensitivity/pre_experiment_validity_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DATASETS = ("beers", "flights", "hospital", "rayyan")

ALPHA_OUTPUTS = (
    "alpha_row_scores.csv",
    "alpha_cleaner_medians.csv",
    "alpha_clusterer_medians.csv",
    "alpha_combo_medians.csv",
    "alpha_rank_correlation.csv",
    "alpha_top_combo_stability.csv",
    "alpha_sensitivity_summary.json",
    "alpha_sensitivity_report.md",
)

SEED_OUTPUTS = (
    "seed_sensitivity_runs.csv",
    "seed_sensitivity_group_summary.csv",
    "seed_sensitivity_summary.json",
    "seed_sensitivity_report.md",
)

# Keep error-type sensitivity out of the reviewer-facing artifact unless the
# paper is explicitly revised to discuss it.  The default policy is only WARN so
# the script can run on a local working tree that still contains generated files.
ERROR_MODEL_PATTERNS = (
    "generated_error_model_data",
    "error_model_sensitivity_*",
)

GENERATED_DATA_PATTERNS = (
    "generated_seed_data",
    "generated_error_model_data",
)

TRUTHY = {"1", "true", "t", "yes", "y", "same", "preserved", "match", "matched"}
FALSY = {"0", "false", "f", "no", "n", "different", "changed", "mismatch", "not_preserved"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay TRACE pre-experiment outputs and validate alpha/seed sensitivity evidence."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="TRACE repository root. Default: parent of scripts/.",
    )
    parser.add_argument(
        "--pre-source-csv",
        type=Path,
        default=Path("data/pre_experiment/alpha_metrics.csv"),
        help="Committed alpha calibration CSV used by the pre-experiment replay.",
    )
    parser.add_argument(
        "--pre-output-dir",
        type=Path,
        default=Path("results/pre_experiment"),
        help="Output directory for replayed pre-experiment tables/manifests.",
    )
    parser.add_argument(
        "--pre-figure-dir",
        type=Path,
        default=Path("figures/pre_experiment"),
        help="Output directory for replayed pre-experiment figures.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("analysis/validity_sensitivity/inputs/analysis_results"),
        help="Directory containing beers/flights/hospital/rayyan summary workbooks.",
    )
    parser.add_argument(
        "--validity-dir",
        type=Path,
        default=Path("analysis/validity_sensitivity"),
        help="Directory containing/writing validity-sensitivity evidence.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.25, 0.47, 0.50, 0.75],
        help="Alpha values to replay for sensitivity checking.",
    )
    parser.add_argument(
        "--base-alpha",
        type=float,
        default=0.47,
        help="Paper alpha value used as the base ranking.",
    )
    parser.add_argument(
        "--min-alpha-spearman",
        type=float,
        default=0.88,
        help="Minimum accepted Spearman correlation for alpha sensitivity.",
    )
    parser.add_argument(
        "--min-top-stability",
        type=float,
        default=0.833,
        help="Minimum accepted optimal-combination stability rate.",
    )
    parser.add_argument(
        "--min-seed-preserved-rate",
        type=float,
        default=0.708,
        help="Minimum accepted seed trend-direction preservation rate.",
    )
    parser.add_argument(
        "--rate-tolerance",
        type=float,
        default=0.005,
        help="Tolerance for rounded percentage claims such as 70.8%.",
    )
    parser.add_argument(
        "--min-score-sanity",
        type=float,
        default=0.999,
        help="Minimum accepted rank/Pearson sanity between workbook Combined Score and recomputed H_alpha."
        " If unavailable, this produces a warning, not a hard failure.",
    )
    parser.add_argument(
        "--skip-pre-experiment",
        action="store_true",
        help="Do not run scripts/38_build_pre_experiment_outputs.py; validate existing files only.",
    )
    parser.add_argument(
        "--skip-alpha-rebuild",
        action="store_true",
        help="Do not run scripts/80_build_alpha_sensitivity.py; validate existing alpha files only.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Shortcut for --skip-pre-experiment --skip-alpha-rebuild.",
    )
    parser.add_argument(
        "--error-model-policy",
        choices=("ignore", "warn", "fail"),
        default="warn",
        help="How to handle error-model sensitivity files in the reviewer-facing package.",
    )
    parser.add_argument(
        "--generated-data-policy",
        choices=("ignore", "warn", "fail"),
        default="warn",
        help="How to handle generated sensitivity data directories such as generated_seed_data/.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("analysis/validity_sensitivity/pre_experiment_validity_report.json"),
        help="Combined validation report JSON.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def run_python(root: Path, args: list[str]) -> dict[str, Any]:
    print("[TRACE] >>>", " ".join([sys.executable] + args))
    proc = subprocess.run(
        [sys.executable] + args,
        cwd=root,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return {
        "command": " ".join([sys.executable] + args),
        "returncode": proc.returncode,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str = "", **extra: Any) -> None:
    row = {"name": name, "status": "PASS" if ok else "FAIL", "ok": bool(ok), "detail": detail}
    row.update(extra)
    checks.append(row)


def add_warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        s = value.strip().replace("%", "")
        try:
            v = float(s)
        except Exception:
            return None
        if "%" in value or v > 1.0:
            v = v / 100.0
        return v if math.isfinite(v) else None
    return None


def walk_json(obj: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield prefix, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_json(v, prefix + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_json(v, prefix + (str(i),))


def get_path(obj: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def find_seed_rate_from_json(obj: dict[str, Any]) -> tuple[float | None, str]:
    exact_keys = {
        "trend_direction_preserved_rate",
        "trend_direction_preservation_rate",
        "direction_preserved_rate",
        "direction_preservation_rate",
        "trend_preserved_rate",
        "trend_preservation_rate",
        "preserved_direction_rate",
        "same_direction_rate",
        "trend_match_rate",
        "direction_match_rate",
    }

    candidates: list[tuple[float, str, int]] = []
    for path, value in walk_json(obj):
        if not path:
            continue
        key = path[-1].lower()
        val = numeric_or_none(value)
        if val is None or not (0.0 <= val <= 1.0):
            continue
        joined = ".".join(path)
        score = 0
        if key in exact_keys:
            score += 100
        if "seed" in joined.lower():
            score += 10
        if "trend" in joined.lower():
            score += 10
        if "direction" in joined.lower():
            score += 10
        if "preserv" in joined.lower():
            score += 10
        if "rate" in joined.lower() or "ratio" in joined.lower() or "fraction" in joined.lower():
            score += 5
        if score >= 20:
            candidates.append((val, joined, score))

    if candidates:
        candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
        val, source, _ = candidates[0]
        return val, f"json:{source}"

    # Common count-based fallback.
    count_key_groups = [
        ("preserved_group_count", "test_group_count"),
        ("preserved_groups", "test_groups"),
        ("same_direction_count", "test_group_count"),
        ("trend_preserved_count", "test_group_count"),
        ("matched_count", "total_count"),
    ]
    flat = {".".join(path).lower(): value for path, value in walk_json(obj) if path}
    for num_key, den_key in count_key_groups:
        num = None
        den = None
        num_source = den_source = ""
        for k, v in flat.items():
            if k.endswith(num_key):
                num = numeric_or_none(v)
                num_source = k
            if k.endswith(den_key):
                den = numeric_or_none(v)
                den_source = k
        if num is not None and den not in (None, 0):
            return float(num / den), f"json:{num_source}/{den_source}"

    return None, "not_found"


def parse_boolish(value: str) -> int | None:
    s = str(value).strip().lower()
    if s in TRUTHY:
        return 1
    if s in FALSY:
        return 0
    try:
        v = float(s)
    except Exception:
        return None
    if v in (0.0, 1.0):
        return int(v)
    return None


def find_seed_rate_from_group_csv(path: Path) -> tuple[float | None, str]:
    if not path.exists():
        return None, "csv_missing"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None, "csv_no_header"
        fields = reader.fieldnames
        lower_fields = {field: field.lower() for field in fields}
        candidate_fields = [
            field
            for field, low in lower_fields.items()
            if (
                "preserv" in low
                or "same_direction" in low
                or "direction_match" in low
                or "trend_match" in low
                or "trend_direction" in low
            )
        ]
        rows = list(reader)
    for field in candidate_fields:
        vals = [parse_boolish(row.get(field, "")) for row in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            return float(sum(vals) / len(vals)), f"csv:{path.name}:{field}"
    return None, "csv_no_candidate_column"


def find_seed_rate_from_text(text: str) -> tuple[float | None, str]:
    # Prefer explicit 70.8%-style statements.  This supports existing reports
    # without forcing a specific JSON schema.
    for match in re.finditer(r"(?P<num>\d+(?:\.\d+)?)\s*%", text):
        num = float(match.group("num")) / 100.0
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        ctx = text[start:end].lower()
        if ("seed" in ctx or "trend" in ctx or "direction" in ctx) and (
            "preserv" in ctx or "same" in ctx or "match" in ctx
        ):
            return num, "report_md:percentage"
    for match in re.finditer(r"(?<!\d)(0\.\d+)(?!\d)", text):
        num = float(match.group(1))
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        ctx = text[start:end].lower()
        if ("seed" in ctx or "trend" in ctx or "direction" in ctx) and (
            "preserv" in ctx or "same" in ctx or "match" in ctx
        ):
            return num, "report_md:decimal"
    return None, "report_md_not_found"


def find_seed_preserved_rate(validity_dir: Path) -> tuple[float | None, str]:
    summary_path = validity_dir / "seed_sensitivity_summary.json"
    if summary_path.exists():
        try:
            rate, source = find_seed_rate_from_json(read_json(summary_path))
            if rate is not None:
                return rate, source
        except Exception:
            pass

    csv_path = validity_dir / "seed_sensitivity_group_summary.csv"
    rate, source = find_seed_rate_from_group_csv(csv_path)
    if rate is not None:
        return rate, source

    report_path = validity_dir / "seed_sensitivity_report.md"
    if report_path.exists():
        rate, source = find_seed_rate_from_text(read_text(report_path))
        if rate is not None:
            return rate, source

    return None, "not_found"


def validate_pre_experiment(root: Path, pre_output_dir: Path, pre_figure_dir: Path, checks: list[dict[str, Any]]) -> None:
    alpha_csv = pre_output_dir / "alpha_metrics.csv"
    manifest = pre_output_dir / "pre_experiment_manifest.json"
    add_check(checks, "pre_experiment_alpha_metrics_exists", alpha_csv.exists(), rel(root, alpha_csv))
    if alpha_csv.exists():
        rows = count_csv_rows(alpha_csv)
        add_check(checks, "pre_experiment_alpha_metrics_has_rows", rows >= 4, f"rows={rows}", rows=rows)
    add_check(checks, "pre_experiment_manifest_exists", manifest.exists(), rel(root, manifest))
    add_check(checks, "pre_experiment_figure_dir_exists", pre_figure_dir.exists(), rel(root, pre_figure_dir))
    if pre_figure_dir.exists():
        figures = [p for p in pre_figure_dir.iterdir() if p.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}]
        add_check(checks, "pre_experiment_figures_exist", len(figures) > 0, f"figure_count={len(figures)}", figure_count=len(figures))


def validate_alpha(root: Path, validity_dir: Path, args: argparse.Namespace, checks: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    for name in ALPHA_OUTPUTS:
        path = validity_dir / name
        add_check(checks, f"alpha_file_exists:{name}", path.exists(), rel(root, path), size_bytes=path.stat().st_size if path.exists() else 0)
        if path.suffix.lower() == ".csv" and path.exists():
            rows = count_csv_rows(path)
            add_check(checks, f"alpha_csv_has_rows:{name}", rows > 0, f"rows={rows}", rows=rows)

    summary_path = validity_dir / "alpha_sensitivity_summary.json"
    if not summary_path.exists():
        return {}
    summary = read_json(summary_path)

    status = str(summary.get("status", "")).upper()
    add_check(checks, "alpha_summary_status_PASS", status == "PASS", f"status={status}")

    base_alpha = numeric_or_none(summary.get("base_alpha"))
    add_check(
        checks,
        "alpha_base_value_matches_paper",
        base_alpha is not None and abs(base_alpha - args.base_alpha) <= 1e-9,
        f"base_alpha={base_alpha}, expected={args.base_alpha}",
        value=base_alpha,
    )

    observed_alphas = [numeric_or_none(v) for v in summary.get("alphas", []) if numeric_or_none(v) is not None]
    missing_alphas = [a for a in args.alphas if not any(abs(a - b) <= 1e-9 for b in observed_alphas)]
    add_check(
        checks,
        "alpha_tested_values_match_paper",
        not missing_alphas,
        f"observed={observed_alphas}, missing={missing_alphas}",
        observed=observed_alphas,
        missing=missing_alphas,
    )

    for key in ("min_combo_spearman", "min_cleaner_spearman", "min_clusterer_spearman"):
        val = numeric_or_none(summary.get(key))
        ok = val is not None and val >= args.min_alpha_spearman
        add_check(checks, f"alpha_{key}_ge_{args.min_alpha_spearman}", ok, f"{key}={val}", value=val)

    top_rate = numeric_or_none(summary.get("top_combo_stability_rate"))
    ok = top_rate is not None and top_rate + args.rate_tolerance >= args.min_top_stability
    add_check(checks, f"alpha_top_combo_stability_ge_{args.min_top_stability}", ok, f"top_combo_stability_rate={top_rate}", value=top_rate)

    sanity = summary.get("score_sanity", {})
    if isinstance(sanity, dict) and sanity.get("available"):
        for key in ("spearman_existing_vs_recomputed", "pearson_existing_vs_recomputed"):
            val = numeric_or_none(sanity.get(key))
            ok = val is not None and val >= args.min_score_sanity
            add_check(checks, f"alpha_score_sanity_{key}_ge_{args.min_score_sanity}", ok, f"{key}={val}", value=val)
    else:
        add_warning(warnings, "Alpha score sanity could not be checked because the Combined Score column was unavailable in the summary workbooks.")

    return summary


def validate_seed(root: Path, validity_dir: Path, args: argparse.Namespace, checks: list[dict[str, Any]]) -> dict[str, Any]:
    for name in SEED_OUTPUTS:
        path = validity_dir / name
        add_check(checks, f"seed_file_exists:{name}", path.exists(), rel(root, path), size_bytes=path.stat().st_size if path.exists() else 0)
        if path.suffix.lower() == ".csv" and path.exists():
            rows = count_csv_rows(path)
            add_check(checks, f"seed_csv_has_rows:{name}", rows > 0, f"rows={rows}", rows=rows)

    summary_path = validity_dir / "seed_sensitivity_summary.json"
    if summary_path.exists():
        try:
            seed_status = str(read_json(summary_path).get("status", "")).upper()
            if seed_status:
                add_check(checks, "seed_summary_status_PASS_or_unspecified", seed_status in {"PASS", ""}, f"status={seed_status}")
        except Exception as exc:
            add_check(checks, "seed_summary_parseable", False, str(exc))

    rate, source = find_seed_preserved_rate(validity_dir)
    ok = rate is not None and rate + args.rate_tolerance >= args.min_seed_preserved_rate
    add_check(
        checks,
        f"seed_trend_preserved_rate_ge_{args.min_seed_preserved_rate}",
        ok,
        f"rate={rate}, source={source}",
        value=rate,
        source=source,
    )
    return {"trend_direction_preserved_rate": rate, "source": source}


def check_summary_workbooks(root: Path, summary_dir: Path, checks: list[dict[str, Any]]) -> None:
    add_check(checks, "summary_workbook_dir_exists", summary_dir.exists(), rel(root, summary_dir))
    for dataset in DATASETS:
        path = summary_dir / f"{dataset}_summary.xlsx"
        add_check(checks, f"summary_workbook_exists:{dataset}", path.exists(), rel(root, path), size_bytes=path.stat().st_size if path.exists() else 0)


def find_unwanted(validity_dir: Path, patterns: Iterable[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(p).replace("\\", "/") for p in validity_dir.glob(pattern))
    return sorted(set(found))


def apply_policy(
    root: Path,
    name: str,
    found: list[str],
    policy: str,
    checks: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    rel_found = [rel(root, Path(p)) for p in found]
    if policy == "ignore":
        return
    if found:
        message = f"{name} present: {', '.join(rel_found)}"
        if policy == "warn":
            add_warning(warnings, message)
            add_check(checks, f"{name}_not_present", True, "WARN_ONLY: " + message, found=rel_found)
        else:
            add_check(checks, f"{name}_not_present", False, message, found=rel_found)
    else:
        add_check(checks, f"{name}_not_present", True, "none found")


def write_combined_summary(
    root: Path,
    validity_dir: Path,
    status: str,
    alpha_summary: dict[str, Any],
    seed_summary: dict[str, Any],
    warnings: list[str],
    failures: list[str],
) -> None:
    alpha_claim = {
        "base_alpha": alpha_summary.get("base_alpha"),
        "tested_alphas": alpha_summary.get("alphas"),
        "min_combo_spearman": alpha_summary.get("min_combo_spearman"),
        "min_cleaner_spearman": alpha_summary.get("min_cleaner_spearman"),
        "min_clusterer_spearman": alpha_summary.get("min_clusterer_spearman"),
        "top_combo_stability_rate": alpha_summary.get("top_combo_stability_rate"),
    }
    seed_claim = {
        "trend_direction_preserved_rate": seed_summary.get("trend_direction_preserved_rate"),
        "source": seed_summary.get("source"),
    }
    payload = {
        "generated_at_utc": utc_now(),
        "status": status,
        "scope": "pre-experiment calibration plus alpha/seed validity-sensitivity checks",
        "paper_alignment": {
            "pre_experimental_calibration": "alpha=0.47 replayed from data/pre_experiment/alpha_metrics.csv",
            "alpha_sensitivity": alpha_claim,
            "seed_sensitivity": seed_claim,
            "excluded_from_reviewer_summary": "error-model/type sensitivity is intentionally not included because it is not needed for the paper-level validity statement.",
        },
        "outputs": {
            "alpha_sensitivity_summary": rel(root, validity_dir / "alpha_sensitivity_summary.json"),
            "seed_sensitivity_summary": rel(root, validity_dir / "seed_sensitivity_summary.json"),
            "validity_sensitivity_summary_json": rel(root, validity_dir / "validity_sensitivity_summary.json"),
            "validity_sensitivity_summary_md": rel(root, validity_dir / "validity_sensitivity_summary.md"),
        },
        "warnings": warnings,
        "failures": failures,
    }
    write_json(validity_dir / "validity_sensitivity_summary.json", payload)

    lines = [
        "# TRACE Validity Sensitivity Summary",
        "",
        f"- Generated at UTC: {payload['generated_at_utc']}",
        f"- Status: {status}",
        "- Scope: pre-experiment calibration, alpha sensitivity, and error-injection seed sensitivity.",
        "- Error-model/type sensitivity is intentionally excluded from this reviewer-facing summary.",
        "",
        "## Paper-aligned checks",
        "",
        "| Claim support item | Value | Source |",
        "|---|---:|---|",
        f"| Base alpha | {alpha_claim.get('base_alpha')} | `data/pre_experiment/alpha_metrics.csv` + `alpha_sensitivity_summary.json` |",
        f"| Min combo Spearman | {alpha_claim.get('min_combo_spearman')} | `alpha_sensitivity_summary.json` |",
        f"| Min cleaner Spearman | {alpha_claim.get('min_cleaner_spearman')} | `alpha_sensitivity_summary.json` |",
        f"| Min clusterer Spearman | {alpha_claim.get('min_clusterer_spearman')} | `alpha_sensitivity_summary.json` |",
        f"| Top-combo stability rate | {alpha_claim.get('top_combo_stability_rate')} | `alpha_sensitivity_summary.json` |",
        f"| Seed trend-direction preserved rate | {seed_claim.get('trend_direction_preserved_rate')} | {seed_claim.get('source')} |",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("No warnings.")
    lines += ["", "## Failures", ""]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("No failures.")
    (validity_dir / "validity_sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# TRACE Pre-experiment and Validity Validation Report",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Status: {report['status']}",
        "",
        "## Commands",
        "",
        "| Step | Status | Return code |",
        "|---|---|---:|",
    ]
    for command in report["commands"]:
        lines.append(f"| {command['name']} | {command['status']} | {command['returncode']} |")
    lines += ["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"]
    for check in report["checks"]:
        detail = str(check.get("detail", "")).replace("|", "\\|")
        lines.append(f"| {check['name']} | {check['status']} | {detail} |")
    lines += ["", "## Warnings", ""]
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("No warnings.")
    lines += ["", "## Failures", ""]
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("No failures.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.validate_only:
        args.skip_pre_experiment = True
        args.skip_alpha_rebuild = True

    root = args.project_root.resolve()
    pre_source_csv = to_root(root, args.pre_source_csv)
    pre_output_dir = to_root(root, args.pre_output_dir)
    pre_figure_dir = to_root(root, args.pre_figure_dir)
    summary_dir = to_root(root, args.summary_dir)
    validity_dir = to_root(root, args.validity_dir)
    output_report = to_root(root, args.output_report)

    commands: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    # Static input checks before running the replay scripts.
    add_check(checks, "project_root_exists", root.exists(), str(root))
    add_check(checks, "pre_source_csv_exists", pre_source_csv.exists(), rel(root, pre_source_csv))
    if pre_source_csv.exists():
        rows = count_csv_rows(pre_source_csv)
        add_check(checks, "pre_source_csv_has_rows", rows >= 4, f"rows={rows}", rows=rows)
    check_summary_workbooks(root, summary_dir, checks)

    if not args.skip_pre_experiment:
        commands.append(
            {
                "name": "pre_experiment_replay",
                **run_python(
                    root,
                    [
                        "scripts/38_build_pre_experiment_outputs.py",
                        "--source-csv",
                        rel(root, pre_source_csv),
                        "--output-dir",
                        rel(root, pre_output_dir),
                        "--figure-dir",
                        rel(root, pre_figure_dir),
                    ],
                ),
            }
        )

    if not args.skip_alpha_rebuild:
        alpha_args = [str(a) for a in args.alphas]
        commands.append(
            {
                "name": "alpha_sensitivity_replay",
                **run_python(
                    root,
                    [
                        "scripts/80_build_alpha_sensitivity.py",
                        "--summary-dir",
                        rel(root, summary_dir),
                        "--output-dir",
                        rel(root, validity_dir),
                        "--alphas",
                        *alpha_args,
                        "--base-alpha",
                        str(args.base_alpha),
                    ],
                ),
            }
        )

    validate_pre_experiment(root, pre_output_dir, pre_figure_dir, checks)
    alpha_summary = validate_alpha(root, validity_dir, args, checks, warnings)
    seed_summary = validate_seed(root, validity_dir, args, checks)

    unwanted_error_model = find_unwanted(validity_dir, ERROR_MODEL_PATTERNS)
    apply_policy(root, "error_model_sensitivity_files", unwanted_error_model, args.error_model_policy, checks, warnings)
    unwanted_generated = find_unwanted(validity_dir, GENERATED_DATA_PATTERNS)
    apply_policy(root, "generated_sensitivity_data_dirs", unwanted_generated, args.generated_data_policy, checks, warnings)

    for command in commands:
        if command["status"] != "PASS":
            add_check(checks, f"command_passed:{command['name']}", False, command["command"])
        else:
            add_check(checks, f"command_passed:{command['name']}", True, command["command"])

    failures = [check["name"] + (f": {check.get('detail', '')}" if check.get("detail") else "") for check in checks if not check.get("ok")]
    status = "FAIL" if failures else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    write_combined_summary(root, validity_dir, status, alpha_summary, seed_summary, warnings, failures)

    report = {
        "generated_at_utc": utc_now(),
        "status": status,
        "project_root": str(root),
        "commands": commands,
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
        "outputs": {
            "combined_json": rel(root, output_report),
            "combined_markdown": rel(root, output_report.with_suffix(".md")),
            "validity_summary_json": rel(root, validity_dir / "validity_sensitivity_summary.json"),
            "validity_summary_md": rel(root, validity_dir / "validity_sensitivity_summary.md"),
        },
    }
    write_json(output_report, report)
    write_markdown_report(output_report.with_suffix(".md"), report)

    print(
        json.dumps(
            {
                "status": status,
                "failure_count": len(failures),
                "warning_count": len(warnings),
                "report": rel(root, output_report),
                "validity_summary": rel(root, validity_dir / "validity_sensitivity_summary.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[TRACE] Pre-experiment/validity report written to: {output_report}")
    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()


