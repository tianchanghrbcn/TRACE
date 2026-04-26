#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate TRACE against a blind randomized exhaustive-search schedule.

This script is a post-hoc evaluation layer. It does not rerun cleaning or
clustering. It reads the trial ledger emitted by scripts/30_replay_trace.py and
constructs a distribution of blind baselines by randomly permuting the order of
cleaner×clusterer paths. Within each path, the saved trial order is preserved.

Why path-level shuffling?
-------------------------
The raw baseline sequence is often determined by implementation/job order. That
order is not a meaningful operational baseline for an early-stop policy. A blind
searcher would not know which cleaner×clusterer pair is promising in advance,
so a path-level permutation is a more honest comparison for the question:
"does TRACE provide useful search guidance beyond unguided method enumeration?"

Inputs expected in --trace-output-dir:
  - trace_dataset_summary.csv
  - trace_baseline_sequence.csv
  - trace_replay_trials.csv

Outputs:
  - trace_blind_random_dataset_summary.csv
  - trace_blind_random_fixed_budget.csv
  - trace_blind_random_aggregate_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


@dataclass
class DatasetSummary:
    dataset_id: int
    dataset_name: str
    h_full: float
    full_trial_count: int
    trace_trials_consumed: int
    trace_hit95_progress: float | None
    trace_score_retention: float | None
    baseline_hit95_progress: float | None
    baseline_retention_at_trace_budget: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TRACE with blind random path-order baselines."
    )
    parser.add_argument(
        "--trace-output-dir",
        type=Path,
        required=True,
        help="Directory containing Stage-4 TRACE replay CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <trace-output-dir>/blind_random.",
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=500,
        help="Number of random blind schedules per dataset. Default: 500.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260424,
        help="Base random seed. Default: 20260424.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Hit threshold as a fraction of H_full. Default: 0.95.",
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="*",
        default=[0.01, 0.02, 0.05, 0.10, 0.20],
        help="Fixed budget fractions for retention curves.",
    )
    parser.add_argument(
        "--path-granularity",
        choices=["path", "trial"],
        default="path",
        help=(
            "Randomization granularity. 'path' preserves trial order inside each "
            "cleaner×clusterer path and is the recommended blind-search baseline. "
            "'trial' fully shuffles individual trials as an additional stress test."
        ),
    )
    return parser.parse_args()


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    return val


def to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def quantile(values: list[float], q: float) -> float | None:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1 - frac) + clean[hi] * frac


def safe_median(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(median(clean))


def load_dataset_summaries(path: Path) -> dict[int, DatasetSummary]:
    rows = read_csv_dicts(path)
    out: dict[int, DatasetSummary] = {}
    for row in rows:
        dataset_id = to_int(row.get("dataset_id"), -1)
        if dataset_id < 0:
            continue
        h_full = to_float(row.get("h_full")) or 0.0
        full_trial_count = to_int(row.get("full_trial_count"), 0)
        trace_trials_consumed = to_int(row.get("trace_trials_consumed"), 0)
        out[dataset_id] = DatasetSummary(
            dataset_id=dataset_id,
            dataset_name=str(row.get("dataset_name", "")),
            h_full=h_full,
            full_trial_count=full_trial_count,
            trace_trials_consumed=trace_trials_consumed,
            trace_hit95_progress=to_float(row.get("trace_hit95_progress")),
            trace_score_retention=to_float(row.get("trace_score_retention")),
            baseline_hit95_progress=to_float(row.get("baseline_hit95_progress")),
            baseline_retention_at_trace_budget=to_float(row.get("baseline_retention_at_trace_budget")),
        )
    return out


def row_score(row: dict[str, str]) -> float | None:
    score = to_float(row.get("score"))
    if score is None:
        return None
    valid = str(row.get("valid_trial", "")).strip().lower()
    if valid in {"false", "0", "no"}:
        return None
    status = str(row.get("trial_status", "")).strip().lower()
    if status and status not in {"ok", "complete", "completed", "valid", "success"}:
        # Empty status is tolerated because older ledgers may omit it.
        return None
    return score


def sort_by_step(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: (to_int(r.get("global_step"), 0), to_int(r.get("path_step"), 0), to_int(r.get("trial_number"), 0)))


def group_baseline_paths(rows: list[dict[str, str]]) -> dict[int, list[list[dict[str, str]]]]:
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset_id = to_int(row.get("dataset_id"), -1)
        if dataset_id < 0:
            continue
        key = (dataset_id, str(row.get("cleaner", "")), str(row.get("clusterer", "")))
        grouped[key].append(row)
    by_dataset: dict[int, list[list[dict[str, str]]]] = defaultdict(list)
    for (dataset_id, _cleaner, _clusterer), path_rows in grouped.items():
        by_dataset[dataset_id].append(sort_by_step(path_rows))
    # Preserve the original implementation order as the deterministic base order.
    for dataset_id, paths in by_dataset.items():
        paths.sort(key=lambda p: to_int(p[0].get("global_step"), 0) if p else 10**12)
    return by_dataset


def group_trace_rows(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by_dataset: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset_id = to_int(row.get("dataset_id"), -1)
        if dataset_id >= 0:
            by_dataset[dataset_id].append(row)
    return {dataset_id: sort_by_step(rows) for dataset_id, rows in by_dataset.items()}


def flatten_paths(paths: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for path_rows in paths:
        out.extend(path_rows)
    return out


def curve_metrics(
    rows: list[dict[str, str]],
    *,
    h_full: float,
    full_trial_count: int,
    threshold: float,
    trace_budget: int,
    fixed_budgets: list[float],
) -> dict[str, Any]:
    if h_full <= 0 or full_trial_count <= 0:
        return {
            "hit95_step": None,
            "hit95_progress": None,
            "retention_at_trace_budget": None,
            "auc_retention": None,
            "fixed_retention": {b: None for b in fixed_budgets},
        }

    target = threshold * h_full
    best = -math.inf
    hit_step: int | None = None
    best_by_step: list[float] = []

    # Use full_trial_count as the denominator. If rows are shorter than the full
    # budget because failed paths had no detailed rows, the best-so-far is held
    # constant for the remaining budget.
    for step in range(1, full_trial_count + 1):
        if step <= len(rows):
            score = row_score(rows[step - 1])
            if score is not None and score > best:
                best = score
        if hit_step is None and best >= target:
            hit_step = step
        best_by_step.append(0.0 if not math.isfinite(best) else best)

    def retention_at_step(step: int) -> float:
        if step <= 0:
            return 0.0
        step = min(step, full_trial_count)
        value = best_by_step[step - 1] if best_by_step else 0.0
        return value / h_full if h_full > 0 else float("nan")

    fixed_retention = {
        b: retention_at_step(max(1, int(math.ceil(b * full_trial_count))))
        for b in fixed_budgets
    }
    budget_step = max(1, min(trace_budget, full_trial_count)) if trace_budget > 0 else 0
    ret_trace_budget = retention_at_step(budget_step) if budget_step > 0 else 0.0
    auc = sum((v / h_full) for v in best_by_step) / full_trial_count

    return {
        "hit95_step": hit_step,
        "hit95_progress": (hit_step / full_trial_count) if hit_step is not None else 1.0,
        "retention_at_trace_budget": ret_trace_budget,
        "auc_retention": auc,
        "fixed_retention": fixed_retention,
    }


def evaluate_dataset(
    summary: DatasetSummary,
    baseline_paths: list[list[dict[str, str]]],
    trace_rows: list[dict[str, str]],
    *,
    random_seeds: int,
    base_seed: int,
    threshold: float,
    fixed_budgets: list[float],
    path_granularity: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    h_full = summary.h_full
    full_trial_count = summary.full_trial_count
    trace_budget = summary.trace_trials_consumed

    trace_curve = curve_metrics(
        trace_rows,
        h_full=h_full,
        full_trial_count=full_trial_count,
        threshold=threshold,
        trace_budget=trace_budget,
        fixed_budgets=fixed_budgets,
    )

    rng = random.Random(base_seed + summary.dataset_id * 1000003)
    random_metrics: list[dict[str, Any]] = []

    all_rows = flatten_paths(baseline_paths)
    for seed_idx in range(random_seeds):
        if path_granularity == "path":
            permuted_paths = list(baseline_paths)
            rng.shuffle(permuted_paths)
            schedule = flatten_paths(permuted_paths)
        else:
            schedule = list(all_rows)
            rng.shuffle(schedule)
        random_metrics.append(
            curve_metrics(
                schedule,
                h_full=h_full,
                full_trial_count=full_trial_count,
                threshold=threshold,
                trace_budget=trace_budget,
                fixed_budgets=fixed_budgets,
            )
        )

    hit_values = [m["hit95_progress"] for m in random_metrics if m["hit95_progress"] is not None]
    ret_values = [m["retention_at_trace_budget"] for m in random_metrics if m["retention_at_trace_budget"] is not None]
    auc_values = [m["auc_retention"] for m in random_metrics if m["auc_retention"] is not None]

    trace_hit = summary.trace_hit95_progress
    if trace_hit is None:
        trace_hit = trace_curve["hit95_progress"]
    trace_ret = summary.trace_score_retention
    if trace_ret is None:
        trace_ret = trace_curve["retention_at_trace_budget"]

    fixed_rows: list[dict[str, Any]] = []
    for b in fixed_budgets:
        rand_fixed = [m["fixed_retention"].get(b) for m in random_metrics]
        rand_fixed_clean = [float(x) for x in rand_fixed if x is not None and math.isfinite(float(x))]
        trace_fixed = trace_curve["fixed_retention"].get(b)
        fixed_rows.append(
            {
                "dataset_id": summary.dataset_id,
                "dataset_name": summary.dataset_name,
                "budget_fraction": b,
                "trace_retention": trace_fixed,
                "blind_random_retention_median": safe_median(rand_fixed_clean),
                "blind_random_retention_p10": quantile(rand_fixed_clean, 0.10),
                "blind_random_retention_p90": quantile(rand_fixed_clean, 0.90),
                "trace_better_than_random_share": (
                    sum(1 for x in rand_fixed_clean if trace_fixed is not None and trace_fixed >= x)
                    / len(rand_fixed_clean)
                    if rand_fixed_clean
                    else None
                ),
            }
        )

    trace_hit_better_share = None
    if trace_hit is not None and hit_values:
        # Smaller hit95 progress is better. Treat equality as not worse.
        trace_hit_better_share = sum(1 for x in hit_values if trace_hit <= x) / len(hit_values)

    trace_ret_better_share = None
    if trace_ret is not None and ret_values:
        # Larger retention is better.
        trace_ret_better_share = sum(1 for x in ret_values if trace_ret >= x) / len(ret_values)

    row = {
        "dataset_id": summary.dataset_id,
        "dataset_name": summary.dataset_name,
        "h_full": h_full,
        "full_trial_count": full_trial_count,
        "trace_trials_consumed": trace_budget,
        "trace_budget_fraction": (trace_budget / full_trial_count) if full_trial_count else None,
        "trace_hit95_progress": trace_hit,
        "trace_score_retention": trace_ret,
        "trace_auc_retention": trace_curve["auc_retention"],
        "original_baseline_hit95_progress": summary.baseline_hit95_progress,
        "original_baseline_retention_at_trace_budget": summary.baseline_retention_at_trace_budget,
        "blind_random_hit95_progress_median": safe_median(hit_values),
        "blind_random_hit95_progress_p10": quantile(hit_values, 0.10),
        "blind_random_hit95_progress_p90": quantile(hit_values, 0.90),
        "blind_random_retention_at_trace_budget_median": safe_median(ret_values),
        "blind_random_retention_at_trace_budget_p10": quantile(ret_values, 0.10),
        "blind_random_retention_at_trace_budget_p90": quantile(ret_values, 0.90),
        "blind_random_auc_retention_median": safe_median(auc_values),
        "blind_random_auc_retention_p10": quantile(auc_values, 0.10),
        "blind_random_auc_retention_p90": quantile(auc_values, 0.90),
        "trace_hit95_not_worse_than_random_share": trace_hit_better_share,
        "trace_retention_not_worse_than_random_share": trace_ret_better_share,
        "random_seeds": random_seeds,
        "path_granularity": path_granularity,
    }
    return row, fixed_rows


def main() -> int:
    args = parse_args()
    trace_dir = args.trace_output_dir.resolve()
    output_dir = (args.output_dir or (trace_dir / "blind_random")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = trace_dir / "trace_dataset_summary.csv"
    baseline_path = trace_dir / "trace_baseline_sequence.csv"
    trace_trials_path = trace_dir / "trace_replay_trials.csv"

    for path in [summary_path, baseline_path, trace_trials_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    summaries = load_dataset_summaries(summary_path)
    baseline_paths = group_baseline_paths(read_csv_dicts(baseline_path))
    trace_rows = group_trace_rows(read_csv_dicts(trace_trials_path))

    dataset_rows: list[dict[str, Any]] = []
    fixed_rows_all: list[dict[str, Any]] = []

    for dataset_id, summary in sorted(summaries.items()):
        paths = baseline_paths.get(dataset_id, [])
        if not paths:
            print(f"[WARN] No baseline paths for dataset_id={dataset_id}; skip.")
            continue
        row, fixed_rows = evaluate_dataset(
            summary,
            paths,
            trace_rows.get(dataset_id, []),
            random_seeds=max(1, int(args.random_seeds)),
            base_seed=int(args.seed),
            threshold=float(args.threshold),
            fixed_budgets=[float(b) for b in args.budgets],
            path_granularity=args.path_granularity,
        )
        dataset_rows.append(row)
        fixed_rows_all.extend(fixed_rows)

    dataset_columns = [
        "dataset_id",
        "dataset_name",
        "h_full",
        "full_trial_count",
        "trace_trials_consumed",
        "trace_budget_fraction",
        "trace_hit95_progress",
        "trace_score_retention",
        "trace_auc_retention",
        "original_baseline_hit95_progress",
        "original_baseline_retention_at_trace_budget",
        "blind_random_hit95_progress_median",
        "blind_random_hit95_progress_p10",
        "blind_random_hit95_progress_p90",
        "blind_random_retention_at_trace_budget_median",
        "blind_random_retention_at_trace_budget_p10",
        "blind_random_retention_at_trace_budget_p90",
        "blind_random_auc_retention_median",
        "blind_random_auc_retention_p10",
        "blind_random_auc_retention_p90",
        "trace_hit95_not_worse_than_random_share",
        "trace_retention_not_worse_than_random_share",
        "random_seeds",
        "path_granularity",
    ]
    write_csv(output_dir / "trace_blind_random_dataset_summary.csv", dataset_rows, dataset_columns)

    fixed_columns = [
        "dataset_id",
        "dataset_name",
        "budget_fraction",
        "trace_retention",
        "blind_random_retention_median",
        "blind_random_retention_p10",
        "blind_random_retention_p90",
        "trace_better_than_random_share",
    ]
    write_csv(output_dir / "trace_blind_random_fixed_budget.csv", fixed_rows_all, fixed_columns)

    aggregate = {
        "trace_output_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "n_datasets": len(dataset_rows),
        "random_seeds": int(args.random_seeds),
        "base_seed": int(args.seed),
        "threshold": float(args.threshold),
        "path_granularity": args.path_granularity,
        "median_trace_hit95_progress": safe_median(r.get("trace_hit95_progress") for r in dataset_rows),
        "median_blind_random_hit95_progress": safe_median(r.get("blind_random_hit95_progress_median") for r in dataset_rows),
        "median_trace_score_retention": safe_median(r.get("trace_score_retention") for r in dataset_rows),
        "median_blind_random_retention_at_trace_budget": safe_median(r.get("blind_random_retention_at_trace_budget_median") for r in dataset_rows),
        "median_trace_auc_retention": safe_median(r.get("trace_auc_retention") for r in dataset_rows),
        "median_blind_random_auc_retention": safe_median(r.get("blind_random_auc_retention_median") for r in dataset_rows),
        "median_trace_hit95_not_worse_than_random_share": safe_median(r.get("trace_hit95_not_worse_than_random_share") for r in dataset_rows),
        "median_trace_retention_not_worse_than_random_share": safe_median(r.get("trace_retention_not_worse_than_random_share") for r in dataset_rows),
        "fixed_budgets": [float(b) for b in args.budgets],
    }
    with (output_dir / "trace_blind_random_aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"[TRACE] Blind-random evaluation written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
