#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast evaluation of TRACE against blind randomized exhaustive-search schedules.

This is a post-hoc evaluation layer. It reads Stage-4 TRACE replay CSVs and
constructs blind baselines by randomly permuting cleaner×clusterer paths while
preserving trial order within each path. Compared with the first implementation,
this version precomputes path-level curves and prints progress, which makes it
suitable for full 60-dataset replay outputs.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

NEG_INF = float("-inf")


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


@dataclass
class PathCurve:
    cleaner: str
    clusterer: str
    rows_count: int
    scores: list[float]
    prefix_best_raw: list[float]
    prefix_curve_nonneg: list[float]
    prefix_curve_sum: list[float]
    path_max: float
    path_max_nonneg: float
    first_hit_step_for_target: int | None = None

    @classmethod
    def from_rows(cls, cleaner: str, clusterer: str, rows: list[dict[str, str]]) -> "PathCurve":
        scores: list[float] = []
        best = NEG_INF
        prefix_best_raw: list[float] = []
        prefix_curve_nonneg: list[float] = []
        running_sum = 0.0
        prefix_curve_sum = [0.0]
        for row in sort_by_step(rows):
            score = row_score(row)
            if score is None:
                scores.append(NEG_INF)
            else:
                scores.append(float(score))
                if score > best:
                    best = float(score)
            prefix_best_raw.append(best)
            curve_value = 0.0 if not math.isfinite(best) else max(0.0, best)
            prefix_curve_nonneg.append(curve_value)
            running_sum += curve_value
            prefix_curve_sum.append(running_sum)
        path_max = max(scores) if scores else NEG_INF
        path_max_nonneg = 0.0 if not math.isfinite(path_max) else max(0.0, path_max)
        return cls(
            cleaner=cleaner,
            clusterer=clusterer,
            rows_count=len(scores),
            scores=scores,
            prefix_best_raw=prefix_best_raw,
            prefix_curve_nonneg=prefix_curve_nonneg,
            prefix_curve_sum=prefix_curve_sum,
            path_max=path_max,
            path_max_nonneg=path_max_nonneg,
        )

    def set_target(self, target: float) -> None:
        self.first_hit_step_for_target = None
        for i, v in enumerate(self.prefix_best_raw, start=1):
            if math.isfinite(v) and v >= target:
                self.first_hit_step_for_target = i
                return

    def contribution_given_best(self, current_best: float) -> float:
        if self.rows_count <= 0:
            return 0.0
        c = 0.0 if not math.isfinite(current_best) else max(0.0, current_best)
        # prefix_curve_nonneg is non-decreasing because it is a cumulative best curve.
        idx = bisect.bisect_right(self.prefix_curve_nonneg, c)
        return c * idx + (self.prefix_curve_sum[-1] - self.prefix_curve_sum[idx])

    def best_at_prefix(self, prefix_len: int) -> float:
        if prefix_len <= 0 or not self.prefix_best_raw:
            return NEG_INF
        prefix_len = min(prefix_len, self.rows_count)
        return self.prefix_best_raw[prefix_len - 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TRACE with blind random path-order baselines.")
    parser.add_argument("--trace-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-seeds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--budgets", type=float, nargs="*", default=[0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--path-granularity", choices=["path", "trial"], default="path")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N datasets. Default: 1.")
    parser.add_argument("--flush", action="store_true", help="Flush progress output immediately.")
    return parser.parse_args()


def log(msg: str, *, flush: bool = True) -> None:
    print(msg, flush=flush)


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
        out[dataset_id] = DatasetSummary(
            dataset_id=dataset_id,
            dataset_name=str(row.get("dataset_name", "")),
            h_full=to_float(row.get("h_full")) or 0.0,
            full_trial_count=to_int(row.get("full_trial_count"), 0),
            trace_trials_consumed=to_int(row.get("trace_trials_consumed"), 0),
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
        return None
    return score


def sort_by_step(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: (to_int(r.get("global_step"), 0), to_int(r.get("path_step"), 0), to_int(r.get("trial_number"), 0)))


def group_baseline_paths(rows: list[dict[str, str]]) -> dict[int, list[PathCurve]]:
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset_id = to_int(row.get("dataset_id"), -1)
        if dataset_id < 0:
            continue
        cleaner = str(row.get("cleaner", ""))
        clusterer = str(row.get("clusterer", ""))
        grouped[(dataset_id, cleaner, clusterer)].append(row)

    by_dataset: dict[int, list[PathCurve]] = defaultdict(list)
    for (dataset_id, cleaner, clusterer), path_rows in grouped.items():
        curve = PathCurve.from_rows(cleaner, clusterer, path_rows)
        by_dataset[dataset_id].append(curve)

    # Preserve original implementation order as deterministic base order.
    for dataset_id, paths in by_dataset.items():
        paths.sort(key=lambda p: min((to_int(r.get("global_step"), 0) for r in grouped[(dataset_id, p.cleaner, p.clusterer)]), default=10**12))
    return by_dataset


def group_trace_scores(rows: list[dict[str, str]]) -> dict[int, list[float]]:
    by_dataset_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset_id = to_int(row.get("dataset_id"), -1)
        if dataset_id >= 0:
            by_dataset_rows[dataset_id].append(row)
    out: dict[int, list[float]] = {}
    for dataset_id, dataset_rows in by_dataset_rows.items():
        scores: list[float] = []
        for row in sort_by_step(dataset_rows):
            score = row_score(row)
            scores.append(NEG_INF if score is None else float(score))
        out[dataset_id] = scores
    return out


def curve_metrics_from_scores(
    scores: list[float],
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
    best = NEG_INF
    hit_step: int | None = None
    best_by_step: list[float] = []
    n_scores = len(scores)
    for step in range(1, full_trial_count + 1):
        if step <= n_scores:
            score = scores[step - 1]
            if math.isfinite(score) and score > best:
                best = score
        if hit_step is None and best >= target:
            hit_step = step
        best_by_step.append(0.0 if not math.isfinite(best) else best)

    def retention_at_step(step: int) -> float:
        if step <= 0:
            return 0.0
        step = min(step, full_trial_count)
        return best_by_step[step - 1] / h_full if h_full > 0 else float("nan")

    fixed_retention = {b: retention_at_step(max(1, int(math.ceil(b * full_trial_count)))) for b in fixed_budgets}
    budget_step = max(1, min(trace_budget, full_trial_count)) if trace_budget > 0 else 0
    return {
        "hit95_step": hit_step,
        "hit95_progress": (hit_step / full_trial_count) if hit_step is not None else 1.0,
        "retention_at_trace_budget": retention_at_step(budget_step) if budget_step > 0 else 0.0,
        "auc_retention": sum((v / h_full) for v in best_by_step) / full_trial_count,
        "fixed_retention": fixed_retention,
    }


def schedule_metrics_path_level(
    paths: list[PathCurve],
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
    for p in paths:
        p.set_target(target)

    current_best = NEG_INF
    step_sofar = 0
    hit_step: int | None = None
    auc_sum = 0.0

    # Budget points include trace budget and fixed budget fractions.
    budget_steps: dict[str, int] = {"trace": max(1, min(trace_budget, full_trial_count)) if trace_budget > 0 else 0}
    for b in fixed_budgets:
        budget_steps[f"fixed:{b}"] = max(1, int(math.ceil(b * full_trial_count)))
    budget_values: dict[str, float] = {}
    sorted_budget_items = sorted((step, key) for key, step in budget_steps.items() if step > 0)
    budget_idx = 0

    def current_best_nonneg() -> float:
        return 0.0 if not math.isfinite(current_best) else max(0.0, current_best)

    for path in paths:
        if step_sofar >= full_trial_count:
            break
        path_len = min(path.rows_count, full_trial_count - step_sofar)
        if path_len <= 0:
            continue

        # Fill budget points that fall inside this path.
        while budget_idx < len(sorted_budget_items) and sorted_budget_items[budget_idx][0] <= step_sofar + path_len:
            budget_step, key = sorted_budget_items[budget_idx]
            inside = budget_step - step_sofar
            local_best = path.best_at_prefix(inside)
            best_here = max(current_best, local_best)
            budget_values[key] = 0.0 if not math.isfinite(best_here) else best_here
            budget_idx += 1

        if (
            hit_step is None
            and current_best < target
            and path.first_hit_step_for_target is not None
            and path.first_hit_step_for_target <= path_len
        ):
            # If the visible prefix of the path crosses the target and no earlier path did, this is the hit.
            hit_step = step_sofar + path.first_hit_step_for_target

        # AUC contribution of this path given best-so-far before the path.
        if path_len == path.rows_count:
            auc_sum += path.contribution_given_best(current_best)
        else:
            # Rare case: full_trial_count cuts through the final path. Fall back to small loop.
            for i in range(path_len):
                local_best = path.prefix_best_raw[i]
                best_here = max(current_best, local_best)
                auc_sum += 0.0 if not math.isfinite(best_here) else best_here

        visible_path_max = path.path_max if path_len == path.rows_count else path.best_at_prefix(path_len)
        if visible_path_max > current_best:
            current_best = visible_path_max
        step_sofar += path_len

    # If successful rows are shorter than full_trial_count, hold best-so-far constant.
    if step_sofar < full_trial_count:
        c = current_best_nonneg()
        remaining = full_trial_count - step_sofar
        auc_sum += c * remaining
        while budget_idx < len(sorted_budget_items):
            _budget_step, key = sorted_budget_items[budget_idx]
            budget_values[key] = c
            budget_idx += 1

    # If budget point occurs before any row, its value remains zero.
    for _step, key in sorted_budget_items:
        budget_values.setdefault(key, 0.0)

    fixed_retention = {b: (budget_values.get(f"fixed:{b}", 0.0) / h_full) for b in fixed_budgets}
    trace_budget_value = budget_values.get("trace", 0.0)
    return {
        "hit95_step": hit_step,
        "hit95_progress": (hit_step / full_trial_count) if hit_step is not None else 1.0,
        "retention_at_trace_budget": trace_budget_value / h_full,
        "auc_retention": auc_sum / h_full / full_trial_count,
        "fixed_retention": fixed_retention,
    }


def schedule_metrics_trial_level(
    scores: list[float],
    *,
    h_full: float,
    full_trial_count: int,
    threshold: float,
    trace_budget: int,
    fixed_budgets: list[float],
) -> dict[str, Any]:
    # Kept for stress tests; path-level is recommended and much faster.
    return curve_metrics_from_scores(scores, h_full=h_full, full_trial_count=full_trial_count, threshold=threshold, trace_budget=trace_budget, fixed_budgets=fixed_budgets)


def evaluate_dataset(
    summary: DatasetSummary,
    baseline_paths: list[PathCurve],
    trace_scores: list[float],
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

    trace_curve = curve_metrics_from_scores(
        trace_scores,
        h_full=h_full,
        full_trial_count=full_trial_count,
        threshold=threshold,
        trace_budget=trace_budget,
        fixed_budgets=fixed_budgets,
    )

    rng = random.Random(base_seed + summary.dataset_id * 1000003)
    random_metrics: list[dict[str, Any]] = []
    all_scores = [s for p in baseline_paths for s in p.scores]

    for _seed_idx in range(random_seeds):
        if path_granularity == "path":
            permuted_paths = list(baseline_paths)
            rng.shuffle(permuted_paths)
            random_metrics.append(
                schedule_metrics_path_level(
                    permuted_paths,
                    h_full=h_full,
                    full_trial_count=full_trial_count,
                    threshold=threshold,
                    trace_budget=trace_budget,
                    fixed_budgets=fixed_budgets,
                )
            )
        else:
            schedule_scores = list(all_scores)
            rng.shuffle(schedule_scores)
            random_metrics.append(
                schedule_metrics_trial_level(
                    schedule_scores,
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

    trace_hit = summary.trace_hit95_progress if summary.trace_hit95_progress is not None else trace_curve["hit95_progress"]
    trace_ret = summary.trace_score_retention if summary.trace_score_retention is not None else trace_curve["retention_at_trace_budget"]

    fixed_rows: list[dict[str, Any]] = []
    for b in fixed_budgets:
        rand_fixed_clean = [float(m["fixed_retention"].get(b)) for m in random_metrics if m["fixed_retention"].get(b) is not None and math.isfinite(float(m["fixed_retention"].get(b)))]
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
                    sum(1 for x in rand_fixed_clean if trace_fixed is not None and trace_fixed >= x) / len(rand_fixed_clean)
                    if rand_fixed_clean else None
                ),
            }
        )

    trace_hit_better_share = None
    if trace_hit is not None and hit_values:
        trace_hit_better_share = sum(1 for x in hit_values if trace_hit <= x) / len(hit_values)

    trace_ret_better_share = None
    if trace_ret is not None and ret_values:
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
    start_time = time.time()
    trace_dir = args.trace_output_dir.resolve()
    output_dir = (args.output_dir or (trace_dir / "blind_random")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = trace_dir / "trace_dataset_summary.csv"
    baseline_path = trace_dir / "trace_baseline_sequence.csv"
    trace_trials_path = trace_dir / "trace_replay_trials.csv"
    for path in [summary_path, baseline_path, trace_trials_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    log(f"[TRACE] Loading summaries: {summary_path}")
    summaries = load_dataset_summaries(summary_path)
    log(f"[TRACE] Loading baseline sequence: {baseline_path}")
    baseline_paths = group_baseline_paths(read_csv_dicts(baseline_path))
    log(f"[TRACE] Loading TRACE trials: {trace_trials_path}")
    trace_scores = group_trace_scores(read_csv_dicts(trace_trials_path))
    log(f"[TRACE] Evaluating {len(summaries)} datasets with {int(args.random_seeds)} blind schedules each")

    dataset_rows: list[dict[str, Any]] = []
    fixed_rows_all: list[dict[str, Any]] = []
    dataset_items = sorted(summaries.items())
    for idx, (dataset_id, summary) in enumerate(dataset_items, start=1):
        paths = baseline_paths.get(dataset_id, [])
        if not paths:
            log(f"[WARN] No baseline paths for dataset_id={dataset_id}; skip.")
            continue
        if idx == 1 or idx % max(1, int(args.progress_every)) == 0 or idx == len(dataset_items):
            log(
                f"[TRACE] Blind-random dataset {idx}/{len(dataset_items)}: "
                f"dataset_id={dataset_id}, name={summary.dataset_name}, paths={len(paths)}, "
                f"full_trials={summary.full_trial_count}"
            )
        row, fixed_rows = evaluate_dataset(
            summary,
            paths,
            trace_scores.get(dataset_id, []),
            random_seeds=max(1, int(args.random_seeds)),
            base_seed=int(args.seed),
            threshold=float(args.threshold),
            fixed_budgets=[float(b) for b in args.budgets],
            path_granularity=args.path_granularity,
        )
        dataset_rows.append(row)
        fixed_rows_all.extend(fixed_rows)

    dataset_columns = [
        "dataset_id", "dataset_name", "h_full", "full_trial_count", "trace_trials_consumed", "trace_budget_fraction",
        "trace_hit95_progress", "trace_score_retention", "trace_auc_retention",
        "original_baseline_hit95_progress", "original_baseline_retention_at_trace_budget",
        "blind_random_hit95_progress_median", "blind_random_hit95_progress_p10", "blind_random_hit95_progress_p90",
        "blind_random_retention_at_trace_budget_median", "blind_random_retention_at_trace_budget_p10", "blind_random_retention_at_trace_budget_p90",
        "blind_random_auc_retention_median", "blind_random_auc_retention_p10", "blind_random_auc_retention_p90",
        "trace_hit95_not_worse_than_random_share", "trace_retention_not_worse_than_random_share",
        "random_seeds", "path_granularity",
    ]
    write_csv(output_dir / "trace_blind_random_dataset_summary.csv", dataset_rows, dataset_columns)

    fixed_columns = [
        "dataset_id", "dataset_name", "budget_fraction", "trace_retention",
        "blind_random_retention_median", "blind_random_retention_p10", "blind_random_retention_p90",
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
        "runtime_seconds": time.time() - start_time,
    }
    with (output_dir / "trace_blind_random_aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    log(json.dumps(aggregate, ensure_ascii=False, indent=2))
    log(f"[TRACE] Blind-random evaluation written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
