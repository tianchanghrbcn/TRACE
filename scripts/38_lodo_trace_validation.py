#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leave-one-dataset-out validation for TRACE Stage 4.

This script performs a lightweight held-out validation using already generated
cached-cleaning clustering logs.  For each fold, it learns q-regime entry gates
(tau_turn, tau_hi) and cleaner priority orders from three dataset names, then
replays TRACE on the held-out dataset name.  It keeps the runtime TRACE transition
rules unchanged and only calibrates the entry ordering from the training tables.

Typical use on Linux:
    python scripts/38_lodo_trace_validation.py \
      --results-dir results/trace_cluster_replay_all \
      --base-trace-output-dir results/processed/trace/cluster_replay_all \
      --config configs/trace.yaml \
      --output-dir results/processed/trace/lodo \
      --random-seeds 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc


SKIP_CLEANERS = {"", "mode", "groundtruth", "GroundTruth"}


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        val = float(text)
    except ValueError:
        return default
    return val if math.isfinite(val) else default


def to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def safe_median(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.median(clean)) if clean else None


def regime_for(q_tot: float, tau_turn: float, tau_hi: float) -> str:
    if q_tot < tau_turn:
        return "low"
    if q_tot < tau_hi:
        return "mid"
    return "high"


def score_from_row(row: dict[str, str]) -> float | None:
    score = to_float(row.get("score"), None)
    if score is None:
        return None
    valid = str(row.get("valid_trial", "")).strip().lower()
    if valid in {"false", "0", "no"}:
        return None
    status = str(row.get("trial_status", "")).strip().lower()
    if status and status not in {"ok", "complete", "completed", "valid", "success"}:
        return None
    return score


def load_metadata(summary_rows: list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    meta: dict[int, dict[str, Any]] = {}
    for row in summary_rows:
        dataset_id = to_int(row.get("dataset_id"), -1)
        if dataset_id < 0:
            continue
        meta[dataset_id] = {
            "dataset_id": dataset_id,
            "dataset_name": str(row.get("dataset_name", "")),
            "q_tot": to_float(row.get("q_tot"), 0.0) or 0.0,
            "h_full": to_float(row.get("h_full"), 0.0) or 0.0,
        }
    return meta


def cleaner_best_retention_by_dataset(
    baseline_rows: list[dict[str, str]],
    meta: dict[int, dict[str, Any]],
) -> dict[int, dict[str, float]]:
    best: dict[tuple[int, str], float] = {}
    for row in baseline_rows:
        dataset_id = to_int(row.get("dataset_id"), -1)
        cleaner = str(row.get("cleaner", "")).strip()
        if dataset_id < 0 or cleaner in SKIP_CLEANERS:
            continue
        score = score_from_row(row)
        if score is None:
            continue
        key = (dataset_id, cleaner)
        if key not in best or score > best[key]:
            best[key] = score

    out: dict[int, dict[str, float]] = {}
    for (dataset_id, cleaner), score in best.items():
        h_full = float(meta.get(dataset_id, {}).get("h_full", 0.0) or 0.0)
        if h_full <= 0:
            continue
        out.setdefault(dataset_id, {})[cleaner] = score / h_full
    return out


def rank_cleaners(values: dict[str, list[float]], fallback_order: list[str]) -> list[str]:
    stats: list[tuple[str, float, float, float, int]] = []
    for cleaner, vals in values.items():
        clean = [float(v) for v in vals if math.isfinite(float(v))]
        if not clean:
            continue
        med = float(statistics.median(clean))
        mean = float(sum(clean) / len(clean))
        top95 = sum(1 for v in clean if v >= 0.95) / len(clean)
        stats.append((cleaner, med, top95, mean, len(clean)))
    stats.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], x[0]))
    order = [x[0] for x in stats]
    for cleaner in fallback_order:
        if cleaner not in order:
            order.append(cleaner)
    return order


def learn_orders_for_thresholds(
    *,
    train_dataset_ids: set[int],
    meta: dict[int, dict[str, Any]],
    cleaner_retention: dict[int, dict[str, float]],
    tau_turn: float,
    tau_hi: float,
) -> dict[str, list[str]]:
    global_values: dict[str, list[float]] = {}
    regime_values: dict[str, dict[str, list[float]]] = {"low": {}, "mid": {}, "high": {}}

    for dataset_id in train_dataset_ids:
        m = meta.get(dataset_id)
        if not m:
            continue
        reg = regime_for(float(m["q_tot"]), tau_turn, tau_hi)
        for cleaner, retention in cleaner_retention.get(dataset_id, {}).items():
            global_values.setdefault(cleaner, []).append(retention)
            regime_values[reg].setdefault(cleaner, []).append(retention)

    fallback = rank_cleaners(global_values, [])
    return {reg: rank_cleaners(regime_values[reg], fallback) for reg in ["low", "mid", "high"]}


def threshold_objective(
    *,
    train_dataset_ids: set[int],
    meta: dict[int, dict[str, Any]],
    cleaner_retention: dict[int, dict[str, float]],
    tau_turn: float,
    tau_hi: float,
) -> tuple[float, dict[str, list[str]]]:
    orders = learn_orders_for_thresholds(
        train_dataset_ids=train_dataset_ids,
        meta=meta,
        cleaner_retention=cleaner_retention,
        tau_turn=tau_turn,
        tau_hi=tau_hi,
    )
    scores: list[float] = []
    for dataset_id in train_dataset_ids:
        vals = cleaner_retention.get(dataset_id, {})
        if not vals:
            continue
        m = meta.get(dataset_id)
        if not m:
            continue
        reg = regime_for(float(m["q_tot"]), tau_turn, tau_hi)
        order = orders.get(reg, [])
        best_cleaner = max(vals.items(), key=lambda kv: kv[1])[0]
        if best_cleaner in order:
            rank_score = 1.0 / (1.0 + order.index(best_cleaner))
        else:
            rank_score = 0.0
        top_cleaner = order[0] if order else None
        top_retention = vals.get(top_cleaner, 0.0) if top_cleaner else 0.0
        scores.append(rank_score + 0.1 * top_retention)
    return (sum(scores) / len(scores) if scores else 0.0), orders


def learn_entry_policy(
    *,
    train_dataset_ids: set[int],
    meta: dict[int, dict[str, Any]],
    cleaner_retention: dict[int, dict[str, float]],
    tau_turn_candidates: list[float],
    tau_hi_candidates: list[float],
    default_tau_turn: float,
    default_tau_hi: float,
) -> tuple[float, float, dict[str, list[str]], float]:
    best_tuple: tuple[float, float, float, dict[str, list[str]], float] | None = None
    for tau_turn in tau_turn_candidates:
        for tau_hi in tau_hi_candidates:
            if tau_turn >= tau_hi:
                continue
            score, orders = threshold_objective(
                train_dataset_ids=train_dataset_ids,
                meta=meta,
                cleaner_retention=cleaner_retention,
                tau_turn=tau_turn,
                tau_hi=tau_hi,
            )
            # Tie-break toward the paper/default thresholds for stability.
            distance = abs(tau_turn - default_tau_turn) + abs(tau_hi - default_tau_hi)
            candidate = (score, -distance, tau_turn, orders, tau_hi)
            if best_tuple is None or candidate > best_tuple:
                best_tuple = candidate
    if best_tuple is None:
        orders = learn_orders_for_thresholds(
            train_dataset_ids=train_dataset_ids,
            meta=meta,
            cleaner_retention=cleaner_retention,
            tau_turn=default_tau_turn,
            tau_hi=default_tau_hi,
        )
        return default_tau_turn, default_tau_hi, orders, 0.0
    score, _neg_distance, tau_turn, orders, tau_hi = best_tuple
    return float(tau_turn), float(tau_hi), orders, float(score)


def deep_update_policy(base_cfg: dict[str, Any], tau_turn: float, tau_hi: float, orders: dict[str, list[str]]) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("trace", {})["tau_turn"] = float(tau_turn)
    cfg.setdefault("trace", {})["tau_hi"] = float(tau_hi)
    cfg.setdefault("trace", {})["entry_orders"] = {
        "low": orders.get("low", []),
        "mid": orders.get("mid", []),
        "high": orders.get("high", []),
    }
    cfg.setdefault("trace", {}).setdefault("entry_bias", {})
    return cfg


def run_blind_random(
    *,
    project_root: Path,
    trace_output_dir: Path,
    output_dir: Path,
    random_seeds: int,
    seed: int,
    budgets: list[float],
    progress_every: int,
) -> None:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "36_eval_trace_blind_random.py"),
        "--trace-output-dir", str(trace_output_dir),
        "--output-dir", str(output_dir),
        "--random-seeds", str(random_seeds),
        "--seed", str(seed),
        "--budgets", *[str(b) for b in budgets],
        "--progress-every", str(progress_every),
        "--flush",
    ]
    subprocess.run(cmd, check=True, cwd=str(project_root))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-dataset-out TRACE held-out validation.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, required=True, help="Cluster replay results dir used by scripts/30_replay_trace.py")
    parser.add_argument("--base-trace-output-dir", type=Path, required=True, help="Existing full Stage-4 output dir used for learning entry policies")
    parser.add_argument("--config", type=Path, default=Path("configs/trace.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed/trace/lodo"))
    parser.add_argument("--random-seeds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--budgets", type=float, nargs="*", default=[0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--tau-turn-candidates", type=float, nargs="*", default=[0.08, 0.10, 0.12, 0.15])
    parser.add_argument("--tau-hi-candidates", type=float, nargs="*", default=[0.18, 0.20, 0.25, 0.30])
    parser.add_argument("--heldout-names", nargs="*", default=None, help="Optional subset of dataset names to hold out")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = args.project_root.resolve() if args.project_root else script_path.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.analysis.trace_replay import replay_trace

    results_dir = args.results_dir.resolve()
    base_trace_dir = args.base_trace_output_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = args.config if args.config.is_absolute() else project_root / args.config
    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    default_tau_turn = float(base_cfg.get("trace", {}).get("tau_turn", 0.10))
    default_tau_hi = float(base_cfg.get("trace", {}).get("tau_hi", 0.20))

    summary_rows = read_csv_dicts(base_trace_dir / "trace_dataset_summary.csv")
    baseline_rows = read_csv_dicts(base_trace_dir / "trace_baseline_sequence.csv")
    meta = load_metadata(summary_rows)
    cleaner_retention = cleaner_best_retention_by_dataset(baseline_rows, meta)

    dataset_names = sorted({m["dataset_name"] for m in meta.values() if m.get("dataset_name")})
    if args.heldout_names:
        dataset_names = [x for x in dataset_names if x in set(args.heldout_names)]
    print(f"[TRACE] LODO dataset names: {', '.join(dataset_names)}", flush=True)

    fold_rows: list[dict[str, Any]] = []
    combined_dataset_rows: list[dict[str, Any]] = []
    combined_blind_rows: list[dict[str, Any]] = []

    for heldout_name in dataset_names:
        heldout_ids = sorted(dataset_id for dataset_id, m in meta.items() if m.get("dataset_name") == heldout_name)
        train_ids = {dataset_id for dataset_id, m in meta.items() if m.get("dataset_name") != heldout_name}
        train_names = sorted({m["dataset_name"] for dataset_id, m in meta.items() if dataset_id in train_ids})

        tau_turn, tau_hi, orders, train_score = learn_entry_policy(
            train_dataset_ids=train_ids,
            meta=meta,
            cleaner_retention=cleaner_retention,
            tau_turn_candidates=[float(x) for x in args.tau_turn_candidates],
            tau_hi_candidates=[float(x) for x in args.tau_hi_candidates],
            default_tau_turn=default_tau_turn,
            default_tau_hi=default_tau_hi,
        )

        fold_dir = output_dir / f"heldout_{heldout_name}"
        trace_dir = fold_dir / "trace"
        blind_dir = trace_dir / "blind_random"
        cfg_out = fold_dir / "trace_lodo_config.yaml"
        fold_dir.mkdir(parents=True, exist_ok=True)
        cfg = deep_update_policy(base_cfg, tau_turn, tau_hi, orders)
        cfg_out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

        print(
            f"[TRACE] Fold heldout={heldout_name}: train={train_names}, ids={heldout_ids}, "
            f"tau=({tau_turn:.3f},{tau_hi:.3f}), orders={json.dumps(orders, ensure_ascii=False)}",
            flush=True,
        )

        if not (args.skip_existing and (trace_dir / "trace_aggregate_summary.json").exists()):
            replay_trace(
                project_root=project_root,
                results_dir=results_dir,
                config_path=cfg_out,
                output_dir=trace_dir,
                dataset_ids=heldout_ids,
            )
        if not (args.skip_existing and (blind_dir / "trace_blind_random_aggregate_summary.json").exists()):
            run_blind_random(
                project_root=project_root,
                trace_output_dir=trace_dir,
                output_dir=blind_dir,
                random_seeds=int(args.random_seeds),
                seed=int(args.seed),
                budgets=[float(b) for b in args.budgets],
                progress_every=5,
            )

        trace_agg = load_json(trace_dir / "trace_aggregate_summary.json")
        blind_agg = load_json(blind_dir / "trace_blind_random_aggregate_summary.json")
        fold_rows.append({
            "heldout_dataset_name": heldout_name,
            "train_dataset_names_json": json.dumps(train_names, ensure_ascii=False),
            "heldout_dataset_ids_json": json.dumps(heldout_ids),
            "tau_turn": tau_turn,
            "tau_hi": tau_hi,
            "train_policy_score": train_score,
            "entry_orders_json": json.dumps(orders, ensure_ascii=False),
            "n_datasets": trace_agg.get("n_datasets"),
            "median_trace_hit95_progress": trace_agg.get("median_trace_hit95_progress"),
            "median_trace_score_retention": trace_agg.get("median_trace_score_retention"),
            "median_blind_random_hit95_progress": blind_agg.get("median_blind_random_hit95_progress"),
            "median_trace_auc_retention": blind_agg.get("median_trace_auc_retention"),
            "median_blind_random_auc_retention": blind_agg.get("median_blind_random_auc_retention"),
            "median_trace_hit95_not_worse_than_random_share": blind_agg.get("median_trace_hit95_not_worse_than_random_share"),
            "datasets_missing_hit95_json": json.dumps(trace_agg.get("datasets_missing_hit95", [])),
        })

        for row in read_csv_dicts(trace_dir / "trace_dataset_summary.csv"):
            row = dict(row)
            row["heldout_dataset_name"] = heldout_name
            combined_dataset_rows.append(row)
        for row in read_csv_dicts(blind_dir / "trace_blind_random_dataset_summary.csv"):
            row = dict(row)
            row["heldout_dataset_name"] = heldout_name
            combined_blind_rows.append(row)

    write_csv(output_dir / "lodo_folds.csv", fold_rows, [
        "heldout_dataset_name", "train_dataset_names_json", "heldout_dataset_ids_json",
        "tau_turn", "tau_hi", "train_policy_score", "entry_orders_json",
        "n_datasets", "median_trace_hit95_progress", "median_trace_score_retention",
        "median_blind_random_hit95_progress", "median_trace_auc_retention",
        "median_blind_random_auc_retention", "median_trace_hit95_not_worse_than_random_share",
        "datasets_missing_hit95_json",
    ])

    if combined_dataset_rows:
        write_csv(output_dir / "lodo_trace_dataset_summary.csv", combined_dataset_rows, list(combined_dataset_rows[0].keys()))
    if combined_blind_rows:
        write_csv(output_dir / "lodo_blind_random_dataset_summary.csv", combined_blind_rows, list(combined_blind_rows[0].keys()))

    aggregate = {
        "output_dir": str(output_dir),
        "n_folds": len(fold_rows),
        "n_datasets": len(combined_blind_rows),
        "random_seeds": int(args.random_seeds),
        "median_trace_hit95_progress": safe_median([to_float(r.get("trace_hit95_progress")) for r in combined_blind_rows]),
        "median_blind_random_hit95_progress": safe_median([to_float(r.get("blind_random_hit95_progress_median")) for r in combined_blind_rows]),
        "median_trace_auc_retention": safe_median([to_float(r.get("trace_auc_retention")) for r in combined_blind_rows]),
        "median_blind_random_auc_retention": safe_median([to_float(r.get("blind_random_auc_retention_median")) for r in combined_blind_rows]),
        "median_trace_hit95_not_worse_than_random_share": safe_median([to_float(r.get("trace_hit95_not_worse_than_random_share")) for r in combined_blind_rows]),
        "folds_csv": str(output_dir / "lodo_folds.csv"),
        "trace_dataset_summary_csv": str(output_dir / "lodo_trace_dataset_summary.csv"),
        "blind_random_dataset_summary_csv": str(output_dir / "lodo_blind_random_dataset_summary.csv"),
    }
    (output_dir / "lodo_aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)
    print(f"[TRACE] LODO validation written to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
