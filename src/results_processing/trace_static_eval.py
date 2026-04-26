#!/usr/bin/env python3
"""Static TRACE entry-screening evaluation from archived final result tables.

This module evaluates TRACE's *entry-screening* policy without re-running the
baseline search and without requiring per-trial archives. It operates on the
canonical result tables already produced by Stage 3.

Input tables
------------
- trials.csv
- result_metrics.csv

Core idea
---------
For each dirty instance D_q, we first collapse cleaner x clusterer results to a
single best score per cleaner:

    H*_c(D_q) = max_a H*(a; D_{q,c})

We then compare the TRACE start set C_start(q_tot) against the full archived
candidate cleaner space (excluding the reference cleaner c0 and GroundTruth).
This gives four static validation metrics:

1. Coverage_top1: whether the globally best candidate cleaner is inside the
   TRACE start set.
2. Coverage_95: whether the start set contains a cleaner whose best archived
   score reaches at least 95% of the archived candidate optimum.
3. Retention_start: best score in the start set divided by the archived
   candidate optimum.
4. Reduction: how much the candidate cleaner space is reduced by the start set.

These metrics validate whether TRACE's *initial pruning* is sensible. They do
not attempt to replay runtime gating or claim wall-clock speedups.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "reference_cleaner": "mode",
    "exclude_cleaners": ["GroundTruth"],
    "score_column": "final_combined_score",
    "thresholds": {
        "tau_turn": 0.10,
        "tau_hi": 0.20,
    },
    "group_map": {
        "norm": ["Unified", "Horizon", "BigDansing"],
        "sem": ["Baran", "HoloClean"],
        "strong": ["BoostClean", "SCAReD"],
    },
    "start_rules": {
        "low": ["norm"],
        "mid": ["norm", "sem"],
        "high": ["sem", "strong"],
    },
    "aliases": {
        "unified": "Unified",
        "horizon": "Horizon",
        "bigdansing": "BigDansing",
        "baran": "Baran",
        "holoclean": "HoloClean",
        "boostclean": "BoostClean",
        "scared": "SCAReD",
        "mode": "mode",
        "groundtruth": "GroundTruth",
    },
    "score_tolerance": 1e-12,
    "paper_snippet": {
        "coverage_percent_decimals": 1,
        "retention_percent_decimals": 1,
        "reduction_percent_decimals": 1,
    },
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML config files.")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("TRACE static config must be a mapping.")
    return _merge_dict(config, raw)


@dataclass(frozen=True)
class NameNormalizer:
    aliases: dict[str, str]

    @staticmethod
    def _key(name: Any) -> str:
        if name is None:
            return ""
        s = str(name).strip().lower()
        return "".join(ch for ch in s if ch.isalnum())

    def canon(self, name: Any) -> str:
        key = self._key(name)
        if not key:
            return ""
        return self.aliases.get(key, str(name).strip())


@dataclass
class DatasetMeta:
    dataset_id: str
    dataset_name: str
    q_tot: float | None
    error_rate: float | None
    missing_rate: float | None
    noise_rate: float | None
    csv_file: str

    @property
    def error_dominance(self) -> str:
        m = self.missing_rate or 0.0
        n = self.noise_rate or 0.0
        if math.isclose(m, n, rel_tol=0.0, abs_tol=1e-15):
            return "balanced"
        return "missing" if m > n else "noise"


@dataclass
class CleanerBest:
    cleaner: str
    best_score: float
    best_clusterer: str


@dataclass
class DatasetEvaluation:
    row: dict[str, Any]


def _infer_q_tot(row: dict[str, str]) -> float | None:
    error_rate = _safe_float(row.get("error_rate"))
    if error_rate is not None:
        return error_rate
    missing_rate = _safe_float(row.get("missing_rate")) or 0.0
    noise_rate = _safe_float(row.get("noise_rate")) or 0.0
    total = missing_rate + noise_rate
    return total if total > 0 else None


def _load_dataset_meta(trials_rows: list[dict[str, str]]) -> dict[str, DatasetMeta]:
    out: dict[str, DatasetMeta] = {}
    for row in trials_rows:
        dataset_id = str(row.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        if dataset_id in out:
            continue
        out[dataset_id] = DatasetMeta(
            dataset_id=dataset_id,
            dataset_name=str(row.get("dataset_name", "")).strip(),
            q_tot=_infer_q_tot(row),
            error_rate=_safe_float(row.get("error_rate")),
            missing_rate=_safe_float(row.get("missing_rate")),
            noise_rate=_safe_float(row.get("noise_rate")),
            csv_file=str(row.get("csv_file", "")).strip(),
        )
    return out


def _regime(q_tot: float | None, tau_turn: float, tau_hi: float) -> str:
    if q_tot is None:
        return "unknown"
    if q_tot < tau_turn:
        return "low"
    if q_tot < tau_hi:
        return "mid"
    return "high"


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(sorted({v for v in values if v}), ensure_ascii=False)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.{decimals}f}"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _collapse_cleaner_scores(
    result_rows: list[dict[str, str]],
    normalizer: NameNormalizer,
    score_column: str,
) -> dict[str, dict[str, CleanerBest]]:
    """Return dataset_id -> cleaner -> best archived score."""
    grouped: dict[str, dict[str, CleanerBest]] = defaultdict(dict)
    for row in result_rows:
        dataset_id = str(row.get("dataset_id", "")).strip()
        cleaner = normalizer.canon(row.get("cleaner", ""))
        clusterer = str(row.get("clusterer", "")).strip()
        score = _safe_float(row.get(score_column))
        if not dataset_id or not cleaner or score is None:
            continue
        prev = grouped[dataset_id].get(cleaner)
        if prev is None or score > prev.best_score:
            grouped[dataset_id][cleaner] = CleanerBest(
                cleaner=cleaner,
                best_score=score,
                best_clusterer=clusterer,
            )
    return grouped


def _group_cleaners(config: dict[str, Any], normalizer: NameNormalizer) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    raw_group_map = config.get("group_map", {}) or {}
    if not isinstance(raw_group_map, dict):
        raise ValueError("group_map must be a mapping.")
    for group_name, cleaners in raw_group_map.items():
        if not isinstance(cleaners, list):
            raise ValueError(f"group_map[{group_name!r}] must be a list.")
        canon_list = []
        for cleaner in cleaners:
            canon = normalizer.canon(cleaner)
            if canon:
                canon_list.append(canon)
        out[str(group_name)] = canon_list
    return out


def _start_cleaners_for_regime(
    regime: str,
    group_cleaners: dict[str, list[str]],
    config: dict[str, Any],
) -> list[str]:
    start_rules = config.get("start_rules", {}) or {}
    groups = start_rules.get(regime, []) or []
    ordered: list[str] = []
    seen: set[str] = set()
    for group_name in groups:
        for cleaner in group_cleaners.get(str(group_name), []):
            if cleaner not in seen:
                ordered.append(cleaner)
                seen.add(cleaner)
    return ordered


def _nominal_candidate_cleaners(
    group_cleaners: dict[str, list[str]],
    reference_cleaner: str,
    excluded: set[str],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for cleaners in group_cleaners.values():
        for cleaner in cleaners:
            if cleaner == reference_cleaner or cleaner in excluded:
                continue
            if cleaner not in seen:
                ordered.append(cleaner)
                seen.add(cleaner)
    return ordered


def evaluate_trace_static(
    processed_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    dataset_ids: list[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    normalizer = NameNormalizer(
        aliases={NameNormalizer._key(k): v for k, v in (config.get("aliases", {}) or {}).items()}
    )

    trials_path = processed_dir / "trials.csv"
    result_metrics_path = processed_dir / "result_metrics.csv"

    trials_rows = _read_csv_rows(trials_path)
    result_rows = _read_csv_rows(result_metrics_path)

    dataset_meta = _load_dataset_meta(trials_rows)
    collapsed = _collapse_cleaner_scores(
        result_rows=result_rows,
        normalizer=normalizer,
        score_column=str(config.get("score_column", "final_combined_score")),
    )

    selected_ids: set[str] | None = None
    if dataset_ids:
        selected_ids = {str(x) for x in dataset_ids}

    reference_cleaner = normalizer.canon(config.get("reference_cleaner", "mode"))
    excluded = {normalizer.canon(x) for x in (config.get("exclude_cleaners", []) or [])}
    group_cleaners = _group_cleaners(config, normalizer)
    nominal_candidates = _nominal_candidate_cleaners(group_cleaners, reference_cleaner, excluded)
    nominal_candidate_set = set(nominal_candidates)

    tau_turn = float((config.get("thresholds", {}) or {}).get("tau_turn", 0.10))
    tau_hi = float((config.get("thresholds", {}) or {}).get("tau_hi", 0.20))
    tol = float(config.get("score_tolerance", 1e-12))

    dataset_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    optimal_cleaner_events: list[dict[str, Any]] = []

    for dataset_id in sorted(set(dataset_meta) | set(collapsed), key=lambda x: (len(str(x)), str(x))):
        if selected_ids is not None and dataset_id not in selected_ids:
            continue

        meta = dataset_meta.get(dataset_id)
        if meta is None:
            warnings.append(
                {
                    "dataset_id": dataset_id,
                    "kind": "missing_trials_row",
                    "message": "dataset_id exists in result_metrics.csv but not in trials.csv",
                }
            )
            # create a minimal placeholder; q_tot unknown -> regime unknown
            meta = DatasetMeta(
                dataset_id=dataset_id,
                dataset_name="",
                q_tot=None,
                error_rate=None,
                missing_rate=None,
                noise_rate=None,
                csv_file="",
            )

        regime = _regime(meta.q_tot, tau_turn, tau_hi)
        start_cleaners_nominal = _start_cleaners_for_regime(regime, group_cleaners, config)
        per_cleaner = collapsed.get(dataset_id, {})

        available_all = sorted(per_cleaner)
        available_candidate = [
            c for c in available_all if c != reference_cleaner and c not in excluded and c in nominal_candidate_set
        ]
        # If a cleaner is present in results but absent from the nominal group table,
        # keep it in the archived candidate set only if it is neither reference nor excluded.
        archived_only = [
            c for c in available_all if c != reference_cleaner and c not in excluded and c not in nominal_candidate_set
        ]
        available_candidate_all = available_candidate + archived_only
        available_candidate_all = sorted(dict.fromkeys(available_candidate_all))

        available_start = [c for c in start_cleaners_nominal if c in per_cleaner]
        available_reference = reference_cleaner in per_cleaner

        if not available_candidate_all:
            warnings.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": meta.dataset_name,
                    "kind": "no_candidate_cleaners",
                    "message": "No archived candidate cleaners found after excluding reference and GroundTruth.",
                }
            )
            best_candidate_score = None
            best_candidate_cleaners: list[str] = []
            best_start_score = None
            best_start_cleaner = ""
            coverage_top1 = None
            coverage95 = None
            retention_start = None
            reduction_available = None
            reduction_nominal = None
            reference_score = per_cleaner[reference_cleaner].best_score if available_reference else None
        else:
            cleaner_scores = {c: per_cleaner[c].best_score for c in available_candidate_all}
            best_candidate_score = max(cleaner_scores.values())
            best_candidate_cleaners = sorted(
                [c for c, score in cleaner_scores.items() if abs(score - best_candidate_score) <= tol]
            )
            if available_start:
                best_start_cleaner = max(available_start, key=lambda c: per_cleaner[c].best_score)
                best_start_score = per_cleaner[best_start_cleaner].best_score
            else:
                best_start_cleaner = ""
                best_start_score = None
                warnings.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_name": meta.dataset_name,
                        "kind": "empty_start_set_in_archive",
                        "message": (
                            "TRACE start set is empty after intersecting with archived results. "
                            f"regime={regime}, nominal_start={start_cleaners_nominal}"
                        ),
                    }
                )
            coverage_top1 = 1 if any(c in available_start for c in best_candidate_cleaners) else 0
            coverage95 = (
                1
                if (best_start_score is not None and best_start_score >= 0.95 * best_candidate_score - tol)
                else 0
            )
            retention_start = (
                best_start_score / best_candidate_score if best_start_score is not None and best_candidate_score else None
            )
            reduction_available = 1.0 - (len(available_start) / len(available_candidate_all))
            denom_nominal = len(nominal_candidates)
            reduction_nominal = 1.0 - (len(start_cleaners_nominal) / denom_nominal) if denom_nominal else None
            reference_score = per_cleaner[reference_cleaner].best_score if available_reference else None

            for cleaner in best_candidate_cleaners:
                optimal_cleaner_events.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_name": meta.dataset_name,
                        "cleaner": cleaner,
                        "covered_by_start": 1 if cleaner in available_start else 0,
                        "best_candidate_score": best_candidate_score,
                        "regime": regime,
                    }
                )

        row = {
            "dataset_id": dataset_id,
            "dataset_name": meta.dataset_name,
            "csv_file": meta.csv_file,
            "q_tot": meta.q_tot,
            "error_rate": meta.error_rate,
            "missing_rate": meta.missing_rate,
            "noise_rate": meta.noise_rate,
            "error_dominance": meta.error_dominance,
            "regime": regime,
            "reference_cleaner": reference_cleaner,
            "reference_score": reference_score,
            "nominal_candidate_cleaners_json": _json_list(nominal_candidates),
            "available_candidate_cleaners_json": _json_list(available_candidate_all),
            "nominal_start_cleaners_json": _json_list(start_cleaners_nominal),
            "available_start_cleaners_json": _json_list(available_start),
            "n_nominal_candidate_cleaners": len(nominal_candidates),
            "n_available_candidate_cleaners": len(available_candidate_all),
            "n_nominal_start_cleaners": len(start_cleaners_nominal),
            "n_available_start_cleaners": len(available_start),
            "best_candidate_cleaners_json": _json_list(best_candidate_cleaners),
            "best_candidate_score": best_candidate_score,
            "best_start_cleaner": best_start_cleaner,
            "best_start_score": best_start_score,
            "coverage_top1": coverage_top1,
            "coverage95": coverage95,
            "retention_start": retention_start,
            "reduction_available": reduction_available,
            "reduction_nominal": reduction_nominal,
        }
        dataset_rows.append(row)

    def _summary_from_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        cov_top1_values = [float(r["coverage_top1"]) for r in rows if r.get("coverage_top1") in (0, 1)]
        cov95_values = [float(r["coverage95"]) for r in rows if r.get("coverage95") in (0, 1)]
        retention_values = [float(r["retention_start"]) for r in rows if r.get("retention_start") is not None]
        red_avail_values = [float(r["reduction_available"]) for r in rows if r.get("reduction_available") is not None]
        red_nominal_values = [float(r["reduction_nominal"]) for r in rows if r.get("reduction_nominal") is not None]
        return {
            "label": label,
            "n_instances": len(rows),
            "coverage_top1_hits": int(sum(cov_top1_values)),
            "coverage_top1_rate": _mean(cov_top1_values),
            "coverage95_hits": int(sum(cov95_values)),
            "coverage95_rate": _mean(cov95_values),
            "median_retention_start": _median(retention_values),
            "mean_retention_start": _mean(retention_values),
            "median_reduction_available": _median(red_avail_values),
            "mean_reduction_available": _mean(red_avail_values),
            "median_reduction_nominal": _median(red_nominal_values),
            "mean_reduction_nominal": _mean(red_nominal_values),
        }

    aggregate = _summary_from_rows(dataset_rows, "all")

    regime_summary_rows: list[dict[str, Any]] = []
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        by_regime[str(row.get("regime", "unknown"))].append(row)
    for regime_name in ["low", "mid", "high", "unknown"]:
        rows = by_regime.get(regime_name, [])
        if not rows:
            continue
        regime_summary_rows.append(_summary_from_rows(rows, regime_name))

    optimal_summary_rows: list[dict[str, Any]] = []
    by_cleaner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in optimal_cleaner_events:
        by_cleaner[str(event["cleaner"])].append(event)
    for cleaner, events in sorted(by_cleaner.items()):
        hits = sum(int(e["covered_by_start"]) for e in events)
        scores = [float(e["best_candidate_score"]) for e in events]
        optimal_summary_rows.append(
            {
                "cleaner": cleaner,
                "n_best_candidate": len(events),
                "n_best_candidate_covered": hits,
                "coverage_rate_when_best": (hits / len(events)) if events else None,
                "mean_best_candidate_score": _mean(scores),
            }
        )

    dataset_columns = [
        "dataset_id",
        "dataset_name",
        "csv_file",
        "q_tot",
        "error_rate",
        "missing_rate",
        "noise_rate",
        "error_dominance",
        "regime",
        "reference_cleaner",
        "reference_score",
        "nominal_candidate_cleaners_json",
        "available_candidate_cleaners_json",
        "nominal_start_cleaners_json",
        "available_start_cleaners_json",
        "n_nominal_candidate_cleaners",
        "n_available_candidate_cleaners",
        "n_nominal_start_cleaners",
        "n_available_start_cleaners",
        "best_candidate_cleaners_json",
        "best_candidate_score",
        "best_start_cleaner",
        "best_start_score",
        "coverage_top1",
        "coverage95",
        "retention_start",
        "reduction_available",
        "reduction_nominal",
    ]
    regime_columns = [
        "label",
        "n_instances",
        "coverage_top1_hits",
        "coverage_top1_rate",
        "coverage95_hits",
        "coverage95_rate",
        "median_retention_start",
        "mean_retention_start",
        "median_reduction_available",
        "mean_reduction_available",
        "median_reduction_nominal",
        "mean_reduction_nominal",
    ]
    optimal_columns = [
        "cleaner",
        "n_best_candidate",
        "n_best_candidate_covered",
        "coverage_rate_when_best",
        "mean_best_candidate_score",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "trace_static_dataset_summary.csv"
    regime_path = output_dir / "trace_static_regime_summary.csv"
    optimal_path = output_dir / "trace_static_optimal_cleaner_summary.csv"
    warnings_path = output_dir / "trace_static_warnings.json"
    aggregate_path = output_dir / "trace_static_aggregate_summary.json"
    paper_numbers_path = output_dir / "trace_static_paper_numbers.json"
    paper_snippet_path = output_dir / "trace_static_paper_snippets.md"

    _write_csv(dataset_path, dataset_rows, dataset_columns)
    _write_csv(regime_path, regime_summary_rows, regime_columns)
    _write_csv(optimal_path, optimal_summary_rows, optimal_columns)
    _write_json(warnings_path, warnings)

    snippet_cfg = config.get("paper_snippet", {}) or {}
    cov_dec = int(snippet_cfg.get("coverage_percent_decimals", 1))
    ret_dec = int(snippet_cfg.get("retention_percent_decimals", 1))
    red_dec = int(snippet_cfg.get("reduction_percent_decimals", 1))

    paper_numbers = {
        "n_instances": aggregate.get("n_instances", 0),
        "coverage_top1_rate_pct": _pct(aggregate.get("coverage_top1_rate"), cov_dec),
        "coverage95_rate_pct": _pct(aggregate.get("coverage95_rate"), cov_dec),
        "median_retention_start_pct": _pct(aggregate.get("median_retention_start"), ret_dec),
        "median_reduction_nominal_pct": _pct(aggregate.get("median_reduction_nominal"), red_dec),
        "median_reduction_available_pct": _pct(aggregate.get("median_reduction_available"), red_dec),
        "tau_turn_pct": _pct(tau_turn, 1),
        "tau_hi_pct": _pct(tau_hi, 1),
        "reference_cleaner": reference_cleaner,
    }
    _write_json(paper_numbers_path, paper_numbers)

    aggregate_payload = {
        "project_processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "config_path": str(config_path) if config_path else "",
        "reference_cleaner": reference_cleaner,
        "nominal_candidate_cleaners": nominal_candidates,
        "thresholds": {"tau_turn": tau_turn, "tau_hi": tau_hi},
        "aggregate": aggregate,
        "regime_summary": regime_summary_rows,
        "warnings_count": len(warnings),
    }
    _write_json(aggregate_path, _to_jsonable(aggregate_payload))

    regime_map = {row["label"]: row for row in regime_summary_rows}
    low = regime_map.get("low")
    mid = regime_map.get("mid")
    high = regime_map.get("high")

    low_text = (
        "Coverage_top1="
        + _pct(low.get("coverage_top1_rate") if low else None, cov_dec)
        + r"\%, Coverage_95="
        + _pct(low.get("coverage95_rate") if low else None, cov_dec)
        + r"\%, median retention="
        + _pct(low.get("median_retention_start") if low else None, ret_dec)
        + r"\%"
    ) if low else "无实例"
    mid_text = (
        "Coverage_top1="
        + _pct(mid.get("coverage_top1_rate") if mid else None, cov_dec)
        + r"\%, Coverage_95="
        + _pct(mid.get("coverage95_rate") if mid else None, cov_dec)
        + r"\%, median retention="
        + _pct(mid.get("median_retention_start") if mid else None, ret_dec)
        + r"\%"
    ) if mid else "无实例"
    high_text = (
        "Coverage_top1="
        + _pct(high.get("coverage_top1_rate") if high else None, cov_dec)
        + r"\%, Coverage_95="
        + _pct(high.get("coverage95_rate") if high else None, cov_dec)
        + r"\%, median retention="
        + _pct(high.get("median_retention_start") if high else None, ret_dec)
        + r"\%"
    ) if high else "无实例"

    snippet = rf"""# TRACE 静态评测可直接改写到论文的段落

## 1) 摘要中的替换句

静态验证基于第 5 章已完成的完整搜索结果进行，不需要重跑 baseline。结果显示，TRACE 的入口起步集合在 **{paper_numbers['coverage_top1_rate_pct']}\%** 的含错实例上覆盖了全局最优的非参考 cleaner，并在 **{paper_numbers['coverage95_rate_pct']}\%** 的实例上包含了一个可达到 archived candidate optimum **95\%** 以上得分的 cleaner；其中，起步集合对 archived candidate optimum 的中位得分保留率为 **{paper_numbers['median_retention_start_pct']}\%**，对应的名义 cleaner 搜索空间中位缩减为 **{paper_numbers['median_reduction_nominal_pct']}\%**。

## 2) 贡献 4 中的替换句

我们将上述规律整理成一套面向清洗--聚类方法选择的行动建议 TRACE。考虑到当前工件稳定保留的是完整搜索的最终结果表，而不是所有 trial 级运行档案，本文首先对 TRACE 的入口筛选做静态验证：它直接利用第 5 章已完成的 baseline 结果，检查 TRACE 的起步集合是否覆盖值得继续投入预算的 cleaner。结果显示，TRACE 的入口集合在 **{paper_numbers['coverage_top1_rate_pct']}\%** 的实例上覆盖了全局最优的非参考 cleaner，并在 **{paper_numbers['coverage95_rate_pct']}\%** 的实例上保留了至少 95\% archived candidate optimum 的可行路径；其中中位得分保留率为 **{paper_numbers['median_retention_start_pct']}\%**，名义 cleaner 搜索空间中位缩减为 **{paper_numbers['median_reduction_nominal_pct']}\%**。

## 3) 第 6.4 节建议替换标题

`\\subsection{{TRACE入口筛选的静态验证}}`

## 4) 第 6.4 节建议替换正文（LaTeX）

```tex
最后，我们先对 TRACE 的入口筛选做静态验证，而不把本工作扩展为一次新的完整在线搜索实验。原因在于，当前工件已经稳定保存了第 5 章完整 baseline 搜索的最终结果表，但并未系统保留所有 path-level trial 档案。因此，这里的目标不是重新评测一个独立的搜索算法，而是检验 TRACE 在搜索开始前给出的起步集合，是否已经覆盖了值得优先投入预算的 cleaner。

具体地，对每个含错实例 $D_q$，我们先把每个 cleaner 在所有聚类算法上的最好得分记为
\[
H^\star_c(D_q)=\max_{{a\in\mathcal{{A}}}} H^\star(a;D_{{q,c}}).
\]
考虑到入口筛选只作用于参考 cleaner $c_0$ 之外的候选分支，我们再定义 archived candidate optimum 为
\[
H^\star_{{cand}}(D_q)=\max_{{c\in\mathcal{{C}}\setminus\{{c_0,\mathrm{{GroundTruth}}\}}}} H^\star_c(D_q).
\]
在此基础上，我们报告四个只依赖已保存最终结果表的指标。第一，\emph{{Coverage}}$_{{top1}}$：若全局最优的非参考 cleaner 属于 $\mathcal{{C}}_{{start}}(q_{{tot}})$，则记为 1，否则为 0。第二，\emph{{Retention}}$_{{start}}$：
\[
\mathrm{{Retention}}_{{start}}(D_q)=
\frac{{\max_{{c\in\mathcal{{C}}_{{start}}(q_{{tot}})}} H^\star_c(D_q)}}{{H^\star_{{cand}}(D_q)}}.
\]
第三，\emph{{Coverage}}$_{{95}}$：若起步集合中存在 cleaner 使其最好得分达到 $0.95H^\star_{{cand}}(D_q)$，则记为 1。第四，\emph{{Reduction}}：入口集合相对于候选 cleaner 空间的缩减比例。四个指标均先在每个 $D_q$ 上计算，再做整体汇总。

基于第 5 章已完成的 archived baseline 结果，TRACE 的入口起步集合在 {paper_numbers['coverage_top1_rate_pct']}\% 的含错实例上覆盖了全局最优的非参考 cleaner，并在 {paper_numbers['coverage95_rate_pct']}\% 的实例上包含了一个达到 archived candidate optimum 95\% 以上得分的 cleaner。进一步地，起步集合对 archived candidate optimum 的中位得分保留率为 {paper_numbers['median_retention_start_pct']}\%，而名义 cleaner 搜索空间的中位缩减为 {paper_numbers['median_reduction_nominal_pct']}\%。这说明第 5 章总结出的错误率分段规律，已经足以在不重新运行全部 baseline 的前提下，把搜索起点收缩到一个更小、但仍能较高概率保留强候选的 cleaner 子集。

需要说明的是，这里验证的是 TRACE 的入口筛选是否合理，而不是一次完整的运行期回放。后者还需要系统保留 path-level trial 日志，属于下一步工作。就当前论文的定位而言，上述静态验证已经能够说明：TRACE 并非从结果反推规则，而是能够直接作用于已有的完整搜索结果，并给出与第 5 章规律一致的起步建议。
```

## 5) 可选补充句（按 error band 分层写）

- 低错误段：{low_text}
- 转折带附近：{mid_text}
- 高错误段：{high_text}
"""
    paper_snippet_path.write_text(snippet, encoding="utf-8")

    return {
        "aggregate": aggregate,
        "dataset_summary_path": str(dataset_path),
        "regime_summary_path": str(regime_path),
        "optimal_cleaner_summary_path": str(optimal_path),
        "warnings_path": str(warnings_path),
        "aggregate_path": str(aggregate_path),
        "paper_numbers_path": str(paper_numbers_path),
        "paper_snippet_path": str(paper_snippet_path),
    }
