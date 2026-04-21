#!/usr/bin/env python3
"""Estimated progress monitor for long-running TRACE validation."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path


DEFAULT_STEPS = {
    "setup_mode_b": 10,
    "setup_mode_c": 20,
    "method_registry": 10,
    "static_checks": 10,
    "mode_b_smoke": 90,
    "mode_b_smoke_manifest": 5,
    "clusterer_coverage": 600,
    "clusterer_coverage_check": 5,
    "torch110_dependency_probe": 60,
    "torch110_dependency_probe_check": 5,
    "boostclean_import_probe": 30,
    "holoclean_import_probe": 30,
    "holoclean_db_check": 10,
    "cleaner_mode": 60,
    "cleaner_baran": 16200,
    "cleaner_holoclean": 420,
    "cleaner_bigdansing": 180,
    "cleaner_boostclean": 180,
    "cleaner_horizon": 180,
    "cleaner_scared": 900,
    "cleaner_unified": 3900,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor estimated TRACE validation progress.")
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--reference", type=Path, default=Path("configs/runtime_reference.yaml"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-minutes", type=float, default=15.0)
    return parser.parse_args()


def load_reference(path: Path) -> dict[str, int]:
    if not path.exists():
        return DEFAULT_STEPS

    try:
        import yaml
        with path.open("r", encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
        steps = data.get("stage2_strict", {}).get("steps", {})
        if isinstance(steps, dict) and steps:
            return {str(k): int(v) for k, v in steps.items()}
    except Exception:
        pass

    return DEFAULT_STEPS


def latest_validation_dir(log_dir: Path) -> Path | None:
    candidates = [
        path for path in log_dir.glob("stage2_strict_*")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def parse_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def parse_start_time(validation_dir: Path) -> datetime | None:
    try:
        stamp = validation_dir.name.replace("stage2_strict_", "")
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    except Exception:
        return None


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def report_once(log_dir: Path, reference_path: Path) -> bool:
    steps = load_reference(reference_path)
    validation_dir = latest_validation_dir(log_dir)

    if validation_dir is None:
        print("[TRACE] No stage2_strict_* validation directory found.")
        return False

    result_path = validation_dir / "RESULT"
    summary_path = validation_dir / "summary.tsv"
    summary = parse_summary(summary_path)

    total_ref = sum(steps.values())
    completed_ref = sum(seconds for name, seconds in steps.items() if summary.get(name) == "PASS")

    start_time = parse_start_time(validation_dir)
    elapsed = 0.0
    if start_time is not None:
        elapsed = (datetime.now() - start_time).total_seconds()

    if result_path.exists():
        result = result_path.read_text(encoding="utf-8-sig", errors="replace").strip()
        progress = 100.0 if result == "PASSED" else min(99.0, 100.0 * completed_ref / max(total_ref, 1))
        print(f"[TRACE] Validation directory: {validation_dir}")
        print(f"[TRACE] Result: {result}")
        print(f"[TRACE] Estimated progress: {progress:.1f}%")
        print(f"[TRACE] Elapsed wall time: {format_seconds(elapsed)}")
        return True

    completed_names = [name for name in steps if summary.get(name) == "PASS"]
    next_step = next((name for name in steps if summary.get(name) != "PASS"), "unknown")

    progress = min(99.0, 100.0 * completed_ref / max(total_ref, 1))
    remaining = max(0.0, total_ref - completed_ref)

    print(f"[TRACE] Validation directory: {validation_dir}")
    print(f"[TRACE] Completed steps: {len(completed_names)} / {len(steps)}")
    print(f"[TRACE] Current or next step: {next_step}")
    print(f"[TRACE] Estimated progress: {progress:.1f}%")
    print(f"[TRACE] Elapsed wall time: {format_seconds(elapsed)}")
    print(f"[TRACE] Reference remaining time: {format_seconds(remaining)}")
    print("[TRACE] Note: this is an estimated progress monitor, not an exact algorithmic progress bar.")
    return False


def main() -> None:
    args = parse_args()

    while True:
        done = report_once(args.log_dir, args.reference)
        if not args.watch or done:
            break
        print("")
        time.sleep(max(1.0, args.interval_minutes * 60.0))


if __name__ == "__main__":
    main()

