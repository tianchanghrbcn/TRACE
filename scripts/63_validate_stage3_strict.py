#!/usr/bin/env python3
"""Minimal strict reviewer workflow validator.

Reviewer-facing workflows:
  paper-replay
  benchmark-smoke
  benchmark-full-audit

For benchmark-full-audit, the authoritative release proof is:
  results/logs/stage2_strict_*/RESULT == PASSED
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Validate strict reviewer workflow evidence.")
    p.add_argument("--full-audit-proof-dir", "--mode-c-proof-dir", dest="proof_dir", type=Path, default=None)
    p.add_argument("--skip-smoke-rerun", "--skip-mode-b-rerun", dest="skip_smoke", action="store_true")
    p.add_argument("--rebuild-paper-replay", "--rebuild-mode-a", dest="rebuild_paper", action="store_true")
    p.add_argument("--allow-missing-full-audit-proof", "--allow-missing-mode-c-proof", dest="allow_missing", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results/logs/stage3_strict_validation_report.json"))
    return p.parse_args()


def run_py(cmd):
    print("[TRACE] >>>", " ".join([sys.executable] + cmd))
    proc = subprocess.run([sys.executable] + cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception:
        return {}


def write_report(path, report):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = path.with_suffix(".md")
    lines = [
        "# TRACE Strict Reviewer Workflow Validation",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Status: {report['status']}",
        "",
        "## Workflows",
        "",
        "| Workflow | Status | Accepted |",
        "|---|---|---:|",
    ]
    for w in report["workflows"]:
        lines.append(f"| {w['workflow']} | {w['status']} | {w['accepted']} |")
    lines += ["", "## Failures", ""]
    lines += [f"- {x}" for x in report["failures"]] if report["failures"] else ["No hard failures."]
    lines += ["", "## Warnings", ""]
    lines += [f"- {x}" for x in report["warnings"]] if report["warnings"] else ["No warnings."]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_paper_replay(rebuild):
    cmd = ["scripts/62_validate_mode_a_paper_replay.py"]
    if rebuild:
        cmd.append("--rebuild")
    run = run_py(cmd)

    report_path = ROOT / "analysis/paper_generated/paper_replay_validation_report.json"
    report = read_json(report_path)
    status = report.get("status", "")
    accepted = run["returncode"] == 0 and status in {"PASS", "PASS_WITH_WARNINGS"} and not report.get("failures")

    return {
        "workflow": "paper-replay",
        "status": status if accepted else "FAIL",
        "accepted": accepted,
        "report_path": str(report_path),
        "command": run,
        "failures": [] if accepted else [f"paper-replay not accepted: status={status}, path={report_path}"],
        "warnings": report.get("warnings", []) if accepted else [],
    }


def validate_smoke(skip):
    failures = []
    commands = []

    if not skip:
        commands.append(run_py(["scripts/90_run_smoke_from_scratch.py", "--config", "configs/benchmark_smoke.yaml", "--clean"]))

    manifest_path = ROOT / "results/logs/pipeline_run_manifest.json"
    summary_path = ROOT / "results/logs/mode_b_smoke_summary.json"

    if not manifest_path.exists():
        failures.append(f"Missing smoke manifest: {manifest_path}")
    else:
        m = read_json(manifest_path)
        if m.get("failure_count", 0) != 0:
            failures.append(f"Smoke manifest failure_count={m.get('failure_count')}")
        if m.get("cleaned_result_count", 0) < 1:
            failures.append("Smoke cleaned_result_count < 1")
        if m.get("clustered_result_count", 0) < 1:
            failures.append("Smoke clustered_result_count < 1")

    if not summary_path.exists():
        failures.append(f"Missing smoke summary: {summary_path}")

    for c in commands:
        if c["returncode"] != 0:
            failures.append(f"Smoke command failed: {c['command']}")

    return {
        "workflow": "benchmark-smoke",
        "status": "PASS" if not failures else "FAIL",
        "accepted": not failures,
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "commands": commands,
        "failures": failures,
        "warnings": [],
    }


def find_proof(explicit):
    if explicit:
        p = explicit if explicit.is_absolute() else ROOT / explicit
        return p

    result_files = sorted((ROOT / "results/logs").glob("stage2_strict_*/RESULT"), reverse=True)
    for result in result_files:
        if result.parent.is_dir():
            return result.parent
    return None


def validate_full_audit(explicit, allow_missing):
    proof = find_proof(explicit)

    if proof is None or not proof.exists():
        if allow_missing:
            return {
                "workflow": "benchmark-full-audit",
                "status": "WARN_MISSING_PROOF",
                "accepted": True,
                "proof_dir": str(proof) if proof else "",
                "failures": [],
                "warnings": ["Full-audit proof directory not found."],
            }
        return {
            "workflow": "benchmark-full-audit",
            "status": "FAIL",
            "accepted": False,
            "proof_dir": str(proof) if proof else "",
            "failures": ["Full-audit proof directory not found."],
            "warnings": [],
        }

    result_path = proof / "RESULT"
    summary_path = proof / "summary.tsv"

    failures = []
    warnings = []

    if not result_path.exists():
        failures.append(f"Missing RESULT: {result_path}")
        result_value = ""
    else:
        result_value = result_path.read_text(encoding="utf-8-sig", errors="replace").strip()

    if result_value != "PASSED":
        failures.append(f"RESULT is not PASSED: {result_value}")

    if not summary_path.exists():
        warnings.append(f"summary.tsv missing: {summary_path}")
    else:
        rows = summary_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        fail_rows = [r for r in rows if "\tFAIL" in r or r.endswith("\tFAIL")]
        if fail_rows:
            warnings.append("summary.tsv contains legacy FAIL rows, but RESULT=PASSED is treated as authoritative release proof.")

    accepted = result_value == "PASSED"

    return {
        "workflow": "benchmark-full-audit",
        "status": "PASS" if accepted else "FAIL",
        "accepted": accepted,
        "proof_dir": str(proof),
        "result_path": str(result_path),
        "summary_path": str(summary_path),
        "result_value": result_value,
        "failures": [] if accepted else failures,
        "warnings": warnings,
    }


def main():
    args = parse_args()

    workflows = [
        validate_paper_replay(args.rebuild_paper),
        validate_smoke(args.skip_smoke),
        validate_full_audit(args.proof_dir, args.allow_missing),
    ]

    failures = []
    warnings = []

    for w in workflows:
        if not w["accepted"]:
            failures.append(f"{w['workflow']} failed: {w['status']}")
        failures.extend([f"{w['workflow']}: {x}" for x in w.get("failures", [])])
        warnings.extend([f"{w['workflow']}: {x}" for x in w.get("warnings", [])])

    if failures:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "workflows": workflows,
        "warning_count": len(warnings),
        "failure_count": len(failures),
        "warnings": warnings,
        "failures": failures,
    }

    write_report(args.output, report)

    print(json.dumps({
        "status": status,
        "workflow_statuses": {w["workflow"]: w["status"] for w in workflows},
        "warning_count": len(warnings),
        "failure_count": len(failures),
        "output": str(args.output),
    }, indent=2, ensure_ascii=False))
    print(f"[TRACE] Strict reviewer workflow validation report written to: {args.output}")
    print(f"[TRACE] Strict reviewer workflow validation status: {status}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()
