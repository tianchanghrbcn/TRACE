from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------
# Files that must stay because current release validation / home page
# depend on them.
# ----------------------------------------------------------------------
KEEP = {
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "data/README.md",
    "docs/data_policy.md",
    "docs/hardware_runtime.md",
    "docs/release_packaging.md",
    "docs/stage1_to_stage4_plan.md",
    "docs/terminal_interface.md",
    "docs/mode_a_paper_replay_validation.md",
    "docs/stage3_strict_validation.md",
    "docs/paper_output_traceability.md",
    "scripts/00_trace_home.py",
    "scripts/44_build_release_assets.py",
    "scripts/45_validate_data_availability.py",
    "scripts/62_validate_mode_a_paper_replay.py",
    "scripts/63_validate_stage3_strict.py",
    "scripts/98_validate_release_package.py",
    "scripts/97_validate_stage2_strict.sh",
    "scripts/90_run_smoke_from_scratch.py",
    "configs/mode_b_smoke.yaml",
    "configs/mode_c_full.yaml",
}

# ----------------------------------------------------------------------
# Conservative but meaningful first-sweep removals.
# These are migration/audit/planning scaffolds or retired helpers.
# ----------------------------------------------------------------------
DELETE_FILES = [
    # Stage 3 migration / audit docs that are no longer reviewer-facing
    "docs/legacy_migration_plan.md",
    "docs/figure_candidate_details.md",
    "docs/pre_experiment_migration_plan.md",
    "docs/visual_demo_migration_plan.md",
    "docs/release_cleanup.md",
    "docs/stage3_paper_exact_replay_plan.md",
    "docs/mode_a_generated_paper_tables.md",
    "docs/mode_a_generated_summary_replay.md",
    "docs/mode_a_paper_exact_replay.md",
    "docs/mode_a_paper_figure_replay.md",
    "docs/paper_table_equivalence.md",
    "docs/paper_table_traceability_layers.md",
    "docs/paper_figure_harness.md",
    "docs/advisor_handoff.md",
    "docs/stage1_to_stage3_summary.md",

    # Retired audit / cleanup helpers
    "scripts/29_audit_legacy_sources.py",
    "scripts/34_audit_figure_candidates.py",
    "scripts/37_audit_pre_experiment.py",
    "scripts/39_audit_visual_demo.py",
    "scripts/41_audit_release_cleanup.py",
    "scripts/42_normalize_release_source_text.py",
    "scripts/43_clean_generated_artifacts.py",

    # Empty root placeholder
    "analysis/.gitkeep",

    # Tracked empty output placeholders
    "figures/png/.gitkeep",
    "figures/pdf/.gitkeep",
    "figures/pre_experiment/.gitkeep",
    "figures/visual_demo/.gitkeep",
    "results/processed/.gitkeep",
    "results/tables/.gitkeep",
    "results/pre_experiment/.gitkeep",
    "results/visual_demo/.gitkeep",
]

# Directories we should never remove even if temporarily empty.
PROTECTED_DIRS = {
    ".git",
    ".github",
    "configs",
    "data",
    "data/raw",
    "data/raw/train",
    "data/pre_experiment",
    "docs",
    "scripts",
    "src",
    "release",
}

def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()

def should_keep(path: Path) -> bool:
    r = rel(path)
    return r in KEEP

def patch_trace_home(apply: bool) -> dict:
    path = ROOT / "scripts" / "00_trace_home.py"
    if not path.exists():
        return {"changed": False, "path": str(path), "reason": "missing"}

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    original = text

    # Remove PLANNED dictionary contents by replacing with empty dict.
    text = re.sub(
        r"PLANNED\s*=\s*\{.*?\}\n\n",
        "PLANNED = {}\n\n",
        text,
        flags=re.S,
    )

    # Remove loop that prints PLANNED menu entries, if present.
    text = re.sub(
        r"\n\s*for key in sorted\(PLANNED, key=lambda value: int\(value\)\):\n\s*print\(f\"  \{int\(key\):2d\}\. \{PLANNED\[key\]\} \[planned/info\]\"\)\n",
        "\n",
        text,
    )

    changed = text != original
    if changed and apply:
        path.write_text(text, encoding="utf-8")

    return {
        "changed": changed,
        "path": rel(path),
        "reason": "remove Stage 4 / claim placeholder menu entries",
    }

def remove_files(apply: bool) -> list[dict]:
    actions = []

    for item in DELETE_FILES:
        path = ROOT / item
        if not path.exists():
            continue
        if should_keep(path):
            continue

        actions.append({
            "path": item,
            "type": "file",
            "action": "delete",
        })

        if apply:
            path.unlink()

    return actions

def remove_empty_dirs(apply: bool) -> list[dict]:
    actions = []

    # deepest-first walk
    all_dirs = sorted(
        [p for p in ROOT.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    )

    for path in all_dirs:
        r = rel(path)

        if r in PROTECTED_DIRS:
            continue

        if any(
            r == protected or r.startswith(protected + "/")
            for protected in {".git", ".github"}
        ):
            continue

        try:
            entries = list(path.iterdir())
        except Exception:
            continue

        if entries:
            continue

        actions.append({
            "path": r,
            "type": "dir",
            "action": "rmdir",
        })

        if apply:
            path.rmdir()

    return actions

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-sweep cleanup for Stage 4 start.")
    parser.add_argument("--apply", action="store_true", help="Actually delete/patch files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/logs/cleanup_stage4_preflight.json"),
        help="Write cleanup summary JSON here.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    patch_result = patch_trace_home(args.apply)
    file_actions = remove_files(args.apply)
    dir_actions = remove_empty_dirs(args.apply)

    summary = {
        "apply": args.apply,
        "patch_trace_home": patch_result,
        "deleted_files": file_actions,
        "removed_empty_dirs": dir_actions,
        "kept_required_paths": sorted(KEEP),
        "note": (
            "This is a conservative first cleanup before Stage 4. "
            "It removes migration scaffolding, retired audit helpers, "
            "tracked empty output placeholders, and Stage 4 placeholder menu entries. "
            "A second cleanup can be done after Stage 4."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(
        {
            "apply": summary["apply"],
            "patched_home": patch_result["changed"],
            "deleted_file_count": len(file_actions),
            "removed_empty_dir_count": len(dir_actions),
            "output": str(args.output),
        },
        indent=2,
        ensure_ascii=False,
    ))

    print("[TRACE] Cleanup report written to:", args.output)

if __name__ == "__main__":
    main()
