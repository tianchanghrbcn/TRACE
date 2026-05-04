#!/usr/bin/env python3
"""Build all registered paper tables.

This is the reviewer-facing entry point for Stage 3 table regeneration.
Individual table builders live under src/paper_artifact/tables/ and are
registered automatically when they expose ARTIFACT and build(ctx).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_artifact.io import BuildContext, ArtifactResult, copy_outputs, ensure_dir, find_project_root, utc_now_iso, write_json
from src.paper_artifact.registry import discover_specs, run_spec, select_specs


def parse_csv_list(values: List[str] | None) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for item in values:
        for part in str(item).split(','):
            part = part.strip()
            if part:
                out.append(part)
    return out


def build_manifest(ctx: BuildContext, specs, results, copied_outputs, errors) -> dict:
    counts = {"success": 0, "skipped": 0, "failed": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    counts["failed"] = counts.get("failed", 0) + len(errors)
    return {
        "kind": "tables",
        "generated_at_utc": utc_now_iso(),
        "project_root": ctx.rel(ctx.project_root),
        "input_root": ctx.rel(ctx.input_root),
        "output_dir": ctx.rel(ctx.output_dir),
        "paper_dir": ctx.rel(ctx.paper_dir) if ctx.paper_dir else None,
        "strict": ctx.strict,
        "counts": counts,
        "registered_specs": [s.to_dict() for s in specs],
        "results": [r.to_manifest_dict(ctx) for r in results],
        "copied_outputs": [ctx.rel(p) for p in copied_outputs],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all registered paper tables.")
    parser.add_argument("--project-root", default=None, help="Repository root. Defaults to auto-detect.")
    parser.add_argument("--input-root", default="results", help="Input root containing processed results. Default: results")
    parser.add_argument("--output-dir", default="results/paper_artifact/tables", help="Output directory for generated tables.")
    parser.add_argument("--paper-table-dir", default=None, help="Optional paper table directory to copy final outputs into.")
    parser.add_argument("--only", nargs="*", default=None, help="Only build these artifact ids or paper ids. Comma-separated values are allowed.")
    parser.add_argument("--skip", nargs="*", default=None, help="Skip these artifact ids or paper ids. Comma-separated values are allowed.")
    parser.add_argument("--include-disabled", action="store_true", help="Include builders whose ARTIFACT has enabled=False.")
    parser.add_argument("--list", action="store_true", help="List registered table builders and exit.")
    parser.add_argument("--strict", action="store_true", help="Fail if any builder fails or if no builders are registered.")
    parser.add_argument("--dry-run", action="store_true", help="List selected builders without running them.")
    parser.add_argument("--clean-output", action="store_true", help="Remove old manifest only; builder outputs are not deleted.")
    args = parser.parse_args()

    project_root = find_project_root(args.project_root or PROJECT_ROOT)
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = project_root / input_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    paper_dir = Path(args.paper_table_dir) if args.paper_table_dir else None
    if paper_dir and not paper_dir.is_absolute():
        paper_dir = project_root / paper_dir

    ensure_dir(output_dir)
    if paper_dir:
        ensure_dir(paper_dir)

    ctx = BuildContext(
        project_root=project_root,
        input_root=input_root,
        output_dir=output_dir,
        paper_dir=paper_dir,
        strict=args.strict,
        dry_run=args.dry_run,
    )

    specs_all = discover_specs("table")
    specs = select_specs(
        specs_all,
        only=parse_csv_list(args.only),
        skip=parse_csv_list(args.skip),
        include_disabled=args.include_disabled,
    )

    if args.list:
        print("Registered table builders:")
        if not specs_all:
            print("  (none yet)")
        for s in specs_all:
            flag = "enabled" if s.enabled else "disabled"
            print(f"  - {s.artifact_id} [{flag}] paper_id={s.paper_id or '-'} module={s.module}")
        return 0

    if args.clean_output:
        manifest_path = output_dir / "tables_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()

    print(f"[TABLES] Project root: {project_root}")
    print(f"[TABLES] Input root:   {input_root}")
    print(f"[TABLES] Output dir:   {output_dir}")
    if paper_dir:
        print(f"[TABLES] Paper dir:    {paper_dir}")
    print(f"[TABLES] Builders:     {len(specs)} selected / {len(specs_all)} registered")

    if args.dry_run:
        for s in specs:
            print(f"[TABLES] DRY-RUN {s.artifact_id}: {s.label}")
        manifest = build_manifest(ctx, specs, [], [], [])
        write_json(output_dir / "tables_manifest.json", manifest)
        return 0

    if not specs:
        msg = "No table builders registered yet. Add modules under src/paper_artifact/tables/."
        print(f"[TABLES] {msg}")
        manifest = build_manifest(ctx, specs, [], [], [])
        manifest["message"] = msg
        write_json(output_dir / "tables_manifest.json", manifest)
        return 1 if args.strict else 0

    results: List[ArtifactResult] = []
    errors = []
    copied_outputs = []

    for i, spec in enumerate(specs, start=1):
        print(f"[TABLES] ({i}/{len(specs)}) {spec.artifact_id}: {spec.label}")
        try:
            result = run_spec(spec, ctx)
            results.append(result)
            print(f"[TABLES]   -> {result.status}: {result.message or len(result.outputs)} output(s)")
            if paper_dir and result.status == "success" and result.outputs:
                copied = copy_outputs(result.outputs, paper_dir)
                copied_outputs.extend(copied)
                print(f"[TABLES]   -> copied {len(copied)} output(s) to paper dir")
        except Exception as exc:
            err = {
                "artifact_id": spec.artifact_id,
                "module": spec.module,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            print(f"[TABLES]   -> failed: {exc}")
            if args.strict:
                break

    manifest = build_manifest(ctx, specs, results, copied_outputs, errors)
    manifest_path = write_json(output_dir / "tables_manifest.json", manifest)
    print(f"[TABLES] Manifest written to: {manifest_path}")

    if errors and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
