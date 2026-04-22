#!/usr/bin/env python3
"""Build local TRACE release assets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


HANDOFF_DOCS = [
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "data/README.md",
    "docs/advisor_handoff.md",
    "docs/stage1_to_stage3_summary.md",
    "docs/stage1_to_stage4_plan.md",
    "docs/artifact_overview.md",
    "docs/reproducibility_modes.md",
    "docs/data_policy.md",
    "docs/hardware_runtime.md",
    "docs/release_checklist.md",
    "docs/stage2_validation.md",
    "docs/results_replay.md",
    "docs/pre_experiment.md",
    "docs/visual_demo.md",
    "docs/known_limitations.md",
    "docs/runtime_progress.md",
    "docs/terminal_interface.md",
    "docs/release_packaging.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TRACE release assets.")
    parser.add_argument("--version", default="v0.1.1-advisor")
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[TRACE]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def zip_dir(source_dir: Path, output_zip: Path) -> None:
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir))


def main() -> None:
    args = parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    source_zip = out / f"TRACE-{args.version}-source.zip"
    run(["git", "archive", "--format=zip", "--output", str(source_zip), args.source_ref])

    handoff_dir = out / f"TRACE-{args.version}-handoff"
    if handoff_dir.exists():
        shutil.rmtree(handoff_dir)
    handoff_dir.mkdir(parents=True)

    for rel in HANDOFF_DOCS:
        src = ROOT / rel
        if src.exists():
            dst = handoff_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    (handoff_dir / "README_FOR_ADVISOR.txt").write_text(
        "\n".join([
            f"TRACE advisor-review package {args.version}",
            "",
            "Recommended entry:",
            "  python scripts/00_trace_home.py",
            "  python scripts/00_trace_home.py --interactive",
            "  python scripts/98_validate_release_package.py",
            "",
            "Stage 1-3 are complete for advisor review.",
            "Stage 4 remains: TRACE validation code, new algorithm extension test, and new dataset extension test.",
            "",
        ]),
        encoding="utf-8",
    )

    handoff_zip = out / f"TRACE-{args.version}-handoff.zip"
    zip_dir(handoff_dir, handoff_zip)

    print(f"[TRACE] Source archive: {source_zip}")
    print(f"[TRACE] Handoff archive: {handoff_zip}")
    print("[TRACE] Stage 2 strict proof should be copied from the Linux machine if it is not already local.")


if __name__ == "__main__":
    main()

