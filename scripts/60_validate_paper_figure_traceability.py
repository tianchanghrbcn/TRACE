#!/usr/bin/env python3
"""Validate paper-figure traceability for Mode A.

This script links three sources:

1. LaTeX-referenced archived paper figures.
2. Selected archived/reference figure outputs.
3. Generated outputs captured by the paper figure harness.

It does not claim pixel-identical equivalence. For figures, exact hash matches
are rare because PDF metadata, fonts, and rendering environments can differ.
The goal of this step is traceability and reference coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FIGURE_EXTENSIONS = {".pdf", ".png", ".svg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paper figure traceability.")
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=Path("analysis/paper_figure_audit/paper_figure_source_selection.csv"),
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path("analysis/paper_generated/paper_figures/paper_figure_script_run_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper_generated/paper_figures/paper_figure_traceability_report.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def norm_stem(value: str) -> str:
    stem = Path(value).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def norm_path_key(value: str) -> str:
    text = value.replace("\\", "/").lower()
    text = re.sub(r"[^a-z0-9/_.-]+", "_", text)
    return text


def selected_archived_figures(selection_rows: list[dict[str, str]]) -> list[dict]:
    out = []

    for row in selection_rows:
        group = row.get("selection_group", "")
        if group not in {
            "paper_figures_latex",
            "paper_figures_reference",
            "paper_figures_word_screenshot",
        }:
            continue

        src = Path(row["root"]) / row["relative_path"]
        if not src.exists() or src.suffix.lower() not in FIGURE_EXTENSIONS:
            continue

        out.append(
            {
                "source_kind": group,
                "path": str(src),
                "relative_path": row["relative_path"],
                "file_name": src.name,
                "stem": norm_stem(src.name),
                "path_key": norm_path_key(row["relative_path"]),
                "extension": src.suffix.lower(),
                "sha256": sha256_file(src),
                "size_bytes": src.stat().st_size,
            }
        )

    return out


def selected_tex_refs(selection_rows: list[dict[str, str]]) -> list[dict]:
    refs = []

    for row in selection_rows:
        if row.get("selection_group") != "paper_tex":
            continue

        raw = row.get("latex_references", "")
        if not raw:
            continue

        for ref in raw.split(" || "):
            ref = ref.strip()
            if not ref:
                continue

            refs.append(
                {
                    "tex_file": row.get("relative_path", ""),
                    "reference": ref,
                    "stem": norm_stem(ref),
                    "path_key": norm_path_key(ref),
                }
            )

    # Deduplicate.
    seen = set()
    out = []
    for row in refs:
        key = (row["tex_file"], row["reference"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    return out


def generated_figures(run_manifest: dict) -> list[dict]:
    out = []

    for item in run_manifest.get("collected_outputs", []):
        output_path = Path(item.get("figure_output_path") or item.get("output_path", ""))
        if not output_path.exists() or output_path.suffix.lower() not in FIGURE_EXTENSIONS:
            continue

        out.append(
            {
                "script": item.get("script", ""),
                "path": str(output_path),
                "file_name": output_path.name,
                "stem": norm_stem(output_path.name),
                "path_key": norm_path_key(str(output_path)),
                "extension": output_path.suffix.lower(),
                "sha256": sha256_file(output_path),
                "size_bytes": output_path.stat().st_size,
            }
        )

    seen = set()
    unique = []
    for row in out:
        key = row["path"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    return unique


def build_index(rows: list[dict]) -> dict[str, list[dict]]:
    index = defaultdict(list)
    for row in rows:
        index[row["stem"]].append(row)
    return dict(index)


def match_tex_refs(tex_refs: list[dict], archived: list[dict], generated: list[dict]) -> list[dict]:
    archived_by_stem = build_index(archived)
    generated_by_stem = build_index(generated)

    rows = []

    for ref in tex_refs:
        stem = ref["stem"]
        archive_matches = archived_by_stem.get(stem, [])
        generated_matches = generated_by_stem.get(stem, [])

        # If exact stem matching fails, allow containment matching.
        if not archive_matches:
            archive_matches = [
                row for row in archived
                if stem and (stem in row["stem"] or row["stem"] in stem)
            ]

        if not generated_matches:
            generated_matches = [
                row for row in generated
                if stem and (stem in row["stem"] or row["stem"] in stem)
            ]

        status = "PASS"
        if not archive_matches and not generated_matches:
            status = "FAIL_NO_REFERENCE"
        elif not archive_matches:
            status = "WARN_NO_ARCHIVED_REFERENCE"
        elif not generated_matches:
            status = "WARN_ARCHIVED_ONLY"
        else:
            status = "PASS_ARCHIVED_AND_GENERATED"

        rows.append(
            {
                "tex_file": ref["tex_file"],
                "reference": ref["reference"],
                "stem": stem,
                "status": status,
                "archived_match_count": len(archive_matches),
                "generated_match_count": len(generated_matches),
                "archived_matches": archive_matches[:10],
                "generated_matches": generated_matches[:10],
            }
        )

    return rows


def match_generated_to_archived(generated: list[dict], archived: list[dict]) -> list[dict]:
    archived_by_stem = build_index(archived)

    rows = []
    for gen in generated:
        candidates = archived_by_stem.get(gen["stem"], [])

        exact = [
            row for row in candidates
            if row.get("sha256") == gen.get("sha256")
        ]

        if exact:
            status = "EXACT"
            best = exact[0]
        elif candidates:
            status = "NAME_MATCH"
            best = candidates[0]
        else:
            # containment fallback
            candidates = [
                row for row in archived
                if gen["stem"] and (gen["stem"] in row["stem"] or row["stem"] in gen["stem"])
            ]
            if candidates:
                status = "FUZZY_NAME_MATCH"
                best = candidates[0]
            else:
                status = "GENERATED_ONLY"
                best = {}

        rows.append(
            {
                "generated_path": gen["path"],
                "script": gen["script"],
                "file_name": gen["file_name"],
                "stem": gen["stem"],
                "status": status,
                "best_archived_reference": best.get("path", ""),
                "generated_size_bytes": gen["size_bytes"],
                "archived_size_bytes": best.get("size_bytes", ""),
            }
        )

    return rows


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Paper Figure Traceability Report",
        "",
        f"- Generated at UTC: {report['generated_at_utc']}",
        f"- Status: {report['status']}",
        f"- LaTeX references: {report['tex_reference_count']}",
        f"- Archived figures: {report['archived_figure_count']}",
        f"- Generated figures: {report['generated_figure_count']}",
        "",
        "## LaTeX reference status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]

    for key, value in sorted(report["tex_reference_status_counts"].items()):
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Generated-to-archived status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]

    for key, value in sorted(report["generated_match_status_counts"].items()):
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## LaTeX references needing attention",
        "",
    ]

    attention = [
        row for row in report["tex_reference_matches"]
        if row["status"] != "PASS_ARCHIVED_AND_GENERATED"
    ]

    if attention:
        for row in attention[:80]:
            lines.append(
                f"- `{row['reference']}`: {row['status']} "
                f"(archived={row['archived_match_count']}, generated={row['generated_match_count']})"
            )
    else:
        lines.append("No LaTeX figure references need attention.")

    lines += [
        "",
        "## Scope note",
        "",
        "This report validates figure traceability. It does not claim pixel-identical equivalence.",
        "Exact visual equivalence requires a later source-data or rendered-image comparison step.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    selection_rows = read_csv_rows(args.selection_csv)
    manifest = read_json(args.run_manifest)

    archived = selected_archived_figures(selection_rows)
    tex_refs = selected_tex_refs(selection_rows)
    generated = generated_figures(manifest)

    tex_matches = match_tex_refs(tex_refs, archived, generated)
    gen_matches = match_generated_to_archived(generated, archived)

    tex_counts = Counter(row["status"] for row in tex_matches)
    gen_counts = Counter(row["status"] for row in gen_matches)

    failures = [
        row for row in tex_matches
        if row["status"] == "FAIL_NO_REFERENCE"
    ]

    status = "PASS"
    if failures:
        status = "FAIL"
    elif any(row["status"].startswith("WARN") for row in tex_matches):
        status = "PASS_WITH_WARNINGS"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "selection_csv": str(args.selection_csv),
        "run_manifest": str(args.run_manifest),
        "tex_reference_count": len(tex_refs),
        "archived_figure_count": len(archived),
        "generated_figure_count": len(generated),
        "tex_reference_status_counts": dict(tex_counts),
        "generated_match_status_counts": dict(gen_counts),
        "tex_reference_matches": tex_matches,
        "generated_to_archived_matches": gen_matches,
        "failures": failures,
        "note": (
            "PASS means every LaTeX figure reference has at least archived traceability. "
            "PASS_WITH_WARNINGS means no missing paper figure reference, but some references are archived-only or generated-only."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    write_markdown(args.output.with_suffix(".md"), report)

    print(json.dumps(
        {
            "status": report["status"],
            "tex_reference_count": report["tex_reference_count"],
            "archived_figure_count": report["archived_figure_count"],
            "generated_figure_count": report["generated_figure_count"],
            "tex_reference_status_counts": report["tex_reference_status_counts"],
            "generated_match_status_counts": report["generated_match_status_counts"],
            "output": str(args.output),
        },
        indent=2,
        ensure_ascii=False,
    ))

    print(f"[TRACE] Paper figure traceability report written to: {args.output}")
    print(f"[TRACE] Paper figure traceability status: {report['status']}")

    raise SystemExit(0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1)


if __name__ == "__main__":
    main()

