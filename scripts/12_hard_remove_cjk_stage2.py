#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Han ideographs + CJK compatibility ideographs + common CJK punctuation/fullwidth forms.
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]+")

TARGET_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".sh",
}

SKIP_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    "results",
    "datasets",
    "data",
    "figures",
    "artifact",
    "legacy",
}


def iter_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []

    for root in roots:
        if not root.exists():
            continue

        if root.is_file():
            if root.suffix in TARGET_EXTENSIONS:
                files.append(root)
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix not in TARGET_EXTENSIONS:
                continue
            files.append(path)

    return sorted(set(files))


def normalize_content(text: str) -> str:
    # Replace CJK text runs with a neutral English placeholder.
    text = CJK_RE.sub("text", text)

    # Clean up common artifacts caused by replacing inline notes.
    text = text.replace("# text", "# Legacy implementation note.")
    text = text.replace("// text", "// Legacy implementation note.")

    # Avoid repeating the replacement phrase too many times on one line.
    text = re.sub(r"(text\s*){4,}", "text", text)

    return text


def scan(files: list[Path]) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []

    for path in files:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            if CJK_RE.search(line):
                hits.append((path, line_no, line.strip()))

    return hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove CJK text from Stage-2-and-earlier source files."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "src/cleaning",
            "src/clustering",
            "src/pipeline",
            "scripts",
            "configs",
            "README.md",
            "docs",
        ],
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    roots = [Path(x) for x in args.roots]
    files = iter_files(roots)

    changed: list[Path] = []

    if args.apply:
        for path in files:
            if path.name in {
                "10_check_cjk.py",
                "11_normalize_stage2_text.py",
                "12_hard_remove_cjk_stage2.py",
            }:
                continue

            old = path.read_text(encoding="utf-8-sig", errors="replace")
            new = normalize_content(old)

            if new != old:
                path.write_text(new, encoding="utf-8")
                changed.append(path)

        print(f"[TRACE] Files changed: {len(changed)}")
        for path in changed:
            print(f"[CHANGED] {path}")

    hits = scan(files)
    hits = [
        item for item in hits
        if item[0].name not in {
            "10_check_cjk.py",
            "11_normalize_stage2_text.py",
            "12_hard_remove_cjk_stage2.py",
        }
    ]

    if hits:
        print("[TRACE] Remaining CJK lines:")
        for path, line_no, line in hits[:300]:
            print(f"{path}:{line_no}: {line}")
        if len(hits) > 300:
            print(f"[TRACE] ... {len(hits) - 300} more line(s)")
        return 1

    print("[TRACE] Stage-2 source CJK cleanup passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
