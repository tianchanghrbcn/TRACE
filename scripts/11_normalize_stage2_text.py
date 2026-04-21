#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from pathlib import Path


CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
TARGET_EXTENSIONS = {".py", ".md", ".txt", ".yaml", ".yml", ".sh"}
SKIP_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    ".pytest_cache",
    "results",
    "datasets",
    "data",
    "figures",
    "artifact",
    "legacy",
}


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def replace_cjk_runs(text: str, replacement: str = "text") -> str:
    return CJK_RE.sub(replacement, text)


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


def normalize_python_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8-sig", errors="replace")
    if not has_cjk(original):
        return False

    try:
        tokens = []
        reader = io.StringIO(original).readline

        for token in tokenize.generate_tokens(reader):
            token_text = token.string

            if token.type == tokenize.COMMENT and has_cjk(token_text):
                token_text = "# Legacy implementation note."
                tokens.append((token.type, token_text))
                continue

            if token.type == tokenize.STRING and has_cjk(token_text):
                # Preserve string syntax as much as possible while removing CJK text.
                token_text = replace_cjk_runs(token_text, "text")
                tokens.append((token.type, token_text))
                continue

            tokens.append((token.type, token_text))

        updated = tokenize.untokenize(tokens)
    except Exception as exc:
        print(f"[WARN] Token-level rewrite failed for {path}: {exc}")
        updated = normalize_text_file_content(original)

    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True

    return False


def normalize_text_file_content(text: str) -> str:
    lines = []

    for line in text.splitlines():
        if not has_cjk(line):
            lines.append(line)
            continue

        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        if stripped.startswith("#"):
            lines.append(indent + "# Legacy implementation note.")
        elif stripped.startswith("//"):
            lines.append(indent + "// Legacy implementation note.")
        else:
            lines.append(replace_cjk_runs(line, "text"))

    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def normalize_text_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8-sig", errors="replace")
    if not has_cjk(original):
        return False

    updated = normalize_text_file_content(original)

    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True

    return False


def scan_remaining(files: list[Path]) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []

    for path in files:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            if has_cjk(line):
                hits.append((path, line_no, line.strip()))

    return hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize Stage-2-and-earlier source text by removing CJK characters."
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
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, only reports remaining CJK.",
    )
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    files = iter_files(roots)

    print(f"[TRACE] Files selected: {len(files)}")

    changed: list[Path] = []

    if args.apply:
        for path in files:
            if path.name == "11_normalize_stage2_text.py":
                continue

            if path.suffix == ".py":
                did_change = normalize_python_file(path)
            else:
                did_change = normalize_text_file(path)

            if did_change:
                changed.append(path)

        print(f"[TRACE] Files changed: {len(changed)}")
        for path in changed:
            print(f"[CHANGED] {path}")

    hits = scan_remaining(files)

    # Ignore this script itself because it intentionally contains Unicode-range logic.
    hits = [
        hit for hit in hits
        if hit[0].name != "11_normalize_stage2_text.py"
    ]

    if hits:
        print("[TRACE] Remaining CJK lines:")
        for path, line_no, line in hits[:300]:
            print(f"{path}:{line_no}: {line}")
        if len(hits) > 300:
            print(f"[TRACE] ... {len(hits) - 300} more line(s)")
        return 1

    print("[TRACE] Stage-2 source text normalization passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
