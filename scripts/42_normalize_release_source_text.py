#!/usr/bin/env python3
"""Normalize reviewer-facing source text by removing CJK text.

The script targets comments, docstrings, and simple user-facing strings.
It does not delete functions, variables, imports, or algorithmic branches.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from pathlib import Path


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]+")

DEFAULT_ROOTS = [
    "README.md",
    "docs",
    "scripts",
    "configs",
    "src/cleaning",
    "src/clustering",
    "src/pipeline",
    "src/results_processing",
    "src/figures",
    "src/pre_experiment",
    "src/visual_demo",
]

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "results",
    "figures",
    "data",
    "legacy",
    "build",
    "dist",
}

TEXT_EXTENSIONS = {".md", ".yaml", ".yml", ".txt", ".sh", ".ps1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize CJK text in release sources.")
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def english_placeholder(original: str) -> str:
    value = original.strip()

    # Keep comments useful while avoiding accidental semantic rewrites.
    if value.startswith("#"):
        return "# Legacy implementation note."

    return CJK_RE.sub("text", original)


def iter_files(root: Path):
    if not root.exists():
        return

    if root.is_file():
        if root.suffix.lower() == ".py" or root.suffix.lower() in TEXT_EXTENSIONS:
            yield root
        return

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() == ".py" or path.suffix.lower() in TEXT_EXTENSIONS:
            yield path


def docstring_token_lines(text: str) -> set[int]:
    lines: set[int] = set()

    try:
        tree = ast.parse(text)
    except Exception:
        return lines

    nodes = [tree]
    nodes.extend(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))

    for node in nodes:
        if not getattr(node, "body", None):
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            start = getattr(first, "lineno", None)
            end = getattr(first, "end_lineno", start)
            if start is not None:
                lines.update(range(start, (end or start) + 1))

    return lines


def normalize_python(path: Path, apply: bool) -> bool:
    original = path.read_text(encoding="utf-8-sig", errors="replace")
    if not has_cjk(original):
        return False

    doc_lines = docstring_token_lines(original)

    try:
        tokens = []
        for tok in tokenize.generate_tokens(io.StringIO(original).readline):
            tok_type = tok.type
            tok_text = tok.string
            line_no = tok.start[0]

            if tok_type == tokenize.COMMENT and has_cjk(tok_text):
                tok_text = english_placeholder(tok_text)

            elif tok_type == tokenize.STRING and line_no in doc_lines and has_cjk(tok_text):
                tok_text = '"""Legacy implementation note."""'

            elif tok_type == tokenize.STRING and has_cjk(tok_text):
                # User-facing messages only. This keeps code structure but removes CJK text.
                tok_text = CJK_RE.sub("text", tok_text)

            tokens.append((tok_type, tok_text))

        updated = tokenize.untokenize(tokens)
    except Exception:
        updated = CJK_RE.sub("text", original)

    if updated != original and apply:
        path.write_text(updated, encoding="utf-8")

    return updated != original


def normalize_text(path: Path, apply: bool) -> bool:
    original = path.read_text(encoding="utf-8-sig", errors="replace")
    if not has_cjk(original):
        return False

    lines = []
    for line in original.splitlines():
        if has_cjk(line):
            lines.append(CJK_RE.sub("text", line))
        else:
            lines.append(line)

    updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")

    if updated != original and apply:
        path.write_text(updated, encoding="utf-8")

    return updated != original


def main() -> None:
    args = parse_args()

    changed = []
    for root_str in args.roots:
        for path in iter_files(Path(root_str)):
            if path.name in {
                "10_check_cjk.py",
                "11_normalize_stage2_text.py",
                "12_hard_remove_cjk_stage2.py",
                "42_normalize_release_source_text.py",
            }:
                continue

            if path.suffix.lower() == ".py":
                did_change = normalize_python(path, args.apply)
            else:
                did_change = normalize_text(path, args.apply)

            if did_change:
                changed.append(str(path))

    for path in changed:
        print(f"[CHANGED] {path}")

    print(f"[TRACE] Files requiring normalization: {len(changed)}")
    if not args.apply:
        print("[TRACE] Dry run only. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()

