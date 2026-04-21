#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check source files for Chinese/CJK characters."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["src/cleaning"],
        help="Directories or files to scan.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".py", ".md", ".txt", ".yaml", ".yml"],
        help="File extensions to scan.",
    )
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Exit with non-zero status if CJK characters are found.",
    )
    return parser.parse_args()


def iter_files(root: Path, extensions: set[str]):
    if root.is_file():
        if root.suffix in extensions:
            yield root
        return

    for path in root.rglob("*"):
        if path.is_file() and path.suffix in extensions:
            if "__pycache__" in path.parts:
                continue
            yield path


def main() -> int:
    args = parse_args()
    extensions = set(args.extensions)
    hits = []

    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists():
            continue

        for path in iter_files(root, extensions):
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                if CJK_PATTERN.search(line):
                    hits.append((path, line_no, line.strip()))

    for path, line_no, line in hits:
        print(f"{path}:{line_no}: {line}")

    if hits:
        print(f"[TRACE] CJK check found {len(hits)} line(s).")
        return 1 if args.fail else 0

    print("[TRACE] CJK check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
