#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe cleaner dependencies for TRACE.

This script scans registered cleaner implementations and reports imported
third-party modules. It can also check whether those modules are importable in a
target conda environment.

The goal is to build cleaner-specific environments from observed dependencies
instead of relying on a large legacy environment.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.method_registry import load_default_registry


IMPORT_TO_PACKAGE = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "mwparserfromhell": "mwparserfromhell",
    "py7zr": "py7zr",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "networkx": "networkx",
    "torch": "torch",
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "requests": "requests",
    "sqlalchemy": "SQLAlchemy",
    "psycopg2": "psycopg2-binary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe cleaner dependencies.")
    parser.add_argument(
        "--cleaners",
        nargs="*",
        default=["group:lightweight"],
        help="Cleaner tokens resolved by configs/methods.yaml.",
    )
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Optional conda environment name used to check imports.",
    )
    parser.add_argument(
        "--output",
        default="results/logs/cleaner_dependency_probe.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Allow probing disabled registry entries if explicitly selected.",
    )
    return parser.parse_args()


def stdlib_modules() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", set()))

    stdlib_path = Path(sysconfig.get_paths().get("stdlib", ""))
    if stdlib_path.exists():
        for path in stdlib_path.glob("*.py"):
            names.add(path.stem)

    names.update(sys.builtin_module_names)
    names.update(
        {
            "__future__",
            "typing",
            "pathlib",
            "dataclasses",
            "argparse",
            "json",
            "os",
            "sys",
            "time",
            "math",
            "re",
            "subprocess",
            "shutil",
            "tempfile",
            "collections",
            "itertools",
            "functools",
            "statistics",
            "copy",
            "pickle",
            "random",
            "warnings",
        }
    )
    return names


STDLIB = stdlib_modules()


def collect_py_files(implementation: Path) -> list[Path]:
    """
    Collect implementation file and sibling Python files.

    This captures local helper modules without scanning the whole repository.
    """
    if implementation.is_dir():
        root = implementation
    else:
        root = implementation.parent

    ignored_parts = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        ".eggs",
    }

    files = []
    for path in root.rglob("*.py"):
        if any(part in ignored_parts for part in path.parts):
            continue
        files.append(path)

    if implementation.is_file() and implementation not in files:
        files.insert(0, implementation)

    return sorted(set(files))


def local_module_names(files: list[Path]) -> set[str]:
    names = set()
    for path in files:
        names.add(path.stem)
        if path.name == "__init__.py":
            names.add(path.parent.name)
    return names


def parse_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
    except SyntaxError:
        return set()

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])

        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                imports.add(node.module.split(".", 1)[0])

    return imports


def classify_imports(files: list[Path]) -> dict[str, Any]:
    local_names = local_module_names(files)

    file_imports: dict[str, list[str]] = {}
    all_imports: set[str] = set()

    for path in files:
        imports = parse_imports(path)
        relative = str(path.relative_to(PROJECT_ROOT))
        file_imports[relative] = sorted(imports)
        all_imports.update(imports)

    third_party = []
    local_or_stdlib = []

    for module in sorted(all_imports):
        if module in STDLIB or module in local_names:
            local_or_stdlib.append(module)
            continue
        if module.startswith("_"):
            local_or_stdlib.append(module)
            continue
        third_party.append(module)

    return {
        "files": [str(path.relative_to(PROJECT_ROOT)) for path in files],
        "file_imports": file_imports,
        "third_party_imports": third_party,
        "local_or_stdlib_imports": sorted(local_or_stdlib),
        "suggested_packages": sorted(
            {IMPORT_TO_PACKAGE.get(module, module) for module in third_party}
        ),
    }


def conda_executable() -> str | None:
    return os.environ.get("CONDA_EXE") or shutil.which("conda")


def check_imports_in_current_env(modules: list[str]) -> dict[str, dict[str, Any]]:
    results = {}

    for module in modules:
        try:
            imported = importlib.import_module(module)
            package = IMPORT_TO_PACKAGE.get(module, module)
            try:
                version = metadata.version(package)
            except Exception:
                version = getattr(imported, "__version__", "unknown")
            results[module] = {
                "status": "ok",
                "package": package,
                "version": version,
            }
        except Exception as exc:
            results[module] = {
                "status": "missing",
                "package": IMPORT_TO_PACKAGE.get(module, module),
                "error": str(exc),
            }

    return results


def check_imports_in_conda_env(env_name: str, modules: list[str]) -> dict[str, dict[str, Any]]:
    conda = conda_executable()
    if not conda:
        raise RuntimeError("Conda executable was not found.")

    code = r"""
import importlib
import importlib.metadata as metadata
import json

modules = __MODULES__
package_map = __PACKAGE_MAP__

results = {}

for module in modules:
    package = package_map.get(module, module)
    try:
        imported = importlib.import_module(module)
        try:
            version = metadata.version(package)
        except Exception:
            version = getattr(imported, "__version__", "unknown")
        results[module] = {"status": "ok", "package": package, "version": version}
    except Exception as exc:
        results[module] = {"status": "missing", "package": package, "error": str(exc)}

print(json.dumps(results))
"""
    code = code.replace("__MODULES__", repr(modules))
    code = code.replace("__PACKAGE_MAP__", repr(IMPORT_TO_PACKAGE))

    completed = subprocess.run(
        [conda, "run", "-n", env_name, "python", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        return {
            "__conda_run__": {
                "status": "error",
                "package": None,
                "error": completed.stderr.strip() or completed.stdout.strip(),
            }
        }

    return json.loads(completed.stdout.strip())


def main() -> None:
    args = parse_args()
    registry = load_default_registry(PROJECT_ROOT)

    specs = registry.resolve("cleaners", args.cleaners)

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "cleaner_tokens": args.cleaners,
        "conda_env": args.conda_env,
        "cleaners": {},
    }

    all_third_party: set[str] = set()
    all_packages: set[str] = set()

    for spec in specs:
        implementation = PROJECT_ROOT / spec.implementation
        files = collect_py_files(implementation)
        info = classify_imports(files)

        all_third_party.update(info["third_party_imports"])
        all_packages.update(info["suggested_packages"])

        report["cleaners"][spec.name] = {
            "id": spec.id,
            "legacy_id": spec.legacy_id,
            "implementation": spec.implementation,
            **info,
        }

    sorted_modules = sorted(all_third_party)

    if args.conda_env:
        import_check = check_imports_in_conda_env(args.conda_env, sorted_modules)
    else:
        import_check = check_imports_in_current_env(sorted_modules)

    missing_packages = sorted(
        {
            row.get("package")
            for row in import_check.values()
            if row.get("status") == "missing" and row.get("package")
        }
    )

    report["aggregate"] = {
        "third_party_imports": sorted_modules,
        "suggested_packages": sorted(all_packages),
        "import_check": import_check,
        "missing_packages": missing_packages,
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[TRACE] Dependency probe written to: {output_path}")
    print("[TRACE] Third-party imports:", ", ".join(sorted_modules))
    print("[TRACE] Suggested packages:", ", ".join(sorted(all_packages)))
    if missing_packages:
        print("[TRACE] Missing packages:", ", ".join(missing_packages))
    else:
        print("[TRACE] Missing packages: none")


if __name__ == "__main__":
    main()