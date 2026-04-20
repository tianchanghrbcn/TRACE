#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method registry for TRACE cleaning and clustering components.

The registry makes the pipeline easier to extend. New methods should first be
registered in configs/methods.yaml. The runner can then resolve method names,
aliases, numeric ids, legacy ids, and method groups from a single source of
truth.

Current convention:
- id: TRACE-native id, 0-based for both cleaners and clusterers.
- legacy_id: original AutoMLClustering id, used for archived results and
  compatibility with older pipeline logic.
- name: recommended command-line and documentation identifier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import yaml


MethodKind = Literal["cleaners", "clusterers"]


def normalize_name(value: str) -> str:
    """Normalize names and aliases for stable lookup."""
    return str(value).strip().lower().replace("-", "").replace("_", "")


@dataclass(frozen=True)
class MethodSpec:
    """One cleaner or clusterer registered in configs/methods.yaml."""

    kind: MethodKind
    name: str
    id: int
    legacy_id: int
    display_name: str
    enabled: bool
    group: str
    implementation: str
    output_contract: str
    default_env: str
    raw: dict[str, Any]

    @property
    def normalized_name(self) -> str:
        return normalize_name(self.name)

    def implementation_path(self, project_root: str | Path) -> Path:
        return Path(project_root).resolve() / self.implementation


class MethodRegistry:
    """Load and query cleaner/clusterer metadata from configs/methods.yaml."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()

        with self.path.open("r", encoding="utf-8-sig") as f:
            self.data = yaml.safe_load(f) or {}

        self._methods: dict[MethodKind, dict[str, MethodSpec]] = {
            "cleaners": {},
            "clusterers": {},
        }
        self._ids: dict[MethodKind, dict[int, MethodSpec]] = {
            "cleaners": {},
            "clusterers": {},
        }
        self._legacy_ids: dict[MethodKind, dict[int, MethodSpec]] = {
            "cleaners": {},
            "clusterers": {},
        }
        self._aliases: dict[MethodKind, dict[str, str]] = {
            "cleaners": {},
            "clusterers": {},
        }

        self._load()

    def _load(self) -> None:
        for kind in ("cleaners", "clusterers"):
            raw_methods = self.data.get(kind, {}) or {}

            for name, info in raw_methods.items():
                method_id = int(info["id"])
                legacy_id = int(info.get("legacy_id", method_id))

                spec = MethodSpec(
                    kind=kind,
                    name=str(name),
                    id=method_id,
                    legacy_id=legacy_id,
                    display_name=str(info.get("display_name", name)),
                    enabled=bool(info.get("enabled", True)),
                    group=str(info.get("group", "")),
                    implementation=str(info.get("implementation", "")),
                    output_contract=str(info.get("output_contract", "")),
                    default_env=str(info.get("default_env", "")),
                    raw=dict(info),
                )

                key = normalize_name(name)
                if key in self._methods[kind]:
                    raise ValueError(f"Duplicate method name in {kind}: {name}")
                if spec.id in self._ids[kind]:
                    raise ValueError(f"Duplicate TRACE id in {kind}: {spec.id}")
                if spec.legacy_id in self._legacy_ids[kind]:
                    raise ValueError(f"Duplicate legacy id in {kind}: {spec.legacy_id}")

                self._methods[kind][key] = spec
                self._ids[kind][spec.id] = spec
                self._legacy_ids[kind][spec.legacy_id] = spec

        aliases = self.data.get("aliases", {}) or {}
        for kind in ("cleaners", "clusterers"):
            raw_aliases = aliases.get(kind, {}) or {}
            self._aliases[kind] = {
                normalize_name(alias): normalize_name(target)
                for alias, target in raw_aliases.items()
            }

            for alias, target in self._aliases[kind].items():
                if target not in self._methods[kind]:
                    raise ValueError(
                        f"Alias `{alias}` in {kind} points to unknown method `{target}`."
                    )

    def list_methods(self, kind: MethodKind, enabled_only: bool = True) -> list[MethodSpec]:
        methods = list(self._methods[kind].values())
        if enabled_only:
            methods = [method for method in methods if method.enabled]
        return sorted(methods, key=lambda item: item.id)

    def get_by_id(self, kind: MethodKind, method_id: int) -> MethodSpec:
        try:
            return self._ids[kind][int(method_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown {kind[:-1]} TRACE id: {method_id}") from exc

    def get_by_legacy_id(self, kind: MethodKind, legacy_id: int) -> MethodSpec:
        try:
            return self._legacy_ids[kind][int(legacy_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown {kind[:-1]} legacy id: {legacy_id}") from exc

    def get_by_name(self, kind: MethodKind, name: str) -> MethodSpec:
        key = normalize_name(name)
        key = self._aliases[kind].get(key, key)

        try:
            return self._methods[kind][key]
        except KeyError as exc:
            raise KeyError(f"Unknown {kind[:-1]} name: {name}") from exc

    def resolve(self, kind: MethodKind, tokens: Optional[Iterable[str]]) -> list[MethodSpec]:
        """
        Resolve CLI tokens into MethodSpec objects.

        If tokens is None or empty, return all enabled methods. Tokens may be:

        - TRACE ids, for example: 0
        - method names, for example: mode, HC
        - aliases, for example: hierarchical
        - groups, for example: group:full
        - legacy ids, for example: legacy:1
        """
        if not tokens:
            return self.list_methods(kind, enabled_only=True)

        resolved: list[MethodSpec] = []

        for token in tokens:
            raw = str(token).strip()
            if not raw:
                continue

            if raw.startswith("group:"):
                group_name = raw.split(":", 1)[1].strip()
                resolved.extend(self.resolve_group(kind, group_name))
                continue

            if raw.startswith("legacy:"):
                legacy_id = int(raw.split(":", 1)[1].strip())
                resolved.append(self.get_by_legacy_id(kind, legacy_id))
                continue

            if raw.isdigit():
                resolved.append(self.get_by_id(kind, int(raw)))
                continue

            resolved.append(self.get_by_name(kind, raw))

        return deduplicate_specs(resolved)

    def resolve_group(self, kind: MethodKind, group_name: str) -> list[MethodSpec]:
        groups = ((self.data.get("groups", {}) or {}).get(kind, {}) or {})
        if group_name not in groups:
            raise KeyError(f"Unknown {kind[:-1]} group: {group_name}")

        return [self.get_by_name(kind, name) for name in groups[group_name]]

    def ids(self, kind: MethodKind, tokens: Optional[Iterable[str]]) -> list[int]:
        return [method.id for method in self.resolve(kind, tokens)]

    def legacy_ids(self, kind: MethodKind, tokens: Optional[Iterable[str]]) -> list[int]:
        return [method.legacy_id for method in self.resolve(kind, tokens)]

    def names(self, kind: MethodKind, tokens: Optional[Iterable[str]]) -> list[str]:
        return [method.name for method in self.resolve(kind, tokens)]

    def validate_implementation_paths(
        self,
        project_root: str | Path,
        enabled_only: bool = False,
    ) -> list[str]:
        """
        Return missing implementation paths.

        This function does not raise by default because some oracle or optional
        methods may be intentionally disabled.
        """
        root = Path(project_root).resolve()
        missing: list[str] = []

        for kind in ("cleaners", "clusterers"):
            for spec in self.list_methods(kind, enabled_only=enabled_only):
                path = root / spec.implementation
                if not path.exists():
                    missing.append(str(path.relative_to(root)))

        return missing


def deduplicate_specs(specs: Iterable[MethodSpec]) -> list[MethodSpec]:
    seen: set[tuple[str, int]] = set()
    output: list[MethodSpec] = []

    for spec in specs:
        key = (spec.kind, spec.id)
        if key in seen:
            continue
        seen.add(key)
        output.append(spec)

    return output


def load_default_registry(project_root: str | Path | None = None) -> MethodRegistry:
    if project_root is None:
        root = Path(__file__).resolve().parents[2]
    else:
        root = Path(project_root).resolve()

    return MethodRegistry(root / "configs" / "methods.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TRACE method registry.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="TRACE project root. Default: current directory.",
    )
    parser.add_argument(
        "--kind",
        choices=["all", "cleaners", "clusterers"],
        default="all",
        help="Which method kind to print.",
    )
    parser.add_argument(
        "--show-disabled",
        action="store_true",
        help="Include disabled methods such as GroundTruth.",
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Check implementation paths.",
    )
    return parser.parse_args()


def _print_methods(registry: MethodRegistry, kind: MethodKind, show_disabled: bool) -> None:
    print(f"{kind}:")
    for spec in registry.list_methods(kind, enabled_only=not show_disabled):
        print(
            f"  id={spec.id:<2} "
            f"legacy_id={spec.legacy_id:<2} "
            f"name={spec.name:<12} "
            f"enabled={str(spec.enabled):<5} "
            f"impl={spec.implementation}"
        )


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    registry = load_default_registry(root)

    if args.kind in ("all", "cleaners"):
        _print_methods(registry, "cleaners", args.show_disabled)

    if args.kind in ("all", "clusterers"):
        _print_methods(registry, "clusterers", args.show_disabled)

    print("groups:")
    print("  cleaner smoke:", registry.names("cleaners", ["group:smoke"]))
    print("  cleaner full:", registry.names("cleaners", ["group:full"]))
    print("  clusterer smoke:", registry.names("clusterers", ["group:smoke"]))
    print("  clusterer full:", registry.names("clusterers", ["group:full"]))

    if args.check_paths:
        missing = registry.validate_implementation_paths(root, enabled_only=False)
        if missing:
            print("missing implementation paths:")
            for path in missing:
                print(f"  {path}")
            raise SystemExit(1)
        print("implementation paths: OK")


if __name__ == "__main__":
    main()
