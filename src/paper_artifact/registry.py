from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from .io import ArtifactResult, BuildContext, normalize_result


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    kind: str  # figure | table
    label: str
    module: str
    function: str = "build"
    paper_id: str = ""
    description: str = ""
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "paper_id": self.paper_id,
            "label": self.label,
            "module": self.module,
            "function": self.function,
            "description": self.description,
            "enabled": self.enabled,
        }


def _iter_modules(package_name: str) -> Iterable[str]:
    package = importlib.import_module(package_name)
    package_path = Path(package.__file__).parent
    for mod in pkgutil.iter_modules([str(package_path)]):
        if mod.name.startswith("_"):
            continue
        yield f"{package_name}.{mod.name}"


def discover_specs(kind: str) -> List[ArtifactSpec]:
    if kind not in {"figure", "table"}:
        raise ValueError(f"Unknown artifact kind: {kind}")

    package_name = "src.paper_artifact.figures" if kind == "figure" else "src.paper_artifact.tables"
    specs: List[ArtifactSpec] = []

    for module_name in _iter_modules(package_name):
        module = importlib.import_module(module_name)
        artifact = getattr(module, "ARTIFACT", None)
        has_build = callable(getattr(module, "build", None))
        if artifact is None and not has_build:
            continue

        if artifact is None:
            artifact = {}
        artifact_id = str(artifact.get("id") or module_name.rsplit(".", 1)[-1])
        enabled = bool(artifact.get("enabled", True))
        specs.append(
            ArtifactSpec(
                artifact_id=artifact_id,
                kind=kind,
                label=str(artifact.get("label") or artifact_id),
                module=module_name,
                function=str(artifact.get("function") or "build"),
                paper_id=str(artifact.get("paper_id") or ""),
                description=str(artifact.get("description") or ""),
                enabled=enabled,
            )
        )

    specs.sort(key=lambda s: (s.paper_id or "zzz", s.artifact_id))
    return specs


def select_specs(
    specs: Sequence[ArtifactSpec],
    only: Optional[Sequence[str]] = None,
    skip: Optional[Sequence[str]] = None,
    include_disabled: bool = False,
) -> List[ArtifactSpec]:
    only_set = set(only or [])
    skip_set = set(skip or [])
    selected: List[ArtifactSpec] = []
    for spec in specs:
        if not include_disabled and not spec.enabled:
            continue
        if only_set and spec.artifact_id not in only_set and spec.paper_id not in only_set:
            continue
        if spec.artifact_id in skip_set or spec.paper_id in skip_set:
            continue
        selected.append(spec)
    return selected


def run_spec(spec: ArtifactSpec, ctx: BuildContext) -> ArtifactResult:
    module = importlib.import_module(spec.module)
    fn = getattr(module, spec.function, None)
    if not callable(fn):
        raise AttributeError(f"Builder {spec.module}.{spec.function} is not callable")

    raw = fn(ctx)
    result = normalize_result(raw, artifact_id=spec.artifact_id)
    if not result.artifact_id:
        result.artifact_id = spec.artifact_id
    return result
