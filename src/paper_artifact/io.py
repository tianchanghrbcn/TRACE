from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class BuildContext:
    """Shared context passed to every paper figure/table builder."""

    project_root: Path
    input_root: Path
    output_dir: Path
    paper_dir: Optional[Path] = None
    config: Dict[str, Any] = field(default_factory=dict)
    strict: bool = False
    dry_run: bool = False

    def resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.project_root / p

    def rel(self, path: str | Path) -> str:
        p = Path(path)
        try:
            return str(p.resolve().relative_to(self.project_root.resolve())).replace("\\", "/")
        except Exception:
            return str(p).replace("\\", "/")


@dataclass
class ArtifactResult:
    """Standard return value for a figure/table builder."""

    artifact_id: str
    status: str = "success"  # success | skipped | failed
    outputs: List[Path] = field(default_factory=list)
    inputs: List[Path] = field(default_factory=list)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_manifest_dict(self, ctx: BuildContext) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "status": self.status,
            "message": self.message,
            "outputs": [ctx.rel(p) for p in self.outputs],
            "inputs": [ctx.rel(p) for p in self.inputs],
            "metadata": self.metadata,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_project_root(start: str | Path | None = None) -> Path:
    """Find repository root by walking upward until scripts/ and configs/ exist."""

    cur = Path(start or Path.cwd()).resolve()
    for candidate in [cur] + list(cur.parents):
        if (candidate / "scripts").exists() and (candidate / "configs").exists():
            return candidate
    return cur


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return p


def copy_outputs(outputs: Iterable[Path], destination: str | Path) -> List[Path]:
    dest = ensure_dir(destination)
    copied: List[Path] = []
    for src in outputs:
        src = Path(src)
        if not src.exists() or src.is_dir():
            continue
        target = dest / src.name
        shutil.copy2(src, target)
        copied.append(target)
    return copied


def normalize_result(raw: Any, artifact_id: str) -> ArtifactResult:
    """Allow builders to return ArtifactResult or a plain dict."""

    if isinstance(raw, ArtifactResult):
        return raw
    if isinstance(raw, dict):
        return ArtifactResult(
            artifact_id=str(raw.get("artifact_id", artifact_id)),
            status=str(raw.get("status", "success")),
            outputs=[Path(p) for p in raw.get("outputs", [])],
            inputs=[Path(p) for p in raw.get("inputs", [])],
            message=str(raw.get("message", "")),
            metadata=dict(raw.get("metadata", {})),
        )
    if raw is None:
        return ArtifactResult(artifact_id=artifact_id, status="success")
    raise TypeError(f"Unsupported builder return value for {artifact_id}: {type(raw)!r}")
