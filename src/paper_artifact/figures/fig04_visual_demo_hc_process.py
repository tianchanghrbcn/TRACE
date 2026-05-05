from __future__ import annotations

"""
Stage 3 builder for paper Figure 4.

This module is intended to live at:
    src/paper_artifact/figures/fig04_visual_demo_hc_process.py

It rebuilds the HC process panels for the visual demo.

Input
-----
Preferred visual demo snapshot:

    <ctx.input_root>/visual_demo/customer_segments/demo_results/clustered_data/HC/

Fallback:

    <project_root>/demo_results/clustered_data/HC/

Required folders:

    HC/mode/clustered_2/
    HC/holoclean/clustered_2/
    HC/baran/clustered_2/

Each folder should contain files like:

    *_summary.json
    *_tree_profile.csv
    *_tree.npz

Output
------
Exactly six PDF files under:

    <ctx.output_dir>/figure_4/

Files:

    dendrogram_mode.pdf
    sse_mode.pdf
    dendrogram_holoclean.pdf
    sse_holoclean.pdf
    dendrogram_baran.pdf
    sse_baran.pdf

Trial trajectory panels are intentionally not generated.
No PNG, zip, or auxiliary outputs are generated.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.paper_artifact.io import ArtifactResult, BuildContext


ARTIFACT = {
    "id": "fig04_visual_demo_hc_process",
    "paper_id": "Figure 4",
    "label": "Figure 4: HC visual demo process panels",
    "description": "Build dendrogram and SSE process panels for the visual demo.",
    "enabled": True,
}


warnings.filterwarnings("ignore", category=UserWarning)

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

LABEL_SIZE = 20
TICK_SIZE = 18
LINE_WIDTH = 1.2

FIG_W = 5.0
FIG_H = 3.8
DENDRO_W = 5.0
DENDRO_H = 5.2

CLEANING_ALGOS = ["mode", "holoclean", "baran"]
BUNDLE_DATASET_ID = 2

DENDRO_YLIM_BY_ALGO = {
    "mode": (0, 2),
    "holoclean": (0, 2),
    "baran": (0, 6),
}


def _paths(base_dir: Path, stem: str, state: str) -> dict[str, Path]:
    p = lambda suf: base_dir / f"{stem}_{state}_{suf}"
    return {
        "summary": p("summary.json"),
        "tree_profile": p("tree_profile.csv"),
        "tree_npz": p("tree.npz"),
    }


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _detect_stems_states(folder: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for f in folder.glob("*_summary.json"):
        name = f.name[:-len("_summary.json")]
        if "_" not in name:
            continue
        stem, state = name.rsplit("_", 1)
        mapping.setdefault(stem, [])
        if state not in mapping[stem]:
            mapping[stem].append(state)
    return mapping


def _prefer_state(states: list[str]) -> str | None:
    if "cleaned" in states:
        return "cleaned"
    if "raw" in states:
        return "raw"
    return None


def _style_axis(ax):
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=TICK_SIZE,
        length=4,
        width=1.0,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _save_pdf(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _resolve_root(ctx: BuildContext) -> Path:
    candidates = [
        ctx.input_root / "visual_demo" / "customer_segments" / "demo_results" / "clustered_data" / "HC",
        ctx.input_root / "demo_results" / "clustered_data" / "HC",
        ctx.project_root / "results" / "visual_demo" / "customer_segments" / "demo_results" / "clustered_data" / "HC",
        ctx.project_root / "demo_results" / "clustered_data" / "HC",
    ]

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    raise FileNotFoundError(
        "Cannot find visual-demo HC clustered_data root. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def _resolve_stem_state(base_dir: Path) -> tuple[str, str]:
    mapping = _detect_stems_states(base_dir)
    if not mapping:
        raise FileNotFoundError(f"No *_summary.json found in {base_dir}")

    stem = sorted(mapping.keys())[0]
    state = _prefer_state(mapping[stem])
    if state is None:
        raise FileNotFoundError(f"No cleaned/raw state found in {base_dir}")

    return stem, state


def plot_sse_profile(base_dir: Path, stem: str, state: str, out_path: Path) -> Path:
    paths = _paths(base_dir, stem, state)
    if not (paths["tree_profile"].exists() and paths["summary"].exists()):
        raise FileNotFoundError(f"SSE profile missing files: {paths}")

    prof = pd.read_csv(paths["tree_profile"])

    if "n_clusters" not in prof.columns or "sse_ratio" not in prof.columns:
        raise ValueError(f"SSE profile missing required columns: {paths['tree_profile']}")

    x = prof["n_clusters"].to_numpy(dtype=float)
    y = prof["sse_ratio"].to_numpy(dtype=float)

    positive = y[y > 0]
    if len(positive) == 0:
        raise ValueError(f"SSE profile has no positive sse_ratio: {paths['tree_profile']}")

    min_positive = positive.min()
    y = np.where(y > 0, y, min_positive * 0.5)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(x, y, linewidth=LINE_WIDTH)

    ax.set_yscale("log")
    ax.set_xlim(300, 0)
    ax.set_xticks([300, 250, 200, 150, 100, 50, 0])

    ax.set_xlabel("Number of Clusters k", fontsize=LABEL_SIZE)
    ax.set_ylabel("Relative SSE", fontsize=LABEL_SIZE)

    _style_axis(ax)
    return _save_pdf(fig, out_path)


def plot_dendrogram(base_dir: Path, stem: str, state: str, out_path: Path) -> Path:
    from scipy.cluster.hierarchy import dendrogram

    paths = _paths(base_dir, stem, state)
    if not (paths["tree_npz"].exists() and paths["summary"].exists()):
        raise FileNotFoundError(f"Dendrogram missing files: {paths}")

    npz = np.load(paths["tree_npz"])
    children = npz["children"]
    distances = np.nan_to_num(npz["distances"].astype(float), nan=0.0)
    counts = npz["counts"].astype(float)

    sm = _load_json(paths["summary"])
    k_star = sm.get("best_k", None)

    n_samples = children.shape[0] + 1
    Z = np.zeros((children.shape[0], 4), dtype=float)

    for i, (a, b) in enumerate(children):
        Z[i, 0] = a
        Z[i, 1] = b
        Z[i, 2] = distances[i] if i < len(distances) else 0.0
        Z[i, 3] = counts[n_samples + i] if (n_samples + i) < len(counts) else 2.0

    color_th = None
    if k_star is not None:
        step_cut = max(n_samples - int(k_star) - 1, 0)
        if 0 <= step_cut < Z.shape[0]:
            color_th = Z[step_cut, 2]

    fig, ax = plt.subplots(figsize=(DENDRO_W, DENDRO_H))
    dendrogram(
        Z,
        color_threshold=color_th,
        no_labels=True,
        count_sort=True,
        ax=ax,
    )

    if color_th is not None:
        ax.axhline(color_th, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Samples", fontsize=LABEL_SIZE)
    ax.set_ylabel("Merge Distance", fontsize=LABEL_SIZE)

    algo = base_dir.parent.name.lower()
    y_min, y_max = DENDRO_YLIM_BY_ALGO.get(algo, (0, 6))
    ax.set_ylim(y_min, y_max)

    _style_axis(ax)
    return _save_pdf(fig, out_path)


def build(ctx: BuildContext) -> ArtifactResult:
    root_dir = _resolve_root(ctx)
    out_dir = ctx.output_dir / "figure_4"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()

    outputs: list[Path] = []
    inputs: list[Path] = [root_dir]

    for algo in CLEANING_ALGOS:
        base_dir = root_dir / algo / f"clustered_{BUNDLE_DATASET_ID}"
        if not base_dir.exists():
            raise FileNotFoundError(f"Missing visual demo folder: {base_dir}")

        stem, state = _resolve_stem_state(base_dir)

        outputs.append(
            plot_dendrogram(
                base_dir,
                stem,
                state,
                out_dir / f"dendrogram_{algo}.pdf",
            )
        )
        outputs.append(
            plot_sse_profile(
                base_dir,
                stem,
                state,
                out_dir / f"sse_{algo}.pdf",
            )
        )

    expected_names = {
        "dendrogram_mode.pdf",
        "sse_mode.pdf",
        "dendrogram_holoclean.pdf",
        "sse_holoclean.pdf",
        "dendrogram_baran.pdf",
        "sse_baran.pdf",
    }
    actual_names = {p.name for p in outputs}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise RuntimeError(f"Output file mismatch. Missing={missing}; extra={extra}")

    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="success",
        outputs=outputs,
        inputs=inputs,
        message=f"Built Figure 4 with {len(outputs)} PDF files under {out_dir}.",
        metadata={
            "output_subdir": "figure_4",
            "expected_output_count": 6,
            "actual_output_count": len(outputs),
            "dataset_id": BUNDLE_DATASET_ID,
            "algorithms": CLEANING_ALGOS,
        },
    )
