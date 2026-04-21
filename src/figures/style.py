#!/usr/bin/env python3
"""Shared figure style and save helpers for TRACE."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


DEFAULT_DPI = 200
DEFAULT_FIGSIZE = (7.0, 4.2)


def prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    """Create standard figure output directories."""
    root = Path(output_root)
    png_dir = root / "png"
    pdf_dir = root / "pdf"

    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    return {"png": png_dir, "pdf": pdf_dir}


def apply_axis_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply a simple, reviewer-friendly axis style."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)


def save_figure(fig, output_root: Path, stem: str) -> dict[str, str]:
    """Save one figure as PNG and PDF."""
    dirs = prepare_output_dirs(output_root)

    png_path = dirs["png"] / f"{stem}.png"
    pdf_path = dirs["pdf"] / f"{stem}.pdf"

    fig.tight_layout()
    fig.savefig(png_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {"png": str(png_path), "pdf": str(pdf_path)}


def shorten_labels(labels: Iterable[str], max_len: int = 28) -> list[str]:
    """Shorten long labels for compact figures."""
    out = []
    for label in labels:
        value = str(label)
        if len(value) > max_len:
            value = value[: max_len - 3] + "..."
        out.append(value)
    return out

