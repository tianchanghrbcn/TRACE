#!/usr/bin/env python3
"""Framework-level TRACE figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.figures.style import save_figure


def plot_trace_layer_diagram(output_root: Path) -> dict[str, str]:
    """Create a lightweight TRACE layer diagram."""
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.axis("off")

    boxes = [
        ("Data", 0.10),
        ("Process", 0.33),
        ("Results", 0.56),
        ("Hyper-parameters", 0.79),
    ]

    for label, x in boxes:
        rect = plt.Rectangle((x, 0.45), 0.17, 0.25, fill=False, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + 0.085, 0.575, label, ha="center", va="center", fontsize=10)

    for _, x in boxes[:-1]:
        ax.annotate(
            "",
            xy=(x + 0.22, 0.575),
            xytext=(x + 0.17, 0.575),
            arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        )

    ax.text(
        0.5,
        0.24,
        "TRACE organizes cleaning effects as data-level, process-level, result-level, and hyperparameter-level evidence.",
        ha="center",
        va="center",
        wrap=True,
        fontsize=9,
    )

    ax.set_title("TRACE Four-Level Analysis Framework")
    return save_figure(fig, output_root, "framework_trace_layers")

