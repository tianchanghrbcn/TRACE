# Flowchart for Fig. 4-1 (English labels) — uses only matplotlib (no seaborn; no explicit colors).
# Run this script to produce 'fig4_1_architecture.pdf' (and a PNG twin).
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

def add_rect(ax, cx, cy, w, h, text, fontsize=10, lw=1.5):
    """Add a rounded rectangle node with centered text."""
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw, fill=False
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)
    return rect

def add_diamond(ax, cx, cy, w, h, text, fontsize=10, lw=1.5):
    """Add a diamond (decision) node with centered text."""
    verts = [(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)]
    poly = Polygon(verts, closed=True, linewidth=lw, fill=False)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)
    return poly

def add_arrow(ax, x0, y0, x1, y1, lw=1.2):
    """Add a directed arrow from (x0,y0) to (x1,y1)."""
    arr = FancyArrowPatch((x0, y0), (x1, y1),
                          arrowstyle="->", mutation_scale=12, linewidth=lw)
    ax.add_patch(arr)
    return arr

def draw_flowchart(save_pdf="fig4_1_architecture.pdf", save_png="fig4_1_architecture.png"):
    # Canvas
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Node geometry
    W, H = 0.16, 0.11
    W_small, H_small = 0.16, 0.11
    W_d, H_d = 0.16, 0.13  # diamond size

    # Top row (evaluation sub-pipeline)
    nodes = {}
    nodes["data"]   = add_rect(ax, 0.08, 0.70, W, H, "Data Intake\n& Binning")
    nodes["clean"]  = add_rect(ax, 0.26, 0.70, W, H, "Cleaner\nExecutor")
    nodes["proc"]   = add_rect(ax, 0.44, 0.70, W, H, "Process\nLogger")
    nodes["eval"]   = add_rect(ax, 0.62, 0.70, W, H, "External\nEvaluator")
    nodes["agg"]    = add_rect(ax, 0.80, 0.70, W, H, "Result\nAggregator")
    nodes["export"] = add_rect(ax, 0.92, 0.70, 0.12, H_small, "Exporter\n(Logs/Report)")

    # Bottom row (AutoML control)
    nodes["cand"]   = add_rect(ax, 0.26, 0.30, W, H, "Candidate Generator\n(Default branches)")
    nodes["sched"]  = add_rect(ax, 0.44, 0.30, W, H, "Search Coordinator\n(Budget-fair)")
    nodes["gate"]   = add_diamond(ax, 0.62, 0.30, W_d, H_d, "Process Gate\n(Early-stop / Rollback / Switch)")

    # Horizontal arrows (top row)
    add_arrow(ax, 0.16, 0.70, 0.18, 0.70)  # data -> clean (short nudge)
    add_arrow(ax, 0.34, 0.70, 0.36, 0.70)  # clean -> proc
    add_arrow(ax, 0.52, 0.70, 0.54, 0.70)  # proc -> eval
    add_arrow(ax, 0.70, 0.70, 0.72, 0.70)  # eval -> agg
    add_arrow(ax, 0.88, 0.70, 0.90, 0.70)  # agg -> export

    # Control-plane arrows (bottom row)
    add_arrow(ax, 0.08, 0.64, 0.26, 0.36)  # data -> candidate generator (diagonal)
    add_arrow(ax, 0.26, 0.30, 0.36, 0.30)  # cand -> sched
    add_arrow(ax, 0.44, 0.36, 0.26, 0.64)  # sched -> cleaner (loop into evaluation)
    add_arrow(ax, 0.62, 0.64, 0.62, 0.36)  # evaluator -> gate (vertical)
    add_arrow(ax, 0.62, 0.30, 0.52, 0.30)  # gate -> sched (feedback)

    # Optional legends (kept minimal to avoid redundancy)
    ax.text(0.08, 0.77, "Evidence chain", ha="center", va="center", fontsize=9)
    ax.text(0.44, 0.22, "AutoML control loop", ha="center", va="center", fontsize=9)

    # Save outputs
    fig.tight_layout()
    fig.savefig(save_pdf, bbox_inches="tight")
    fig.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    draw_flowchart()
