"""
Training curves for the paper, from the saved training histories (no models are run).

    Figure 1  SegFormer-B2 fine-tuning: validation mIoU / risk mIoU per epoch, and training loss
              (final model = cross-entropy, vs the focal + class-balanced variant)
    Figure 2  UniDepthV2 fine-tuning: validation AbsRel per epoch, overall and per distance range, and training loss
              (final model vs the distance-weighted variant)
    Figure 3  Learned fusion (point 3): validation AP per epoch for the three seeds, and training loss

Output: results/figures/fig*_*.png (300 dpi) and .pdf

Run (any Python with matplotlib, from the project folder):
    python code/plot_training.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "figures"
# Categorical slots in fixed order (validated palette): 1 blue, 2 orange, 3 aqua.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#262624", "#5f5e5a", "#e4e2dc"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "savefig.dpi": 300,
    "figure.dpi": 110, "font.family": "DejaVu Sans"})


def load(run):
    h = json.load(open(HERE / "runs" / run / "history.json"))
    return h if isinstance(h, list) else h.get("history", h)


def series(h, key):
    pts = [(r["epoch"], r[key]) for r in h if r.get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def line(ax, x, y, color, label, marker="o"):
    ax.plot(x, y, color=color, label=label, marker=marker, markersize=4.5, markeredgecolor="white", markeredgewidth=0.8)


def best_mark(ax, x, y, color, mode="max", fmt="{:.3f}", at=None, below=False, dx=0):
    """Circle one point (the best, or the epoch `at`) and write its value above or below it."""
    if at is not None:
        i = x.index(at)
    else:
        i = max(range(len(y)), key=lambda j: y[j]) if mode == "max" else min(range(len(y)), key=lambda j: y[j])
    ax.plot([x[i]], [y[i]], marker="o", markersize=8, markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.6)
    ax.annotate(fmt.format(y[i]), (x[i], y[i]), textcoords="offset points", xytext=(dx, -14 if below else 8),
                ha="center", fontsize=8, color=INK)


def headroom(ax, frac=0.12):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - frac * (hi - lo), hi + frac * (hi - lo))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


def fig_segmentation():
    a, b = load("seg_b2_native"), load("seg_b2_focal")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, key, title in ((axes[0], "risk_mIoU", "Validation risk mIoU (8 risk classes)"), (axes[1], "mIoU", "Validation mIoU (19 classes)")):
        for h, c, lab in ((a, C1, "cross-entropy (final)"), (b, C2, "focal + class-balanced")):
            x, y = series(h, key)
            line(ax, x, y, c, lab)
            best_mark(ax, x, y, c, below=(c == C2))
        headroom(ax)
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
    axes[0].legend(frameon=False, loc="lower right")
    ax = axes[2]
    for h, c, lab in ((a, C1, "cross-entropy"), (b, C2, "focal + class-balanced")):
        x, y = series(h, "loss")
        line(ax, x, y, c, lab, marker=None)
    ax.set_title("Training loss (different losses: compare shapes, not levels)", loc="left")
    ax.set_xlabel("epoch")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.suptitle("Figure 1. SegFormer-B2 fine-tuning on nuScenes front camera (epoch 0 = Cityscapes model before training; circle = peak)",
                 x=0.01, ha="left", fontsize=10, y=1.04)
    save(fig, "fig1_segformer_training")


def fig_depth():
    a, b = load("depth_s"), load("depth_s_far")
    panels = [("all", "Overall"), ("0-20 m", "0-20 m"), ("20-50 m", "20-50 m"), ("50-80 m", "50-80 m")]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 2.9))
    # The saved model (best.pt) is the epoch with the lowest OVERALL AbsRel; mark that same epoch in every panel.
    saved = {id(h): min(h, key=lambda r: r["val"]["all"]["AbsRel"])["epoch"] for h in (a, b)}
    for ax, (key, title) in zip(axes[:4], panels):
        # value of each run at its saved epoch: the lower point gets its label below, the higher one above
        at_saved = {id(h): next(r["val"][key]["AbsRel"] for r in h if r["epoch"] == saved[id(h)]) for h in (a, b)}
        for h, other, c, lab in ((a, b, C1, "standard loss (final model)"), (b, a, C2, "distance-weighted loss, far x3")):
            x = [r["epoch"] for r in h]
            y = [r["val"][key]["AbsRel"] for r in h]
            line(ax, x, y, c, lab)
            best_mark(ax, x, y, c, at=saved[id(h)], below=at_saved[id(h)] <= at_saved[id(other)])
        headroom(ax, 0.15)
        ax.set_title(f"Validation AbsRel, {title}", loc="left")
        ax.set_xlabel("epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.06))
    ax = axes[4]
    for h, c, lab in ((a, C1, "standard"), (b, C2, "distance-weighted")):
        x, y = series(h, "train_loss")
        line(ax, x, y, c, lab, marker=None)
    ax.set_title("Training loss (weighted loss is larger by design)", loc="left")
    ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.suptitle("Figure 2. UniDepthV2-small fine-tuning, LiDAR depth truth (lower is better; epoch 0 = pretrained; circle = saved model)",
                 x=0.01, ha="left", fontsize=10, y=1.06)
    save(fig, "fig2_unidepth_training")


def fig_fusion():
    f = HERE / "runs" / "fusion" / "fusion_val.json"
    if not f.exists():
        print("no runs/fusion/fusion_val.json, skipping figure 3")
        return
    logs = json.load(open(f))["training"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.9))
    for (seed, log), c in zip(sorted(logs.items()), (C1, C2, C3)):
        h = log["history"]
        line(axes[0], [r["epoch"] for r in h], [r["val_AP"] for r in h], c, f"seed {seed}", marker=None)
        line(axes[1], [r["epoch"] for r in h], [r["train_loss"] for r in h], c, f"seed {seed}", marker=None)
    axes[0].axhline(0.033, color=INK2, linewidth=1.2, linestyle="--")
    axes[0].annotate("fixed formula 0.033", (1, 0.033), textcoords="offset points", xytext=(0, 4), fontsize=8, color=INK2)
    axes[0].set_title("Validation AP, objects with TTC < 4 s (1 drive)", loc="left")
    axes[1].set_title("Training loss", loc="left")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend(frameon=False, loc="upper left", ncol=3)
    fig.suptitle("Figure 3. Learned fusion (not adopted: test AP 0.006, chance level)", x=0.01, ha="left", fontsize=10, y=1.05)
    fig.tight_layout()
    save(fig, "fig3_fusion_training")


if __name__ == "__main__":
    fig_segmentation()
    fig_depth()
    fig_fusion()
