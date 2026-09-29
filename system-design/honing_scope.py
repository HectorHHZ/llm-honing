"""Honing's scope: four modules from a base model to a deployed one, the data plane, and the one gate (matplotlib)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_suffix(".png")

# The palette of honing_architecture.py, so a module has one color across every diagram.
C = {
    "data": ("#dbe7f4", "#3d6fa8"),
    "train": ("#fdeeda", "#d07b19"),
    "compression": ("#fbe3e0", "#c0392b"),
    "harness": ("#e0f0ee", "#2a8a80"),
    "gate": ("#e6f0dd", "#4c8a35"),
    "serve": ("#ececec", "#7f7f7f"),
    "model": ("#efe6f5", "#7d4fa0"),
    "ink": "#24292e",
    "sub": "#57606a",
}

fig, ax = plt.subplots(figsize=(17, 9.6), dpi=160)
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis("off")
fig.patch.set_facecolor("white")


def panel(x, y, w, h, kind, dashed=False):
    fc, ec = C[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
            linestyle=(0, (5, 3)) if dashed else "-",
            zorder=2,
        )
    )


def text(x, y, s, fs=9, color=None, bold=False, ha="center", va="center", style="normal"):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color or C["ink"],
        fontweight="bold" if bold else "normal",
        style=style,
        zorder=3,
    )


def arrow(p0, p1, color, lw=2.0, rad=0.0, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=lw,
            color=color,
            linestyle=(0, (5, 3)) if dashed else "-",
            zorder=4,
        )
    )


def label(x, y, s, color, fs=8):
    ax.text(
        x,
        y,
        s,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        fontweight="bold",
        zorder=5,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.92},
    )


# ---------------------------------------------------------------- title
text(50, 54, "Honing: from a base model to a customized, deployed model", fs=18, bold=True)
text(50, 51.3, "Use all four modules, or only the ones you need", fs=10.5, color=C["sub"], style="italic")

# ---------------------------------------------------------------- start and end
panel(1, 31, 9.5, 12, "model")
text(5.75, 40.8, "START", fs=10.5, bold=True)
text(5.75, 37.2, "an open model\n(e.g. 122B)", fs=8.5, color=C["sub"])
text(5.75, 33.6, "or an API model", fs=8.5, color=C["sub"])

panel(89.5, 31, 9.5, 12, "model")
text(94.25, 40.8, "END", fs=10.5, bold=True)
text(94.25, 36.2, "a customized,\ndeployed model", fs=8.5, color=C["sub"])

# ---------------------------------------------------------------- the four modules
MODULES = [
    ("compression", "COMPRESSION", "make the model smaller", ["Pruning", "Quantization", "Recovery training"]),
    (
        "train",
        "POST-TRAINING",
        "make it better at your task",
        ["SFT · RL (PPO, GRPO, DPO)", "Distillation", "On-policy distillation", "QAT · LoRA / SMT"],
    ),
    (
        "harness",
        "HARNESS SELF-EVOLVE",
        "improve prompts, skills and tools",
        ["Harness evolve", "Auto research", "GEPA · Meta-Harness"],
    ),
    ("serve", "SERVING", "deploy anywhere", ["SGLang · vLLM", "GPU cluster", "Mac / mobile"]),
]
X0, W, GAP, Y, H = 13.5, 16.5, 2.5, 26, 22
centers = []
for i, (kind, title, goal, methods) in enumerate(MODULES):
    x = X0 + i * (W + GAP)
    centers.append(x + W / 2)
    panel(x, Y, W, H, kind)
    text(x + W / 2, Y + H - 1.8, title, fs=11, bold=True)
    text(x + W / 2, Y + H - 4.3, goal, fs=8.8, color=C[kind][1], style="italic")
    for j, method in enumerate(methods):
        cy = Y + H - 8.2 - j * 3.3
        ax.add_patch(
            FancyBboxPatch(
                (x + 1.4, cy - 1.15),
                W - 2.8,
                2.3,
                boxstyle="round,pad=0.1,rounding_size=0.5",
                facecolor="white",
                edgecolor=C[kind][1],
                linewidth=0.9,
                zorder=3,
            )
        )
        text(x + W / 2, cy, method, fs=8.6)

# flow from start through the modules to the end
arrow((10.9, 37), (X0 - 0.3, 37), C["model"][1])
for i in range(3):
    left = X0 + i * (W + GAP) + W + 0.3
    arrow((left, 37), (left + GAP - 0.6, 37), C["ink"], lw=1.6)
arrow((X0 + 4 * W + 3 * GAP + 0.3, 37), (89.2, 37), C["model"][1])

# ---------------------------------------------------------------- data plane
panel(13.5, 12.5, 73, 8.5, "data")
text(50, 19.6, "DATA PLANE  (shared by every module)", fs=10.5, bold=True)
text(
    31,
    16.2,
    "Offline data for training\nSFT · preference · distillation sets",
    fs=8.6,
    color=C["sub"],
)
text(
    69,
    16.2,
    "Online data for self-evolve\nrollouts · user feedback · serving logs",
    fs=8.6,
    color=C["sub"],
)

# offline batches feed post-training; online data feeds the harness loop; serving writes logs and telemetry
arrow((centers[1], 21.4), (centers[1], Y - 0.4), C["data"][1])
label(centers[1] + 4.6, 23.6, "training batches", C["data"][1], fs=7.6)
arrow((centers[2], 21.4), (centers[2], Y - 0.4), C["data"][1])
label(centers[2] + 5.2, 23.6, "episodes, feedback", C["data"][1], fs=7.6)
arrow((centers[3], Y - 0.4), (centers[3], 21.4), C["serve"][1], dashed=True)
label(centers[3] + 4.2, 23.6, "logs, telemetry", C["serve"][1], fs=7.6)
arrow((centers[0], 21.4), (centers[0], Y - 0.4), C["compression"][1], dashed=True)
label(centers[0] + 5.2, 23.6, "routing telemetry", C["compression"][1], fs=7.6)

# ---------------------------------------------------------------- the one gate
panel(1, 1.2, 98, 8.2, "gate")
text(50, 8.1, "Every change is one gated, versioned candidate that can be rolled back", fs=10.5, bold=True)
STAGES = [
    (18, "Candidate", "a train step · a prune · a quantization · a harness edit"),
    (50, "Gate", "quality must win; cost must not regress"),
    (82, "Release chain", "versioned model + harness; joint rollback"),
]
for cx, name, detail in STAGES:
    ax.add_patch(
        FancyBboxPatch(
            (cx - 14, 2.1),
            28,
            4.2,
            boxstyle="round,pad=0.15,rounding_size=0.6",
            facecolor="white",
            edgecolor=C["gate"][1],
            linewidth=1.1,
            zorder=3,
        )
    )
    text(cx, 5.2, name, fs=9.5, bold=True)
    text(cx, 3.3, detail, fs=8.2, color=C["sub"])
arrow((32.3, 4.2), (35.7, 4.2), C["gate"][1])
arrow((64.3, 4.2), (67.7, 4.2), C["gate"][1])

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
