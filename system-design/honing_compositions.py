"""Which Honing modules different users turn on, and what each gets out (matplotlib)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_suffix(".png")

# The palette of honing_architecture.py, so a module has one color across every diagram.
C = {
    "train": ("#fdeeda", "#d07b19"),
    "compression": ("#fbe3e0", "#c0392b"),
    "harness": ("#e0f0ee", "#2a8a80"),
    "serve": ("#ececec", "#7f7f7f"),
    "model": ("#efe6f5", "#7d4fa0"),
    "gate": ("#e6f0dd", "#4c8a35"),
    "skip": ("#f6f8fa", "#c9ced4"),
    "ink": "#24292e",
    "sub": "#57606a",
}

COLUMNS = [
    ("model", "Base model"),
    ("compression", "Compression"),
    ("train", "Post-training"),
    ("harness", "Harness evolve"),
    ("serve", "Serving"),
]
# One row per user: who they are, one cell per column (None = the module is skipped), and what they end with.
ROWS = [
    (
        "Company with GPUs",
        ["122B open model", "prune + quantize", "SFT / RL / distill", "harness evolve", "SGLang / vLLM cluster"],
        "A domain-expert model, much smaller\nthan where it started, served in-house",
    ),
    (
        "Individual, no GPUs",
        ["API model", None, None, "harness self-evolve\non your own data", "Mac / mobile"],
        "A personal agent that keeps improving\nfrom daily use, with no training compute",
    ),
    (
        "Any other subset",
        ["open model", None, "LoRA / SMT", None, "vLLM"],
        "A post-trained model served with vLLM",
    ),
]

fig, ax = plt.subplots(figsize=(17, 7.2), dpi=160)
ax.set_xlim(0, 104)
ax.set_ylim(0, 42)
ax.axis("off")
fig.patch.set_facecolor("white")


def cell(x, y, w, h, kind, s, fs=8.8, bold=False, italic=False):
    fc, ec = C[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.15,rounding_size=0.6",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5 if kind != "skip" else 1.0,
            linestyle=(0, (4, 3)) if kind == "skip" else "-",
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        s,
        ha="center",
        va="center",
        fontsize=fs,
        color=C["sub"] if kind == "skip" else C["ink"],
        fontweight="bold" if bold else "normal",
        style="italic" if italic else "normal",
        zorder=3,
    )


ax.text(52, 40.3, "Compose what you need", ha="center", fontsize=18, fontweight="bold", color=C["ink"])
ax.text(
    52,
    37.6,
    "The same infra serves very different users; each turns on only the modules they need",
    ha="center",
    fontsize=10.5,
    color=C["sub"],
    style="italic",
)

LABEL_W, COL_W, COL_GAP, RESULT_W = 13, 12, 1.0, 21.5
x_cols = [LABEL_W + 1.5 + i * (COL_W + COL_GAP) for i in range(len(COLUMNS))]
x_result = x_cols[-1] + COL_W + 3.2

# column headings
HEAD_Y = 32.2
for x, (kind, name) in zip(x_cols, COLUMNS, strict=True):
    ax.text(x + COL_W / 2, HEAD_Y, name, ha="center", va="center", fontsize=10, fontweight="bold", color=C[kind][1])
ax.text(x_result + RESULT_W / 2, HEAD_Y, "Result", ha="center", va="center", fontsize=10, fontweight="bold")

ROW_H, ROW_GAP = 7.4, 2.2
for r, (who, cells, result) in enumerate(ROWS):
    y = 22.4 - r * (ROW_H + ROW_GAP)
    ax.text(0.8, y + ROW_H / 2, who, ha="left", va="center", fontsize=10.5, fontweight="bold", color=C["ink"])
    for x, (kind, _), content in zip(x_cols, COLUMNS, cells, strict=True):
        if content is None:
            cell(x, y, COL_W, ROW_H, "skip", "skip", italic=True)
        else:
            cell(x, y, COL_W, ROW_H, kind, content)
    ax.add_patch(
        FancyArrowPatch(
            (x_cols[-1] + COL_W + 0.4, y + ROW_H / 2),
            (x_result - 0.4, y + ROW_H / 2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=C["gate"][1],
            zorder=4,
        )
    )
    cell(x_result, y, RESULT_W, ROW_H, "gate", result, fs=8.8)

ax.text(
    52,
    0.6,
    "Any subset connects, because every module shares one data plane and one release chain.",
    ha="center",
    fontsize=9.5,
    color=C["sub"],
)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
