"""Honing infra system design diagram (matplotlib, no external assets)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).with_suffix(".png")

# ---------------------------------------------------------------- palette
C = {
    "data": ("#dbe7f4", "#3d6fa8"),  # blue
    "train": ("#fdeeda", "#d07b19"),  # orange
    "compression": ("#fbe3e0", "#c0392b"),  # red
    "harness": ("#e0f0ee", "#2a8a80"),  # teal
    "campaign": ("#fdf5d3", "#a67c00"),  # amber
    "gate": ("#e6f0dd", "#4c8a35"),  # green
    "serve": ("#ececec", "#7f7f7f"),  # gray (reserved)
    "model": ("#efe6f5", "#7d4fa0"),  # purple
    "ink": "#24292e",
    "sub": "#57606a",
}

fig, ax = plt.subplots(figsize=(17.5, 13.4), dpi=180)
ax.set_xlim(0, 100)
ax.set_ylim(-3.5, 86)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, kind, title, lines=(), dashed=False, tfs=10.5, lfs=7.6, lh=1.45):
    fc, ec = C[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.35,rounding_size=0.9",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
            linestyle=(0, (5, 3)) if dashed else "-",
            zorder=2,
        )
    )
    ax.text(
        x + w / 2, y + h - 1.0, title, ha="center", va="top", fontsize=tfs, fontweight="bold", color=C["ink"], zorder=3
    )
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 2.8 - i * lh, ln, ha="center", va="top", fontsize=lfs, color=C["sub"], zorder=3)


def arrow(p0, p1, color, label=None, rad=0.0, lw=2.0, dashed=False, lab=(0, 0), fs=7.8):
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
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + lab[0], (p0[1] + p1[1]) / 2 + lab[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
            zorder=5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.92},
        )


def polyline(pts, color, dashed=False, lw=1.9):
    ls = (0, (5, 3)) if dashed else "-"
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1], strict=True)
        ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, zorder=4, solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15, linewidth=lw, color=color, linestyle=ls, zorder=4
        )
    )


def note(x, y, text, color, fs=7.4, rotation=0):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        fontweight="bold",
        rotation=rotation,
        zorder=5,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
    )


# ---------------------------------------------------------------- title
ax.text(50, 84.6, "HONING — System Design", ha="center", fontsize=19, fontweight="bold", color=C["ink"])
ax.text(
    50,
    82.0,
    "Planes meet only through honing.core contracts  ·  every step is one UpdateCandidate  ·  "
    "one gate  ·  one release chain  ·  Slime is the one training dependency",
    ha="center",
    fontsize=9.5,
    color=C["sub"],
    style="italic",
)

# ---------------------------------------------------------------- top row
box(
    2,
    68.5,
    22,
    11,
    "data",
    "DATA PLANE",
    [
        "one TrainingBatch, two sources:",
        "DatasetProvider (offline)",
        "SFT · preferences · distill sets",
        "FeedbackProvider (online)",
        "RolloutProvider: samples name",
        "their release and harness",
    ],
    lfs=7.2,
    lh=1.3,
)

box(
    26.5,
    68.5,
    33,
    11,
    "model",
    "MODEL SOURCE",
    [
        "open checkpoint, used as-is",
        "or CompressedInit: a large model thinned first",
        "",
        "master weights: HF safetensors",
        "(the one format every editor and trainer reads)",
    ],
    lfs=7.3,
    lh=1.32,
)

# ---------------------------------------------------------------- compression plane (right column)
box(64, 44.5, 34, 35, "compression", "COMPRESSION PLANE   (standalone)", [], tfs=10.5)
box(
    66,
    58.5,
    30,
    17.5,
    "compression",
    "PRUNE",
    [
        "ImportanceScorer → PruneSelector → ModelEditor",
        "signals: telemetry · calibration · weights · gradients",
        "removes: layers · experts · heads",
        "FFN / hidden channels · parameter masks",
        "PrunePlan (before, after, scores) kept on the release",
        "v1: uniform layers, refused before editing",
        "recovery = a Train job (RecoverySpec)",
    ],
    tfs=9.2,
    lfs=7.1,
    lh=1.4,
)
box(
    66,
    46,
    30,
    11,
    "compression",
    "QUANTIZE",
    [
        "Quantizer (PTQ): master → ServingWeights",
        "never modifies the master",
        "one QuantScheme shared with QAT and the engine",
        "a cut makes serving weights stale",
    ],
    tfs=9.2,
    lfs=7.1,
    lh=1.4,
)

# ---------------------------------------------------------------- harness (left column)
box(
    2,
    17.5,
    20,
    46.5,
    "harness",
    "HARNESS EVOLVE",
    ["HarnessSpec: files per agent", "EvolveMethod: propose · record", "EpisodeRunner on slime.agent"],
    lfs=7.2,
    lh=1.3,
)
box(
    3.8,
    47.5,
    16.4,
    7.2,
    "harness",
    "PRE — before training",
    ["instructions and skills", "searched on the base model"],
    tfs=8.4,
    lfs=7.0,
    lh=1.3,
)
box(
    3.8,
    37.5,
    16.4,
    8.2,
    "harness",
    "DURING — co-adapt",
    ["retuned between train windows;", "one harness release per window", "(mixed batches refused)"],
    tfs=8.4,
    lfs=7.0,
    lh=1.3,
)
box(
    3.8,
    28.0,
    16.4,
    7.7,
    "harness",
    "POST — the harness that ships",
    ["evolved against the", "serving weights"],
    tfs=8.4,
    lfs=7.0,
    lh=1.3,
)
ax.text(12, 24.6, "gate episodes and training", ha="center", fontsize=7.2, color=C["harness"][1], fontweight="bold")
ax.text(12, 23.2, "rollouts use the same runner", ha="center", fontsize=7.2, color=C["harness"][1], fontweight="bold")
ax.text(12, 20.4, "methods plug in: GEPA,", ha="center", fontsize=7.0, color=C["sub"])
ax.text(12, 19.1, "SkillClaw, Meta-Harness, ...", ha="center", fontsize=7.0, color=C["sub"])

# ---------------------------------------------------------------- center pipeline
# The dashed outline marks the training backends: the only code allowed to import a framework.
ax.add_patch(
    Rectangle((25.4, 44.2), 35.2, 21.0, fill=False, edgecolor=C["sub"], linewidth=1.2, linestyle=(0, (2, 2)), zorder=1)
)
ax.text(26.0, 64.3, "honing/backends/", ha="left", va="center", fontsize=7.0, color=C["sub"], style="italic", zorder=5)

box(
    26.5,
    45.2,
    33,
    17.4,
    "train",
    "TRAIN PLANE",
    [
        "algorithm: SFT · distill · OPD · GKD · PPO · GRPO · DPO",
        "update policy: full · LoRA · param subset · sparse mask",
        "numerics: bf16 · fp8 · fake-quant QAT (QuantScheme)",
        "rollout: harness episodes · direct · none",
        "teacher: lineage · release · external  (engine / in-process)",
        "Slime runs all but DPO  ·  DPO: local trainer",
        "safetensors ⇄ torch_dist around each Slime job",
        "BackendSupport declares what each backend can do",
    ],
    tfs=10.0,
    lfs=7.2,
    lh=1.58,
)

box(
    26.5,
    35.6,
    33,
    6.2,
    "campaign",
    "CAMPAIGN   —   scheduler + ordering rules",
    ["every move becomes one candidate; rules refuse it", "before any weight is trained, edited, or quantized"],
    tfs=9.4,
    lfs=7.2,
    lh=1.35,
)

box(
    26.5,
    26.6,
    33,
    6.2,
    "gate",
    "GATE   —   one gate for every candidate",
    ["scores serving weights when they exist (what ships)", "win/loss for quality · no-regression for cost"],
    tfs=9.4,
    lfs=7.2,
    lh=1.35,
)

box(
    26.5,
    14.6,
    33,
    9.2,
    "gate",
    "RELEASE CHAIN",
    [
        "HoneRelease = master ⊕ serving ⊕ architecture",
        "⊕ transforms ⊕ harness ⊕ method state",
        "record written before the head moves",
        "joint rollback  ·  honing curve",
    ],
    tfs=9.6,
    lfs=7.2,
    lh=1.4,
)

box(
    26.5,
    3.6,
    33,
    7.6,
    "serve",
    "SERVE PLANE   (reserved)",
    ["loads serving weights when they exist", "SGLang engines shared with rollout · /v1 routes"],
    dashed=True,
    tfs=9.4,
    lfs=7.2,
    lh=1.4,
)

box(
    64,
    3.6,
    34,
    11.5,
    "compression",
    "TELEMETRY STORE",
    [
        "per-expert routing counts, layer activation norms",
        "aggregated per window, not stored per request",
        "filled by rollout and serving engines alike",
    ],
    tfs=9.4,
    lfs=7.2,
    lh=1.45,
)

# ---------------------------------------------------------------- arrows
# data -> train: leaves the data box's right edge so it never crosses the records loop on the left channel
arrow((24.2, 71.0), (29.5, 62.9), C["data"][1], None, rad=0.25)
note(33.8, 66.4, "TrainingBatch", C["data"][1], fs=7.2)
# model source -> train
arrow((47, 68.5), (47, 63.0), C["model"][1], "release 0", lab=(5.2, 0.4))
# model source -> compression (compressed init)
arrow((59.9, 74), (63.6, 74), C["compression"][1], None, dashed=True)
note(61.8, 71.4, "Compressed\nInit", C["compression"][1], fs=6.2)
# train -> campaign
arrow((43, 45.2), (43, 42.2), C["train"][1], "train / recovery candidate", lab=(0, 0), fs=7.2)
# compression -> campaign
polyline([(64.0, 50.0), (62.2, 48.5), (62.2, 39.2), (59.9, 38.7)], C["compression"][1])
note(62.2, 43.8, "prune / quantize\ncandidate", C["compression"][1], fs=6.8, rotation=90)
# harness -> campaign
arrow((22.2, 38.7), (26.1, 38.7), C["harness"][1], None)
# phase attach hints
arrow((20.2, 51), (26.1, 54.5), C["harness"][1], None, rad=-0.1, dashed=True, lw=1.3)
arrow((20.2, 41.6), (26.1, 48.0), C["harness"][1], None, rad=-0.08, dashed=True, lw=1.3)
# campaign -> gate
arrow((43, 35.6), (43, 33.2), C["campaign"][1], "passed the rules", lab=(7.4, 0), fs=7.2)
# gate -> release
arrow((43, 26.6), (43, 24.2), C["gate"][1], "selected", lab=(5.0, 0), fs=7.2)
# release -> serve
arrow((43, 14.6), (43, 11.6), C["serve"][1], "activate (reserved)", dashed=True, lab=(8.4, 0), fs=7.2)
# serve -> telemetry
arrow((59.9, 7.5), (63.6, 8.2), C["compression"][1], None, lw=1.8)
# telemetry -> compression
arrow((81, 15.1), (81, 44.1), C["compression"][1], None, lw=2.0)
note(87.8, 30.5, "utilization window\n→ ImportanceScorer", C["compression"][1], fs=7.3)
# serve -> data: the records loop, through the left channel
polyline([(26.1, 7.5), (23.3, 9.5), (23.3, 66.0), (15.0, 68.3)], C["data"][1], dashed=True, lw=1.6)
note(17.2, 66.3, "records + receipts", C["data"][1], fs=6.8)

# ---------------------------------------------------------------- footnotes
notes = [
    "①  A training step, a prune, a quantization, and a harness change are each one UpdateCandidate:"
    " one gate, one commit record, one rollback.",
    "②  One QuantScheme is shared by fake-quant training, export, and the serving engine; exporting under"
    " another scheme is refused before quantizing.",
    "③  After a shape-changing cut, the recovery teacher (typically the pre-cut release) is served by its own"
    " engine, since an in-process teacher must share the student's shape.",
    "④  v1: uniform layers only · DPO in a local trainer · Slime is the only framework dependency"
    " (Reef is a design reference, not imported).",
]
for i, text in enumerate(notes):
    ax.text(2, 1.0 - i * 1.55, text, fontsize=8.0, color=C["ink"])

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
