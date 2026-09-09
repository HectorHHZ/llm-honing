"""Honing infra system design diagram (matplotlib, no external assets)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_suffix(".png")

# ---------------------------------------------------------------- palette
C = {
    "data": ("#dbe7f4", "#3d6fa8"),  # blue
    "train": ("#fdeeda", "#d07b19"),  # orange
    "prune": ("#fbe3e0", "#c0392b"),  # red
    "harness": ("#e0f0ee", "#2a8a80"),  # teal
    "gate": ("#e6f0dd", "#4c8a35"),  # green
    "serve": ("#ececec", "#7f7f7f"),  # gray (reserved)
    "model": ("#efe6f5", "#7d4fa0"),  # purple
    "ink": "#24292e",
    "sub": "#57606a",
}

fig, ax = plt.subplots(figsize=(17.5, 12.6), dpi=180)
ax.set_xlim(0, 100)
ax.set_ylim(0, 81)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, kind, title, lines=(), dashed=False, tfs=10.5, lfs=7.9, lh=1.5):
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
        x + w / 2, y + h - 1.1, title, ha="center", va="top", fontsize=tfs, fontweight="bold", color=C["ink"], zorder=3
    )
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 3.0 - i * lh, ln, ha="center", va="top", fontsize=lfs, color=C["sub"], zorder=3)


def arrow(p0, p1, color, label=None, rad=0.0, lw=2.0, dashed=False, lab=(0, 0), fs=8.0):
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


# ---------------------------------------------------------------- title
ax.text(50, 79.3, "HONING — System Design", ha="center", fontsize=19, fontweight="bold", color=C["ink"])
ax.text(
    50,
    76.6,
    "Versioned spine (Scenario / Gate / Commit / releases)  +  Prune plane  +  "
    "Algo x UpdatePolicy x Rollout matrix  +  3-phase harness co-adaptation",
    ha="center",
    fontsize=9.5,
    color=C["sub"],
    style="italic",
)

# ---------------------------------------------------------------- top row
box(
    2,
    64,
    26.5,
    10.5,
    "data",
    "DATA PLANE",
    [
        "two sources, one TrainingBatch:",
        "FeedbackProvider (online):",
        "records + receipts + /hone/report",
        "DatasetProvider (offline):",
        "SFT · preference pairs · distill sets",
        "DataContract checked at boot",
    ],
    lfs=7.4,
    lh=1.32,
)

box(
    31.5,
    64,
    28.5,
    10.5,
    "model",
    "MODEL SOURCE",
    [
        "DirectCheckpoint(",
        "    open-source ckpt )   — train as-is",
        "PrunedInit(",
        "    large MoE → one-shot prune )",
        "either way: init model = release 0",
    ],
    lfs=7.4,
    lh=1.32,
)

box(
    67,
    47,
    31,
    27.5,
    "prune",
    "PRUNE PLANE   (standalone system)",
    [
        "PruneMethod registry:",
        "expert-drop (MoE router stats) · layer-drop",
        "width / head · structured sparsity",
        "distill-then-prune",
        "",
        "analyze(model, telemetry) → PrunePlan",
        "(declarative, audited manifest)",
        "modes:  one-shot CLI | init pipeline | in-loop",
        "HonePolicy: trigger · max_regression",
        "· recovery · capacity floor",
    ],
    lfs=7.6,
    lh=1.42,
)

# ---------------------------------------------------------------- left column
box(
    2,
    20,
    20,
    38,
    "harness",
    "HARNESS EVOLVE",
    ["cordis engine: propose /", "gate / publish — GEPA,", "SkillClaw, Meta-H, self-prop."],
    lfs=7.4,
    lh=1.32,
)
box(
    4,
    43.5,
    16,
    6.3,
    "harness",
    "PRE — before training",
    ["optimize prompts/skills on base;", "better data distribution first"],
    tfs=8.6,
    lfs=7.1,
    lh=1.35,
)
box(
    4,
    35,
    16,
    7.5,
    "harness",
    "DURING — co-adapt",
    ["CoAdaptScheduler alternation;", "1 train window = 1 harness", "release (mixed batch discarded)"],
    tfs=8.6,
    lfs=7.1,
    lh=1.35,
)
box(
    4,
    27,
    16,
    6.5,
    "harness",
    "POST — final polish",
    ["evolve against frozen", "pruned weights"],
    tfs=8.6,
    lfs=7.1,
    lh=1.35,
)
ax.text(
    12,
    25.4,
    "all phases' candidates settle",
    ha="center",
    fontsize=7.6,
    color=C["harness"][1],
    fontweight="bold",
    zorder=3,
)
ax.text(
    12, 23.9, "through the same gate  →", ha="center", fontsize=7.6, color=C["harness"][1], fontweight="bold", zorder=3
)

# ---------------------------------------------------------------- center pipeline
box(
    26,
    40,
    36,
    17.5,
    "train",
    "TRAIN PLANE   (Slime / Megatron backend)",
    [
        "Algo zoo:  SFT · Distill · OPSD · GKD · PPO · GRPO · DPO",
        "Update policy:  full · LoRA · sparse-mask (⇄ PrunePlan repr.)",
        "RolloutProvider:  Harness episodes | Direct (SGLang) | none",
        "TeacherBinding:  endpoint | local engine | lineage:parent",
        "loss families load in driver/workers only — never in service",
    ],
    lfs=7.7,
    lh=1.62,
)

box(
    26,
    31,
    36,
    6,
    "gate",
    "GATE  —  one gate for every candidate",
    ["candidate vs incumbent on eval tasks · CandidateSelector · max_regression"],
    lfs=7.6,
)

box(
    26,
    20.5,
    36,
    7.5,
    "gate",
    "COMMIT  /  RELEASE CHAIN",
    [
        "HoneRelease = weights ⊕ prune-manifest ⊕ harness  (joint rollback)",
        "durable-before-compaction · pending→promote · honing curve",
    ],
    lfs=7.6,
)

box(
    26,
    8.5,
    36,
    7.5,
    "serve",
    "SERVE PLANE   (reserved — phase 2)",
    [
        "SGLang engines (shared with rollout) · /v1/* routes · Surface protocols",
        "transport-free RequestService ⇒ /v1 mounts later, no train/prune change",
    ],
    dashed=True,
    lfs=7.4,
)

box(
    67,
    8.5,
    31,
    11,
    "prune",
    "TELEMETRY STORE",
    [
        "router / expert activation stats, aggregated",
        "ring buffer (not SQLite records); collected at",
        "serving AND rollout — same SGLang engines",
    ],
    tfs=9.5,
    lfs=7.6,
    lh=1.5,
)

# ---------------------------------------------------------------- arrows
# data -> train
arrow((15, 64), (30, 55.5), C["data"][1], "TrainingBatch", rad=-0.12, lab=(3.5, -1.6))
# model -> train
arrow((45, 64), (45, 57.5), C["model"][1], "init model", lab=(6.2, 0.6))
# model -> prune (one-shot)
arrow((60.4, 69.5), (66.6, 69.5), C["prune"][1], None, dashed=True)
ax.text(63.5, 71.6, "one-shot", ha="center", fontsize=7.2, color=C["prune"][1], fontweight="bold", zorder=5)
ax.text(63.5, 70.4, "prune", ha="center", fontsize=7.2, color=C["prune"][1], fontweight="bold", zorder=5)
# train -> gate
arrow((42, 40), (42, 37), C["train"][1], "trained candidate", lab=(8.6, 0))
# prune -> gate (in-loop candidate) via right channel
polyline([(70, 47), (64.6, 43.5), (64.6, 34.5), (62.2, 34)], C["prune"][1])
ax.text(
    66.1,
    39.3,
    "pruned candidate (in-loop)",
    rotation=90,
    ha="center",
    va="center",
    fontsize=7.0,
    color=C["prune"][1],
    fontweight="bold",
    zorder=5,
)
# harness -> gate
arrow((22, 30.5), (26, 33), C["harness"][1], None)
# phase attach hints
arrow((20, 47), (26, 49.5), C["harness"][1], None, rad=-0.1, dashed=True, lw=1.4)
arrow((20, 39), (26, 43.5), C["harness"][1], None, rad=-0.08, dashed=True, lw=1.4)
# gate -> commit
arrow((42, 31), (42, 28), C["gate"][1], "selected", lab=(5.4, 0))
# commit -> serve
arrow((42, 20.5), (42, 16), C["serve"][1], "activate (reserved)", dashed=True, lab=(9.2, 0))
# serve -> telemetry
arrow((62, 12.5), (67, 13.2), C["prune"][1], None, lw=1.8)
# telemetry -> prune plane
arrow((82.5, 19.5), (82.5, 47), C["prune"][1], None, lw=2.0)
ax.text(88.6, 33, "utilization window", ha="center", fontsize=7.6, color=C["prune"][1], fontweight="bold", zorder=5)
ax.text(88.6, 31.5, "→ analyze()", ha="center", fontsize=7.6, color=C["prune"][1], fontweight="bold", zorder=5)
# serve -> data (records loop) via left channel
polyline([(26, 12.5), (23.2, 14.5), (23.2, 60.5), (9.5, 64)], C["data"][1], dashed=True, lw=1.7)
ax.text(
    13.5,
    61.4,
    "records + receipts (loop)",
    ha="center",
    fontsize=7.3,
    color=C["data"][1],
    fontweight="bold",
    zorder=5,
    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
)

# ---------------------------------------------------------------- footnotes
notes = [
    "①  A prune step IS an UpdateCandidate — the gate / commit / rollback machinery is"
    " shared verbatim across weight training, pruning, and harness evolution.",
    "②  sparse-mask tuning and PrunePlan share one param-selection representation: masked"
    " training is reversible prune rehearsal, made structural after repeated gate wins.",
    '③  The release chain doubles as a teacher registry: TeacherBinding "lineage:parent"'
    " serves the pre-prune ancestor for OPSD/GKD recovery — no extra deployment.",
    "④  Honing curve = release-chain metrics over time (params, $/1k tok, latency,"
    " in-domain score) — GET /hone/curve.",
]
for i, note in enumerate(notes):
    ax.text(2, 5.6 - i * 1.7, note, fontsize=8.2, color=C["ink"])

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
