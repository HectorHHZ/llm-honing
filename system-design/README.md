# Honing

**Continual specialization infra: a large model that sheds its own dead weight while it learns.**

Honing is post-training infrastructure that turns a large general model into a leaner
domain-specialized one *over time*. As the model serves real traffic and real data, the
system keeps the capacity that carries the work, removes what it no longer needs, and
publishes each result as a versioned, rollback-able release. The model gets better
in-domain while getting cheaper to run.

Three framings of the same mechanism:

| Audience | The claim |
|---|---|
| **Researchers** | A post-training process that lets a large MoE model continuously shed dead weight while it learns and works — keeping the few parts doing real work, cutting the rest, so the model becomes smaller, faster, and cheaper on its own. |
| **Practitioners** | A general model becomes a specialized model through use. It keeps the capacity that matters, drops what it doesn't, and delivers similar or better in-domain intelligence at a fraction of serving cost. |
| **Decision makers** | Your AI gets smarter for your business while getting cheaper to run — instead of paying forever for an oversized general model. Because honing runs on your own servers, proprietary data never leaves your infrastructure. |

> **Status.** This directory is a design specification, not an implementation. Interfaces
> below are the contract to build against. Comparison claims about Reef are verified
> against its source at `016b4670`; see [Where Honing goes beyond Reef](#where-honing-goes-beyond-reef).

---

## The five planes

![Honing system design](honing_architecture.png)

Honing is five subsystems on one shared spine: the scenario aggregate, the gate, the
commit protocol, and the versioned release chain.

| Plane | Owns | Independent? |
|---|---|---|
| **Model Source** | Where the initial trainable model comes from | boot-time only |
| **Data Plane** | Two data sources (online feedback, offline dataset) behind one batch type | — |
| **Train Plane** | Algorithm × update policy × rollout source | — |
| **Prune Plane** | Structural capacity reduction and its audit trail | ✅ standalone package + CLI |
| **Harness Evolve** | Prompt/skill/tool composition search, at three training phases | ✅ reusable engine |
| **Serve Plane** | Live traffic, artifact activation | ⏳ reserved for phase 2 |

The one idea that makes this small enough to build:

> **A prune step *is* an `UpdateCandidate`.** The gate / commit / rollback machinery is
> shared verbatim across weight training, pruning, and harness evolution. A bad prune is
> a rejected candidate or a rolled-back release — not a new failure mode needing new
> machinery.

Read [`system-design.md`](system-design.md) for the subsystem contracts, the coupling
model, and worked lifecycles.

---

## How Honing differs from the surrounding stack

### Capability comparison

| Ability | Inference engine<br>(vLLM, SGLang) | RL framework<br>(Slime, veRL, AReaL) | Reef | **Honing** |
|---|:---:|:---:|:---:|:---:|
| Serves live traffic | ✅ | ❌ | ✅ | ⏳ phase 2 |
| Trains weights | ❌ | ✅ | ✅ | ✅ |
| Offline dataset training (SFT, DPO, distillation) | ❌ | ✅ | ❌ | ✅ |
| Online feedback training (receipts → steps) | ❌ | ❌ | ✅ | ✅ |
| Versioned artifacts, rollback | ❌ | ❌ | ✅ | ✅ |
| Stays live through updates | ❌ | ❌ | ✅ | ⏳ phase 2 |
| Evolves the harness (prompts, skills, tools) | ❌ | ❌ | ✅ | ✅ |
| **Harness × weight co-adaptation** | ❌ | ❌ | ❌ | ✅ |
| **Structural pruning / model compression** | ❌ | ❌ | ❌ | ✅ |
| **Serving cost falls as the model learns** | ❌ | ❌ | ❌ | ✅ |
| Ablation control (isolate what caused a gain) | ❌ | ❌ | ❌ | ✅ |

Inference engines and RL frameworks are **components** Honing orchestrates, not
competitors: SGLang serves rollouts and (later) traffic; Slime/Megatron is the training
backend. The comparison is about *what layer owns the loop*.

### Where Honing goes beyond Reef

Reef is the closest system in scope, so the five axes below are where Honing's
capabilities differ. Each row was verified against the Reef source, with the deciding
file cited so a reader can check the claim.

| Axis | Reef today | Honing | Evidence in Reef |
|---|---|---|---|
| **Training algorithms** | 3 production loss families — `sao`, `tttd`, `openclawrl` — all test-time-training RL variants. No SFT, no DPO, no distillation. (CORAL reuses `tttd`'s family rather than adding one.) | 7-algorithm zoo across three families: supervised (`sft`), distillation (`distill`, `opsd`, `gkd`), RL (`ppo`, `grpo`, `dpo`) | `recipes/{sao,tttd,openclawrl}/__init__.py` register the only three; `recipes/coral/recipe.py:36` sets `loss_family="tttd"` |
| **Gradient update policy** | Full fine-tune or LoRA only, selected by one flag | `full` · `lora` · `sparse_mask`, as a first-class registry orthogonal to the algorithm | `megatron_lora_rank` is the sole switch (`reef/train/slime_backend/reef_adapters/megatron/lora.py:113`) |
| **Data sources** | Records only. Both `ReportedFeedbackProcessor` and `ComputedFeedbackProcessor` read the record store; there is no offline dataset path. | Two providers behind one `TrainingBatch`: online feedback **and** offline dataset (corpora, preference pairs, distillation sets) | `reef/train/processors/{reported,computed}.py`; no dataset/corpus loader exists under `reef/train/` |
| **Model compression** | None. The model only ever grows or changes in place; serving cost is fixed by the checkpoint you started with. | Prune Plane as a standalone system: expert-drop, layer-drop, width/head, structured sparsity, distill-then-prune | no pruning code in `reef/` (matches are file cleanup and HF-export quantization, not model compression) |
| **Harness × weights** | Separate recipes. A deployment picks `CordisRecipe` (harness) *or* a `WeightTrainingRecipe` (weights) — never both, and no ablation control to attribute a gain. | Harness evolution as three lifecycle phases (`pre` / `during` / `post`) around weight training, with a co-adaptation scheduler and ablation-arm configuration | `reef/train/cordis_backend/recipe.py` and `reef/recipe/base.py:161` are disjoint branches of `Recipe` |

---

## Directory layout

```text
system-design/
  README.md                  this file: what Honing is, the five planes, how it compares
  system-design.md           the design: subsystem contracts, coupling, lifecycles, endpoints
  honing_architecture.py     the diagram source (matplotlib, self-contained)
  honing_architecture.png    the rendered diagram, regenerated by the script beside it
```

Regenerate the diagram after editing the script:

```bash
uv run --no-project --with matplotlib python honing_architecture.py
```

It writes the PNG next to the script, so it is safe to run from any working directory.
