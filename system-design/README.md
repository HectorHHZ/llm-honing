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

---

## The planes

![Honing system design](honing_architecture.png)

Honing is a set of planes on one release spine: a campaign that decides the next move,
one gate, and a release chain that records every change before it takes effect. The
planes meet only through the contracts in `honing/core`.

| Plane | Owns | In code today |
|---|---|---|
| **Core** | Value types and protocols every plane is built against; standard library only | `honing/core` |
| **Model Source** | The initial model: an open checkpoint, or a large model compressed first | protocol + fake |
| **Data** | Online feedback and offline datasets behind one batch type | protocol + fake |
| **Train** | Algorithm, update policy, numerics, rollout source, and teacher; run by Slime, or by a local trainer for DPO | protocol + fake |
| **Compression** | Pruning as score → select → edit, and post-training quantization; usable standalone | protocol + fake |
| **Harness Evolve** | Harness specs, episode runs, and evolution methods before, during, and after training | protocol + fake |
| **Campaign** | The scheduler, and the ordering rules every move passes before it touches weights | `honing/campaign` |
| **Gate / Release** | One gate for every candidate; joint releases; the commit record; rollback | `honing/release` |
| **Serve** | Live traffic and artifact activation | reserved |

The one idea that makes this small enough to build:

> **Every step is an `UpdateCandidate`.** A training step, a prune, a quantization, and a
> harness change pass the same gate and settle into the same release chain. A bad prune
> is a rejected candidate or a rolled-back release — not a new failure mode needing new
> machinery.

Read [`system-design.md`](system-design.md) for the subsystem contracts, the coupling
model, the ordering rules, and worked lifecycles.

---

## Dependencies

### Slime — the one training dependency

[Slime](https://github.com/THUDM/slime) is pinned at commit
`41014d1f29e201137fdffce737bb8bac65bc5219` (Apache-2.0) in the `runtime` dependency
group. Honing uses Slime's published arguments and extension hooks and does not copy its
code or mirror its structure. The table was verified against Slime's source at that
commit.

| Honing needs | Slime at the pinned commit | Where in Slime |
|---|---|---|
| SFT | native: `--loss-type sft_loss` | `slime/utils/arguments.py` |
| GRPO, PPO (and GSPO, CISPO, REINFORCE++) | native advantage estimators | `slime/utils/arguments.py` |
| On-policy (self-)distillation | native: `--use-opd`, teacher on an SGLang engine or loaded in Megatron | `docs/en/advanced/on-policy-distillation.md` |
| Offline logit distillation, GKD | not built in; runs through `--custom-loss-function-path` | `slime/utils/arguments.py` |
| DPO | **absent** — Honing runs it in a local trainer | — |
| LoRA | Megatron arguments passed through | Slime README, "native Megatron argument pass-through" |
| Train a subset of parameters | native: `--freeze-params-name-list`, `only_train_params_name_list` | `slime/backends/megatron_utils/model_provider.py` |
| Element-level sparse masks | not built in; Megatron train-step hooks exist, signatures undocumented | `slime/utils/arguments.py` |
| Models with every layer the same shape | native: built from one configuration | `slime/backends/megatron_utils/model_provider.py` |
| Models with layers of different shapes | only through `--spec` or `--custom-model-provider-path`; checkpoint conversion and SGLang loading unverified | `slime/backends/megatron_utils/model_provider.py` |
| BF16 training with FP8 rollout | stable | `docs/en/advanced/low-precision.md` |
| FP8 training; INT4 fake-quant training | experimental; beta | `docs/en/advanced/low-precision.md` |
| Post-training quantization | FP8 and INT4 conversion tools | `tools/convert_hf_to_fp8.py`, `tools/convert_hf_to_int4.py` |
| Agent rollouts with token-level capture | `BaseHarness`, the `Sandbox` protocol, Anthropic and OpenAI adapters | `slime/agent/` |
| **Pruning** | **absent** — the Compression plane supplies it | — |

Two Slime constraints shape the design directly. A teacher loaded in Megatron must share
the student's architecture, so after a cut the recovery teacher is served by its own
engine. And quantization-aware training cannot be combined with Slime's DeepGEMM MoE
forward path.

### Why Reef is not a dependency

[Reef](https://github.com/Human-Agent-Society/reef) is the closest system in scope, and
its principles shape this design. It is not imported, for three reasons verified against
its source at `016b4670`:

1. **Its harness evolution methods do not ship.** GEPA, SkillClaw, and Meta-Harness live
   in Reef's `recipes/`, which its wheel excludes; depending on Reef yields the mechanism
   but not the methods.
2. **Its evolution engine is bound to its own trainer and scenario**, while Honing needs
   one gate and one joint release across weights, compression, and harness.
3. **Training rollouts and gate evaluations must run the same harness runner.** Slime's
   agent layer captures token-level training samples; Reef's episode runner reads
   session logs for scoring. Using both would evaluate one runner and train on another.

Reef would be reconsidered if Honing needs agents beyond Claude Code and Codex or
harness edits that are code, if Reef packages its evolution methods, if its harness
layer stabilizes behind tagged releases, or if the Serve plane needs Reef's install and
release delivery.

### The dependency boundary

`honing/core` imports only the standard library. Planes depend on `honing.core` and never
on each other. Only `honing/backends/` may import a training framework.
`tests/contracts/test_boundaries.py` enforces all three.

---

## How Honing differs from the surrounding stack

### Capability comparison

| Ability | Inference engine<br>(vLLM, SGLang) | RL framework<br>(Slime, veRL, AReaL) | Reef | **Honing** |
|---|:---:|:---:|:---:|:---:|
| Serves live traffic | ✅ | ❌ | ✅ | ⏳ reserved |
| Trains weights | ❌ | ✅ | ✅ | ✅ |
| Offline dataset training (SFT, DPO, distillation) | ❌ | ✅ | ❌ | ✅ |
| Online feedback training (receipts → steps) | ❌ | ❌ | ✅ | ✅ |
| Versioned artifacts, rollback | ❌ | ❌ | ✅ | ✅ |
| Stays live through updates | ❌ | ❌ | ✅ | ⏳ reserved |
| Evolves the harness (prompts, skills, tools) | ❌ | ❌ | ✅ | ✅ |
| **Harness × weight co-adaptation** | ❌ | ❌ | ❌ | ✅ |
| **Structural pruning and quantization, recorded per release** | ❌ | ❌ | ❌ | ✅ |
| **Serving cost falls as the model learns** | ❌ | ❌ | ❌ | ✅ |
| Ablation control (isolate what caused a gain) | ❌ | ❌ | ❌ | ✅ |

Inference engines and RL frameworks are **components** Honing orchestrates, not
competitors: SGLang serves rollouts and (later) traffic; Slime is the training backend.
The comparison is about *what layer owns the loop*.

### Where Honing goes beyond Reef

Reef is the closest system in scope, so the five axes below are where Honing's
capabilities differ. Each row was verified against the Reef source at `016b4670`, with
the deciding file cited so a reader can check the claim.

| Axis | Reef | Honing | Where in Reef's source |
|---|---|---|---|
| **Training algorithms** | 3 production loss families — `sao`, `tttd`, `openclawrl` — all test-time-training RL variants. No SFT, no DPO, no distillation. (CORAL reuses `tttd`'s family rather than adding one.) | SFT, offline distillation, on-policy distillation, GKD, PPO, GRPO, and DPO | `recipes/{sao,tttd,openclawrl}/__init__.py` register the only three; `recipes/coral/recipe.py:36` sets `loss_family="tttd"` |
| **Gradient update policy** | Full fine-tune or LoRA only, selected by one flag | `full` · `lora` · `param_subset` · `sparse_mask`, independent of the algorithm | `megatron_lora_rank` is the sole switch (`reef/train/slime_backend/reef_adapters/megatron/lora.py:113`) |
| **Data sources** | Records only. Both `ReportedFeedbackProcessor` and `ComputedFeedbackProcessor` read the record store; there is no offline dataset path. | Online feedback **and** offline datasets behind one `TrainingBatch` | `reef/train/processors/{reported,computed}.py`; no dataset loader exists under `reef/train/` |
| **Model compression** | None. Serving cost is fixed by the checkpoint a deployment starts with. | The Compression plane: expert, layer, head, and channel pruning with recovery; post-training quantization; quantization-aware training | no pruning code in `reef/` (matches are file cleanup and export-time quantization) |
| **Harness × weights** | Separate recipes. A deployment picks `CordisRecipe` (harness) *or* a `WeightTrainingRecipe` (weights), never both, and has no ablation control. | Harness evolution as three phases around weight training, with a co-adaptation scheduler | `reef/train/cordis_backend/recipe.py` and `reef/recipe/base.py:161` are disjoint branches of `Recipe` |

---

## Acknowledgements

- [Slime](https://github.com/THUDM/slime) (Apache-2.0) — the training backend Honing
  depends on, pinned at `41014d1f`.
- [SGLang](https://github.com/sgl-project/sglang) and Megatron-Core — the inference and
  training engines, reached through Slime.
- [Reef](https://github.com/Human-Agent-Society/reef) (Apache-2.0) — the design reference
  for publishing every change as a gated candidate, writing the commit record before the
  head moves, and attributing each sample to the release that produced it. No Reef code
  is imported or copied.

---

## Directory layout

```text
system-design/
  README.md                  this file: what Honing is, its planes, dependencies, how it compares
  system-design.md           the design: contracts, coupling, rules, lifecycles, implementation steps
  honing_architecture.py     the diagram source (matplotlib, self-contained)
  honing_architecture.png    the rendered diagram, regenerated by the script beside it

honing/
  core/                      value types and protocols; imports only the standard library
  release/                   the gate, selection policies, episode evaluator, release chain
  campaign/                  the campaign, the scheduler, and the ordering rules

tests/
  fakes/                     CPU stand-ins for every plane
  contracts/                 protocol coverage and dependency boundaries
  test_walking_skeleton.py   one campaign across every plane
```

Regenerate the diagram after editing the script:

```bash
uv run --no-project --with matplotlib python honing_architecture.py
```

It writes the PNG next to the script, so it is safe to run from any working directory.
