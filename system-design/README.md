# Honing

One infra to customize, train, and deploy a model end to end. Use all of its
modules, or only the ones you need.

Honing takes an open model, or a model behind an API, to a model customized for
one user or one company and deployed where they run it. Compression makes the
model smaller, post-training makes it better at the task, harness self-evolve
improves the prompts, skills, and tools around it, and serving deploys it. A
user turns on only the modules they need. This document describes the planes
behind those modules and how they depend on Slime.

![Honing scope: four modules, the data plane, and the gate](honing_scope.png)

What each audience gets from it:

| Audience | The claim |
|---|---|
| Researchers | Compression, post-training, and harness evolution are moves under one gate. Each move is measured against the release it would replace and can be rolled back, so the source of a gain can be isolated. |
| Practitioners | One infra from a base model to a deployed one. A company with GPUs can prune, quantize, and post-train a large open model into a smaller domain expert; an individual with no GPUs can evolve the harness around an API model on their own data. |
| Decision makers | A model that improves on your own data and, with compression turned on, costs less to serve. With an open model on your own servers, proprietary data never leaves your infrastructure. |

---

## The planes

![Honing system design](honing_architecture.png)

In code, the modules are planes on one release spine: a campaign that decides the
next move, one gate, and a release chain that records every change before it takes
effect. The planes meet only through the contracts in `honing/core`, which is what
lets a user turn one module on without the others.

| Plane | Module | Owns | In code today |
|---|---|---|---|
| Core | | Value types and protocols every plane is built against; standard library only | `honing/core` |
| Model Source | | The initial model: an open checkpoint, or a large model compressed first | protocol + fake |
| Data | data plane | Online feedback and offline datasets behind one batch type | contracts; RA2 demonstration and rollout providers |
| Train | post-training | Algorithm, update policy, numerics, rollout source, and teacher; run by Slime, or by a local trainer for DPO | `LocalJevTrainer` for decision models; `SlimeTrainer` (SFT, OPD, GRPO) for language models |
| Compression | compression | Pruning as score → select → edit, and post-training quantization; usable standalone | protocol + fake |
| Harness Evolve | harness self-evolve | Harness specs, episode runs, and evolution methods before, during, and after training | contracts; RA2 runner, grader, and harness |
| Campaign | | The scheduler, and the ordering rules every move passes before it touches weights | `honing/campaign` |
| Gate / Release | | One gate for every candidate; joint releases; the commit record; rollback | `honing/release` |
| Serve | serving | Live traffic and artifact activation | a decision server for Laya models; live traffic reserved |

The one idea that makes this small enough to build:

> Every step is an `UpdateCandidate`. A training step, a prune, a quantization, and a
> harness change pass the same gate and settle into the same release chain, so a bad
> prune is a rejected candidate or a rolled-back release and needs no machinery of its
> own.

Read [`system-design.md`](system-design.md) for the subsystem contracts, the coupling
model, the ordering rules, and worked lifecycles.

---

## Compose what you need

![Which modules different users turn on](honing_compositions.png)

A company with GPUs can use every module. It starts from a large open model, prunes and
quantizes it, post-trains it on its domain data, evolves the harness, and serves the
result in-house, ending with a domain expert much smaller than the model it started from.

An individual with no GPUs can skip compression and post-training. They start from an API
model, let the harness self-evolve on their own data, and deploy on a Mac or phone, which
gives a personal agent that keeps improving from daily use with no training compute.

Any subset in between works, for example post-training with LoRA or SMT and serving with
vLLM, because every module shares one data plane and one release chain. The decision-model
loop in the top-level README is this design with post-training, harness self-evolve, and
serving turned on and compression turned off.

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

---

## Acknowledgements

- [Slime](https://github.com/THUDM/slime) (Apache-2.0) — the training backend Honing
  depends on, pinned at `41014d1f`.
- [SGLang](https://github.com/sgl-project/sglang) and Megatron-Core — the inference and
  training engines, reached through Slime.

---

## Directory layout

```text
system-design/
  README.md                  this file: what Honing is, its planes, dependencies
  system-design.md           the design: contracts, coupling, rules, lifecycles, implementation steps
  honing_architecture.py     the system design diagram (matplotlib, self-contained)
  honing_scope.py            the four modules, the data plane, and the gate
  honing_compositions.py     which modules different users turn on
  *.png                      each rendered diagram, regenerated by the script of the same name

honing/
  core/                      value types and protocols; imports only the standard library
  release/                   the gate, selection policies, episode evaluator, release chain
  campaign/                  the campaign, the scheduler, and the ordering rules

tests/
  fakes/                     CPU stand-ins for every plane
  contracts/                 protocol coverage and dependency boundaries
  test_walking_skeleton.py   one campaign across every plane
```

Regenerate a diagram after editing its script:

```bash
uv run --no-project --with matplotlib python honing_architecture.py
```

Each script writes its PNG next to itself, so it is safe to run from any working directory.
