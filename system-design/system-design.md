# Honing — System Design

The subsystem contracts, how they couple, and what runs when.

Read [`README.md`](README.md) first for what Honing is and how it compares to the
surrounding stack. This document is the design others build against.

**Contents**

1. [Design principles](#1-design-principles)
2. [The subsystems](#2-the-subsystems)
3. [How they couple](#3-how-they-couple)
4. [Lifecycles](#4-lifecycles)
5. [HTTP surface](#5-http-surface)
6. [Configuration](#6-configuration)
7. [Implementation phases](#7-implementation-phases)
8. [Open questions](#8-open-questions)

---

## 1. Design principles

Five rules decide most of the structure below. The first four are established practice
for continual-learning infrastructure; the fifth is what Honing adds.

**P1 — One aggregate root, one assembly interface.** Every subsystem hangs off
`Scenario`, and a method reaches the framework only through the recipe's hooks. No
subsystem calls another directly. This is why a plane can be developed, tested, and
shipped independently.

**P2 — Candidates are uniform.** Anything that wants to change what is served becomes an
`UpdateCandidate`, is judged by one gate, and settles through one commit protocol.
Weight steps, prune steps, and harness mutations all take this path. *This is the single
decision that keeps Honing's scope tractable* — pruning inherits evaluation, publication,
provenance, and rollback for free.

**P3 — Durable before destructive.** The commit record reaches disk before any record is
compacted, any checkpoint is deleted, or any head moves. A crash mid-commit is
recoverable by construction.

**P4 — Version-frozen attribution.** A sample is stamped with the release that produced
it, before the generating call is made. Batches that span releases are discarded whole,
because rewards from different policies are not comparable. Honing extends this: a batch
spanning *harness* releases is equally incomparable and equally discarded.

**P5 — Capacity is a first-class, reversible dimension.** Model capacity is not fixed at
boot. It is proposed, measured, published, and — when it turns out to be wrong — rolled
back, exactly like weights. Every reduction carries a manifest saying what was removed
and on what evidence.

---

## 2. The subsystems

### 2.1 Model Source — where the initial model comes from

Honing supports two entry points, so a team can start from an open checkpoint *or* from
a large model they want thinned first. Both produce release 0 of the scenario's chain.

```python
class ModelSource(Protocol):
    """Materialize the initial trainable model. Runs once, at boot."""
    def materialize(self, work: Path) -> ModelArtifact: ...


DirectCheckpoint(path="Qwen3-235B-A22B")        # train the open model as-is

PrunedInit(                                      # thin a large model, then train it
    base="huge-moe-ckpt",
    pipeline=[ExpertDrop(keep_ratio=0.5), LayerDrop(n=8)],
    calibration=DatasetRef("domain-calib") | TelemetryRef("prod-30d"),
)
```

`PrunedInit` delegates to the Prune Plane and records the resulting manifest on release 0,
so a model that arrived pre-thinned is as auditable as one thinned later in the loop.

### 2.2 Prune Plane — capacity reduction as an independent system

The Prune Plane is a standalone Python package (`honing.prune`). It has no dependency on
the training loop, the service, or a running deployment; the loop depends on *it*.

```python
class PruneMethod(Protocol):
    name: str    # registry key

    def analyze(self, model: ModelHandle, telemetry: UtilizationTelemetry) -> PrunePlan:
        """What to remove and why, given observed utilization."""

    def apply(self, model: ModelHandle, plan: PrunePlan) -> ModelArtifact:
        """Execute the plan; produce a smaller model."""

    def estimate(self, plan: PrunePlan) -> CostDelta:
        """Δparams, ΔFLOPs, Δmemory, Δlatency — before paying for the apply."""
```

Registered methods:

| Method | Target | Signal it reads |
|---|---|---|
| `expert_drop` | MoE experts | router activation counts per expert, per domain |
| `layer_drop` | transformer blocks | layer-wise activation norm / representational redundancy |
| `width_head` | attention heads, FFN width | head utility, neuron activation statistics |
| `structured_sparsity` | block-sparse patterns | magnitude + gradient saliency under the domain mix |
| `distill_then_prune` | composite | trains a smaller student before removing the teacher's capacity |

A `PrunePlan` is **declarative and audited**, never an opaque side effect:

```python
@dataclass(frozen=True)
class PrunePlan:
    method: str
    removals: tuple[ParamAddress, ...]   # which experts / layers / heads / blocks
    evidence: TelemetrySummary           # the window and statistics that justified it
    estimated: CostDelta
```

The plan is published as `prune_manifest` on the release. **The manifest chain is the
complete provenance of every cut ever made** — which is what makes "your model got 40%
cheaper" an auditable claim rather than an assertion.

Three usage modes, all the same code:

| Mode | Trigger | Use |
|---|---|---|
| **Standalone CLI** | `hone prune -c prune.yaml` | one-shot compression, no service running |
| **Init pipeline** | `PrunedInit(...)` at boot | large model → thinned starting point |
| **In-loop** | `HonePolicy` during training | continuous honing as utilization is observed |

```python
@dataclass(frozen=True)
class HonePolicy:
    trigger: PruneTrigger          # every_n_commits(n) | utilization_stationary(window) | manual
    method: str
    max_regression: float          # in-domain score drop the gate will tolerate
    recovery: RecoverySpec | None  # e.g. opsd(teacher="lineage:parent", steps=k)
    floor: CapacityFloor           # min experts / layers — the stopping condition
```

`floor` matters: without it, honing has no termination condition and will happily cut
until the model is useless. The floor makes "how small" an explicit operator decision.

### 2.3 Data Plane — two sources, one batch type

Honing needs both online feedback *and* offline corpora (SFT sets, preference pairs,
distillation data). Record-driven systems support only the first, so both sit behind one
protocol here:

```python
class DataProvider(Protocol):
    def next_batch(self, contract: DataContract) -> TrainingBatch | None: ...
```

| Provider | Source | Status |
|---|---|---|
| `FeedbackProvider` | record store + receipts + `POST /hone/report` | standard online path |
| `DatasetProvider` | dataset manifest (local or HF) + sampler (epoch, shuffle, packing) | **new** |

**The Train Plane cannot tell which provider fed it.** Both emit `TrainingBatch`. The
difference is confined to a boot-time contract check: each algorithm declares the
`DataContract` it needs, and a provider that cannot satisfy it fails at startup rather
than several hours into a run.

Mixing is legal and expected: SFT on a corpus to establish the domain, then RL on live
feedback to keep improving — two providers feeding one trainer in sequence.

### 2.4 Train Plane — three orthogonal registries

Systems in this space typically bind one algorithm per deployment. Honing separates three
concerns that are usually conflated, so any combination is expressible without new code:

```python
@dataclass(frozen=True)
class HoneTrainingSpec:
    algo: str                        # sft | distill | opsd | gkd | ppo | grpo | dpo
    update_policy: str               # full | lora(rank, targets) | sparse_mask(selection)
    rollout: str | None              # harness | direct | None
    teacher: TeacherBinding | None   # endpoint(url) | engine(local) | lineage:parent
```

Static declarations, validated as a matrix at boot:

| `algo` | `DataContract` | Rollout | Teacher | Loss family |
|---|---|:---:|:---:|---|
| `sft` | tokens + loss_mask | ✗ | ✗ | `sft` |
| `distill` | tokens (teacher logits fetched online) | ✗ | ✓ | `kd` |
| `opsd` | on-policy samples + logprobs | ✓ | ✓ (may be lineage) | `opsd` |
| `gkd` | on-policy samples | ✓ | ✓ | `gkd` |
| `ppo` | samples + logprobs + reward + value | ✓ | ✗ | `ppo` |
| `grpo` | **grouped** samples + logprobs + reward | ✓ | ✗ | `grpo` |
| `dpo` | preference pairs | ✗ | ✗ (ref = frozen self) | `dpo` |

An unsatisfiable cell (`sft` + `rollout="harness"`, `gkd` without a teacher) is rejected
at startup with the reason, rather than failing hours into a run.

**Update policy** is orthogonal to all of the above:

```python
class UpdatePolicy(Protocol):
    def trainable_selection(self, model: ModelHandle) -> ParamSelection: ...
    def export(self, step: int) -> CheckpointSpec: ...   # full ckpt | adapter | masked delta
```

> **`sparse_mask` and `PrunePlan` share one parameter-addressing representation.**
> This makes masked training a *reversible rehearsal* for pruning: train with capacity
> masked out but structurally present, and once the gate has repeatedly accepted the
> masked model, the Prune Plane converts the mask into a structural removal. This is the
> shortest path from "training and pruning are two systems" to "training and pruning
> co-adapt", and it costs one shared data structure.

Slime/Megatron is the training backend. Loss families are imported **only** by the driver
and workers, so the service process never loads the torch stack.

### 2.5 Rollout — with or without a harness

On-policy algorithms need samples from the current policy. Where those samples come from
is a separate, swappable decision:

```python
class RolloutProvider(Protocol):
    def collect(self, tasks, policy: ModelBinding, n_per: int) -> tuple[RolloutGroup, ...]: ...
```

| Provider | What a sample is | Cost | Use when |
|---|---|---|---|
| `HarnessRolloutProvider` | a full agent episode: real binary, tool calls, scored trajectory | high | training an agent in its deployed harness |
| `DirectRolloutProvider` | `n` completions from SGLang, token-native logprobs | low | training the model itself; ablation baselines |
| `None` | — | — | offline algorithms (SFT, DPO, offline distillation) |

The algorithm code is identical across providers. The same GRPO implementation trains an
agent on harness episodes or a bare model on completions — the provider decides.

Having both is also what makes **ablation control** possible: run the same algorithm with
`direct` and with `harness` rollouts and the difference attributes the gain to the harness
rather than to the weights. Reef cannot express this comparison at all.

### 2.6 Harness Evolve — three phases, one engine

Harness evolution is usually a *parallel kind of deployment*: a system runs harness search
or weight training, never both. Honing makes it a set of lifecycle hooks around weight
training instead, over one propose / gate / publish engine — so any composition-search
method (GEPA, SkillClaw, Meta-Harness, failure-driven self-proposal) plugs into the same
seam.

```python
Phase = Literal["pre_training", "during_training", "post_training"]

@dataclass(frozen=True)
class HarnessPhaseConfig:
    trigger: PhaseTrigger        # pre: once | during: every_n_commits(n), on_plateau(metric) | post: once
    budget: EvolveBudget         # max evolve steps / episodes / tokens
    evolve: CordisConfig         # propose / evaluate / selection — any of the four methods
    freeze_weights: bool = True  # during-phase: alternate rather than update jointly
```

| Phase | When | Why it belongs there |
|---|---|---|
| `pre_training` | before the first weight step | prompt/skill optimization is orders of magnitude cheaper than gradient steps, and a better harness yields a better data distribution for everything after |
| `during_training` | interleaved, on a schedule or plateau | weights and harness co-adapt; each is retuned against the other's current state |
| `post_training` | after the final prune and recovery | one last polish against frozen, thinned weights — the harness that ships |

**The co-adaptation invariant.** This is the one place the design must be strict. A batch
whose samples span weight releases is already discarded (P4). A harness change alters the
reward distribution just as a weight change does, so:

> A training window samples against exactly one harness release. A batch spanning harness
> releases is discarded whole, with the mixed release ids recorded — the same rule, the
> same telemetry, extended to a second axis.

Alternation is driven by an explicit scheduler, so the campaign's shape is inspectable
rather than emergent:

```python
class CoAdaptScheduler(Protocol):
    def next_move(self, state: CampaignState) -> Literal["train", "prune", "harness", "stop"]: ...
```

The default policy: pre-harness once → *N* train commits → when utilization goes
stationary, attempt a prune → recovery distillation against `lineage:parent` → resume →
stop at the capacity floor → post-harness polish.

### 2.7 Serve Plane — reserved

Not built in phase 1, but its contracts are fixed now so that adding it later is purely
additive: the `Surface` protocol family (loader, activator, inference hooks, lease),
`InferenceRuntime`, and the `/v1/*` routes.

Phase 1 implements only what rollouts need: an `EngineHandle` over SGLang, plus telemetry
collection. Because request handling is transport-free by construction, mounting `/v1`
later touches no training or pruning code.

**Telemetry** is the coupling line between serving and pruning, and it is new:

```python
class TelemetryStore(Protocol):
    def observe(self, scenario: str, stats: EngineStats) -> None: ...
    def window(self, scenario: str, span: TimeSpan) -> UtilizationTelemetry: ...
```

Per-expert activation counts and layer activation norms, aggregated into a bounded ring
buffer — **deliberately not the SQLite record store**, whose per-request row model would
be swamped by per-token statistics. Rollout engines and serving engines are the same
SGLang processes, so utilization is observable even before the Serve Plane exists.

---

## 3. How they couple

### 3.1 The spine

No subsystem calls another. They are assembled onto `Scenario` by the recipe at boot, and
communicate through two data lines.

```
                     Recipe  ← the only assembly interface
                       │ build()
                       ▼
    ┌──────────────  Scenario  ──────────────┐
    │        binding + release chain          │
    ▼           ▼            ▼                ▼
 runtime     Surface      Trainer      CommitProtocol
 (rollout)  (delivery)   (learning)   (transactional order)
```

- **Horizontal line — receipts.** A record carries the `artifact_ref` of the release that
  produced it, frozen *before* the generating call. Feedback references the receipt.
  Attribution survives concurrent publication.
- **Vertical line — releases.** Every accepted candidate becomes a release in the chain.
  Ancestors stay addressable, which is what makes `lineage:parent` teachers and
  joint rollback possible.

### 3.2 The seven interfaces

| # | Interface | Couples | Contract |
|---|---|---|---|
| 1 | `Recipe` | everything | the only place a method touches the framework: runtime, report type, trainer, surface, base artifact, validator |
| 2 | `Scenario` | all planes | aggregate root; owns binding, release chain, records, trainer, commit protocol |
| 3 | `TrainingBatch` | Data Plane → Train Plane | one type for online and offline; source is invisible downstream |
| 4 | `UpdateCandidate` | Train / Prune / Harness → Gate | the uniform candidate — *the key unification* |
| 5 | `HoneRelease` | Gate → Commit → Serve | joint artifact: weights ⊕ prune manifest ⊕ harness |
| 6 | `ParamSelection` | `sparse_mask` ⇄ `PrunePlan` | shared addressing makes masked training a reversible prune rehearsal |
| 7 | `UtilizationTelemetry` | Serve/Rollout → Prune | observed capacity usage; the evidence in every prune manifest |

### 3.3 The joint release

After co-adaptation, "roll back" must be unambiguous. Rolling back weights while keeping a
harness that was evolved *for* those weights produces a combination that was never
evaluated. So the unit of publication is a triple:

```python
@dataclass(frozen=True)
class HoneRelease:
    model: ArtifactRef                  # weights, or base + adapter/delta
    prune_manifest: ArtifactRef | None  # what this version removed, and on what evidence
    harness: ArtifactRef | None
    metrics: Mapping[str, Any]          # params, $/1k tok, latency, in-domain score
```

Rollback restores the triple. The commit protocol is otherwise unchanged by this — only
the artifact it publishes is composite.

**The honing curve is `metrics` over the release chain.** Parameter count, serving cost,
latency, and in-domain score per release, exposed at `GET /hone/curve`. It is the
system's own evidence for the claim in the README, and it is a projection of data the
system already stores rather than a separate reporting path.

---

## 4. Lifecycles

Three worked examples showing which planes activate.

### 4.1 Offline specialization (no feedback, no harness)

The simplest useful deployment — a team with a corpus and a cost target.

```
DatasetProvider(sft_corpus)
  → Train(algo=sft, update_policy=lora, rollout=None)
  → Gate (held-out eval)  → Commit → release 1..N
  → HonePolicy fires: analyze(telemetry from eval rollouts) → PrunePlan
  → Prune candidate → Gate (max_regression=0.02) → Commit → release N+1
  → recovery: Train(algo=opsd, teacher=lineage:parent, rollout=direct)
  → repeat until CapacityFloor
```

No records, no receipts, no harness. Demonstrates that the Data Plane split is real:
the trainer never learns which provider fed it.

### 4.2 Online RL with harness rollouts

The online deployment shape, plus honing.

```
FeedbackProvider (receipts + /hone/report)
  → Train(algo=grpo, update_policy=full, rollout=harness)
  → Gate → Commit → release, served → more traffic → more feedback
  ⤷ every N commits: HonePolicy → prune candidate → same gate
  ⤷ on plateau: harness during-phase → harness candidate → same gate
```

All three candidate kinds queue at one gate. The commit chain interleaves them, and each
release records which kind produced it.

### 4.3 Full co-adaptation campaign

```
1. pre_training harness phase        GEPA over prompts/skills on the base model
2. train window                      grpo + harness rollouts, one harness release only
3. utilization stationary?           → expert_drop plan → gate → commit
4. recovery                          opsd against lineage:parent (pre-prune ancestor)
5. during_training harness phase     re-evolve harness for the thinned model
6. goto 2 until CapacityFloor
7. post_training harness phase       final polish against frozen weights
8. publish                           HoneRelease: thinned weights ⊕ manifest chain ⊕ harness
```

Step 4 is where the release chain pays for itself: the distillation teacher is the
pre-prune ancestor, already stored, already addressable. No second deployment, no separate
teacher checkpoint management.

---

## 5. HTTP surface

Routes new to Honing are marked in bold.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions`, `/v1/messages` | provider-compatible inference *(reserved — phase 2)* |
| `POST /hone/report` | feedback: `{score?, feedback?, references[], metadata?}` |
| `POST /hone/train` | explicit training instruction (`training_mode: manual` / `hybrid`) |
| `GET`/`POST` `/hone/scenarios` | list / create |
| `GET /hone/scenarios/{s}/releases` | version history |
| `POST /hone/scenarios/{s}/rollback` | restore a release triple |
| `POST /hone/scenarios/{s}/promote` | approve a `pending` release |
| `GET /hone/status` | training, storage, engine state |
| `GET /healthz` | unauthenticated liveness |
| **`GET`/`POST` `/hone/datasets`** | register / list offline dataset manifests |
| **`GET /hone/prune/plans`** | inspect proposed and applied plans with their evidence |
| **`POST /hone/prune`** | trigger a hone step manually |
| **`GET /hone/telemetry`** | current utilization window |
| **`GET /hone/curve`** | the honing curve: release metrics over time |

Bearer-token auth on everything except `/healthz`. The token *is* the service boundary,
and several tokens may be accepted at once so a gateway can rotate without downtime.

---

## 6. Configuration

One YAML per deployment: a flat settings section plus a `services` list the orchestrator
starts in dependency order with readiness probes.

```yaml
implementation: honing.recipe:HoningRecipe

model_source:
  kind: pruned_init
  base: ${MODEL_PATH}
  pipeline:
    - method: expert_drop
      keep_ratio: 0.5
  calibration: {kind: dataset, ref: domain-calib}

data:
  providers:
    - {kind: dataset, manifest: ./data/sft.jsonl, epochs: 2}
    - {kind: feedback}                      # switches to live traffic once the corpus is drained

training:
  algo: grpo
  update_policy: {kind: lora, rank: 32, targets: [q_proj, v_proj]}
  rollout: {kind: harness, adapter: pi, tasks: ./tasks.yaml}
  teacher: {kind: lineage, ref: parent}

hone:
  trigger: {kind: utilization_stationary, window: 200}
  method: expert_drop
  max_regression: 0.02
  recovery: {algo: opsd, steps: 500}
  floor: {min_experts: 16, min_layers: 24}

harness:
  pre_training:      {trigger: once, evolve: {propose: gepa, budget: {steps: 10}}}
  during_training:   {trigger: {on_plateau: in_domain_score}, freeze_weights: true}
  post_training:     {trigger: once, budget: {steps: 5}}

services:
  - {name: sglang,  ready: {http: /health}}
  - {name: teacher, ready: {http: /health}}      # only when teacher.kind == engine
  - {name: hone,    ready: {http: /healthz}}
```

Every block is optional. Omitting `hone:` gives a training system with no pruning;
omitting `harness:` gives weight-only training; omitting both leaves a plain continual
training loop over the full algorithm zoo.

---

## 7. Implementation phases

| Phase | Deliverable | Depends on |
|---|---|---|
| **1** | Spine: Scenario, gate, commit, release chain, transport-free service | — |
| **2** | Data Plane: `DatasetProvider` + `DataContract` boot validation | 1 |
| **3** | Train Plane: algorithm registry (`sft`, `dpo`, `distill` first — no rollout needed) | 2 |
| **4** | Prune Plane standalone + `hone prune` CLI + `expert_drop` | — *(parallelizable)* |
| **5** | Telemetry store; `PrunedInit` wiring | 4 |
| **6** | Rollout providers; on-policy algorithms (`grpo`, `ppo`, `opsd`, `gkd`) | 3, 5 |
| **7** | In-loop `HonePolicy`; prune-as-candidate through the gate | 4, 6 |
| **8** | Harness phases + `CoAdaptScheduler` | 6, 7 |
| **9** | `sparse_mask` update policy ⇄ `PrunePlan` unification | 7 |
| **10** | Serve Plane | 6 |

Phase 4 has no dependency on the rest and produces a shippable artifact on its own: a
pruning toolkit usable without any Honing deployment. Starting there de-risks the novel
part while the spine port proceeds in parallel.

---

## 8. Open questions

Honest gaps a reviewer should press on.

1. **Does prune-then-recover actually preserve in-domain quality?** The whole thesis rests
   on it. The gate protects each individual step, but repeated cuts may compound in ways a
   per-step `max_regression` doesn't catch. Needs an early empirical answer on one MoE
   model before phase 7 — this is the highest-risk assumption in the design.
2. **What is the right utilization signal?** Expert activation counts are the obvious
   choice for MoE routing, but frequency is not utility: a rarely-fired expert may be
   essential to a rare-but-critical input. Saliency-based signals may be necessary, at
   higher cost.
3. **Co-adaptation may be unstable.** Alternating optimization over two coupled surfaces
   can oscillate. `freeze_weights` and the one-release-per-window invariant are the
   safeguards; whether they suffice is empirical.
4. **Sparse-mask ⇄ prune equivalence is asserted, not proven.** A masked model and a
   structurally pruned model differ in numerics (padding, kernel selection). The
   conversion needs a validation step that the gate can see, or the "reversible rehearsal"
   framing overstates the guarantee.
5. **Telemetry retention vs. plan quality.** A bounded ring buffer may not span enough
   traffic to justify a large cut. The window length is a tunable with no principled
   default yet.
6. **Ablation cost.** Running `direct` and `harness` rollout arms to attribute gains
   doubles rollout spend. Whether that is worth it per-deployment, or only during method
   development, is an operator decision the config should make explicit.
