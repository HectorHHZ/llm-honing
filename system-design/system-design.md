# Honing — System Design

The subsystem contracts, how they couple, the rules every move passes, and what runs when.

Read [`README.md`](README.md) first for what Honing is, why Slime is its one training
dependency, and how it compares to the surrounding stack. This document is the design
others build against. Where a contract is already code, the section links to it.

**Contents**

1. [Design principles](#1-design-principles)
2. [The subsystems](#2-the-subsystems)
3. [How they couple](#3-how-they-couple)
4. [Lifecycles](#4-lifecycles)
5. [HTTP surface](#5-http-surface)
6. [Configuration](#6-configuration)
7. [Implementation steps](#7-implementation-steps)
8. [Open questions](#8-open-questions)
9. [The walking skeleton](#9-the-walking-skeleton)

---

## 1. Design principles

Five rules decide most of the structure below. The first four are established practice
for continual-learning infrastructure; the fifth is what Honing adds.

**P1 — One spine, planes meet only through contracts.** Every plane is assembled onto
one campaign and its release chain. No plane calls another directly; each depends only
on the types and protocols in `honing/core`. This is why a plane can be developed,
faked, tested, and replaced on its own.

**P2 — Candidates are uniform.** Anything that wants to change what is served becomes an
`UpdateCandidate`, is judged by one gate, and settles through one release chain. Training
steps, prunes, quantizations, and harness changes all take this path. *This is the single
decision that keeps Honing's scope tractable* — compression inherits evaluation,
publication, and rollback instead of reimplementing them.

**P3 — Durable before destructive.** The commit record of a change reaches disk before
the head moves, before any record is compacted, and before any checkpoint is deleted. A
crash mid-commit is recoverable by construction.

**P4 — Version-frozen attribution.** A sample names the release and the harness that
produced it. A batch that spans releases or harnesses is refused whole, because rewards
earned under different weights or different harnesses are not comparable.

**P5 — Capacity and precision are reversible dimensions.** Model size and numeric
precision are not fixed at boot. They are proposed, measured, published, and — when a
change turns out wrong — rolled back like weights. Every reduction records its plan on
the release that applied it.

### 1.1 The dependency boundary

- `honing/core` imports only the standard library, so every plane can depend on it
  without pulling in a framework.
- Planes depend on `honing.core` and never on each other.
- Only `honing/backends/` may import a training framework. Slime is the one such
  dependency; Reef is a design reference and is never imported.

[`tests/contracts/test_boundaries.py`](../tests/contracts/test_boundaries.py) enforces
all three.

---

## 2. The subsystems

### 2.1 Core — the contracts

[`honing/core`](../honing/core/__init__.py) holds value types and protocols only. Value
types are frozen dataclasses that validate themselves on construction and raise
`ContractError`, so an impossible value never exists long enough to reach a trainer.

| Module | Holds |
|---|---|
| `structure` | `StructureAddress` (layer, expert, attention head, FFN channels, hidden channels, parameter mask) and `ArchitectureSpec` |
| `numerics` | `QuantScheme`, `NumericsPolicy` |
| `weights` | `CheckpointRef`, `ModelSnapshot`, `ServingWeights`, the master weight format |
| `data` | `DataContract`, `PolicySample`, `PreferencePair`, `TrainingBatch` |
| `training` | `TrainJobSpec`, `UpdatePolicy`, `TeacherRef`, `RecoverySpec`, `ALGORITHM_REQUIREMENTS` |
| `compression` | `ImportanceMap`, `CompressionBudget`, `PrunePlan`, `QuantizePlan`, `CompressionStep` |
| `harness` | `HarnessSpec`, `Task`, `EpisodeOutcome`, `Grade`, `EpisodeRecord`, `ProposeResult`, `HarnessPhaseConfig`, `RunnerSupport`, `MethodRequirements` |
| `chat`, `endpoints`, `state` | `ChatMessage`, `ChatReply`; `PolicyEndpoint`; the method-state rules (`validate_method_state`) |
| `candidate` | `UpdateCandidate`, `PolicyRef`, `EvaluationResult`, `SelectionDecision` |
| `release` | `HoneRelease` |
| `backend` | `BackendSupport` |
| `schedule` | `Move`, `CampaignState` |
| `protocols` | every plane protocol (listed per plane below) |

### 2.2 Model Source — where the initial model comes from

```python
class ModelSource(Protocol):
    def materialize(self, work_dir: Path) -> ModelSnapshot: ...
```

A campaign starts from either an open checkpoint used as-is, or a large model the
Compression plane thins first (`CompressedInit`, planned). Either way the result is
release 0, and its master weights are **Hugging Face safetensors** — the one master
format every editor, quantizer, and trainer in Honing reads. A `CompressedInit` records
its plans on release 0, so a model that arrived thinned is as traceable as one thinned
later.

### 2.3 Data Plane — two sources, one batch type

```python
class DataProvider(Protocol):
    def next_batch(self, contract: DataContract) -> TrainingBatch | None: ...

class RolloutProvider(Protocol):
    def collect(self, tasks, policy: PolicyRef, harness: HarnessSpec | None,
                release_id: str | None, samples_per_task: int) -> TrainingBatch: ...
```

Offline datasets (SFT corpora, preference pairs, distillation sets) and online feedback
both produce a `TrainingBatch`, and **the trainer cannot tell which one fed it**. Each
algorithm declares the `DataContract` it consumes — token sequences, on-policy samples,
grouped on-policy samples, or preference pairs — and the campaign checks the batch against
it before training starts.

Rollout samples carry `source_release` and `source_harness`; offline samples carry
neither. That attribution is what the `batch_from_one_release` rule reads (P4).

### 2.4 Train Plane — five independent choices

```python
@dataclass(frozen=True)
class TrainJobSpec:
    algorithm: AlgorithmName          # sft | distill | opd | gkd | ppo | grpo | dpo
    update_policy: UpdatePolicy       # full | lora | param_subset | sparse_mask
    numerics: NumericsPolicy          # bf16 | fp8 | fake_quant_qat (with its QuantScheme)
    rollout: RolloutKind              # none | direct | harness
    teacher: TeacherRef | None        # lineage | release | external, served by an engine or in process
```

[`ALGORITHM_REQUIREMENTS`](../honing/core/training.py) states what each algorithm needs
and which executor runs it. The campaign checks a job against it before calling any
trainer.

| `algorithm` | Consumes | Rollout | Teacher | Executor | How Slime runs it |
|---|---|:---:|:---:|---|---|
| `sft` | token sequences | ✗ | ✗ | slime | native `--loss-type sft_loss` |
| `distill` | token sequences | ✗ | ✓ | slime | custom-loss hook |
| `opd` | on-policy samples | ✓ | ✓ | slime | native `--use-opd`; self-distillation when the teacher is a lineage release |
| `gkd` | on-policy samples | ✓ | ✓ | slime | custom-loss hook |
| `ppo` | on-policy samples with rewards | ✓ | ✗ | slime | native estimator |
| `grpo` | grouped samples with rewards | ✓ | ✗ | slime | native estimator |
| `dpo` | preference pairs | ✗ | ✗ | local | — Slime's data flow has no preference-pair path |

**Update policy.** `full` trains everything. `lora` passes Megatron's LoRA arguments
through Slime. `param_subset` selects parameters by name pattern, which Slime supports
natively. `sparse_mask` selects elements with the same `ParamMask` a pruning plan uses,
which makes masked training a reversible rehearsal for a structural cut; Slime has no
built-in path for it, so it runs through a train-step hook.

**Numerics.** `bf16` is the default. `fp8` trains in FP8 (experimental in Slime).
`fake_quant_qat` keeps BF16 master weights but quantizes them in the forward pass under
one `QuantScheme` (INT4 is beta in Slime), so the result tolerates exactly that scheme
when served.

**Teachers.** A teacher served by its own `engine` needs only a compatible tokenizer. One
loaded `in_process` shares the student's model definition and must have the same
architecture — a constraint Slime's in-process teacher mode imposes.

**Master format.** Slime trains from Megatron's `torch_dist` checkpoints. A `WeightConverter`
converts the master to `torch_dist` before each Slime job and back to safetensors after
it (Slime ships `tools/convert_hf_to_torch_dist.py`); the conversion never reaches a
release. For a large MoE this conversion is a real per-job cost (see §8).

**Backends declare what they support.**

```python
@dataclass(frozen=True)
class BackendSupport:
    name: str
    architectures: Literal["uniform", "heterogeneous"]
    train_numerics: frozenset[NumericsMode]
    weight_formats: frozenset[WeightFormat]
    algorithms: tuple[tuple[AlgorithmName, Literal["native", "hook", "unsupported"]], ...]

class Trainer(Protocol):
    support: BackendSupport
    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult: ...
```

Two backends are planned: `backends/slime` for everything but DPO, and
`backends/local_dpo` for DPO. v1 backends declare `uniform` architectures only.

### 2.5 Compression Plane — pruning and quantization

The Compression plane has no dependency on training, serving, or a running campaign; the
campaign depends on *it*. It is usable as a standalone command (`hone compress`, planned),
as the `CompressedInit` pipeline, or in the loop.

**Pruning is three stages, each with its own registry**, so methods combine instead of
multiplying:

```python
class ImportanceScorer(Protocol):
    signal: SignalKind                 # telemetry | calibration | weights | gradients
    def score(self, model: ModelSnapshot, telemetry: UtilizationTelemetry | None) -> ImportanceMap: ...

class PruneSelector(Protocol):
    def select(self, importance: ImportanceMap, model: ModelSnapshot, budget: CompressionBudget) -> PrunePlan: ...

class ModelEditor(Protocol):
    def apply(self, model: ModelSnapshot, plan: PrunePlan) -> ModelSnapshot: ...
```

What a plan can remove is a `StructureAddress`: a `Layer`, an `Expert`, an
`AttentionHead`, `FFNChannels` (of a dense FFN or one expert), `HiddenChannels` (the
residual width, which reshapes every layer at once), or a `ParamMask` that keeps the
shape. Layer drop, expert drop, head and channel pruning, and unstructured masks are
therefore scorer and selector pairs, not new plane types.

```python
@dataclass(frozen=True)
class PrunePlan:
    removals: tuple[StructureAddress, ...]
    before: ArchitectureSpec
    after: ArchitectureSpec
    importance: ImportanceMap          # the scores the selection was made from
    estimated: CostDelta
```

The plan is stored on the release it produces, so every cut a model has taken can be read
back from its release chain.

**v1 accepts uniform architectures only**: every layer keeps the same shape, although
*which* experts or heads each layer keeps may differ. Slime builds its model from one
configuration; a plan that leaves layers of different shapes is refused by
`architecture_supported` before the editor runs. `ArchitectureSpec` keeps a shape per
layer regardless, so heterogeneous support is a backend change, not a type change.

**Recovery is training.** A compression step names a `RecoverySpec`; the Train plane runs
it.

```python
@dataclass(frozen=True)
class CompressionStep:
    transform: PrunePlan | QuantizePlan
    recovery: RecoverySpec | None = None
    settle: Literal["after_recovery", "per_stage"] = "after_recovery"
```

`after_recovery` is the default because a freshly pruned model almost always regresses;
judging it before recovery would reject good cuts. `per_stage` publishes the raw cut and
leaves recovery to a later train move, for measuring what a cut costs on its own. After a
shape change the recovery teacher — typically the pre-cut ancestor — must be served by an
engine (`recovery_teacher_fits_new_shape`).

**Quantization builds serving weights; it never modifies the master.**

```python
class Quantizer(Protocol):
    def quantize(self, model: ModelSnapshot, plan: QuantizePlan) -> ServingWeights: ...

@dataclass(frozen=True)
class ServingWeights:
    checkpoint: CheckpointRef
    scheme: QuantScheme
    source: CheckpointRef              # the master these were quantized from
```

**One `QuantScheme` is shared by three consumers**: quantization-aware training
(`NumericsPolicy`), post-training quantization (`QuantizePlan`), and the engine that loads
the serving weights. Sharing one object is what lets `qat_scheme_matches_export` refuse an
export under a scheme other than the one the weights were trained for — a mismatch that
otherwise fails silently.

Quantization is a subsystem of the Compression plane rather than a plane of its own. A
separate plane would duplicate the gate and the release path, and the ordering
constraints between pruning and quantization — quantize after the last cut; a cut
invalidates serving weights — need one owner.

### 2.6 Harness Evolve — three phases, one contract

```python
@dataclass(frozen=True)
class HarnessSpec:
    agent: str                                       # claude_code | codex | a programmable agent
    files: tuple[tuple[str, HarnessFile], ...]       # path → instructions | skill | command | settings | hook | program

class EpisodeRunner(Protocol):
    support: RunnerSupport
    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome: ...

class TaskGrader(Protocol):
    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade: ...   # score in [0, 1], plus feedback
```

A runner only runs the agent; a grader judges the outcome, because grading depends on the
task and the same grader serves the gate, a method's own minibatches, and training
rollouts. An `EpisodeRecord` is an outcome plus its grade, and `run_graded` is the one
place that pairs them.

A harness is a set of files for one agent. File kinds map onto the node kinds Reef's
harness layer renders, which keeps a Reef-backed runner possible later without changing
this type.

The planned runner is built on Slime's agent layer: `BaseHarness.run` launches the agent
in a sandbox after Honing writes the spec's files into its workspace; the `Sandbox`
protocol lets a local container stand in for Slime's E2B sandbox; the agent's own
transcript (`{workdir}/.harness/trajectory.jsonl`) is what reflection methods read; and
Slime's Anthropic and OpenAI adapters capture token-level samples when the episode is a
training rollout. **Gate evaluations and training rollouts use the same runner**, so the
harness that wins the gate is exactly the harness trained under.

```python
class EvolveMethod(Protocol):
    name: str
    requires: MethodRequirements
    selector: CandidateSelector
    state_schema_version: int
    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]: ...
    def propose(self, context: EvolveContext) -> ProposeResult: ...          # proposal or None, new state, episodes spent
    def record(self, state, proposal, evaluation, decision) -> Mapping[str, object]: ...
```

Methods that reflect or propose call a model through `ChatModel`, and runners find the
engine behind a policy through `PolicyEndpoints`; both are protocols in `honing/core`, and
their implementations live outside it.

Six properties let a future method — GEPA, SkillClaw, Meta-Harness, a failure-driven
self-proposer — plug in without infrastructure changes:

1. **A proposal is a complete spec.** Methods that edit and methods that write whole
   compositions both produce one.
2. **A method may run and grade its own episodes.** `EvolveContext` carries the runner,
   the grader, and an episode budget, and `EvolveContext.episode` runs and grades in one
   call, for minibatches against candidates that are not the head.
3. **Method state is released, and never lost.** `propose` returns the new state even
   when it proposes nothing, and a skip is written to the commit record with the episodes
   it spent. The state must be JSON-native and carry the `schema_version` its method
   declares; state under another version is refused before the method reads it.
4. **Evaluations are per task.** `EvaluationResult` keeps score vectors and episode
   records, which Pareto-style selection needs.
5. **Selection belongs to the method.** Its `selector` replaces the gate's default.
6. **`record` sees every verdict**, accepted or rejected, because archive-based methods
   learn from rejections.

| Phase | When | Why it belongs there |
|---|---|---|
| `pre_training` | before the first weight step | prompt and skill search is far cheaper than gradient steps, and a better harness gives better data |
| `during_training` | interleaved, on a schedule or plateau | weights and harness co-adapt, each retuned against the other |
| `post_training` | after the final compression | a last pass against the weights that ship — the serving weights when they exist |

**The co-adaptation invariant.** A training window samples under exactly one harness
release. A batch spanning harness releases is refused whole (`batch_from_one_release`),
the same rule P4 applies to weight releases.

**v1 limits.** Claude Code and Codex have fixed agent loops, so methods that rewrite the
loop itself need a programmable agent (Slime's `BaseHarness` allows one that drives its
own loop). Evolution from real user traffic waits for the Serve plane. Slime's adapters
forward to a local SGLang engine, so a runner built on them evaluates only locally served
policies.

### 2.7 Campaign — the next move and the rules it must pass

[`honing/campaign`](../honing/campaign/__init__.py) owns the loop:

```python
class CoAdaptScheduler(Protocol):
    def next_move(self, state: CampaignState) -> Move: ...   # train | prune | quantize | harness | stop

campaign.start(source, work_dir, harness_seed)   # release 0
campaign.run(scheduler)                           # one candidate per move, settled by the gate
campaign.rollback(release_id)                     # restore a whole release
```

For each move the campaign builds one candidate, checks the rules, runs the gate, and
publishes or records the rejection. **Rules are checked before any weight is trained,
edited, or quantized**, so a refused step costs no compute and leaves nothing behind.

| Rule | Refuses | Checked |
|---|---|---|
| `architecture_supported` | a shape the backend cannot build (v1: any non-uniform one) | at start; before editing; after training |
| `master_is_hf_safetensors` | a candidate whose master is not in the master format | before the gate |
| `serving_built_from_master` | serving weights quantized from a different master than they ship with | before the gate |
| `qat_scheme_matches_export` | exporting under a scheme other than the one the weights were trained for | before quantizing |
| `recovery_teacher_fits_new_shape` | an in-process teacher after a shape-changing cut | before editing |
| `algorithm_requirements_met` | a job whose rollout, teacher, or precision its algorithm or backend cannot serve | before drawing data or editing |
| `batch_meets_contract` | a batch of a kind the algorithm does not consume, or no batch | before training |
| `batch_from_one_release` | samples from a release or harness other than the head's | before training |
| `method_supported_by_runner`, `spec_runnable` | an evolution method or harness the runner cannot run | at start; per proposal |
| `method_state_schema_matches` | method state written under a schema the method does not read, for example after a rollback | at start; before each proposal |

A harness method's state is carried across rejected and skipped steps in memory and
published with the next accepted release, whatever its kind. A harness move may name its
phase; `HarnessComponents.phases` gives each phase its own tasks, and the phase scheduler
will enforce each phase's budgets.

The default scheduling policy (planned): pre-training harness pass → *N* training commits
→ when utilization is stable, prune with recovery → retune the harness → repeat until the
capacity floor → quantize → post-training harness pass.

### 2.8 Gate and release chain

[`honing/release`](../honing/release/__init__.py) is the spine every candidate settles
through.

**The gate scores what ships.** A candidate with serving weights is evaluated on them; the
gate refuses an evaluation that scored the master instead, because that would hide exactly
the regression quantization can cause. Selection policies: `WinLossSelector` for changes
meant to improve quality (training, harness), `NoRegressionSelector` for changes made for
cost (prune, quantize), `AcceptAll`, or a method's own selector.

**Comparisons are paired.** Every score is in `[0, 1]`, and selectors compare candidate and
incumbent task by task. Every gated candidate, published or rejected, records
`gate_tasks`, both means, and a bootstrap interval on the mean paired difference
(`gate_difference_low`, `gate_difference_high`), so a scheduler can tell a plateau from a
real difference.

**A release is the whole accepted candidate.**

```python
@dataclass(frozen=True)
class HoneRelease:
    release_id: str
    candidate: UpdateCandidate   # master, serving, architecture, transforms, harness, trained numerics, method state, metrics
    decision: SelectionDecision | None
```

Rolling back weights while keeping a harness evolved for them, or restoring a master while
keeping serving weights quantized from another, produces a combination no gate evaluated.
Rollback restores all of it.

**The commit record is written before the head moves (P3).** `ReleaseChain` appends each
publish, rejection, skip, and rollback to a JSON-lines record and flushes it to disk first.
A value JSON cannot represent exactly is an error, never a string.
Rebuilding a chain from that record is not implemented yet; the ordering that makes it
possible is, and is tested.

**The honing curve** is each release's parameter count and metrics over the chain
(`ReleaseChain.curve()`), a projection of data the chain already stores.

### 2.9 Serve Plane and telemetry — reserved

The Serve plane (live traffic, `/v1` routes, artifact activation) is not built. It will
load a release's serving weights when they exist.

```python
class TelemetryStore(Protocol):
    def observe(self, stats: UtilizationTelemetry) -> None: ...
    def window(self) -> UtilizationTelemetry | None: ...
```

Telemetry — per-expert routing counts and layer activation norms — is the line from
serving to compression. It is aggregated per window rather than stored per request, and
it can be filled before a Serve plane exists, because rollouts run on the same engines.

---

## 3. How they couple

### 3.1 The spine

```
ModelSource ──► Campaign.start ──► ReleaseChain: release 0
                    │
          CoAdaptScheduler.next_move
                    │
     ┌──────────────┼───────────────┬───────────────┐
   train          prune          quantize         harness
   Trainer        Scorer →       Quantizer        EvolveMethod
   + Data or      Selector →                      + EpisodeRunner
     Rollout      Editor → recovery (Trainer)
     └──────────────┴───── UpdateCandidate ────────┘
                    │  rules (before any weight is touched)
                    ▼
                  Gate ── Evaluator (serving weights when present) ── Selector
                    │
         ReleaseChain.commit  |  ReleaseChain.record_rejection
```

### 3.2 The interfaces

| Interface | Couples | Contract |
|---|---|---|
| `TrainingBatch` | Data → Train | one type for online and offline data; the source is invisible downstream |
| `UpdateCandidate` | every plane → Gate | the uniform candidate — *the key unification* |
| `HoneRelease` | Gate → Release chain → Serve | the whole candidate, published and restored together |
| `PolicyRef` | Release → Runner and Rollout | the exact weights an episode runs against, master or serving |
| `StructureAddress`, `ArchitectureSpec` | Compression ⇄ Train | what a plan removes and the shape a backend must build |
| `QuantScheme` | Train ⇄ Compression ⇄ Serve | one scheme for fake-quant training, export, and loading |
| `HarnessSpec`, `EpisodeRecord` | Harness ⇄ Gate ⇄ Rollout | the files an agent reads and what one episode produced |
| `BackendSupport`, `RunnerSupport` | backends → Campaign rules | what a backend or runner declares it can do |
| `UtilizationTelemetry` | Serve and Rollout → Compression | observed routing, the signal importance is scored from |

### 3.3 The joint release

The unit of publication is the whole candidate: master weights, serving weights,
architecture, the transforms applied, the harness, the precision the master was trained
in, the evolution method's state, and metrics. A release therefore answers three
questions at once — what is served, how it got that way, and what to restore if it is
wrong.

---

## 4. Lifecycles

### 4.1 Offline specialization (no feedback, no harness evolution)

```
DatasetProvider(sft corpus)
  → train (sft, lora)                          → gate (win/loss)       → release 1..N
  → prune (expert scorer from rollout telemetry, uniform top-k, recovery: distill from
           the pre-cut release served by an engine)   → gate (no regression) → release N+1
  → repeat until the capacity floor
  → quantize (PTQ, int4 group 128)             → gate on serving weights → release M
```

### 4.2 Online RL with harness rollouts

```
RolloutProvider(harness) — samples name the head's release and harness
  → train (grpo, full)   → gate → release, served → new rollouts under that release
  ⤷ every N commits: prune with recovery → same gate
  ⤷ on plateau: harness evolution (during_training) → same gate
```

### 4.3 A full campaign, with both quantization flows

```
1. pre_training harness pass           an evolution method over instructions and skills
2. train window                        grpo under one harness release
3. utilization stable?                 prune → recover (opd from the pre-cut release, engine-served) → gate
4. during_training harness pass        retune the harness for the thinned model
5. repeat 2–4 until the capacity floor
6a. flow A: quantization-aware         train with fake_quant_qat under scheme S → quantize with S
            (a different export scheme is refused)
6b. flow B: post-training              quantize the BF16 master with scheme S
7. post_training harness pass          against the serving weights
8. rollback at any point               restores weights, serving weights, harness, and method state together
```

---

## 5. HTTP surface

Reserved for the Serve plane; none of it is built. New routes are marked in bold.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions`, `/v1/messages` | provider-compatible inference |
| `POST /hone/report` | feedback: `{score?, feedback?, references[], metadata?}` |
| `POST /hone/train` | an explicit training instruction |
| `GET`/`POST` `/hone/campaigns` | list / create |
| `GET /hone/campaigns/{c}/releases` | version history |
| `POST /hone/campaigns/{c}/rollback` | restore a release |
| `GET /hone/status` | training, storage, and engine state |
| `GET /healthz` | unauthenticated liveness |
| **`GET`/`POST` `/hone/datasets`** | register and list offline dataset manifests |
| **`GET /hone/compression/plans`** | proposed and applied plans with their importance scores |
| **`POST /hone/compression`** | trigger a compression move |
| **`GET /hone/telemetry`** | the current utilization window |
| **`GET /hone/curve`** | the honing curve |

---

## 6. Configuration

One YAML per campaign. Every block except `model_source` and `training` is optional:
omitting `compression` gives training without compression; omitting `harness` gives
weight-only training.

```yaml
model_source:
  kind: compressed_init              # or: checkpoint
  base: ${MODEL_PATH}                # Hugging Face safetensors
  pipeline:
    - {scorer: expert_activation, selector: uniform_top_k, keep_ratio: 0.5}

data:
  providers:
    - {kind: dataset, manifest: ./data/sft.jsonl, epochs: 2}
    - {kind: feedback}

training:
  backend: slime                     # dpo runs on backend: local_dpo
  algorithm: grpo
  update_policy: {kind: lora, rank: 32, targets: [q_proj, v_proj]}
  numerics: {mode: bf16}
  rollout: harness

compression:
  trigger: {kind: utilization_stable, window: 200}
  prune:
    scorer: expert_activation
    selector: uniform_top_k
    budget: {keep_ratio: 0.75, min_experts_per_layer: 16}
    recovery: {algorithm: opd, steps: 500, teacher: {kind: lineage, ref: parent, placement: engine}}
    settle: after_recovery
  quantize:
    scheme: {weight_format: int4, granularity: group, group_size: 128}

gate:
  default: {kind: win_loss}
  prune: {kind: no_regression, max_regression: 0.02}
  quantize: {kind: no_regression, max_regression: 0.01}

harness:
  agent: claude_code
  method: {name: failure_driven_self_proposer}
  phases:
    pre_training:    {trigger: once, budget: {steps: 10}}
    during_training: {trigger: {on_plateau: in_domain_score}}
    post_training:   {trigger: once, budget: {steps: 5}}
```

---

## 7. Implementation steps

Steps are ordered by risk: the assumption the whole design rests on is tested first.

| Step | Deliverable | Status |
|---|---|---|
| **0** | Core contracts: value types and protocols in `honing/core` | done |
| **1** | Walking skeleton: campaign, rules, gate, and release chain, with every other plane faked | done |
| **2** | A real expert-drop editor on a small MoE, reading and writing safetensors; `hone compress` | next |
| **3** | `backends/slime`: SFT through Slime with safetensors ⇄ `torch_dist` conversion, plus a test that pins every Slime argument Honing emits | |
| **4** | `DatasetProvider` and `FeedbackProvider` | |
| **5** | A rollout provider and runner on `slime.agent`; GRPO and on-policy distillation | |
| **6** | Quantization: PTQ through Slime's tools; quantization-aware training through `numerics` | |
| **7** | In-loop compression with recovery, and a utilization-driven scheduler | |
| **8** | Harness phases with a first real evolution method; the harness contracts it needs ([#15](https://github.com/HectorHHZ/llm-honing/issues/15)) are in | in progress |
| **9** | `backends/local_dpo` | |
| **10** | `sparse_mask` training unified with `PrunePlan` | |
| **11** | Serve plane | |

Step 2 depends on nothing else and answers the riskiest question (§8.1) on its own.

---

## 8. Open questions

Gaps a reviewer should press on.

1. **Does prune-then-recover preserve in-domain quality?** The whole thesis rests on it.
   The gate protects each step, but repeated cuts may compound in ways a per-step limit
   does not catch. This needs an empirical answer on one MoE model before step 7.
2. **What is the right importance signal?** Routing frequency is not utility: a rarely
   used expert may be essential to a rare but critical input. Saliency signals may be
   necessary, at higher cost.
3. **Co-adaptation may be unstable.** Alternating optimization over two coupled surfaces
   can oscillate; the one-harness-per-window invariant and phase scheduling are the
   safeguards, and whether they suffice is empirical.
4. **Masked training and a structural cut are not numerically identical.** Padding and
   kernel selection differ, so converting a mask into a cut needs a step the gate can see.
5. **Telemetry windows may be too short.** A bounded window may not cover enough traffic
   to justify a large cut; there is no principled default length yet.
6. **Ablation doubles rollout cost.** Running direct and harness rollout arms to attribute
   a gain is worth it during method development; per deployment it is an operator choice.
7. **`slime.agent` is not a declared extension point.** It is used by one Slime example
   and carries no stability promise; step 5 needs a contract test on `BaseHarness`,
   `Sandbox`, and the adapters.
8. **Unverified details.** The Codex transcript format; whether transcripts show which
   skills an agent loaded; whether SGLang loads heterogeneous architectures.
9. **Format conversion per job.** Converting a large MoE between safetensors and
   `torch_dist` around every Slime job may dominate short jobs; caching the converted
   master per release is the likely mitigation.
10. **Chain recovery is not implemented.** The record is written first; replaying it after
    a crash is still to be built.

---

## 9. The walking skeleton

[`tests/test_walking_skeleton.py`](../tests/test_walking_skeleton.py) runs a campaign
through every plane on a CPU. The campaign, the rules, the gate, the selectors, the
episode evaluator, and the release chain are real; everything else is a fake in
[`tests/fakes/`](../tests/fakes/__init__.py) that satisfies the same protocol a real
implementation will.

What it proves:

- **Every seam is crossed.** One campaign trains, prunes with recovery, quantizes, and
  evolves the harness; each step is a candidate judged by the one gate.
- **The joint release holds.** Rolling back to the training release restores its master,
  its lack of serving weights, its architecture, its harness, and its method state.
- **The gate scores what ships.** Quantize and harness candidates are evaluated on
  serving weights.
- **Refusals happen before work.** A non-uniform plan never reaches the editor; a
  mismatched export scheme never reaches the quantizer; a batch from another release
  never reaches the trainer; a rollout job without a rollout provider never edits weights.
- **Rejection leaves the head serving**, and the rejection is recorded.

[`tests/contracts/test_core_protocols.py`](../tests/contracts/test_core_protocols.py)
checks that every protocol has an implementation and that invalid values are refused at
construction; [`tests/test_campaign_rules.py`](../tests/test_campaign_rules.py) gives each
rule a case it allows and one it refuses.

What it does not prove: that any real method works. Each implementation step replaces a
fake and must keep the skeleton passing.
