# Base Laya: Overcooked context and choice experiment

This experiment runs the unmodified English base checkpoint
`convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
The root `model.safetensors` SHA-256 is
`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
This is not `laya-typed-decisions`, a Tank checkpoint, a cloud API or a protocol stand-in.

The feature base is `feat/overcook_ai` at `6f252b2`. All experiment code,
candidate files and results belong to `experiment/overcook_laya-harness`.
The original integration was merged in PR #190. This experiment is a separate
follow-up targeting `main`; it leaves the original feature branch unchanged.

## Result

Completed on November 15, 2026: **15 real-model episodes / 6,000 game decisions**
and 10 native control episodes. Every episode ran 400 ticks. All 25 archives
(120 original files) passed checksums and fresh native action replay.
The post-run weight hash still matches the original published checkpoint.
The observed maximum was 1,310 input tokens; all 6,000 replies retained the
complete state and all six options, with zero truncation flags.

All six harnesses tied on both search layouts; no candidate qualified for the
validation gate. The actual `Gate.judge` was therefore not invoked. The seed was
retained, evaluated once on the validation map, then evaluated on the two
heldout maps after selection was frozen. Other harnesses were not evaluated
on validation/heldout, and no heldout score was used to revise a prompt.

| Harness | Search: cramped_room | Search: asymmetric_advantages | Mean |
| --- | ---: | ---: | ---: |
| seed | 0 | 180 | 90 |
| readable | 0 | 180 | 90 |
| readable_objective | 0 | 180 | 90 |
| readable_local_effects | 0 | 180 | 90 |
| local_progress | 0 | 180 | 90 |
| local_progress_no_history | 0 | 180 | 90 |

The two adaptive candidates were frozen only after inspecting the initial
search traces. `local_progress` asks which immediate effect makes progress and
distinguishes turning from interacting. Its no-history ablation removes recent
joint actions to test repetition anchoring. Neither improved the raw score.
See [adaptive-proposals.json](adaptive-proposals.json) for the rationale and
`candidates/` for the exact delivered instruction/settings files.

| Split / layout | Retained base Laya | Chef 0 always STAY | Rule chef 0 |
| --- | ---: | ---: | ---: |
| Search: cramped_room | 0 | 0 | 40 |
| Search: asymmetric_advantages | 180 | 180 | 320 |
| Validation: coordination_ring | 0 | 140 | 260 |
| Heldout: forced_coordination | 0 | 0 | 0 |
| Heldout: counter_circuit_o_1order | 100 | 100 | 160 |

The retained seed chose EAST on **all 2,000** decisions across its five layouts.
Across every candidate/game, chef 0 never changed its held object and delivered
zero rewarded soups. Chef 1 made all 59 rewarded deliveries across the 15 model
episodes; these are deliveries across different experimental conditions, not
59 independent successes. The search score of 180 and heldout score of 100
match STAY controls. On the validation map, repeated eastward movement reduced
team score by 140 relative to STAY. This experiment does not establish a useful
model contribution or a harness improvement.

Across all game decisions, action counts were EAST 5,231, INTERACT 515,
NORTH 42 and STAY 212; SOUTH and WEST were never selected. Rewording altered
the action distribution without producing ingredient pickup or delivery.
This result concerns these six harnesses with fixed base-Laya weights and
primitive actions; it does not show that all harness designs would fail.

There were three preflight decision requests (see `preflight.json`) and eight
post-selection diagnostic requests: 6,011 recorded model API decisions in total.
The eight post-selection requests are excluded
from the 6,000 game-decision count, as are the three preflight requests. Four simple six-way customer-routing checks
returned the expected labels (including reversed label meanings). This checks
basic semantic inference, not game capability. Four orderings of the same
initial kitchen question selected EAST, EAST, EAST and NORTH. Option order can
affect the answer; that diagnostic was not used to revise candidates or claim
a full-game improvement. All formal games used the existing canonical JSON
wire ordering. Raw requests/replies are in [diagnostics.json](diagnostics.json).

Useful entry points:

- [scores.csv](scores.csv): every episode's raw score and token maxima.
- [summary.json](summary.json): frozen selection, totals and identities.
- [behavior-audit.json](behavior-audit.json): native delivery attribution and inventory changes.
- [weight-verification.json](weight-verification.json): exact before/after weight hashes.
- [archive-verification.json](archive-verification.json): completed native archive verification.
- [examples/readable_local_effects.json](examples/readable_local_effects.json): an exact real request/reply for the designed context and choices.

## Fixed conditions

- Chef 0 is controlled by Laya, choosing an argmax over exactly six native actions.
- Chef 1 uses the same independent `solo` greedy policy on every episode.
- All episodes run exactly 400 joint ticks with manual cooking; two soups is not an early stop.
- Weights, native rules, teammate, action keys and score scale do not change.
- Search uses `cramped_room` and `asymmetric_advantages` only.
- Validation uses `coordination_ring`; only the strict search winner can enter the gate.
- Heldout uses `forced_coordination` and `counter_circuit_o_1order`, after selection is frozen.
- Matched controls use chef 0 always STAY and chef 0 on the public-state rule policy.

Raw team score is the primary metric. Honing's grader uses the monotone
`score / (score + 40)` transform. Team score can come entirely from the fixed
teammate; report controls and behavior before attributing useful play to Laya.
There is one deterministic episode per layout/condition, not repeated independent seeds.
These five layouts cannot establish broad generalization or statistical significance.

## What Codex changes

Codex designs the harness templates and candidates; Laya chooses the executed
actions. Real native trajectories and Honing grades judge those candidates.
No SFT, GRPO or model-weight update occurs.

| Initial candidate | Context | Action question/options |
| --- | --- | --- |
| `seed` | Existing full JSON and four recent joint outcomes | Existing instruction and static descriptions |
| `readable` | English kitchen summary, full ASCII terrain, both chefs, all objects, recipes/rules and history | Same instruction/options as seed |
| `readable_objective` | Same readable context | Explicit recipe-stage instruction; same static options |
| `readable_local_effects` | Same readable context | Same new instruction; each option describes its current local physical effect |

The last variant is a change in controller assistance as well as wording:
its labels expose rule-derived interactions such as taking an onion, adding it
to a pot or collecting ready soup. It keeps all six options, including no-ops.
It does not choose a goal, compute routes, run a future simulator, reveal the
teammate's pending choice or replace Laya's answer. Movement remains primitive.
The preceding ablations help distinguish context, instruction and local-effect changes.

The protocol permits at most two additional candidates based only on search
feedback. Exact delivered files are preserved under `candidates/`; existing
candidates are not overwritten. Validation and heldout feedback cannot revise them.

Search selects a strictly higher mean raw score than seed, with ties retaining
the earlier candidate. A qualifying candidate enters the actual Honing
`Gate(EpisodeEvaluator(...), WinLossSelector())` on the validation task; a tie
rejects it. The selected harness is frozen before either heldout map is run.
Search/heldout episodes use Honing's actual `run_graded`, Overcooked runner and
native-replay grader, not a standalone approximate score.

## Runtime and token accounting

The game uses its existing isolated Windows Python 3.10 environment. Inference
uses the existing Linux/WSL CUDA Python environment on the local RTX 5060 Laptop
GPU (8 GB). The official Laya source is pinned to
`970dc8c5f63d7b886a68409493f37d569424f933` (0.3.20), with torch 2.14.0+cu130,
transformers 5.17.0 and bfloat16 inference. No shared package was upgraded.
Source, model downloads and caches are kept under this integration's ignored
`.cache/` on D:; the existing CUDA interpreter is reused.

The frozen serving limits are 2,048 input tokens and 512 question/option tokens,
the shared Honing loader's defaults. The checkpoint's stored default is 512;
the inference limit is explicitly larger for the full public kitchen input.
This is a runtime setting, not a weight update, and is the same for all candidates.
Every actual reply must audit all six options and report no context truncation;
missing/contradictory audits, trimmed context or bad probabilities stop an episode.
Four-decimal probabilities are preserved and validated with the existing bounded tolerance.

On the initial search frame, real Laya token audits reported 619 state tokens
for JSON and 481 for readable context. These are preflight diagnostics, not
episode scores. Actual episode maxima are retained in `episodes.json`.
Upstream also warned that its unused `choice:11+` temperature is out of range
and was clamped; this experiment always uses six choices, whose bucket is unchanged.

## Running and verifying

### Watch the recorded demo

After installing the game environment using the integration's
[setup instructions](../../README.md#setup-and-play), run from the repository
root. No model download, inference server, API key or GPU is needed for replay:

```powershell
& integrations/overcooked/.venv/python.exe -m integrations.overcooked.laya_demo --open
```

The default is the seed's `asymmetric_advantages` game (final team score 180)
at <http://127.0.0.1:4187/>. To watch its zero-point cramped kitchen instead:

```powershell
& integrations/overcooked/.venv/python.exe -m integrations.overcooked.laya_demo --episode search--seed--cramped_room --port 4188 --open
```

`--help` lists all 15 recorded model episodes, including the six candidates.
The viewer verifies the committed archive checksums, extracts into a fresh
ignored temporary directory under this integration's `.cache/`, then verifies
the native joint-action replay before serving. It never modifies the committed
results. Stop with Ctrl+C to close the viewer and remove its temporary files.

Use Start, Single step, speed selection and the timeline slider. The page shows
the official native image, map/state/rules interfaces, both recorded actions,
the model's exact request and its action probabilities. Tick 0 previews the
first decision; later ticks display the action that produced that frame. This
is recorded model play, not live inference or a rule-policy demonstration.
All rewarded deliveries in these model games came from chef 1.

### Run local model inference

From the repository root, prepare and serve the pinned model with the existing
WSL CUDA interpreter (substitute local interpreter and checkout paths):

```powershell
wsl -d Ubuntu --exec /opt/honing/venvs/nanojev/bin/python /mnt/d/llm-honing/integrations/overcooked/laya_runtime.py prepare
wsl -d Ubuntu --exec /opt/honing/venvs/nanojev/bin/python /mnt/d/llm-honing/integrations/overcooked/laya_runtime.py inspect
wsl -d Ubuntu --exec /opt/honing/venvs/nanojev/bin/python /mnt/d/llm-honing/integrations/overcooked/laya_runtime.py serve --port 8805 --max-len 2048
```

The source checkout must already be cloned to `.cache/laya-source` at the
pinned revision. Model preparation requests only the root English weights,
config, encoder config and tokenizer, and verifies the published LFS SHA-256.
`inspect` writes a local architecture/runtime record for the Windows runner.

In another terminal, `laya_experiment.py init` creates a new experiment directory;
it deliberately refuses to overwrite the committed experiment. For reproduction,
use a separate checkout or a new explicitly named experiment directory in code.
The executed search command for each frozen candidate is:

```powershell
& integrations/overcooked/.venv/python.exe -m integrations.overcooked.laya_experiment search --candidate seed
```

Other modes are `controls`, `gate`, `heldout`, `verify` and `report`. The driver
refuses changed play/audit source or changed experiment-driver hashes after
protocol initialization. A completed exact task/spec can be reused when resuming;
it is not counted as a fresh independent episode.

An independent, repeatable standard-library audit works without GPU/model
dependencies and re-derives scores/counts while checking all archived bytes
and every canonical frame hash:

```powershell
& integrations/overcooked/.venv/python.exe integrations/overcooked/experiments/laya-harness-v1/audit.py
```

The integration tests, including the portable replay demo, passed:
`python -m pytest integrations/overcooked/tests` reported 67 passed, no skips,
with one upstream pygame/pkg_resources warning.

Each archive retains manifest, result, exact harness and gzip-compressed native
steps/decisions with SHA-256 identities. `verify` checks every archived file,
extracts into an ignored temporary directory, and replays every joint action in
the pinned native engine. It makes no model calls. Keep this code/runtime with
the records; historical feature-branch receipts refer to their original source.

Weights & Biases was unavailable: the reused inference environment has no SDK
and no API key is configured. The same per-episode metrics and every raw
decision/probability/audit are retained locally. Source commit and harness/model
identities accompany each archived episode. This is a local Codex-guided
harness experiment, not a GEPA or Meta-Harness run.
