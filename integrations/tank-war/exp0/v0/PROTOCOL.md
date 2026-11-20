# Tank War Exp 0 v0: fixed protocol

This is the v0 protocol for [#183](https://github.com/HectorHHZ/llm-honing/issues/183).
The machine-readable pins are in [freeze.json](freeze.json). This protocol
describes the existing 40-game development sample. A change to the sample,
fault construction, prompt, model setting or primary score requires a new
dataset version and an explicit amendment before further answers are seen.

## Question and unit

For each complete episode, System 0 returns one `harness`, `weights` or
`controller` attribution, or abstains. Compare `read` and `probe` on the same
40 episodes. The primary score is component accuracy over 30 technically
verified changed-component games; an abstention counts as incorrect. The ten
no-injection games are controls and receive a separate unsupported-fault rate.
Decision windows support technical audits but are not a scored v0 answer.
An observed component change does not by itself establish that it caused the
whole-game loss.

## Recorded release and tasks

The frozen 40 source manifests all name game code commit
`31912a1a3378f49e542577fa383d42891a6f2dc3` and one identical source-hash
map. [check-v0-pins.py](../../scripts/check-v0-pins.py) verifies all nine
game-source hashes, the NanoJev source/model revisions in `pins.json`, and
the default controller settings against that commit. The clean NanoJev weights
SHA-256, edited checkpoint SHA-256s, harness/controller fingerprints and Node
version come from those recorded manifests; see `freeze.json`.

Use the planner opponent, both seats, 30 simulated seconds, 200 ms model and
300 ms opponent decisions. The clean agent uses `plans`, `full` context,
`light` assistance, baseline strategy and no shot guard. Do not truncate a
request above the 8192-token model limit. Score win 1, draw/timeout 0.5 and
loss 0. Use native engine events for outcomes; do not infer a missed shot from
an absent hit. The existing 40 episodes span 19 seed/reset groups, so 40 rows
are not 40 independent statistical units. Related seats and clean/fault
counterparts remain in one group.

## Fault construction and verification

Use the game-local builders in `integrations/tank-war/scripts/`:

- `build-harness-faults.py` changes one harness file and records its original
  and altered hashes. The retained `extreme_standoff-v1` changes the strategy
  setting. Its observed behavior can also affect plan revalidation after model
  choice; require a direct changed-input and model-choice comparison for a
  harness label, and disclose the mixed boundary.
- `build-weight-faults.py` constructs real NanoJev checkpoints by editing only
  the 1,024-element `scalar.weight` tensor. The pinned 3.0 and 4.0 reverse
  strengths use two checksum-verified source checkpoints; a wrapper that
  changes logits or output actions is not a weights fault. Match output hashes
  to `freeze.json` before treating a rebuilt checkpoint as the recorded fault.
- `build-controller-faults.py` emits `drop_fire_after_selection-v1`. The
  recorded game runtime at the pinned source commit applies this after model
  selection. Verify selected versus executed controls; do not relabel a
  post-selection change as harness or weights.

The builders create candidates, not labels. A valid per-game component label
requires actual changed input, changed checkpoint response, or changed
selected-to-executed controls at the saved state. Preserve all attempts,
incomplete runs, rejected candidates and exclusions. Keep the paired clean
outcome and its effect status separate from the component label. Uncertain,
mixed and naturally failing episodes stay outside the 30 verified examples.
The ten selected no-injection controls are all losses; this limits the control
outcome mix and must stay visible in the data card.

## Diagnosis and reporting

Freeze the evaluated episode IDs, split manifest, sample/attempt caps, prompt,
output schema, read limit, probe access, model ID/settings and budget before a
real API rerun. The minimum is one `read` and one `probe` call on each of the
same 40 episodes. Probe identity requires the same request/checkpoint to
reproduce option probabilities within 0.01 before using an alternate render
or checkpoint. Keep the alternate checkpoint isolated from the clean release.

Diagnoser inputs may contain ordinary replay evidence but not gold labels,
fault manifests, revealing filenames, reference answers or evaluator metadata.
Audit the exact input allowlist and reject tool or gold access. Record all
requests, replies, probe calls, errors, retries, token usage, wall time and
cost. The prior ChatGPT-login CLI run pinned a model name and effort but could
not expose an immutable internal model revision or temperature; it is not an
API-key run record.

Report component accuracy, macro-F1, per-class counts, confusion, controls,
abstention/coverage, calibration where supported, random/majority/constant
baselines, paired read/probe difference with interval grouped by fault instance
and scenario/reset, plus limits. A negative result passes delivery acceptance.
Do not infer a general three-family effect, independently confirmed result,
or permission for #180 or Exp 1/2 from this v0 release.
