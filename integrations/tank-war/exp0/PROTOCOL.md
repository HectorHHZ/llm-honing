# Tank War Exp 0: registered protocol (draft)

Issue: [#183](https://github.com/HectorHHZ/llm-honing/issues/183), under
[#174](https://github.com/HectorHHZ/llm-honing/issues/174) and
[#138](https://github.com/HectorHHZ/llm-honing/issues/138).
This file is the protocol, not an experimental result. Changes after the first
confirmatory test episode require a new dataset version and must be reported.

## Question and endpoint

Can a diagnoser identify the affected decision window and the verified injected
layer (`harness`, `weights`, `controller`)? Does access to controlled System 1
probes raise accuracy on the _same_ episodes? The primary endpoint is the
paired difference in cause accuracy between the development-selected probe mode
and its read mode with the same few-shot status. Abstentions count as wrong.
No-injection, natural, mixed, and uncertain cases are reported separately.
No-injection loss is not a `chance` label.

The primary diagnoser is `gpt-6-astra`, with `gpt-6-sol` as a fixed replication.
Both run through Codex CLI 0.158.0 at `model_reasoning_effort=high`, standard
service tier only. The `low` effort arm is a secondary sensitivity analysis;
it is never used to select the winning method. Codex with ChatGPT login does
not expose a temperature or immutable model snapshot. Record model ID, CLI
version, request time, JSONL usage, and this limitation. Public API list-price
equivalents are cost estimates, not the user's bill. Freeze a call and
token/cost-equivalent cap after the pilot and before test.

## Frozen Tank War configuration

Use a clean, pinned commit and the release in `integrations/ra2/pins.json`:
NanoJev checkpoint revision `047b927b30882a1138fc504821b82ac145a4b81a`
and source revision `76fdfc9ecdca45a9bcef17991a07d3041a87685a`.
The match uses planner opponent, 30 simulated seconds, model interval 200 ms,
opponent interval 300 ms, `plans` controller, `full` context, light assistance,
and shot guard off. Both seats of a seed are one reset/statistical unit. Use
greedy model choice, no `sample-seed`. Set NanoJev `--max-len 8192`; never
silently truncate an oversized request. Score win 1, draw or timeout 0.5,
loss 0. The engine's own `kill` and `round` events define death, self-hit,
draw and timeout. Do not call a shot a miss without a separate native rule.

Before injecting a fault, replay two independently recorded clean rounds for
each seat of 64 seeds and require matching state and request hashes at every
decision. If identity fails, repair the replay and re-run this gate. For all
later probes, the unchanged checkpoint/request must reproduce each probability
within 0.01 on at least 500 decision questions. Record any failed question;
do not relax the tolerance after seeing test labels.

## Fault validation and sample selection

Use #171 builders and the same clean release. Candidate harness faults remove
one threat/aim field, alter fire/hold wording, or disable assistance. The
already-off shot guard is not a candidate. Candidate weight faults are actual
checkpoints: reversed validated direction edit, SFT on flipped labels for one
question, or an earlier weaker checkpoint. A logits wrapper is not a weight
fault. Candidate controller faults remap a chosen turn, drop fire after
selection, or corrupt the legal-action projection. Every manifest names the
single changed layer, family, instance, expected trigger, and changed hashes.

Every candidate is evaluated on 128 pre-generated seed blocks, both seats,
paired clean/fault, disjoint from pilot and final collection. Fixed-size
sampling avoids optional stopping. The effect is the mean within-block
`fault_score - clean_score`, averaged across both seats. Retain only faults
with point effect <= -0.05 and the upper end of a seed-block bootstrap 95%
interval < 0 (10,000 resamples, fixed RNG seed 183). Log all runs, including
invalid candidates and errors. A clean release loss is a control, not a
verified injected cause.

The pilot contains one retained fault per cause, five verified failed episodes
per fault, and ten no-injection controls. It tests capture, replay, teacher
labels, reviewer workflow, probe identity, and cost; it is not confirmatory.
The initial full collection target is five validated fault instances per cause,
20 verified failures per instance, and 60 controls (360 episodes total). Two
instances per cause are development-only; three are final-test-only. Historical
#173 cases provide few-shot examples. The pilot may use development instances
on different seeds but never final-test instances or seeds.

An episode is verified only if the injected fault triggered in its selected
window and a clean intervention at the same saved raw state changes the local
decision or executed action in the predicted layer. The clean branch's final
outcome is recorded separately. A failed episode without local evidence is
`uncertain`, never used to fill a cause quota. Store one selected window per
episode. Preserve the full episode under a fixed read rule. Every attempted
episode is logged, regardless of selection; cap attempts at 20 times the
target count per fault and split. Exceeding that cap yields an incomplete
stratum, not substitution with adjacent windows.

Keep both seats, clean/fault counterparts, and all windows of a reset group in
one split. No fault instance may occur in both development and test. Test must
include aim/fire and movement/avoidance strata. Hold out a fault family for a
separate transfer analysis if each cause has at least two validated families.
Keep natural losses exploratory. Have Zhuohang and Nick independently inspect
the trigger, cause evidence and window; unresolved disagreement is uncertain.
They blind-diagnose at least 30 opaque test episodes before seeing their gold
manifests, recording elapsed time.

After pilot and development diagnosis, choose the smallest equal-per-cause
test design on the grid of 3–5 final fault instances per cause and 20, 30, 40,
or 50 verified failures per instance that gives simulated >=80% power for a
10-percentage-point paired accuracy gain at two-sided alpha 0.05. Simulate
10,000 datasets by resampling development fault instances and reset groups;
use a fixed RNG seed 183 and the same interval method as final analysis. Add
independent fault instances before increasing episodes per instance. The hard
cap is 750 verified final episodes. If no grid point meets power, report
insufficient power and do not make a go claim. Freeze the chosen counts,
attempt cap, prompts, retrieval, probe count and budget before any final test.

## Blinding and diagnosis

The diagnoser receives an opaque episode ID, a fixed window of five decisions
before and after the selected decision, native events, full option probability
distributions, chosen/decoded/executed actions, and one fixed full-episode
retrieval. It never receives the manifest, cause, fault name, revealing path,
reference action, split assignment, or label status. The reference action can
be a set; its source and reviewer agreement stay in evaluation-only records.
Probe mode sees the same trace plus frozen #170 probes. Harness probes re-render
the saved state. Weight probes re-ask the identical request with an identified
different checkpoint. Run identity first. Score the weight-side direction edit
separately; a probability change is not proof of repaired outcome.

Execute `read`, `read+fewshot`, `probe`, and `probe+fewshot` on identical test
episodes with both high-effort models. Pick probe vs probe+fewshot on development
accuracy gain over its matching read mode; break a tie toward no few-shot.
Use the selected setting unchanged for Sol. Run low-effort versions only after
the high-effort outputs are sealed. Codex runs inside an OS mount sandbox with
only opaque inputs; test labels and manifests are not mounted. Reject any case
whose Codex JSONL contains a tool call or whose input leaks a gold identifier.
No retries based on answer correctness. A failed call can be retried once for
transport error with the same input; preserve both records and count the final
failure as abstention.

## Analysis and decision

Compute paired accuracy difference on the same episode IDs. A two-way
stratified bootstrap resamples fault instances within cause and scenario/reset
groups, 10,000 times with seed 183; never resample decision rows as trials.
The primary Astra comparison is selected probe versus matched read. If
few-shot was selected, additionally require a positive paired interval against
bare `read` to satisfy #174. Go requires the primary 95% lower bound > 0,
accuracy above the majority-class baseline fixed from development data, and
the predeclared power gate. Sol is replication, not a second model-selection
chance. All other inferential comparisons use Holm correction and are clearly
secondary. A null or negative result is a completed no-go result.

Report per-cause accuracy, macro-F1, confusion, harness-vs-weights accuracy,
question/window localization, confidence calibration, abstention coverage,
random/majority/always-harness/always-weights/blind-human baselines, matched
intervals, costs, failures, exclusions, and no-injection/natural outcomes
separately. Link a W&B table if available or attach equivalent machine-readable
tables. The artifact root is `/data/xinkaiz/system0/exp0/tank-war/{dataset_version}/`
on ds-serv2; #174's `/project/flame/haozeh/system0/exp0/` is absent there.
No secrets or final-test gold labels are mounted into diagnosis processes.
