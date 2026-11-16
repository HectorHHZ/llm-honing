# Tank War Exp 0: registered protocol (draft)

Issue: [#183](https://github.com/HectorHHZ/llm-honing/issues/183), under
[#174](https://github.com/HectorHHZ/llm-honing/issues/174) and
[#138](https://github.com/HectorHHZ/llm-honing/issues/138).
This file is the protocol, not an experimental result. Changes after the first
confirmatory test episode require a new dataset version and must be reported.

**Post-test scope amendment, 2026-11-15: one label per game.** Zhuohang clarified
that System 0's requested output is one `harness`, `weights`, or `controller`
attribution for the complete game. Selecting a decision index and judging
whether a specific moment caused the loss are no longer primary endpoints.
The original 40-game prompt and sealed answers are immutable. They already
contain one cause answer per game, so `episode-attribution-rescore-v1` evaluates
those cause answers against independently audited per-game technical labels.
This is an exploratory post-test change of emphasis, not a newly preregistered
blind test. The original prompt also asked for a window; this may have affected
its cause answer. The old window and joint metrics remain in the archive only.
Any new prospective test must freeze a prompt and schema that request just the
per-game cause before seeing model answers. The ten no-injection games remain
controls outside the three edited-fault classes; an ordinary unmodified loss
must not be silently relabeled as an edited-weight fault. Automated technical
label auditing does not satisfy the original two-human-reviewer requirement.

## Question and endpoint

Can a diagnoser identify the affected decision window and the verified injected
layer (`harness`, `weights`, `controller`)? Does access to controlled System 1
probes raise accuracy on the _same_ episodes? The primary endpoint is the
paired difference in verified changed-layer accuracy between bare `probe` and
bare `read` on the same episodes. Abstentions count as wrong.
No-injection, natural, mixed, and uncertain cases are reported separately.
No-injection loss is not a `chance` label.

**Pre-test amendment, 2026-11-13.** Zhuohang stopped further Astra use because
its available usage was nearly exhausted. At that point the 45-case development
selector and read arm had completed, while only seven Astra probe diagnoses had
completed. Preserve all attempts and costs, but do not finish or use this
incomplete arm for a go/no-go decision. Sol becomes the sole primary diagnoser;
the same paired endpoint, split, blinding, and decision thresholds apply. This
changes the original Astra-primary design before any confirmatory test episode
is opened. Report the deviation and the absence of cross-model replication.

**Pre-test amendment, 2026-11-14: at most 50 further human-reviewed games.**
Zhuohang capped all further ordinary one-game reviews at 50. An earlier draft
reserved 20 of these for re-reviewing known development games and 30 for blind
final games. Its fixed 20-case page and manifest were preserved, but saved
progress remained at revision zero. The later amendment below retires that
allocation without deleting its audit record.

**Later pre-test amendment, 2026-11-14: no human cause selector.** The reserved
20-case second development review had not started (saved revision zero), so
preserve its frozen draft but stop requesting it. Zhuohang's 20 prior decision
marks remain the development reference. Reserve all **50 further ordinary
reviews** for independent blind final-test games, one replay and decision
marking per game. The reviewer chooses relevant decisions, `none`, or
`uncertain`; the page asks no question about harness, weights, or controller.
No game is replaced or reviewed twice to repair a label. Freeze all 50 IDs
and source hashes before review, balanced across causes and decision strata.

Determine the technical layer with same-state checks: changed model-visible
input or assistance setting supports `harness`; changed model probabilities
on an identical request supports `weights`; changed command transformation or
execution for an identical response supports `controller`. For an unmodified
baseline game, when checks show the model received the recorded input and its
selected action was executed as specified, a human-marked bad decision is
the model's own choice. Record this separately from an **edited-checkpoint**
`weights` fault: the former can happen in a completely unmodified game. The
reviewer does not assign either label; System 0 gives its own cause answer,
which is checked against independent evidence afterward. In an altered game, an apparently ordinary later model
choice is not automatically a weights fault: an earlier intervention may have
changed the state in which it was made.
An execution change caused by a changed harness setting remains a harness
case; compare controller behavior only while holding the harness setting and
model response fixed. Normal legal-action revalidation, fire timing, and
assistance also require comparison with the same baseline state before a
controller diagnosis.
If the checks fail, multiple layers differ, or no local evidence supports a
layer, keep the case `uncertain`. The private configuration manifest verifies
the experimental change but is not itself a failure-cause label. Never tell
System 0 that changes were deliberately introduced. Measure overlap of the
technical deviation with human-marked failure-relevant decisions separately;
an alteration elsewhere does not prove why a game was lost. This later
amendment supersedes the 20+30 review allocation and all later human-cause
selection wording. The 50-case human-window endpoint still has limited power
for a ten-point improvement; report its achieved interval and power.

The primary diagnoser is `gpt-6-sol` after the 2026-11-13 pre-test amendment
below. It runs through Codex CLI 0.158.0 at `model_reasoning_effort=high`, standard
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
The pilot review page displays a program-selected decision point and the known
injected layer. Zhuohang's `relevant` judgment only says that the displayed
local difference matters to the loss. It is not an independent human label for
the decision index or cause. Keep these three label sources separate in every
analysis and report.
The pilot's exact-index agreement with that program-selected point is only a
machine-reference diagnostic. Persistent harness and weight faults may produce
valid local probe signals at every index in an 11-decision window, while a
controller fault may produce several. A model selecting another signal-bearing
index is not thereby wrong about the failure window. Before the formal window
endpoint, have Zhuohang review candidate windows without a fixed highlighted
point and record all acceptable failure-relevant windows, or mark the case
`uncertain`. Do not report exact machine-reference agreement as human window
localization accuracy.
The independent-selection pilot CSV (SHA-256
`1a99775907afca9897d65d8d8abe87e9db8be78a7a4fd5056dcf60ebb56f93e6`)
contains 14 episodes with marked decisions and one `uncertain`. The original
program point is among Zhuohang's marks in only two episodes. At least one
human mark falls within the old 11-decision detailed packet in eight of the
14 marked episodes; all marks fit in six. This was a workflow check, because
the reviewer had previously seen the fixed-point pilot cases and injected
layers. The old model outputs must not be scored as formal human-window
accuracy: in six marked episodes the detailed packet omitted every human mark.
An exploratory Astra-high selector given only the full pilot replay chose a
first 11-decision inspection window containing a human mark in 12/14 marked
episodes; allowing its second window raised coverage to 14/14. Uniformly
random candidate indices with the same one- or two-window counts would cover
6.12 and 8.00 of these 14 episodes in expectation. However, only one model
candidate index exactly matched a human mark. This supports the two-stage
retrieval workflow, not precise failure-window accuracy or a confirmatory
claim. These exposed pilot episodes cannot set the final-test score.
The first detailed-read prompt after independent selection required a
"verified injected deviation" even when the read arm had no reference
checks. Astra therefore abstained on all 15 cases. Archive that attempt as a
prompt-design failure, not a read baseline. The revised pilot prompt asks
both arms for the most supported inference from the evidence available and
reserves `uncertain` for cases with no favored explanation. Compare read and
probe only when both use the same revised prompt and selected windows.
The 45-case development review page now displays each deliberate injection,
its recorded per-decision triggers, and any same-state action comparison, at
Zhuohang's request. Treat its exported window marks as **informed development
review**, not independent blinded human localization labels. The page exports
`review_context=injection_visible_development_v1`. A separate final-test
review packet must hide injection type, trigger locations, machine-selected
reference point, and technical labels for the 50 blind human window reviews.
The earlier Codex file-preview page lost its in-progress browser-only marks;
it showed 0/45 afterward and no 45-case CSV was found locally. Do not infer or
reconstruct those missing human judgments. The replacement 45-case page is
served on loopback and atomically saves each mark to a local private JSON file;
its save/readback and stale-write checks were tested before reuse.
After reviewing the first eight development episodes, Zhuohang identified
deliberate cancellation of an otherwise permitted fire command as a salient
possible contribution to failure. Keep that as a human review consideration,
not an automatic label: the model must have requested fire, the action must
have permitted it, and the injected controller must actually have changed
fire from true to false. A trigger that only changes movement while fire was
already false, or a normal `fire_permitted=false` veto, is a different event.
The reviewer still decides whether the moment plausibly mattered to the loss.
In the exploratory 15-case self-selected-window pilot, Astra's revised common
prompt gave 13/15 injected-cause matches for read and 15/15 for probe on the
same episodes and its own selected windows; the paired change was two
corrections and no regressions. Its exact human-mark overlap was 2/14 and
3/14 respectively. Sol's independent first inspection window covered a
human mark in 12/14 marked cases; up to two windows covered 14/14, but its
candidate index exactly matched a human mark in only one case. On Sol's own
selected windows, its matched read and probe calls gave 10/15 and 14/15
injected-cause matches: four corrections, no regressions. Both modes' final
decision indices overlapped Zhuohang's earlier marks in 2/14 marked cases.
The stored paired analyses verify that each model's read and probe arms used
the same detailed replay and static prompt; probe alone received the
additional controlled checks. These pilot
numbers are not confirmatory: the reviewer had seen these episodes earlier,
the sample is small, and the detailed-window retrieval is still being frozen
for development and final test. Preserve the invalid first prompt and all
reruns in the artifact history.
The pilot window-selection prompt described every input as a loss because all
15 injected pilot cases were losses. Before development selection and any
formal diagnosis, replace this with an outcome-neutral prompt and include
no-injection wins and losses. The selector must be allowed to return no
window; do not force a suspicious decision in a clean control.
An outcome-neutral pilot was then run on the same 15 injected losses and ten
no-injection controls (eight losses, two wins) using compact full replays and
one prompt for each model. Astra's selected windows covered a previously
marked human decision in 13/14 marked injected cases; Sol's covered 14/14.
Exact candidate-index overlap was 2/14 and 5/14 respectively. Astra selected
at least one window in all ten controls; Sol returned no window for one win
and selected windows in the other nine. A selected window in a clean control
is not itself a false cause attribution: even without an injected fault, a
game can contain a natural mistake. These are exposed-pilot workflow results,
not blinded accuracy, and they do not establish a no-injection false-positive
rate. Preserve both models' calls, the first incomplete replay-build attempt,
and the corrected build with game-engine-derived 400-pixel aim summaries.
On the first 45 fault-injected development episodes, Sol's fixed read prompt
matched the private injection manifest in 29/45 cases (harness 7/21,
controller 22/24). The paired same-state probe prompt matched in 45/45,
correcting 16 cases with no manifest-match regression. This batch contains no
weight fault, and manifest agreement is not human-verified failure-window
accuracy. Among the first eight informed human-reviewed episodes, three had
human-marked recorded cancellation of permitted fire. Sol's read diagnosis
chose such an exact decision with controller cause in 2/3, and probe in 3/3.
Preserve the 45-case paired records and the partial, stopped Astra arm.
On ten separate no-injection pilot controls (eight losses and two wins), Sol's
same read prompt assigned an injected cause to 9/10, including one win. This
is an unacceptable development false-alarm signal; do not proceed to final
test with this prompt without a control-calibration check. The old-prompt
control-probe attempt was stopped before diagnosis. These controls share no seed with
the reserved 60 formal controls.
Zhuohang then required that the diagnoser never be told that episodes were
deliberately altered. The selector and detailed-diagnosis prompts above used
the word "injected"; all their outcomes are retained as disclosed-construction
development history, not evidence from the required blind protocol. Stop any
unfinished calls using those prompts. Rerun Sol with versioned prompts and a
schema that never mention episode construction, injection, a clean condition,
or gold labels. `none` replaces the model-facing `no_injection` category and
means that the model finds no supported consequential problem in these layers.
The private evaluator records that answer without assuming every unmodified
game has no natural failure-relevant weakness.
Provide comparison results as observations from another pinned configuration,
without labeling either side as deliberately modified. Keep selector prose out
of the detailed input so that its speculation cannot prime the second stage.
Audit the exact concatenated prompt, schema, full replay, detailed input, and
probe JSON for construction hints before every new model call. In the revised
construction-blind 25-game pilot, Sol's read arm matched the private layer
manifest in 10/15 altered games and named a layer in 8/10 unmodified controls.
Those eight are **claims needing review**, not automatically false: an
unmodified model can still make a consequential mistake. Resolve them against
the same-state comparisons and limited human failure-relevance review before
freezing final-test prompts. Do not treat the control condition alone as a
human gold label for why a game was won or lost.
The paired probe arm matched the altered layer in 13/15, with three corrections
and no regression, while naming a layer in 6/10 unmodified games. All completed
unmodified same-state checks showed
identical render, checkpoint probabilities, and controller execution in the
selected windows. The model's answers on those games often explicitly describe
natural poor play, rather than a difference between configurations. Score that
question separately. The selector's two 11-decision windows covered at least
one prior informed human mark in all 14 marked altered games; the final exact
decision overlapped a mark in only 2/14 in both read and probe. This is
development evidence for coarse retrieval, not blind exact localization.
In two altered harness games the probe answer differed from the manifest
because it judged the observed harness difference irrelevant to the loss or
favored a natural model decision. A manifest label alone cannot adjudicate
that failure-relevance disagreement; retain the answer and route those cases
to human review instead of silently treating the manifest as a complete
causal explanation.
With the construction-blind prompts on 45 separate development games, Sol's
read arm matched the private layer manifest in 24/45 (harness 2/21,
controller 22/24), and the paired probe arm matched 40/45 (harness 16/21,
controller 24/24). There were 16 corrections and no regression in manifest
agreement. The first eight informed human-reviewed games still gave an exact
controller-cause match on all three marked permitted-fire cancellations in
both arms. A separate 20-game development batch from the first weight
instance gave 18/20 manifest matches in the read arm and 19/20 in the probe
arm. Both calls passed the no-tool-call and input-hash audit; the probe arm
corrected two read-layer mismatches and introduced one new mismatch. These
numbers are exploratory, contain repeated games from
the same instance, and do not establish human-blind localization accuracy.
For the five development games where Sol's probe answer disagreed with a
harness-change manifest, an additional saved-state clean comparison at Sol's
chosen index found a local harness effect in one game and no local change in
four. The check is private evaluation evidence, not model input. Absence of a
local change at that index cannot rule out an earlier harness effect or a
natural problem at the chosen index; human failure-relevance review remains
necessary.
Five harness-manifest disagreements that had not been reviewed in the 45-case
page were packaged into a separate five-case human page with full playback,
decision marks, a cause choice, and its own durable progress file. The page
shows construction details to help adjudicate failure relevance and hides
Sol's answers; its judgments are informed development review, never the
50 blind final-test window reviews. This historical page predated the
no-human-cause-selector amendment and is not used for further reviews.
Zhuohang completed all five: each was marked `weights`, despite a harness
configuration change that measurably altered at least one saved-state action
in three of the five episodes. This records his judgment about which decision
was most relevant to the loss; the same-state change alone cannot settle the
counterfactual game outcome. The sealed Sol read answers agreed with his cause
in 4/5 episodes and the paired probe answers in 3/5; their exact selected
decision appeared among his marked decisions in 2/5 and 1/5, respectively.
The selector's chosen windows covered at least one human mark in all five. These five
were intentionally chosen because probe disagreed with the harness manifest,
so the counts are descriptive and cannot estimate probe benefit on a random
development or final-test sample. The revision-39 human snapshot and audited
answer hashes are recorded in `development-targeted-human-adjudication-v1.json`
on ds-serv2. Do not silently relabel the five as harness from the change
manifest or train a main-test selection rule on this targeted subset alone.
For the three-cause primary endpoint, exclude these five as disputed
harness-versus-natural-model cases: the human did not confirm harness as the
failure-relevant cause, while the altered harness prevents treating them as
pure unmodified weights examples. Retain them for a separately reported
competing-cause analysis.
For the first eight separately reviewed development games, an audited
comparison with the sealed construction-blind Sol outputs found that the
selector's 11-decision windows contained at least one human mark in 7/8 games.
The read and probe answers each named an exact human-marked decision in 4/8.
This eight-game page did not ask Zhuohang to label the cause, so it supplies
window feedback only. It showed construction information and is not the blind
final-test window score. The source hashes, paired IDs and individual answers
are in `development-human-window-first8-v1.json` on ds-serv2 and locally.
The second independently screened weight checkpoint (`reverse_strength_1p5-v1`)
supplied 20 development losses from the first ten paired development seed
groups. Each has a complete deterministic replay; twelve are aim/fire and
eight are movement/avoidance cases. The same construction-blind Sol selector
and diagnosis prompts gave 18/20 manifest-layer matches from reading and
19/20 with probes (one correction, no regression). Both diagnosis arms passed
the no-tool-call and input-hash audit. These checkpoints are distinct fault
instances but edits along one weight direction, so they do not establish
transfer to a new weight-fault family or human-verified failure relevance.
Actual high-effort Sol usage for the construction-blind 25-game pilot and
85-game development set is recorded per episode and stage in
`cost-blind-sol-development-v1/cases.csv` on ds-serv2, including selector,
read, probe, and pinned NanoJev reference-check durations. These are observed
CLI tokens and elapsed seconds, not an API bill; the final call and equivalent
price caps remain to be frozen before opening the final test set.
The 2026-11-13 official [GPT-6 Sol price page](https://developers.openai.com/api/docs/models/gpt-6-sol)
lists Standard API rates of $2 per million input and $10 per million output
tokens, with a higher rate for requests over 272,000 input tokens. A dated
private price snapshot records those rates for later API-equivalent estimates;
it is not a ChatGPT-login charge or a frozen final-test budget. Recheck the
official rate and freeze the exact call, token, and dollar caps once the
power-based sample count is chosen, before any final-test diagnosis.
The `prefer_unsafe_progress` harness candidate stopped after 176 completed
validation games when the next game exceeded the 600-second infrastructure
wall timeout under shared-server load. Its partial run and attempt log are
retained. It is not counted as validated; a pinned GPU 5 resume is queued
after the current GPU 5 screen worker releases the endpoint.
An attempted second weight server on GPU 6 was rejected by the backend's GPU
lock while the `reverse_strength_1p5` collector owned that GPU; no match was
screened in that attempt. The failed startup record is retained. The 2.5
strength candidate was moved to newly idle GPU 2 for immediate screening;
the 4.0 candidate was queued on GPU 6 after the 80-game collector.
That collector finished 80/80 games without a failed run, and its evidence
follower processed all 40 seed groups, yielding 77 loss packets. The GPU 6
4.0 candidate screen is now running; the GPU 2 2.5 candidate server passed
checkpoint and GPU-identity health checks and its screen is running.
The `always_hold` harness candidate completed its full 128-seed, dual-seat
screen: mean fault-minus-clean score was -0.064453125, but the paired 95%
interval was [-0.130859375, 0.001953125]. It therefore failed the frozen
upper-bound-below-zero rule and is excluded from accepted fault instances.
Two further screens stopped at an infrastructure wall timeout before the
256th game: `swap_fire_hold_descriptions` after 143 completed games and
`reverse_strength_2p0` after 15. The latter server also logged a response
count mismatch. Their attempts and partial run directories are preserved;
the 2.0 checkpoint is being retried on GPU 1 with the same seeds and frozen
configuration, and the remaining 3.0/0.5 checkpoint screens are queued after
that retry. Both affected GPU 5 harness screens are queued for ordered retry
after the original GPU 5 worker releases its model service. Neither timeout
is scored as a validated candidate result.
The full 128-seed, dual-seat screens subsequently retained two more
independent candidates: the real `reverse_strength_2p5` checkpoint on GPU 2
(fault-minus-clean mean -0.291015625, paired 95% interval
[-0.349609375, -0.232421875]) and controller
`drop_fire_after_selection` (mean -0.3046875, interval
[-0.361328125, -0.25]). Both completed 256 games per arm. These are
screening results, not human-reviewed failure-cause accuracy. GPU 2 was then
reallocated to the pinned `reverse_strength_3p0` screen; its checkpoint
service passed the pinned health check. The old GPU 1 waiting process was
replaced by a logged follower containing only `reverse_strength_0p5`, so
3.0 will not be screened twice. The queue change and process IDs are recorded
in `queue-weight-3p0-to-gpu2-v1.json` on ds-serv2.
The retained `drop_fire_after_selection` controller instance is queued for
80 development games (40 fixed development seeds, both seats) on GPU 5 after
the two already queued harness resume screens release that GPU. It uses the
existing complete 80-game clean development pair, validates the same release
checkpoint and seed plan, and records all attempts under
`formal-controller-extra-development-v1/drop-fire/`. This is data collection,
not a final-test diagnosis.
The `reverse_turn_words` harness candidate completed all 256 validation games
per arm but was rejected: mean fault-minus-clean score was +0.0078125 and the
paired 95% interval was [-0.0546875, 0.072265625]. The next queued GPU 0
candidate is `fire_when_available`. This rejection leaves a shortage of
independently validated harness instances relative to the five-instance
target; do not fill that quota with failed screens or with multiple windows
from one fault.
Two additional harness-only settings candidates were built from the same
clean release: `wrong_spacing_target` (an 800–1000 pixel preferred range)
and `penalize_safe_shots` (a negative weight for fast verified shots). Each
changes only `tank/controller.json`; the model weights and controller source
stay pinned. Their local and server file hashes match, and the Tank War parser
accepted both while preserving `plans/full/light`. They are queued on GPU 6
after the 4.0 weight screen, for the same complete 128-seed dual-seat gate.
These strategy settings may affect plan selection as well as the rendered
request; report that layer-boundary ambiguity and require saved-state evidence
before using either as a verified harness-cause episode.
On 2026-11-13, future screen processes were given an 1800-second per-match
wall-clock limit (previously 600 seconds for harness/controller and 900 seconds
for weights) because completed 30-second simulations can take several minutes
under shared-server contention. The game duration, decision intervals,
candidate files, seed plan, and selection thresholds did not change. Screens
already running retain the old Python-loaded limit; all timeout attempts and
retries remain in their original logs. The updated screen script SHA-256 values
are `0d2bfae01b737f2dc6ba1475bf0dd3b8cb3bafbeedbb17af34e314626dcf8874`
and `d03bde8fde7db726313260d512620e408185bd3c8eb0ccdcf34b887221d4735b`.
On 2026-11-14, `suppress_selected_turn` stopped at the old 600-second limit
after 167 completed fault games. Its timed-out clean game was archived, and a
guarded retry resumed the same 128-seed paired screen with the 1800-second
runner. The candidate, match build, model health and saved results were checked
before launch. Preserve the original and retry attempts; no validation result
exists until all 256 fault and 256 clean games finish and `screen.json` passes.

On 2026-11-14, the resumed `suppress_selected_turn` screen completed 256
paired games: mean fault-minus-clean score -0.212890625, paired 95% interval
[-0.283203125, -0.14453125], retained. `reverse_strength_4p0` also completed:
-0.291015625, interval [-0.3515625, -0.232421875], retained. The completed
`fire_when_available`, `prefer_unsafe_progress`, and
`swap_fire_hold_descriptions` harness screens were rejected by the same gate.
Five new harness-only settings were queued across free GPUs 0, 5 and 6; no
partial screen is accepted. A first `invert_selected_drive` controller attempt
failed before a valid match because the pinned server build does not recognize
that new fault kind. Preserve its attempted run and log; do not count it as a
rejected gameplay effect or silently alter the fixed build. A second accepted
controller development instance, `drop_fire_after_selection`, completed 80/80
games; its replay and same-state evidence check is running separately.
The new `enable_shot_guard` candidate turns the default-off guard **on**; it
does not reuse the invalid proposal to turn an already-off guard off.

The initial full collection target is five validated fault instances per cause,
20 verified failures per instance, and 60 controls (360 episodes total). Two
instances per cause are development-only; three are final-test-only. Historical
#173 cases were the original intended few-shot source, but #173 remains open
and its required historical diagnoses are not present in this worktree or the
Exp 0 archive. The later 50-review amendment uses the 20 already-marked
pilot/development cases only where separate technical checks support a layer
at a human-marked decision. Never fill an unverified cause from the private
configuration manifest or a model answer. The pilot may use development instances
on different seeds but never final-test instances or seeds.
A local 22-game cause-review packet is prepared from 15 previously reviewed
pilot games and seven previously reviewed development games. Existing human
window marks were copied with source hashes; no new cause labels were inferred.
The corrected v2 page also shows 231 saved-state comparison rows from the
pilot source packets, including 76 locally verified differences, and labels
the old replay's missing fire-command telemetry. The earlier v1 draft and its
progress file remain separate, so moving to v2 cannot overwrite a human mark.
The pilot and development seed groups have zero overlap with the reserved final
test seed group. A fixed 20-case subset of this packet is a candidate
few-shot source after technical checks. The 22-case draft is
archived and is not a request to review all 22. This does not complete the
missing historical #173 deliverable or provide final blind-review evidence.
The frozen IDs, excluded cases and page/progress hashes of the retired
20-case cause-review draft are recorded in
`private-review/development-cause-review-budget50.json`. Its saved progress
stayed at revision zero; no second review was requested.
Without asking Zhuohang to revisit any game, a same-state audit checked all
human marks in the 14 previously marked pilot games retained in that cohort.
Only seven games had a local selected-action, command, or execution difference
at at least one human-marked decision: harness 2/5, weights 2/5, controller
3/4. The six already-marked formal development games were also checked without
new human input: harness 2/3 and controller 2/3 had such a local difference.
Across all 20, the count is 11. The other nine are not relabeled `weights`
merely because the local check was unchanged; an earlier intervention could
have altered the state reaching that point. The audit and per-decision hashes are in
`private-review/technical-human-mark-audit-v1.json` and the server's
`pilot-human-mark-technical-v1/` and
`development-human-mark-technical-v1/`. These are informed development observations,
not blind final-test scores or proof of whole-game causality. Use only the
locally supported examples if constructing the secondary few-shot arm.
An observed-usage planning estimate from 110 completed construction-blind Sol
episodes is in `private-review/sol-formal-budget-planning-v1.json`: four modes
on the initial 240 final episodes, including 60 controls, would involve roughly
1,440 CLI calls and 101 million input tokens before few-shot overhead. This is
not the frozen budget or an API bill. Complete the power calculation and measure
few-shot overhead before setting the final call, token and price-equivalent caps.

An episode is verified only if the injected fault triggered in its selected
window and a clean intervention at the same saved raw state changes the local
decision or executed action in the predicted layer. The clean branch's final
outcome is recorded separately. A failed episode without local evidence is
`uncertain`, never used to fill a cause quota. Store the machine-supported
local deviation separately from all human-marked failure-relevant decisions;
the diagnoser may inspect up to two windows. Preserve the full episode under
a fixed read rule. Every attempted
episode is logged, regardless of selection; cap attempts at 20 times the
target count per fault and split. Exceeding that cap yields an incomplete
stratum, not substitution with adjacent windows.
For final-test evidence, visit seed groups in a SHA-256 order fixed by fault
instance and seed before observing outcomes. Record both seats, but retain at
most one locally verified failed game per seed and fault instance. This avoids
letting the first few chronological losses dominate the 50 human reviews.
The implementation was replay-smoked on a development seed with two eligible
seats: exactly one was retained; neither final-test data nor model answers were
opened for this check.

Keep both seats, clean/fault counterparts, and all windows of a reset group in
one split. No fault instance may occur in both development and test. Test must
include aim/fire and movement/avoidance strata. Hold out a fault family for a
separate transfer analysis if each cause has at least two validated families.
On unmodified controls, ask System 0 the same open cause-and-window question;
do not force `none` or call a `weights` answer wrong solely because no change
was introduced. Keep natural losses exploratory and adjudicate such claims
against behavioral evidence separately from the changed-layer primary score.
Zhuohang is the sole human reviewer for up to 50 opaque final-test episodes.
Each receives one normal game review with independent decision marks and
elapsed time, without a cause selector. The 20 development examples already
have human marks. Final reviews occur before their private configuration
manifests or System 0 answers are shown.
An additional 20-game, independent-seed development check used the retained
controller fire-cancellation instance and the frozen neutral selector and
diagnosis prompt. Sol selected two 11-decision windows without construction
details; 12/20 covered the machine-verified local deviation. On those same
20 games, read matched the changed controller layer in 19 and probe in 18
(zero corrections, one regression); exact machine-reference indices were
6 and 8 respectively. These are machine-side development diagnostics, not
human-reviewed claims that the change caused the loss and not final-test
performance. The result argues against treating probe benefit as assured.
An `uncertain` review counts against the 50-game limit, stays uncertain, and
is not replaced or silently assigned a technical layer. Record the absence
of a second independent reviewer as a measurement limit. All other attempted
games receive programmatic trigger/replay/same-state checks and remain separate
from human-grounded failure-cause claims.

After pilot and development diagnosis, choose the smallest equal-per-cause
machine-side test design on the grid of 3–5 final fault instances per cause
and 20, 30, 40, or 50 verified failures per instance. Simulate 10,000
datasets by resampling development fault instances and reset groups with seed
183, adding instances before episodes, and keep the hard cap of 750 verified
final fault episodes. Report the achieved power for a ten-point paired gain;
do not assert that the 50 human-window reviews have 80% power.
The 105 paired development cases available before final testing give 88.5%
estimated power for three final instances per layer and 20 cases each under
the nominal resampling model (10,000 simulations per grid cell). Doubling
the between-instance variance lowers that estimate to 67.5%; four final
instances per layer and 20 cases each give 81.9% under this sensitivity.
Only one harness development instance exists, so its between-instance
variance was imputed from the largest observed class variance. The first
85 pairs used an earlier diagnosis prompt, while the newest 20 used the
revised prompt. These estimates set a planning range, not a guarantee;
`private-review/development-power-grid-v3.json` is the frozen calculation,
and v2 is a preserved preliminary normal-approximation attempt. Select the
final feasible grid only after screening and evidence availability are known.
A private curator uses sealed manifests only to balance their 50 IDs before any System 0
answers are read: 17 harness, 17 weights, and 16 controller cases, spread
across final instances and both decision strata. The six fixed layer/stratum
quotas are harness 9 aim/fire + 8 movement/avoidance, weights 9 + 8, and
controller 8 + 8. Select deterministically by a pinned SHA-256 rank while
rotating across instances; require at least three distinct final instances
per layer and one review per reset
group where available. If a layer cannot supply its allocation of eligible
games, review only the available fixed cases and report
the shortfall; do not expand human review. Freeze counts, attempt cap,
prompts, retrieval, probe count and budget before any final test.
The selection script freezes source-run hashes and maps private sample IDs to
opaque review IDs before the first final diagnosis or review. The blind page
contains a whole-game replay, 400-pixel ricochet gunlines, selected/commanded/
executed action, and ordinary decision marking only. It contains no cause
selector or construction metadata. A 50-case synthetic save/readback check
confirmed durable progress, rejection of a cause field, and stale-write
protection; a real development run verified page construction and gunlines.
The machine-test freezer separately verifies each accepted 256-game screen,
same-state local evidence, replay hash, and exact source build before creating
opaque IDs. All 160 collected final clean games were checked against the pinned
release; the 60 replay-verified controls contain 45 losses and 15 wins. The
label-free machine replay builder was smoked on one development fault, one
control win, and the longest existing control (140 decisions, 127,743
characters). One-index same-state probes passed on an unmodified control, an
altered harness development case, and the isolated controller-v3 validation
build. These are workflow checks, not confirmatory diagnosis results.
On 2026-11-14, collection began for the accepted controller
`swap_selected_turn-v1` on the first 80 final seed groups, both seats, using
the already collected same-build clean games. A separate clean collection
began for the isolated controller-v2 build needed by
`suppress_selected_turn-v1`. The first attempt to load a weight checkpoint
on GPU 5 exited before collecting any game because the server enforces a
per-user GPU lock even with `--share-gpu`; its logs and process result are
preserved. The weight collections now wait for that lock to be released.
None of these collections constitutes a final System 0 diagnosis or uses any
of the 50 human reviews.
The GPU 5 `point_blank_pursuit-v1` harness screen subsequently completed all
256 paired-seat fault games and passed the gate (mean score change
`-0.150390625`, paired interval `[-0.220703125, -0.080078125]`). The following
`enable_shot_guard` candidate failed immediately because the pinned planned
steering configuration rejects the raw shot guard; preserve that invalid
attempt, and do not count it as a fault instance. The GPU 5 screen then closed
its model service. The controller-v2 clean collection that shared that service
completed 11 of 160 final games and recorded 149 failed attempts. Its original
output remains untouched. A versioned recovery queue has verified the 11
source hashes, links them into a new output, and will collect only the 149
missing games after a GPU 0 collector frees capacity. The controller-v2
`suppress_selected_turn-v1` final fault collector is independently running on
the still healthy clean checkpoint service; its same-state evidence queue
waits for both full 160-game sets. A development-screen game confirmed that
the isolated v2 scanner detects an actual executed-action difference. The
screened point-blank harness final collection was first queued after the
controller collector. Its queued GPU 0 collector and evidence watcher were
subsequently cancelled **before any game or probe** and recorded as such. A
second preflight confirmed the same pinned clean checkpoint and retained
candidate on the GPU 6 service; the versioned GPU 6 collection and evidence
watcher are now live. The evidence watcher still requires all 160 final games
before probing. All failed and superseded attempts remain archived.
The screened `reverse_strength_2p5-v1` and `reverse_strength_4p0-v1` weight
instances have separate final-collection and same-state evidence queues. Each
evidence run waits for its own complete 160-game fault set and checks the
clean reference checkpoint before probing. No output from a partial run
qualifies for the final machine or human selection.
The `reverse_strength_2p5-v1` final fault collection then completed all 160
games with 160 successful attempts and zero failures. Its evidence watcher
waits for reference capacity. The queued `reverse_strength_4p0-v1` collector
has started its pinned model service; neither instance has a completed
20-case same-state evidence set yet.
Because the GPU 5 weight-4 service had spare capacity, a versioned four-worker
prefill now covers final seed indices 64–79, both seats, on that same pinned
checkpoint endpoint. Its preflight checked the complete 256-game screen,
checkpoint and running collector metadata, exact Tank War sources, and absence
of overlapping run directories. The primary collector still verifies all
160 final games and writes the authoritative summary; the separate prefill
log preserves every attempt and validates each completed manifest.
An additional harness-only backup, `reward_shell_danger-v1`, was frozen from
the same clean release before screening. It changes only the planner's shell
clearance weight in `tank/controller.json` to favor low-clearance paths; the
pinned strategy parser accepted it. Its separate 128-seed, dual-seat screen is
running against the existing GPU 6 clean checkpoint service. It is not an
accepted instance unless all 256 fault games finish and the same paired gate
passes. The GPU 6 service is shared with another harness screen and a final
collection; no second checkpoint server or GPU lock was created.
The independent `reverse_strength_3p0-v1` weight screen completed all 256
fault games and passed the paired gate (mean score change `-0.298828125`,
paired interval `[-0.35546875, -0.2421875]`). Its final collector now waits
for the prior GPU 5 weight collections and exclusive checkpoint lock before
starting the first 80 final seed groups, both seats. A separate versioned
same-state evidence watcher was started with the accepted screen and candidate
hashes; it waits for all 160 final fault games and does not expose labels to
the diagnoser. Acceptance of the screen is not a final diagnosis result.
The isolated `invert_selected_drive-v1` controller candidate is now screening
against all 128 validation seed groups, both seats, using exact-source clean
games and the existing pinned GPU 6 clean model service. It is a possible
third final controller instance only if the complete paired gate passes. A
separate queue records the screen result; only if retained does it collect 160
final fault games and then check same-state evidence against the isolated
controller build's recovered 160 clean games. No partial screen qualifies.
The `extreme_standoff-v1` harness screen is also incomplete. A separate queue
has pinned its candidate hashes and waits for the full 256 fault games. Only a
retained screen can start its 160 final games, after the GPU 0 controller-swap
collector releases capacity; same-state evidence is checked afterward. A
rejected or interrupted screen produces no final collection.
The complete `extreme_standoff-v1` screen subsequently passed all 256 paired
games: mean fault-minus-clean score `-0.24609375`, paired interval
`[-0.306640625, -0.185546875]`. Its existing queue waits for the GPU 0
controller collector before starting 160 final fault games and same-state
evidence; the screen result alone is not a final case.
The `wrong_spacing_target-v1` screen and backup `reward_shell_danger-v1`
screen remain incomplete. Their separate queues use the same full-screen gate;
if retained, they collect on the existing GPU 6 clean server after the earlier
GPU 6 final collector releases capacity, then check same-state evidence.
Neither queued candidate counts toward the final allocation yet.
The running `wrong_spacing_target-v1` screen also has a disjoint GPU 2
prefill for validation seed indices 96–127, both seats. Its preflight
verified that those 64 target paths were absent, the primary screen had
not approached them, and the GPU 2 clean checkpoint service matched the
GPU 6 primary service on every model field except GPU identity. It uses
the same Tank War build and harness candidate and records attempts in a
separate versioned directory. The original complete 256-game paired
screen alone decides whether this candidate is retained; the prefill is
not final-test evidence.
One recorded GPU 2 spacing-harness decision was re-asked on the pinned
GPU 6 clean checkpoint service with the same request. Its complete option
probabilities matched exactly (maximum absolute difference 0, versus the
predeclared 0.01 tolerance). The baseline re-render differed, as expected
for this harness candidate. This is a cross-service spot check; the full
screen and later identity/evidence gates remain authoritative.
A conditional successor for the backup `reward_shell_danger-v1` harness
screen now waits for the GPU 2 spacing prefill to finish all 64 games
without failure. Only then may it use the same clean service for its own
disjoint validation seed indices 96–127, both seats. It rechecks the
candidate, Tank War build, checkpoint service, original screen position,
and empty target paths before creating any backup fault run. The complete
original paired screen remains the only retention gate.
Because the isolated controller validation was progressing slowly while the
shared clean service and CPU had headroom, a versioned four-worker prefill was
started for validation seed indices 96–127, both seats. The primary sequential
screen was still at the beginning of the seed list; preflight refused any
existing tail run and verified the same candidate, build, checkpoint, and
server. Prefill attempts are logged separately. The original screen remains
responsible for checking all 256 fault games and applying the paired gate;
prefill alone cannot retain the candidate or create a final-test case.
The first disjoint tail block subsequently completed all 64 target games,
with 64 successful attempts and zero failures. The original 256-game screen
is still incomplete and remains the only acceptance gate.
A second versioned two-worker block covers seed indices 80–95, both seats;
its targets are disjoint from the first prefill and the primary screen's
current prefix. Both parallel blocks use the original match script and same
checkpoint service. The original screen still performs the complete
256-game validation and acceptance decision.
A third versioned two-worker block covers seed indices 64–79, both seats,
using the existing GPU 0 clean checkpoint service. Its preflight compared
the complete service metadata with the GPU 6 service, excluding only GPU
identity, and verified the same isolated controller build, candidate, and
disjoint run paths. Its attempts remain separate; the original 256-game
screen alone decides whether the candidate passes the paired gate.
Once the GPU 0 block for indices 64–79 finishes successfully, a separate
versioned two-worker queue may use that same pinned service for indices
48–63, both seats. It checks the prior process result, refuses any existing
target path, and requires the sequential screen's current seed index below
24 before starting. At launch the queue waited without creating a fault run.
The preceding GPU 0 block then finished 32/32 games with zero failures. The
48–63 successor passed its preflight and began collecting on the same clean
service; its first completed games were valid. The full 256-game screen
remains responsible for the final paired acceptance decision.
The GPU 0 successor then finished 32/32 games with zero failures and
recorded `complete_32_middle_tail_games`. It remains part of the original
256-game screen, which has not yet written its paired result.
The GPU 6 workers likewise have a conditional successor for validation seed
indices 32–47, both seats. It waits for the 80–95 block to finish 32/32
successfully, refuses overlapping target directories, and checks that the
primary sequential screen is still below seed index 16 before beginning.
At launch this queue had created no fault run.
The preceding GPU 6 block then completed 32/32 games with zero failures.
The 32–47 successor passed its overlap, seed-position, candidate, build,
checkpoint, and service checks and began collecting. The complete original
screen remains the only acceptance gate.
The GPU 6 successor then completed 32/32 games with zero failures and
recorded `complete_32_middle_tail_games`. All versioned disjoint inversion
prefills have now finished; the original 256-game screen still has to
verify every game and write the paired retention result.
The optional `reverse_strength_0p5-v1` weight candidate has a separate
versioned final-collection queue. It waits for its full 256-game paired screen
and the preceding GPU 5 weight queue, then checks the exclusive checkpoint
lock before collecting 160 final games. If the screen fails the retained-fault
gate, the queue records that rejection and collects no final games. This
candidate is outside the frozen v1 instance catalog; it cannot enter the
formal allocation without a separately recorded catalog amendment before
the first formal diagnosis.
Its complete screen passed all 256 paired games (mean score change
`-0.306640625`, paired interval `[-0.361328125, -0.251953125]`). A separate
same-state evidence watcher waits for its 160-game final collection and
requires 20 distinct final seed groups with verified local changes.
The disjoint GPU 5 weight-4 final prefill completed 32/32 games with zero
failures. Its separate attempts are available to the original 160-game
collector, which remains responsible for the authoritative final result.
A second disjoint GPU 5 prefill uses the same live weight-4 checkpoint
service for final seed indices 56–63, both seats (16 games). It started
only after confirming those paths were absent and the sequential collector
was below index 40. Its own attempt log and manifest checks are separate;
the original 160-game collector still writes the authoritative summary.
That second block completed 16/16 games with zero failures and recorded
`complete_16_final_tail_games`. The complete 160-game summary and same-state
evidence remain pending.
A proposed third 16-game block for indices 48–55 was uploaded and
launched, but its own preflight refused to start because the sequential
collector reached the safety boundary before it could reserve those paths.
The attempt log is retained; the block created no output directory or
target game. The original collector will cover those seeds.
The original weight-4 final collector then completed all 160/160 games
with zero failures and wrote `collected_160_final_fault_games`. Its
same-state evidence watcher still waits for clean reference capacity;
the completed games alone do not make the instance eligible for the
final machine set. The queued strength-3 checkpoint is next on GPU 5.
The strength-3 queue acquired GPU 5, started its pinned checkpoint
service, and began the 160-game final collection. Its first completed
game matched the checkpoint manifest and paired clean game on source,
harness, and controller hashes; only the weight hash differed. The full
160-game collector and same-state evidence remain pending.
A separate four-worker GPU 5 prefill for strength-3 final seed indices
64–79, both seats, passed its retained-screen, checkpoint-service,
source-hash, and disjoint-path preflight and began collecting 32 games.
It records every attempt separately; the original 160-game collector
will verify and summarize the complete batch.
Before any final diagnosis, `final-instance-catalog-v1.json` fixes candidate
priority and source paths (SHA-256
`e8613f97e2d6caee341dd1c3e74872f3486657db0d871226628e88d469ad31fe`).
A read-only readiness check reports each full screen and 20-case evidence set.
It writes a machine config only after three eligible instances per layer and
the fixed 50-game human allocation are possible. The first check found no
completed final evidence set, so no machine config was written.
A versioned handoff watcher now checks the same readiness rule every five
minutes. It pins the catalog and freezer script hashes; if three eligible
instances per layer and the fixed 50-case allocation become possible, it
freezes the machine set and opaque human review IDs. A rejected candidate or
allocation shortfall is recorded without silently reducing the target. The
watcher does not call a diagnoser, generate human answers, or publish results.
A second versioned queue waits for that handoff's successful process result,
then builds the cause-free 50-game replay page from the frozen private
selection. It pins the page builder and template hashes and verifies the
page contains exactly 50 opaque IDs and no construction labels. It does not
start the human review server or record any human answer.
Before the final 50 cases were available, the page's 400-pixel gunline
generator was run on one completed final replay from each technical layer.
Harness, weights, and controller outputs had 26/26, 75/75, and 29/29
gunline/state rows respectively; all three matched the recorded engine
hash. This checks page input compatibility, not the final 50-case page.
The construction-blind whole-game input builder was also exercised on one
completed loss from each layer. It produced 26, 75, and 29 decision rows
for harness, weights, and controller; serialized inputs were 13,546,
49,295, and 15,332 characters, below the frozen 150,000-character
selector limit. This is an input preflight, not a final System 0 call.

## Blinding and diagnosis

Formal diagnosis uses two frozen stages. First give each diagnoser an opaque
episode ID, a compact timeline of **every** decision, the outcome, map, and
native events. It independently selects up to two 11-decision windows for
inspection or abstains. Do not place a program-selected or human-selected
point in this input. Use the same selected windows for that model's matched
read and probe arms, separately for the two few-shot settings. Freeze the
selection prompt, context fields, window count, tie handling, and output
schema on development data before opening final-test cases. The final chosen
decision index is correct for localization only if it is among Zhuohang's
blinded marks; selected-window coverage is an exploratory retrieval measure.
Report both with the opportunity baseline for two 11-decision windows.

Second give both matched arms the same detailed records for the selected
windows: raw state, rendered request, full option probability distributions,
chosen/decoded/executed actions, and one fixed full-episode retrieval. Probe
mode additionally receives the predeclared controlled checks at those same
indices. Neither arm receives the manifest, cause, fault name, revealing path,
reference action, split assignment, or label status. The reference action can
be a set; its source and reviewer agreement stay in evaluation-only records.
Harness probes re-render the saved state. Weight probes re-ask the identical
request with an identified different checkpoint. Run identity first. Score
the weight-side direction edit separately; a probability change is not proof
of repaired outcome.

Execute `read`, `read+fewshot`, `probe`, and `probe+fewshot` on identical test
episodes with Sol at high effort. Under the 50-review amendment, fix bare
`probe` versus bare `read` as the primary paired comparison before test. The
few-shot modes are secondary: the 20 already-marked development examples cannot also
serve as an unbiased development set for choosing the primary mode, and no
extra human reviews are budgeted for that choice.
The secondary few-shot text is frozen from three previously human-marked
development decisions, one per technical layer, with same-state checks at each
mark. It contains no construction metadata or final-test case. Both few-shot
arms receive the identical examples (`fewshot-blind-v1.json`, SHA-256
`bf05e326c600caa181c8670b02341e95d7d13e863b8dc2be4dac52fa19cf8a83`).
The derived secondary selector prompt retains only three nearby timeline rows
per example, excludes the example's cause answer, and is 7,536 characters
(SHA-256 `b35c1697aced3619a690c44a1b2e24ef24d57aef28a6f3bcffb23e77efa6ca9a`).
The secondary diagnosis prompt includes the full three development examples
and is 67,675 characters (SHA-256
`02803e7d138fc7c04ded2997e80c88d66a632c99ffcfdd12ed840e70d5cbb1c3`).
On three fixed development games, the few-shot selector added exactly 2,429
input tokens per call versus the same bare selector input. The few-shot read
added 20,921 tokens per call; the few-shot probe added 20,854–20,921. These
nine calls measured overhead only: their diagnoses used the existing bare
selector windows, so they are not efficacy comparisons. Raw calls and usage
are preserved in `fewshot-cost-development-v1` on ds-serv2.
The read and probe arms use exactly the same static diagnostic instructions and
output schema. Both include a `PROBES JSON` field; it is `null` for read and
contains the frozen reference-check results for probe. These checks run at
every index of the selected windows. The pilot's old program-selected
11-decision packets are development evidence only.
Run low-effort versions only after
the high-effort outputs are sealed. Codex runs inside an OS mount sandbox with
only opaque inputs; test labels and manifests are not mounted. Reject any case
whose Codex JSONL contains a tool call or whose input leaks a gold identifier.
No retries based on answer correctness. A failed call can be retried once for
transport error with the same input; preserve both records. The frozen budget
reserves at most two such extra calls per stage, still subject to the fixed
token and dollar-equivalent caps. A remaining failed call stops the stage and
is reported as incomplete evidence; it cannot be treated as a successful
model answer or silently replaced.

## Analysis and decision

Compute paired differences on the same episode IDs. On the full, locally
verified final set, the observed changed layer is a technical benchmark,
not a human judgment about why the game was lost. On the fixed 50 blind human
games, score independently marked failure-relevant decisions. An `uncertain`
review remains in the review count and cannot be silently turned into a
decision label. Use 10,000 grouped bootstrap
resamples with seed 183: resample fault instances within each layer, and
resample reset seeds globally so the same seed remains linked across fault
instances. Keep the three layer weights equal in the machine benchmark.
The predeclared majority-class accuracy baseline for that balanced,
three-layer machine benchmark is exactly `1/3`; the 60 no-injection controls
are reported separately and do not change this threshold.
For each `marked` human-reviewed game, the exact-index opportunity baseline is
the number of distinct human-marked decisions divided by that game's total
decision count: one decision chosen uniformly without seeing the marks.
Compare the model's exact-index agreement minus this per-game chance by the
same grouped bootstrap. Separately enumerate all pairs of distinct candidate
centers to calculate the chance that two uniform 11-decision windows cover at
least one human mark; compare that with the model's selected-window coverage.
`none` reviews remain in the 50-game table and measure whether the model
abstains; `uncertain` reviews remain counted but have no exact-index gold.
The go criterion for human-window accuracy uses only the predeclared `marked`
games and the exact-index opportunity baseline. Report how many of the 50
were marked, none, and uncertain, without replacing any case.
The primary comparison is bare Sol `probe` minus bare `read` on the same
machine-verified games. The final report gives the paired changed-layer
interval, exact human-window agreement versus the fixed random-window
opportunity baseline, and the fraction of human marks at a verified local
deviation. Claim **go** only if the changed-layer interval is wholly above
zero, probe accuracy exceeds the predeclared majority baseline, and human-window
accuracy is above its opportunity baseline with a positive grouped interval.
Report the combined correct-layer/exact-window rate and distinguish local
deviations from claims about why the whole game was lost.
Otherwise report **no-go for this evidence standard**, naming a power or label
shortfall explicitly; do not infer that System 0 has no ability. No extra
human reviews are requested to rescue a result. All other comparisons,
including few-shot, receive Holm correction and are secondary. For these
secondary paired comparisons, use the same instance/reset grouped bootstrap;
center its draws on the observed effect to form a two-sided null-tail
probability, then apply Holm across the three few-shot diagnosis comparisons
and the few-shot selector-window comparison. Report the unadjusted intervals
and adjusted probabilities together; do not use them to change the primary
go/no-go decision.
The pre-test final scorer is `score-final-sol.py` (SHA-256
`0aa48f68cd510a1d4de662a0302a5f2015677111ea36564bf8f28385a2ab451a`)
with grouped-statistics helper `final-statistics.py` (SHA-256
`7fff2482cbde5e27cb5ae60d745720ae6b58a7ccdd0d2976e5ef5b277079e961`).
It requires all six sealed Sol stages and all 50 human progress records,
rejects any human cause field, verifies source hashes, and reports the
machine-layer and human-window endpoints separately. Synthetic go/no-go,
cause-field rejection, and preserved transport-retry checks passed before any
formal model answer.

Report per-cause accuracy, macro-F1, confusion, harness-vs-weights accuracy,
exact agreement with a blinded human-marked decision (with adjacent-index
agreement secondary),
confidence calibration, abstention coverage,
random/majority/always-harness/always-weights/blind-human baselines, matched
intervals, costs, failures, exclusions, and no-injection/natural outcomes
separately. Link a W&B table if available or attach equivalent machine-readable
tables. The artifact root is `/data/xinkaiz/system0/exp0/tank-war/{dataset_version}/`
on ds-serv2; #174's `/project/flame/haozeh/system0/exp0/` is absent there.
No secrets or final-test gold labels are mounted into diagnosis processes.
