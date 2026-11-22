# REA-Issac NanoJev agent development

The experiment evidence archives are retained privately, outside this
repository. Archive names and SHA-256 checksums below identify those
historical copies; they are not public downloads. To run the selected agent
from a portable checkpoint directory, follow [REPRODUCE.md](REPRODUCE.md).

The current agent code lives in `honing/backends/rea_issac_agent/`. It supplies
its own harness configuration, controller, data and training scripts, and a
thin decision proxy around the unchanged REA-Issac match runner. New agent
recordings use the unchanged `rea-issac-match-v1` game files plus an
`agent_decisions.jsonl` sidecar and `rea-issac-agent-record-v1` manifest.
The sidecar keeps the model's original probabilities and selected action;
the game recording keeps the executed action. `audit_agent_run` checks both.
The archived runs below predate this packaging change and retain their
original source fingerprint; replay them with the recorded source revision.
No GPU evaluation was rerun after this packaging change.

This work follows the failed Exp 0 preflight. The game engine, public
observation and action schema, and existing task definitions are frozen at the
hashes in `agent_frozen.json`. Only the agent harness, controller, and model
weights are changed. The agent implementation lives in its own package;
the shared decision pipeline has no game-specific branches.

## First controller pilot

`navigate-empty-rooms-v1` queries NanoJev at every step, uses the existing
public-observation rule navigator in an empty current room, and executes the
model-selected action once enemies are present. The recording stores both
selected and executed actions plus the controller reason; replay recomputes
the switch and rejects mismatches.

With the pinned NanoJev checkpoint and the original harness, seeds 1002 and
2001 reached the first combat room at tick 63. Both model runs then recorded
zero hits and zero kills, and lost at ticks 356 and 341, respectively. The
same 900-tick paired rule runs cleared 4 and 2 rooms. Fifteen same-state
instruction probes showed that an explicit coordinate rule could change
shooting from southeast to east in some states, while movement remained
mostly southeast. These are development observations, not held-out agent
scores. The complete pilot and probe archive is
`agent_pilot1.tar.gz` (SHA-256
`4d3a0b35c40381e97a0fc7971b5c6ed723240ecf85a7af1c48ed4d49de069652`).

## Training protocol

The revised harness keeps zero action-history entries and gives explicit
combat, shooting-axis and bomb-release instructions. The fixed public
observation JSON and model question options are unchanged. A rule teacher
generated combat decisions from training seeds `7`, `11`, `1001`, `1002`,
`2002`; seed `2001` is validation; seeds `3001` and `3002` are reserved for
agent evaluation. Forty decisions per train or validation seed were selected
from states with at most 7,000 rendered characters. The 240-decision dataset
has SHA-256
`9ed213c9977f76e89b851ceaaeb8465ddbdf05e23045c9108be95fb5972c912e`.
The teacher is a training signal, not a claim that its actions are uniquely
correct. Longer combat states are excluded from this first SFT pass and must
be assessed separately. The dataset archive is
`agent_demos_v2.tar.gz` (SHA-256
`595f3731acf8ff2c2c697b0b41a27923c9af8c0f99112d3d39e12ef0eb2d9b5b`).
Regenerating it with the checked-in generator yielded the same decision-file
SHA-256.

The first training run updated NanoJev's decision heads only for one epoch,
with one question per batch and a 4,096-token limit. It runs on
`ds-serv2.ucsd.edu` GPU 2, with a 50% per-process GPU-memory cap. During
inference or training, the 18 GiB GPU 2 memory holder is stopped; the
launcher restores it when the job finishes or fails. GPU 2 is shared with an
existing user process, so the launcher requires at least 24 GiB free before
training. It used 600 training question examples and 120 validation question
examples. Validation loss fell from `1.8457` to `1.3656`; validation choice
accuracy was `0.45`. The clean and trained checkpoint SHA-256 values are
`f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b`
and `37029ae8f19d05813adb45c0ab719feb6949556557b40dfd792c80dfc29d9754`.
The first trained checkpoint and full training manifest are retained with the
private run evidence. This run did not use wandb; the checkpoint manifest,
loss history, validation metrics, and
audited game traces provide the equivalent measurements for this pilot.

The trained agent was evaluated on seeds `3001` and `3002` with the original
`navigate-empty-rooms-v1` controller and a 900-tick limit. Both model runs
entered combat but scored zero kills and lost at ticks `405` and `755`.
The paired rule runs cleared 2 and 3 rooms; seed `3001` was truncated and
seed `3002` won. The model fired no shot on 99 of 114 combat decisions in
seed `3001` and 122 of 231 in seed `3002`. Thus lower validation loss did not
establish a competent game agent. These two seeds have now been used for
development; later controller evaluations must use new seeds for a fresh
out-of-sample check.

The next adapter-only controller, `model-assisted-combat-v1`, retains the
rule's public-state movement and shot as a safety suggestion. It accepts
NanoJev's movement when the directions differ by at most approximately 45
degrees, and accepts NanoJev's nonzero shot under the same rule. The model is
queried at every decision. Each recording saves its original choice,
executed action, and separate movement/shot acceptance reason. This is a
hybrid agent; its performance must not be described as NanoJev acting alone.

The same controller and harness with the untrained checkpoint also scored
zero kills: seed `3001` lost at tick `343`, and seed `3002` reached the
900-tick limit without clearing the combat room. The head SFT changed the
choices and survival time but did not yield a useful room-clearance score.
Training metadata and complete first-evaluation recordings are in
`agent_train_and_first_eval.tar.gz` (SHA-256
`e035c3eacd15d910b9c8608034a68ff7b66e1170c3755b214a0b0b90941001e2`).
The 2.3 GB checkpoint weights are excluded from this archive; their checksum
is recorded above.

With the trained checkpoint and assisted controller on the two development
seeds, the model-backed agent won at ticks `806` and `736`, while the paired
rule runs were truncated at `900` and won at `785`. The complete recordings
are in `agent_assisted_dev_v1.tar.gz` (SHA-256
`3079178759ac93fe7de50de88929b381bedb5d9870558ba04695ef360d61e8db`).
In the two model runs, 24 of 211 and 9 of 174 combat decisions actually used
a movement direction different from the rule suggestion. No executed shot
direction differed from the rule. Therefore the wins demonstrate the hybrid
controller's viability, not NanoJev combat competence on its own.

On untouched seeds `4001` and `4002`, both the hybrid and rule runs reached
the 900-tick limit. The hybrid cleared 4 rooms with 6 kills on `4001`, matching
the rule's room and kill counts, but took two damage instead of one. On `4002`
it cleared 4 rooms with 5 kills, compared with the rule's 5 rooms and 6 kills.
The model actually changed movement on 36 of 182 and 16 of 171 combat
decisions, and changed firing on 3 and 0 decisions. This is a useful
model-backed multiroom trajectory, but the pilot does not establish that the
trained weights improve the rule agent. Complete fresh recordings are in
`agent_assisted_fresh_v1.tar.gz` (SHA-256
`7a4590dd13a3833ad81c6802d994e33720779a3333e8ce3f757fba581619c488`).

The clean NanoJev checkpoint was then run on the same two fresh seeds with
the same assisted controller and harness. It cleared 3 rooms with 4 kills on
`4001`, and 1 room with 2 kills on `4002`; both runs reached the 900-tick
limit. Thus the head SFT improved this hybrid agent over the clean checkpoint
on both tested seeds, while the trained hybrid still failed to exceed the
rule baseline on the fresh pair. Complete clean-checkpoint recordings are in
`agent_base_assisted_v1.tar.gz` (SHA-256
`aa530ce1b36a31c0b2dbb62ea91707c91d6eb518e413a93535dc86d875713d8f`).
These two seeds are too few to support a general performance claim.

The remote Linux recorder passed exact replay audit on all four runs. After
download, exact replay on macOS passed for both `4002` runs but stopped at
ticks 429 and 216 on `4001` because a single enemy coordinate differed by
about `1e-14`; every replayed observation on those two runs remained within
`1.5e-14` of the recording and the terminal status matched. Exact state
hashes are therefore platform-sensitive for this frozen engine. A cross-OS
check must distinguish approximate trajectory equivalence from exact hash
reproduction; no game or observation code was changed to mask the difference.
No 40-episode fault-attribution
dataset or Sol diagnosis has been collected. Cross-game diagnostic accuracy
has not been measured; the shared contract tests only check interface
compatibility.

## Multiroom optimization in progress

The first SFT dataset contained only `-1,0` combat states: the 7,000-character
filter excluded every later room. A new deterministic selection spreads 60
decisions per seed across the available rooms and raises the state limit to
12,000 characters. The resulting 360-decision set includes 300 training and
60 validation decisions, 85 training boss states, and shooting labels in all
four cardinal directions. It retains the original seed split and frozen game
files. The decision-file SHA-256 is
`4016873759763a7b70b0ae3daed7dc4ffeef401db3ae96fe53d25e49941d8dab`;
the archive `agent_demos_v3.tar.gz` has SHA-256
`fed5e407ce40d76c6d4acb28d795096b957df69595e67deeb771644ba134ac7c`.
Regenerating the old default dataset still yields its original decision-file
SHA-256. The next run will fine-tune from the first trained checkpoint with
an 8,192-token limit on GPU 2. Its game result is not yet known.
For the next capability check, freeze seeds `7001` through `7004` and a
1,800-tick limit before loading the new weights. Report Warden wins, rooms
cleared, kills, damage, and model-selected versus executed actions for each
seed. Use the same harness and controller for an old-weight comparison. The
goal is a reliable, capable NanoJev-backed baseline for #184, not a claim of
superiority over the scripted teacher.

A 1,800-tick old-controller run on seed `4001` exposed a navigation stall:
after clearing room `0,-1`, the player repeatedly moved east from x≈593.927,
just short of the door trigger at x=594. A cardinal move would cross the wall
bound, so the player never entered the boss room and was truncated at tick
`1800` with 6 kills and 4 cleared rooms. On seed `5001`, the same trained
weights and old controller reached and defeated Warden at tick `1039` with
4 kills and 3 cleared rooms. The complete audited pair and rule runs are in
`agent_assisted_full_v1.tar.gz` (SHA-256
`2839810f0b7e827de89d88be7613371f284710e109cf2cd79f040932a5866905`).
The adapter-only
`model-assisted-combat-v2` controller makes a smaller diagonal step when the
same public-observation geometry appears at any of the four doors. Tests
confirm the old controller remains unchanged and the new action actually
enters each door in the frozen engine. With the same trained weights, v2
entered the boss room and defeated Warden on `4001` at tick `1313`, with 7
kills, 5 cleared rooms and one damage taken. The trace contains one
`door_entry_nudge` and passed exact replay audit on the UCSD recorder. It is
archived in `agent_assisted_full_v2.tar.gz` (SHA-256
`e54782bb5173f765a2b52f00bf61cbfac8078eadb09b5785d5c040175a41aa42`).
The 8,192-token multiroom head SFT is now running on GPU 2; its result is
pending.

While that SFT ran, the previous trained checkpoint was evaluated on the
predeclared `7001`–`7004` seeds using the same v2 controller and harness on
otherwise idle GPU 5. It defeated Warden on `7001` (tick `1434`), `7002`
(tick `765`), and `7003` (tick `1332`). Seed `7004` reached the 1,800-tick
limit with 3 rooms cleared and 5 kills. The full four-seed data, including
paired rule runs, is in `agent_v1_fixed_v2_gpu5.tar.gz` (SHA-256
`b7d44d2538332fccb0d0187e00c03b8e4a7ae091e3fb5441004419f407b6119a`).
This is the current capability baseline for #184: 3/4 Warden wins in a
small fixed-seed pilot. GPU 5 was released immediately after evaluation;
training continued on GPU 2, so this project used at most two GPUs at once.

The multiroom SFT completed on GPU 2. It started from the first trained
checkpoint (`37029ae8…d9754`) and produced an independently hashed checkpoint
`a60db30eb906b46e89499e284a40f31f7e1ea1a790f4a2ac744253b9f90a1c51`.
Validation loss fell from `1.5313` to `1.3667`; validation choice accuracy was
`0.5056` on 180 question examples. It trained on 900 question examples with
an 8,192-token limit. The checkpoint and full manifest are retained with the
private run evidence. The compact
training metadata archive is `agent_train_v3_metadata.tar.gz`
(SHA-256 `97ef977a74aa9f9372b96834adfc8fdd343935390de160cbfdf73110424e8842`);
it excludes the 2.3 GB weight file. GPU 2's 18 GiB idle hold was restored.
Before inspecting the new-weight result, the selection rule was: prefer the checkpoint
with more Warden wins on `7001`–`7004` at 1,800 ticks; if tied, compare rooms
cleared on losses and then median winning tick. This is a small development
comparison, not a confidence claim. For a later controller check, reserve
fresh seeds `8001`–`8004` at 1,800 ticks before running any v3-controller model
game. Evaluate v2 and v3 on the same seeds with the selected new checkpoint.

The new weights with the unchanged v2 controller and harness won on `7001`
(tick `1455`), `7002` (`810`), and `7003` (`1201`), and reached the tick limit
on `7004` with 6 rooms cleared and 9 kills. The old weights also won 3 of 4,
but cleared only 3 rooms and scored 5 kills on the losing `7004` game. Under
the stated selection rule, the new checkpoint is the selected weight baseline.
This is an observed capability improvement on a development seed, not a
statistically established gain. The full audited evaluation is in
`agent_v3_fixed_v2_gpu5.tar.gz` (SHA-256
`f167eebceef50626f4adfc0beda4a197a1bec293e52df6b88b7796c5367b5ad6`).
GPU 5 was released after the run; GPU 2's 18 GiB idle hold remained active.
The model was queried on every decision. Its selected movement changed the
executed movement relative to the rule suggestion on `51`, `20`, `20`, and
`77` decisions respectively across the four seeds. Executed shooting never
departed from the rule suggestion. The baseline is therefore explicitly a
hybrid model-backed agent, with a meaningful weight-sensitive movement path;
the wins do not establish independent model combat skill.

The old-weight agent was also run on `7004` with the task's full 6,000-tick
limit. It still truncated with 5 kills and 3 rooms cleared; the paired rule
run also truncated. The full recording is
`agent_v1_7004_6000_gpu5.tar.gz` (SHA-256
`76c81e65dbbff5ec5005da19f29566ea3f9aacfe92dbdfaf4bb9215a65d6ffc3`).
Its last normal combat room repeatedly alternated movement above and below
a one-shot enemy without aligning a shot. A local frozen-engine continuation
from the recorded tick-1800 state tested a narrow `model-assisted-combat-v3`
controller candidate: it kept moving toward that enemy's shooting lane and
cleared the room by tick `1821`. This test used rule-selected suggestions for
the continuation; it is not a real NanoJev evaluation and v3 had not yet been
deployed.

The selected new weights were subsequently run on `7004` at the full
6,000-tick limit with the v3 controller. The model-backed agent defeated
Warden at tick `1977`, with 7 rooms cleared and 10 kills. The complete
audited recording and launcher are in
`agent_v3_7004_6000_gpu5.tar.gz` (SHA-256
`47d62c104107beb639263f1b30ed436149203d17a35ad53c6796a2fbb5858281`).
Crucially, the v3 cleanup override fired zero times. An offline same-state
check recomputed all 659 executed actions with v2 and found zero differences.
Thus this win supports the new-weight, longer-horizon agent's capability but
does not support a v3-controller improvement.

On the predeclared untouched seeds `8001`–`8004` at 1,800 ticks, the selected
new weights with v2 won all four Warden games (ticks `1689`, `866`, `850`,
`1387`). A same-state offline application of v3 to every one of those 1,599
recorded model decisions found zero executed-action differences. V3 therefore
has no demonstrated benefit for the selected checkpoint on this held-out
batch. The separately run v3 arm also won the same four games at exactly the
same ticks. For all 1,599 paired decisions, the original requests, complete
answer probabilities, selected actions, executed actions, controller reasons,
and pre-action state hashes matched exactly. Only service metadata and timings
differed. Both arms passed exact replay on Linux. The full paired recording,
logs, and launcher are in `agent_v3_controller_pair_gpu5.tar.gz`
(SHA-256 `4f715f1ee5a32129c342b2e12697c89c1c359b4b8c57d4c64cab18cd33f96059`).
V3 is preserved as a tested optional controller candidate, but v2 is selected
for the #184 baseline because v3 did not improve any observed behavior with
the selected weights.

## Selected baseline for #184

The selected agent uses the `model-assisted-combat-v2` agent controller,
the harness in `config/controller.json` and `config/questions.json` (spec ID
`h-c2d1d7d325930098`), and the multiroom SFT NanoJev checkpoint. Place the
six selected checkpoint files in `work/rea_issac_agent/checkpoint/` or pass
another directory with `--checkpoint-dir`; [REPRODUCE.md](REPRODUCE.md)
describes the layout, verification, and launch commands.
The weight file has SHA-256
`a60db30eb906b46e89499e284a40f31f7e1ea1a790f4a2ac744253b9f90a1c51`.
Inference uses the NanoJev tree decision service with an 8,192-token limit;
the game executes each choice for 3 ticks and uses each task's native tick
limit. The frozen game and public interface hashes remain in
`agent_frozen.json`. The historical archives listed above retain the original
checkpoint metadata, raw responses, and complete replay records; they are not
included in this PR. The portable launch commands are in `REPRODUCE.md`.
Exact replay of the new `7004` Linux trace on macOS stopped at tick `120`
because the frozen engine's floating-point state diverged across platforms;
the Linux recorder's exact replay passed. This is the same cross-OS audit
limitation observed earlier, and the engine was not changed to hide it.

The selected checkpoint and controller won 3/4 fixed development seeds at
1,800 ticks and 4/4 subsequently reserved fresh seeds at 1,800 ticks. The
remaining fixed seed `7004` won at tick `1977` when allowed the full 6,000
ticks; a same-state check showed v2 and v3 executed identical actions in
that run. These are small capability pilots, not a population estimate.
The #184 40-episode fault attribution experiment, Codex Sol diagnoses,
fault-family generalization, true API calls, and API cost accounting remain
unperformed. Shared contract behavior was tested with REA-Issac and a
structurally different synthetic action space; cross-game diagnostic accuracy
was not tested.
