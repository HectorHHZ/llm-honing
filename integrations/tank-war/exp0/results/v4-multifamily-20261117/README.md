# #183 second prospective Tank War component-attribution test

## Result and scope

On the pre-frozen 40-game test, `gpt-6-sol` at `high` effort correctly classified
**14/30** technically verified fault losses from the replay alone and **25/30**
with controlled same-state probes. The paired gain is **11/30, or 36.7 percentage
points**; the pre-registered instance-and-reset-group bootstrap 95% interval is
**+19.2 to +51.5 points**. The pre-registered technical-component go rule passes.
This is a result for the sampled constructed faults, not a finding that each
intervention caused its game's final loss. A full #183 go/no-go remains pending
independent human review of decision relevance and the limits below.

The comparison used the same 30 fault losses and 10 unmodified losses in every
mode. Each fault class contributed 10 games. The 40 games span 19 reset seeds;
the 30 fault games span 17. The frozen sample SHA-256 is
`3c7a50d15531c21b8c9937122b9c33a9b998498b189dfa832abf1238c9b746f4`.
Sample selection, model IDs, effort, prompts, schema, modes, score rule, and
limits were fixed before any of these 320 diagnostic answers. The earlier
33-game sample with only three harness cases was archived before any answer and
is not counted here.

| Mode | Correct fault games | Harness | Weights | Controller | Unmodified losses called harness/controller |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sol high, replay | 14/30 | 5/10 | 0/10 | 9/10 | 9/10 |
| Sol high, probe | 25/30 | 7/10 | 9/10 | 9/10 | 8/10 |
| Sol high, replay + examples | 27/30 | 10/10 | 10/10 | 7/10 | 1/10 |
| Sol high, probe + examples | 30/30 | 10/10 | 10/10 | 10/10 | 0/10 |
| Astra medium, replay | 8/30 | 0/10 | 0/10 | 8/10 | 9/10 |
| Astra medium, probe | 30/30 | 10/10 | 10/10 | 10/10 | 4/10 |
| Astra medium, replay + examples | 25/30 | 10/10 | 7/10 | 8/10 | 6/10 |
| Astra medium, probe + examples | 30/30 | 10/10 | 10/10 | 10/10 | 0/10 |

The Sol example-assisted probe gain over its matching replay baseline is
+3/30, with a 95% interval from 0 to +20.8 points. Astra medium's bare probe
gain is +22/30, with a 95% interval from +66.7 to +83.3 points; its
example-assisted gain is +5/30, from +3.0 to +33.8 points. These are secondary
comparisons. A `weights` answer on an ordinary unmodified loss is kept separate
from an unsupported harness/controller call: the model's own poor choice can
be a plausible explanation without an injected fault. The exact confusion
matrices, macro-F1, calibration, abstentions, and per-mode usage are in
[diagnosis-summary.json](diagnosis-summary.json).

## Data and validity

The final private audit at
`/data/xinkaiz/system0/exp0/tank-war/v4-multifamily-progress-20261117/`
indexes **1,186 labeled fault games across 1,330 attempted fault records**.
Of 1,202 complete fault-loss recordings, **1,120 (93.2%)** have a paired clean
game and enter the label table; the other 82 remain in the separate unverified
index with their original logs. Of the 1,120 paired losses, 1,058 have a
verified local layer effect and 62 remain uncertain. The fixed 100-game
extension for the retained
`extreme_standoff-v1` harness fault completed 100/100 without collection errors:
92 losses, 84 with verified local effects, and eight uncertain. The separate
128-seed, two-seat score screen for this fault was −0.2461 game points
(paired 95% interval −0.3066 to −0.1855). The other blind-test instances are
two real edited checkpoints from the reverse-strength family and the
`drop_fire_after_selection-v1` controller fault. Their screens and collection
counts are in [audit-summary.json](audit-summary.json). The exact audit used
to freeze the blind sample, before the next controller family was added, is
retained as [blind-source-audit-summary.json](blind-source-audit-summary.json).
A distinct
`wrong_legal_projection-v1` controller candidate passed its own screen at
−0.2422 points (−0.3066 to −0.1777). Its independent fixed collection
completed 160/160 games with no errors: 152 losses, 126 with verified local
controller effects, and 26 uncertain. It was added to the final dataset audit
but not to the already frozen blind test.

All **40/40** sampled games passed full deterministic replay. Neutral probes
covered all **2,151** recorded decisions. The probe audit passed 10/10 cases
for each fault class and 10/10 unmodified controls at the fixed 0.01
probability tolerance. At each of the ten edited-weight reference decisions,
the model input stayed identical and the selected action changed. The
probability-shift ranges were 0.386–0.412 for strength 3.0 and 0.436–0.470
for strength 4.0. None of the 481 control decisions crossed 0.01 or changed
selected action under the same checkpoint repeat. See
[weight-probe-analysis.json](weight-probe-analysis.json).

All **320/320** diagnosis calls completed with valid JSON and token usage;
there were no tool calls, output-schema errors, or missing usage records. The
calls used Bubblewrap with only anonymous replay/probe inputs and required CLI
files mounted. The sample, fault manifests, technical labels, and reference
decisions were outside the diagnostic sandbox. The Codex CLI version was
`0.158.0`; Sol used `high` and Astra used `medium`. The ChatGPT-login CLI does
not expose a verifiable internal model revision or temperature. The recorded
**23,596,241 input and 96,605 output tokens** are usage measurements, not an
API bill. Summed call durations were 3,883.9 seconds across at most two
concurrent calls; this is not wall-clock time. The per-mode caps of four
million input and 100,000 output tokens were respected.

The recorded private GPU service intervals for this expansion, including the
rejected candidate screens and the clean probe service, total about **11.62
GPU-hours** against the authorized 18 GPU-hour ceiling. This is reserved
service time, not measured utilization. No more than two experiment GPUs were
occupied at once, and all temporary services have exited. The machine-readable
[data card](data-card.json) separates the paired-loss utilization from the
smaller set with verified local effects.

## Error analysis and limits

Sol's bare probe made five fault errors. Two harness games were called
controller after the model noticed selected-versus-executed movement or fire
differences. This is a real boundary ambiguity: the tested harness strategy
also participates in plan revalidation after model choice. A third harness
game was called weights because the model focused on poor retreat and firing
choices. One edited-weight game was called harness after the diagnostic
over-weighted ordinary “Do not fire yet” instruction text; one controller
game was called weights after it focused on the model's hold choices. The
probe produced a net gain of 11 correct fault answers but still made unsupported
harness/controller calls on eight of ten unmodified losses.

The blind test contains one harness instance, two strengths of one weight-edit
family, and one controller instance. It does not establish transfer to unseen
families. The harness instance has a mixed functional boundary, even though
its selected reference cases have direct same-state model-choice evidence.
The local technical label is not proof that the fault caused the final loss.
The human decision-relevance review has not been joined to these case IDs.
No shared pipeline module changed; cross-game generalization was not tested
and is not claimed. These limits prevent an unconditional go for autonomous
Tank War failure attribution.

## Reproduction and private evidence

The run used the frozen private pre-registration copied as
[diagnosis-protocol.json](diagnosis-protocol.json). The actual orchestrator
command was:

```text
python3 /data/xinkaiz/system0/exp0/tank-war/v4-multifamily-progress-20261117/tools/run-tank183-v4-blind-main-v4.py
```

It invoked the repository's `prepare-multifamily-blind.py`,
`replay-multifamily-blind.py`, `build-critical-probes.py`,
`audit-sustained-probes.py`, `analyze-multifamily-weight-probes.py`,
`freeze-multifamily-diagnosis.py`, `run-critical-episode-only.py`, and
`score-multifamily-diagnosis.py` in that order. Exact arguments, logs,
source hashes, the 40-case frozen manifest, per-case requests and replies,
probe files, and raw games remain in the private store. Only aggregate
summaries are committed here.
