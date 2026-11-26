# Exp 0 v0 episode contract (`honing-exp0-episode-v0`, contract version 0.2.0)

How a game collector writes one Exp 0 v0 attempt so that the common tools in this directory can validate,
audit, render, probe, diagnose, score, bundle and verify it the way the accepted Griddly Spiders v0 release
did. Written 2026-11-23. Machine-readable schemas: `schemas/*.schema.json` (JSON Schema
2020-12, checked by `schema_lite.py`, standard library only). Validator: `validate_episode.py`. Shared
guard and helpers: `contract.py`. Adapter interface: `adapter.py`. A complete executable example (both
shapes, all four conditions): `tests/synthetic_game.py`.

Sources the fields come from: Griddly v0 `episode.py`, `audit_set.py`, `episode-index.schema.json`,
`collection.json`, `LABELS.md`; issue #174 and #185 (required identity, runtime, intervention, verified
component, outcome and annotation fields); #170 (probe identity, tolerance 0.01). Where this contract
differs from Griddly the reason is given in place.

Changes are listed in section 11.

## 1. The unit and the four conditions

One **attempt** = one complete run of one episode ("一局") from the scenario's start breakpoint to a native
terminal or a cap, under one condition, in one reset group. It is one directory.

| `condition` | what differs from the clean release | Griddly name |
|---|---|---|
| `no_injection` | nothing (clean checkpoint, clean harness, clean controller) | same |
| `harness` | the harness changes what the model is told or offered, from a trigger rule on | same |
| `weights` | the whole episode is served by the alternate checkpoint (its own decision service) | same |
| `controller` | after the model's selection, the controller executes something else, from a trigger rule on | same |

The brief's `none` is written `no_injection`, as in Griddly. A **reset group** (`reset_group`, Griddly's
`scenario_group`) is one independent start (scenario + seed/start breakpoint); its clean and changed
attempts are paired and share one split. A **slot** is `<reset_group>/<condition>`; a slot may have several
attempts (`attempt.number` 1, 2, ...), all kept.

## 2. Two decision shapes, one record

- `native`: one model decision -> one native action held for k ticks (VVVVVV, Celeste Classic, Taisei,
  Hurrican).
- `option`: one model decision -> an option the harness carries out as many native windows (full HK,
  hallowdale).

Both write the same decision row. Every executed row lists its native **windows**; a `native`-shape
decision has exactly one window (the validator checks this). The common tools therefore have one code
path for both shapes.

## 3. Directory layout of one attempt

```
<collection_root>/
  collection.json            the frozen plan (section 8), byte-identical copy
  <run>/                     one attempt; <run> = private run id, e.g. fk01-harness-a1 (never shown to a model)
    manifest.json            written before the first decision            (schemas/manifest.schema.json)
    decisions.jsonl          one row per decision, appended as it happens  (schemas/decision.schema.json)
    responses/<NNNN>.raw     raw response bytes exactly as received; a retry is <NNNN>-t2.raw
    public_states.jsonl      per model/forced row: the saved public state for re-rendering (private)
    result.json              native outcome, written at the terminal or cap   (schemas/result.schema.json)
    error.json               written when the attempt fails or is interrupted (schemas/error.schema.json)
    extra/                   optional: anything else raw (observations, bridge logs); never a model input
```

Files are UTF-8; JSONL lines are `contract.canonical()`: compact JSON with keys in **insertion order, never
sorted** (no NaN). Key order is data: the order of a request's options is part of what Laya encodes (section 5).
Paths inside the records are relative to `<run>/`. An attempt with `error.json` is **incomplete**: it is kept and listed with its reason,
never scored. Do not delete or overwrite attempts; a new attempt gets a new directory.

## 4. `manifest.json` (per attempt)

| field | meaning |
|---|---|
| `format` | `"honing-exp0-episode-v0"` |
| `contract_version` | `"0.1.0"` |
| `game_id` | dataset game id, e.g. `hollow-knight-forestdango`, `vvvvvv` (= plan `game_id`) |
| `shape` | `native` or `option` |
| `run` | the directory name (private id) |
| `condition` | `no_injection` / `harness` / `weights` / `controller` |
| `reset_group` | the plan group id this attempt starts from |
| `attempt` | `{"number": k, "slot": "<reset_group>/<condition>"}` |
| `scenario` | `{"task", "start": {exact breakpoint: scene, entry, position, HP, resources, abilities, ...}, "seed", "caps": {"decisions", "native_ticks", "wall_seconds"}}` |
| `selection` | `{"rule": "argmax", "seed": null}` or `{"rule": "sample", "seed": <int>}` (use `contract.Sampler`); with a per-group seed in the plan (section 8) the manifest carries its group's seed |
| `pins.game_build` | `{"id", "sha256" or "version"}` of the game binary/build |
| `pins.code` | `{"upstream_sha": <40 hex>}` and/or `{"fingerprint": ...}` of the collector + bridge + adapter code |
| `pins.harness` | `{"id", "fingerprint"}` of the frozen harness snapshot (`contract.fingerprint_tree`) |
| `pins.controller` | `{"id", "fingerprint"}` of the controller/executor code and settings |
| `pins.observation` | `{"condition": <plan observation id>, "knobs": {...}}` (visible profile, units, kinds, ...) |
| `pins.checkpoint` | the decision server's `/v1/health` `metadata` at the start, at least `weightsSha256`, `role`, `repo`, `revision` |
| `clean_weights_sha256` | the plan's clean checkpoint SHA-256 |
| `intervention` | `null` for `no_injection`; else `{"component", "family", "instance", "trigger_rule", "changed_artifact", "original_ref", "altered_ref", "builder"?}` copied from the plan's fault definition |

`original_ref` / `altered_ref` say where the two artifacts are, e.g. harness:
`"decisions.jsonl:clean_request"` / `"decisions.jsonl:request"`; weights: `"checkpoint:<clean sha>"` /
`"checkpoint:<alternate sha>"`; controller: `"decisions.jsonl:model_choice"` /
`"decisions.jsonl:executed"`. The trigger *decision* is only known afterwards: it is `result.trigger_index`.

## 5. `decisions.jsonl` (one row per decision)

`kind` is `model` (the model was asked), `forced` (exactly one option was available, no model call;
Griddly's forced actions) or `auto` (the harness or collector executed native windows without any option:
boot, settle, scene transitions; `model_choice` and `executed.option` are `null`).

| field | meaning |
|---|---|
| `index` | row number from 0 (all kinds) |
| `kind` | `model` / `forced` / `auto` |
| `native_before` | `{"tick", "frame"?, "sim_seconds"?}` native clock when the decision was asked |
| `public_observation` | compact JSON of what a human player sees on the current screen/HUD at this decision, written by the collector from the visible observation **before** the harness renders it; the harness fault must not change it. Required for `model` rows. Recommended size <= 1-2 KB (plan `diagnosis.public_observation_max_chars`). It reaches the diagnoser (section 9) |
| `public_state` | line number in `public_states.jsonl` of this row's saved state, or `null` |
| `request` | the exact `/v1/decide` body sent (`model` rows): `{"id"?, "parts": {...}, "questions": {<question_key>: {"type": "choice", "instructions", "criteria": {option: text}}}}`. Its `parts` are the observation exactly as rendered to the model. Only Laya's sections `essential`, `details`, `memory`, `logs` (other part names are ignored by the model and rejected by the guard). Exactly one choice question, named by the plan's `question_key` |
| `clean_request` | harness episodes only: at every row where the harness fault was applied, the request the clean harness renders from the same state; else `null` |
| `response` | `model` rows: `{"file": <accepted try's raw file>, "tries": [{"file", "status": "accepted" | "rejected" | "transport_error", "reason", "seconds"}]}` |
| `probabilities` | the option probabilities exactly as the server returned them (Laya rounds to 4 decimals) |
| `server_choice` | the server's `choice` |
| `model_choice` | the policy's selection: the server choice (`argmax`) or the seeded draw (`sample`); `forced` rows: the only option |
| `executed` | `{"option": <option the controller actually carried out>, "windows": [window, ...]}`; `null` only for the unexecuted last row of a failed attempt |
| `fault` | `null` when no fault mechanism acted at this row; else `{"component", "changed": true/false/null, "detail"}` (weights: `changed: null`, the effect is unknown until the evidence step) |
| `events` | native events from this decision until the next one: `{"tick", "kind", "text", "visible"}`, kind in `damage, death, heal, kill, goal, checkpoint, room, pickup, terminal, other`; `visible: true` when a human player sees it (only those reach the diagnoser). Take them from native game signals, never inferred from absence |
| `native_after` | native clock after the last window |
| `timing` | `{"model_seconds", "execution_seconds", "wall_seconds"}` |
| `extra` | optional private game-specific fields (never rendered for any model) |

**Key order is data.** Write the request exactly as sent, keys in the order sent. The same options in another
order get other probabilities (measured: a logged HK request re-serialized with sorted keys gave max |dp|
0.0358; a VVVVVV request 0.069). Compare requests only with
`contract.same_request` (order-sensitive; dict `==` is not). The validator reports an error when the raw
reply's option order differs from the recorded request's, i.e. when a row was re-serialized with sorted keys.

A **window**: `{"native_action", "ticks", "start_tick", "frames"?, "intended_action"?, "reason"?, "extra"?}`.
`native_action` is the native input actually sent (a canonical string, e.g. `right`, `left+jump`).
`intended_action` appears **only** when the controller fault sent something other than what the harness
requested in this window. `reason` is the clean harness's own execution reason (private; never written
by fault code; not rendered for the diagnoser). Native non-execution (cooldown, unavailable jump) is not a
controller change: put the game's acknowledgement in `extra`, not in `intended_action` (#185). In the `option`
shape an option whose behaviour ends without sending any native input (e.g. HK King's Pass `jump_over_enemy`:
"cannot clear the enemy from here") is recorded truthfully as `windows: []` with `native_after == native_before`;
never invent a window. A `native`-shape decision has exactly one window; an `auto` row has at least one.

### The guard (before any native action)

Use `contract.guard_response(request, raw_bytes, question_key)` on every reply **before** executing
anything. It raises `ResponseRejected` when: the reply is not a JSON object; it does not echo the request
`id`; it does not answer exactly the asked question; the probabilities do not cover exactly the offered
options, are not finite in [0, 1], or do not sum to one within Laya's rounding (n x 0.5e-4); the server
choice is not an offered option of maximal probability; or the model did not see the full request
(non-Laya part names, `stateText` differs from the joined `essential/details/memory/logs` lines, anything
`trimmed`, or `tokenAudit[q].stateTruncated` not `false`). On rejection: save the raw bytes, record the
try as `rejected` with the reason, execute nothing; re-send the identical request at most
`plan.response_tries - 1` times; if no try is accepted write the row with `executed: null` and
`error.json` (`status: rejected_response`). Never substitute another action for the model's choice.

### Fault records, per condition (checked by the validator)

- `no_injection`: every `fault` and `clean_request` is `null`; `executed.option == model_choice`; no
  window has `intended_action`.
- `harness`: rows where the fault was applied have `fault.component = "harness"`, a `clean_request`, and
  `fault.changed == (clean_request != request)`. The trigger is the first row with `changed: true`.
  Nothing else differs (execution follows the model's choice).
- `controller`: the fault acts after the selection. A changed row has `executed.option != model_choice`
  (the option was replaced) and/or windows with `intended_action` (keys dropped or altered inside the
  option). Every such difference has `fault.changed: true`; no difference without a fault record.
- `weights`: `pins.checkpoint.role = "alternate"` and its `weightsSha256` differs from the clean one;
  `fault` is `null` or `{"component": "weights", "changed": null}`; no `clean_request`; execution follows
  the choice.

No fault may leak into model-facing text except the harness fault itself: the validator refuses fault
family/instance names and checkpoint SHA-256s anywhere in requests, public observations, native actions,
event texts or window reasons, and warns on words like "fault", "inject", "intervention".

## 6. `public_states.jsonl` and the re-render callback (probes)

One line per `model` (and `forced`) row: `{"index": <row index>, "state": {...}}`. `state` is everything the
**clean** harness needs to render that row's request again, restricted to public information (what the
player saw, plus the harness's own memory derived from earlier public observations, e.g. remembered
surfaces). It never reaches a model or the diagnoser directly.

The game supplies an adapter (`adapter.py`): `game_id` and `rerender(state, history) -> request`, where
`history` is `[contract.public_history_entry(row) for row in earlier rows]`. It is deterministic, uses no
fault switch, no network and no game process, and returns the full request including the logged `id`.
For every decision of a `no_injection` episode it must reproduce the logged request exactly; at a
harness-fault row it returns the clean request (`== clean_request`). `probes.py selfcheck` tests both
before any probe call. A game whose harness cannot be re-rendered exactly must say so: its probe
`standard_rerender_request_changed` would then be true for controls too, which would mislead the probe
mode, so it is a blocker to report, not to paper over.

Probes use the first four available model decisions of each episode (#170, Griddly). Choose fault trigger
rules that fire early (ideally within those four), or the probe cannot show the change; the audit reports
how many verified episodes have their trigger outside the probed decisions.

## 7. `result.json` and `error.json`

`result.json`: `status` (`win` / `loss` / `timeout` / `other_terminal`, from native game events; a cap is
`timeout`), `reason` (from the plan's outcome vocabulary, e.g. `boss_defeated`, `player_died`,
`decision_cap`), `terminal_tick`, counts (`rows`, `model_decisions`, `forced_decisions`, `auto_rows`,
`windows`), `native_ticks` (= `terminal_tick` - first row's `native_before.tick`), `native_frames`,
`sim_seconds`, `wall_seconds`, `event_counts` (by kind), `trigger_index` (first row where a harness or
controller fault changed something, else `null`), `fault_changed_rows`, and `checkpoint_end` (the
server's `/v1/health` metadata at the end: the same `weightsSha256` and `role` as at the start, else the
attempt is invalid).

`error.json`: `{"status": rejected_response | transport_error | game_crash | bridge_timeout |
identity_changed | interrupted | collector_error | other, "type", "message", "at_index"}`.

## 8. The frozen collection plan (`collection.json`, one per game)

Frozen before the main collection (`schemas/collection.schema.json`): `game_id`, `dataset_dir`
(`<game-id>/v0/`), `shape`, `question_key`, `observation`, `per_condition` (10), `initial_groups` (10),
`selection` (`{"rule", "seed"}`; for sampling with one seed per reset group, as when the native start is
fixed, `{"rule": "sample", "seed": null, "per_group_seed": "scenario.policy_seed"}` names where each group keeps
its seed), `clean_weights_sha256`, `alternate_weights_sha256`, `weight_difference_tolerance` (0.01),
`probe` (`first_decisions` 4, `identity_tolerance` 0.01, `tie` 0.01), `diagnosis` (`max_tokens` 9000,
`time` = `{"unit": "native_tick" | "sim_seconds", "before", "after"}` for the renderer windows,
`preamble` = one or two game-neutral sentences describing the game's time unit and options for the
diagnoser, `windows.max_runs_listed`, `windows.max_mismatches_listed`), `outcomes` (the reason vocabulary per
status), `faults` (per component: `family`, `instance`, `trigger_rule`, `changed_artifact`), `groups`
(`[{"id", "split": train | val | test, "scenario": {...}}]`, initial groups first, then reserves),
`attempt_cap_per_slot`, `neutral_id_prefix` (e.g. `e` -> `e001`), `shuffle_seed` (200 as Griddly),
`response_tries`.

## 9. What the common tools do with it (the pipeline, Griddly-aligned)

| step | tool | output |
|---|---|---|
| validate every attempt | `validate_episode.py <collection>` | per-attempt `valid` / `incomplete` / `invalid` |
| weights evidence when no identical clean request exists | `audit_set.py reask` (calls the clean service) | `weights-reask.jsonl` (private) |
| audit, evidence, selection, neutral ids | `audit_set.py audit` | `audit/attempts.jsonl`, `audit/episodes.json` (gold; never a model input) |
| read inputs | `diagnosis_inputs.py` | `read-inputs.jsonl` (`{"episode_id", "text"}` only) |
| probes | `probes.py capture` x2, `probes.py compare` | `probe-reference.jsonl`, `probe-alternate.jsonl`, `probes/identity.jsonl`, `probes/probe-inputs.jsonl` |
| diagnosis | `diagnose.py` (Codex CLI or stub) | `diagnosis-<mode>/settings.json`, `diagnoses.jsonl`, attempts |
| scoring | `score.py` | `scores.json` |
| bundle / verify | `bundle.py`, `release_metadata.py`, `verify_release.py` | `<release_root>/<game-id>/v0/` |

**Evidence** (audit, per intervention attempt; the label stays separate from it):
- harness: the trigger row's `request` vs `clean_request` (`method: request_vs_clean_render`) with a
  summary of the changed parts and options;
- controller: the trigger row's `model_choice` vs `executed.option` and window mismatches
  (`method: selected_vs_executed`);
- weights: Griddly's rule first: the first request of the weights attempt that is identical (same request,
  same native tick) to one of the matched clean attempt in its reset group and whose normalized
  probabilities differ by more than 0.01 (`matched_clean_identical_request`); when the game is not
  deterministic enough for that, the logged requests of the weights attempt re-asked under the clean
  checkpoint (`same_request_reask`; #185 allows saved-request probes when exact state restore is
  unavailable). The first decision with max |dp| > 0.01 is the evidence.
`verified_component` and `annotation_status` (`verified` / `control` / `excluded`) are written by the audit,
never by the collector. `outcome_effect` is `not_established` when a paired clean outcome exists, `untested`
otherwise; `release_level_score_drop` is `untested` (v0).

**What the diagnoser sees** (explicit allowlist, `diagnosis_inputs.py`): the plan's preamble; per model
decision the request's rendered parts, one more state section `offered options` with the question's
instructions and every option text (the #168 renderer alone shows only option keys; HK puts state into option
texts), the `public_observation`, the
option probabilities and the model choice, the executed option and a summary of the native windows
(native actions with tick counts; requested-vs-sent pairs when a window records `intended_action`), visible
native events, and the native outcome (status and reason). Never: run names, paths, condition, fault
fields, `clean_request`, checkpoint ids or roles, fingerprints, window `reason`, `extra`, invisible events,
the server's internal state.

**Release content and upload.** A release holds structured text/JSON evidence and our code only: requests,
raw replies, probabilities, executed actions and windows, rendered and public observations, outcomes, probes,
diagnoses, scores, configs and code. No game binaries, game assets, frames, screenshots or videos: `bundle.py`
copies only UTF-8 text/JSON files from `extra/` (others are listed in `dataset.json` as skipped) and
`bundle.py release` refuses any non-text file, credential-like string or absolute local path. Checkpoints are
referenced by repo + revision + weights SHA-256 (`checkpoint-references.json`; artifact URI defaults to the
Hugging Face repo at its revision), never re-uploaded. Uploads to the private `honinglab/exp0-data` are made by
the release owner only, after reviewing a locally verified release: `upload_release.py check|plan` (no token,
no network), then `upload_release.py upload ... --execute` and `download-verify ... --execute`, which read the
token from the file named by `HF_TOKEN_FILE` and never print it.

## 10. Checklist for a game collector

1. Copy the plan into the collection root as `collection.json`; create one directory per attempt.
2. Write `manifest.json` before the first decision (checkpoint = `/v1/health` metadata; drop nothing).
3. Per decision: save the public state, render the request, (harness fault: also render the clean request),
   POST it, save the raw bytes, run `guard_response`, select, (controller fault: change the execution and
   record it), execute, append the row with windows and native events.
4. At the end: re-read `/v1/health`, write `result.json`; on any failure write `error.json` instead.
5. Run `python3 integrations/_shared/exp0/validate_episode.py <collection_root>`; every attempt must be `valid` or
   `incomplete`. Run `python3 integrations/_shared/exp0/probes.py selfcheck <collection_root> --adapter <file.py:attr>`
   on a clean attempt to prove the re-render.

## 11. Changes

- 0.1.0 (2026-11-23 11:10 PST): first version (CONTRACT READY).
- 0.1.1 (11:31): rows keep key order (`canonical()` no longer sorts keys); order-sensitive request comparison
  (`same_request`); validator checks the raw reply's option order against the recorded request. Rows written
  with sorted keys must be re-written from the raw logs.
- 0.1.2 (11:40): `option`-shape rows may have `windows: []` (a behaviour that sent no native input; requested by
  the HK collector); per-reset-group sampling seeds via `selection.per_group_seed` (requested by the small-games collector); release
  content policy (text/JSON only) and the upload tool (`upload_release.py`, release owner only).
- 0.2.0 (2026-11-25 15:45 PST): several questions per decision, window lengths, several fault instances per
  component, dry-run plans (section 12).


## 12. Several questions per decision, window lengths, several fault instances, dry runs (0.2.0)

Added 2026-11-25 for the full-HK raw-key harness (`v9_keys`: every
decision asks `move` and `attack` in one request, as Tank War v0 asked motion and weapon). Everything in 0.1.x
stays valid and unchanged: a plan with `question_key` and rows whose choices are strings validate, render, probe,
audit, score and bundle exactly as before (the 11 tests of 0.1.2 pass and produce the same inputs, probes and
scores; `tests/test_pipeline.py` now has 15).

- **Plan.** Either `question_key` (one question, as before) or `question_keys` (two to 16 questions, in request
  order; exactly one of the two). Every model request asks exactly these questions in this order.
- **Rows with several questions.** The same fields, keyed by question in request order:
  `probabilities` = {question: {option: p, as returned}}, `server_choice` = {question: option},
  `model_choice` = {question: option}, `executed.option` = {question: option actually carried out}. One request and
  one raw reply per decision (Laya answers all questions of a request; the reply's `answers` and `tokenAudit` cover
  every question). `fault.questions` (optional) names the questions whose text or execution a fault changed.
  Counted separately: decisions, and question records (= decisions x questions; validator summary `questions`,
  dataset report `model_questions`, card `MODEL_QUESTIONS`).
- **Guard.** `contract.guard_decision(request, raw, question_keys)` applies every check of `guard_response` to each
  question; one failed check rejects the whole reply (nothing is executed). `guard_response` is unchanged for one
  question. Sampling (`Sampler.choose_all`): one draw per question in request order.
- **Controller faults at the key level.** A window may carry `intended_ticks` (the length the clean controller
  would have used) next to or instead of `intended_action`, when a controller fault held the keys longer or
  shorter. Either field marks a controller change and needs a `fault` record with `changed: true`; a window whose
  intended value equals the sent one is invalid. With near-raw options a key-level change usually changes the
  executed option too (e.g. LEFT sent for a chosen `right`): then `executed.option` names the option of the keys
  actually sent, and the diagnoser sees "Selected move right, attack forward; executed move left, attack forward".
- **Several validated fault instances.** `faults.<component>` may be a list of fault definitions; each attempt's
  `intervention` must equal one of them (by instance).
- **Evidence.** Harness: `change_summary` keeps the part / line changes and adds `questions` (per-question option
  changes) and `questions_changed`. Weights: the difference is the largest over all questions; `clean_probabilities`
  and `alternate_probabilities` are keyed by question. Controller: `first_mismatches` also gives
  `requested_ticks` / `sent_ticks`.
- **Diagnosis inputs and probes.** Each decision shows every question's probabilities and choice (the #168 renderer
  supports several questions); the `offered options` section lists each question's instructions and options; the
  #170 identity check covers every question (its report counts `questions`). Probe inputs with several questions
  key the probabilities by question.
- **Dry runs.** A plan with `"dry_run": true` may have `per_condition` < 10 to run the whole chain (audit, inputs,
  probes, diagnosis, scores, cards, bundle, fresh-environment verify) on pilot or smoke data. The bundle and its
  release manifest say `dry_run: true`, the verifier checks 4 x `per_condition` episodes and reports `dry_run`,
  and `upload_release.py check` refuses such a bundle. Without the mark a bundle needs 10 per condition, as before.
- **Version.** `contract_version` "0.2.0"; the validator accepts 0.1.x and 0.2.x attempts.
