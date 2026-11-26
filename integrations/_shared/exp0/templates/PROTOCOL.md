# {{GAME_NAME}} Exp 0 v0 protocol

Evaluate one component diagnosis per complete native episode. The primary endpoint is component accuracy on 30 verified interventions, with abstentions and invalid outputs scored as incorrect. Compare `read` and `probe` on the same episodes. The ten no-injection controls measure unsupported component accusations. [Issue {{ISSUE}}]({{ISSUE_URL}}) · [Exp 0 #174](https://github.com/HectorHHZ/llm-honing/issues/174)

## Collection

Use {{GAME_BUILD_DESCRIPTION}} (pins in `collection.json` and each attempt's manifest), the `{{OBSERVATION_CONDITION}}` observation ({{OBSERVATION_SUMMARY}}; what a human player sees on the current screen and HUD, as structured text, never pixels), the harness snapshot `{{HARNESS_ID}}` and the controller `{{CONTROLLER_ID}}`. Decision shape: `{{SHAPE}}` ({{SHAPE_SUMMARY}}). The clean checkpoint is `{{CLEAN_CHECKPOINT}}`; the alternate is `{{ALTERNATE_CHECKPOINT}}`. The collector checks the server-reported weights SHA-256 before and after each episode and validates every reply before any native action.

One episode ("一局") starts at {{START_BREAKPOINT}} and ends at the first native terminal: {{END_CONDITIONS}}. Caps: {{CAPS}}. Selection: {{SELECTION_RULE}}. Use the reset groups in `collection.json`, the first ten for each condition; if a group fails validation, collect reserve groups in listed order and keep all attempts (attempt cap {{ATTEMPT_CAP}} per slot). Select non-wins first for intervention conditions, in plan order within an outcome bucket. Include both winning and non-winning controls when the eligible attempts contain them. Keep native loss, other terminal status and caps distinct.

The three interventions are:

- Harness: {{HARNESS_FAULT_RULE}} Preserve the logged request and the clean render of the same state as direct evidence.
- Weights: the alternate checkpoint serves the whole episode. {{WEIGHTS_EVIDENCE_RULE}}
- Controller: {{CONTROLLER_FAULT_RULE}} Preserve the selection, the executed option and every native window.

Validate every attempt against the episode contract; record all incomplete, untriggered, rejected and unselected attempts. {{REPLAY_OR_TIMING}} Use the matched clean result as separate outcome evidence. This v0 collection records per-decision component effects; release-level score drop is untested. Pilots were used for development and are outside the scored set.

## Diagnosis

Render `read` inputs from an explicit allowlist of public observations, displayed model requests, option probabilities, selected/executed actions with a summary of the native windows, and native outcomes. Use neutral episode ids and the shared 9,000-estimated-token renderer budget. Keep the answer-bearing index, file paths, checkpoint ids, experiment metadata and internal native state outside the diagnostic request. Reserve space for the opening and terminal decisions and ordinary selected/executed and requested/sent mismatches. [Shared trace contract #168](https://github.com/HectorHHZ/llm-honing/issues/168)

`probe` uses the same rendered trace and also receives results for the first four available model decisions of every episode. Replay the exact request and the standard re-render from the saved public state and history under both fixed checkpoints. Preserve raw requests, replies and identities privately for scoring and audit. Compare same-checkpoint replay with logged probabilities using the #170 tolerance and tie threshold 0.01. Report identity checks and weight probes separately. The relative probability results are supplied under stable `reference` and `alternate` names, without episode labels or run paths. [Shared probe #170](https://github.com/HectorHHZ/llm-honing/issues/170)

Pin one diagnosis model ({{DIAGNOSIS_MODEL}}), CLI or API version, scored ids, prompt, schema and probe inputs before the run. Each episode and mode starts a fresh, tool-disabled session. Save the raw replies, attempts, reported token usage and timings. Record unavailable billing amounts as unknown. [Diagnoser #172](https://github.com/HectorHHZ/llm-honing/issues/172)

On verified changed-component episodes report accuracy, macro-F1, per-class scores, confusion, abstention, coverage and calibration of stated confidence. Report control false positives and costs. Include random, majority and constant baselines, and paired `probe` minus `read` differences with reset-group and fault-instance bootstrap intervals. The small number of distinct faults limits those intervals to v0 descriptive evidence.
