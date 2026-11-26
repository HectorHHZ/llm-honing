# {{GAME_NAME}} v0 delivery receipt draft

For parent issue [#174](https://github.com/HectorHHZ/llm-honing/issues/174). Fill the publication fields from the completed release before posting (posting needs the user's approval).

| Field | Value |
| --- | --- |
| Issue | {{ISSUE}} |
| GitHub release commit | {{CODE_REVISION}} |
| Dataset | `honinglab/exp0-data`, `{{GAME_ID}}/v0/` |
| Hugging Face commit | Pending upload |
| Alternate checkpoint artifact and immutable version | {{ALTERNATE_CHECKPOINT}} |
| Fresh HF download and verification log | Pending the pinned HF revision |
| Scored set | {{SCORED_EPISODES}} native episodes in {{RESET_GROUPS}} reset groups, {{PER_CONDITION}} per intervention and {{PER_CONDITION}} no-injection controls |
| Observation | `{{OBSERVATION_CONDITION}}` |
| Diagnoser | {{DIAGNOSIS_MODEL}} via {{CLI_OR_API_VERSION}}, one fresh session per episode and mode |
| Recorded usage | {{USAGE_SUMMARY}} |
| Attribution | `read` {{READ_CORRECT}}/30 and `probe` {{PROBE_CORRECT}}/30 on verified interventions |
| Controls | `read` {{READ_CONTROL_FP}}/{{PER_CONDITION}} and `probe` {{PROBE_CONTROL_FP}}/{{PER_CONDITION}} false positives |
