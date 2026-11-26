# {{GAME_NAME}} Exp 0 v0 results

{{DIAGNOSIS_MODEL}} evaluated the same {{SCORED_EPISODES}} native episodes in both modes through {{CLI_OR_API_VERSION}}. Each diagnosis used a fresh session. {{REVISION_AND_COST_AVAILABILITY}} {{ATTEMPT_SUMMARY}} [Protocol](PROTOCOL.md) · [Dataset card](README.md)

| Metric | Read | Probe |
| --- | ---: | ---: |
| Correct on verified interventions | {{READ_CORRECT}}/30 | {{PROBE_CORRECT}}/30 |
| Macro-F1 on verified interventions | {{READ_MACRO_F1}} | {{PROBE_MACRO_F1}} |
| Abstentions on verified interventions | {{READ_ABSTAIN}}/30 | {{PROBE_ABSTAIN}}/30 |
| False positives on no-injection controls | {{READ_CONTROL_FP}}/{{PER_CONDITION}} | {{PROBE_CONTROL_FP}}/{{PER_CONDITION}} |
| Reported input tokens | {{READ_INPUT_TOKENS}} | {{PROBE_INPUT_TOKENS}} |
| Reported output tokens | {{READ_OUTPUT_TOKENS}} | {{PROBE_OUTPUT_TOKENS}} |

{{PER_CLASS_NARRATIVE}}

The paired accuracy difference on the 30 verified cases was {{PAIRED_ESTIMATE}}. The descriptive 95% bootstrap interval over reset groups spans {{GROUP_INTERVAL}}, and over the three fault instances {{INSTANCE_INTERVAL}}. There is one retained fault instance per component, so the results have limited support for generalization to other fault instances or games. The recorded effects are at the decision level; release-level effects on whole-episode scores remain untested. Baselines: uniform {{UNIFORM_ACC}}, majority {{MAJORITY_ACC}}.

`evaluation/scores.json` holds the confusion matrices, per-class results, baselines, calibration and paired uncertainty; `evaluation/weight-probes.json` reports the alternate-checkpoint probability changes separately from diagnosis performance. `tools/verify_release.py` reproduces the scores from the saved replies without an API key.
