# {{GAME_NAME}} Exp 0 v0 dataset card

This local dataset supports the paired `read` and `probe` experiment of {{ISSUE}} under #174. It uses {{GAME_BUILD_DESCRIPTION}} and the `{{OBSERVATION_CONDITION}}` observation condition ({{OBSERVATION_SUMMARY}}). One episode is {{EPISODE_DEFINITION}}. The {{SCORED_EPISODES}} scored episodes come from {{RESET_GROUPS}} reset/scenario groups; a group's clean and changed attempts share one split and are one statistical unit.

| Condition | Complete | Win | Loss | Timeout | Other terminal |
| --- | ---: | ---: | ---: | ---: | ---: |
| No injection | {{PER_CONDITION}} | {{CONTROL_WIN}} | {{CONTROL_LOSS}} | {{CONTROL_TIMEOUT}} | {{CONTROL_OTHER}} |
| Harness | {{PER_CONDITION}} | {{HARNESS_WIN}} | {{HARNESS_LOSS}} | {{HARNESS_TIMEOUT}} | {{HARNESS_OTHER}} |
| Weights | {{PER_CONDITION}} | {{WEIGHTS_WIN}} | {{WEIGHTS_LOSS}} | {{WEIGHTS_TIMEOUT}} | {{WEIGHTS_OTHER}} |
| Controller | {{PER_CONDITION}} | {{CONTROLLER_WIN}} | {{CONTROLLER_LOSS}} | {{CONTROLLER_TIMEOUT}} | {{CONTROLLER_OTHER}} |

Outcomes come from native game events: win = {{WIN_RULE}}; loss = {{LOSS_RULE}}; timeout = {{CAP_RULE}}; other terminal = {{OTHER_RULE}}. {{ATTEMPTS}} formal attempts were made; {{EXCLUDED}} were excluded (reasons in `evaluator/attempts.jsonl`). The scored episodes contain {{MODEL_DECISIONS}} model decisions ({{MODEL_QUESTIONS}} question records), {{NATIVE_WINDOWS}} native windows and {{MODEL_INPUT_TOKENS}} model-server input tokens; their `decisions.jsonl` files total {{DECISION_BYTES}} bytes. Decisions, windows and questions are counted separately from episodes; adjacent windows are not independent episodes.

Labels record the component changed in the actual episode. Harness evidence is the logged request next to the clean render of the same saved state ({{HARNESS_FAULT_SUMMARY}}). Weights evidence is {{WEIGHTS_EVIDENCE_SUMMARY}}; across the {{PER_CONDITION}} weights cases the maximum option probability difference ranges from {{WEIGHTS_DP_MIN}} to {{WEIGHTS_DP_MAX}}, and the most probable option changed in {{WEIGHTS_CHOICE_CHANGED}} of them. Controller evidence is the selected option next to the executed option and native windows ({{CONTROLLER_FAULT_SUMMARY}}). Reference-action annotations are unavailable. Whole-episode effects are recorded separately (`outcome_effect`); release-level score drop is untested.

The versioned `honing-exp0-episode-v0` records (contract {{CONTRACT_VERSION}}, schemas in `schemas/`) live in `evaluator/`: `evaluator/episodes.json` maps the neutral ids to attempts and gold labels, `split-manifest.json` records group membership, `evaluator/attempts.jsonl` records every attempt with its status. `sha256.json` covers the file set. Use the [label guide](LABELS.md) for the evidence and the [runbook](RUNBOOK.md) to reproduce and verify. The dataset is for development and reproducibility, with one verified intervention instance per component and correlated episodes within each reset group. {{KNOWN_LIMITS}}
