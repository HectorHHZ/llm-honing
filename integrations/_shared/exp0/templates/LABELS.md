# {{GAME_NAME}} v0 label guide

`evaluator/episodes.json` is the scored episode index. Each row names the actual intervention in `injection_layer`, the fault family and instance, the changed artifact, and the evidence that supports the verified component. The index is covered by `schemas/episode-index.schema.json` and stays in the evaluator directory; it and the run paths never enter model inputs.

| Component | Evidence |
| --- | --- |
| Harness | At the evidence row, compare the actual model request with the request the clean harness renders from the same saved public state (`clean_request`). The intervention {{HARNESS_FAULT_SUMMARY}}. |
| Weights | {{WEIGHTS_EVIDENCE_RULE}} The two probability vectors cover the same offered options and come from different verified checkpoint SHA-256s. |
| Controller | The model selected {{CONTROLLER_SELECTED}} and the controller executed {{CONTROLLER_EXECUTED}}. The recorded windows and the following native events are the game's acknowledgement. |
| No injection | The reference checkpoint and the standard request and controller ran without the three interventions. No component gold label is assigned to these controls. |

`annotation_provenance` identifies the automated verifier (`audit_set.py` with the contract evidence rules). {{REVISIONS_AND_DISAGREEMENTS}} `reference_actions.status` is `unavailable`. The native outcome and the paired clean outcome are independent evidence fields; `outcome_effect` is `not_established` where a paired clean outcome exists. The experiment verified per-decision effects; paired release-level score drop was untested.

`evaluator/attempts.jsonl` records the complete formal attempt set. {{PILOTS_OUTSIDE_SCORED_SET}} The formal collection keeps all clean and changed attempts with the same start in one reset group and one split. `split-manifest.json` records exact memberships{{HELDOUT_NOTE}}.
