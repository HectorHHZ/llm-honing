# Hollow Knight (ForestDango) v0 delivery receipt draft

For parent issue [#174](https://github.com/HectorHHZ/llm-honing/issues/174). Fill the publication fields from the completed release before posting (posting needs the user's approval).

| Field | Value |
| --- | --- |
| Issue | #185 (Exp 0 v0, Hollow Knight) |
| GitHub release commit | {{CODE_REVISION}} |
| Dataset | `honinglab/exp0-data`, `hollow-knight-forestdango/v0/` |
| Hugging Face commit | Pending upload |
| Checkpoints | clean `convaiinnovations/laya-typed-decisions@1a793eb5` (weights `4fa56de7…`); alternate `jayzou3773/laya-ra2-v33@51d2f5a0` (weights `2ed46ac9…`), referenced, not re-uploaded (`checkpoint-references.json`) |
| Fresh HF download and verification log | Pending the pinned HF revision; the local fresh-venv check is `<release root>-verify-receipt.json` |
| Scored set | {{SCORED_EPISODES}} native episodes in {{RESET_GROUPS}} reset groups (ten starts on one route in King's Pass), {{PER_CONDITION}} per intervention and {{PER_CONDITION}} no-injection controls |
| Observation | `hk-forestdango-visible-window-v12-standing-v1`: the frozen `v12_standing` harness's English text of the visible screen, two questions per decision, no pixels |
| Diagnoser | Codex `gpt-6.2-sol` via {{CLI_VERSION}}, one fresh session per episode and mode |
| Recorded usage | {{USAGE_SUMMARY}} |
| Attribution | {{ATTRIBUTION_SUMMARY}} |
| Controls | {{CONTROLS_SUMMARY}} |
| Pre-registered probe variant | {{VARIANT_SUMMARY}} |
