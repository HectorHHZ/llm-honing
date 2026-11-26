# VVVVVV Exp 0 v0 to-do and limitations

Deliver 40 complete, auditable episodes, then compare `read` and `probe` with one real diagnosis model on the same episodes. Each episode gets one component answer (`harness`, `weights`, `controller` or an abstention); whole-episode outcomes and their causal evidence are recorded separately. v0 is for development and reproducibility; a probe without gain is a valid result. [#185](https://github.com/HectorHHZ/llm-honing/issues/185) · [Exp 0 #174](https://github.com/HectorHHZ/llm-honing/issues/174)

## Status

- [x] Pin the game build, harness, controller, observation and both checkpoints; freeze the collection plan before the first release attempt (`collection.json`, SHA-256 `{{PLAN_SHA}}`; pins in `collector-source/vvvvvv-pins/pins.json` and every manifest).
- [x] Define the episode: native Space Station 1 time-trial start, win = past "Conundrum", deaths respawn, timeout at {{DECISION_CAP}} decisions; seeded sampling per reset group; selection and reserve rule of Griddly v0.
- [x] Pilot the persistent faults on one pilot group and check each from decision 1 (development data, not in the release): {{PILOT_EVIDENCE}}
- [x] Collect the 40 episodes 10/10/10/10 with every attempt kept: {{COLLECTION_STATUS}}
- [x] Validate every attempt; re-render every logged request byte for byte; replay every attempt natively; run the integration's own receiver on every executed reply: {{CHECKS_SUMMARY}}
- [x] Audit, evidence, selection and neutral ids (`audit_set.py`); label-free `read` inputs with the isolation check; probe captures under both checkpoints and the #170 identity check ({{IDENTITY_SHORT}}).
- [{{DIAGNOSIS_BOX}}] Run Codex `gpt-6.2-sol` in `read` and `probe` once each on the 40 episodes (80 calls), then the pre-registered probe variant (40 calls); save replies, attempts, usage and timing; score offline. {{DIAGNOSIS_STATUS}}
- [{{DIAGNOSIS_BOX}}] Bundle the release with the real code revision, verify it in a fresh venv without packages, and pass `upload_release.py check` (`release_v0.py resume`, see the runbook).
- [ ] Upload to `honinglab/exp0-data` under `vvvvvv/v0/`, download the pinned revision in a fresh environment and verify it, and post the receipt to #174 (the release owner, after review).

## Limitations

- **One layout.** The segment has one native start and two rooms; the {{RESET_GROUPS}} reset groups are policy seeds, not layouts. Episodes are correlated within and across groups (every episode's first request is the same, so the weights evidence of all weights episodes is the same request); intervals over groups describe this sample only.
- **One fault instance per component**, each persistent for the whole episode. The fault-instance bootstrap therefore has three units.
- **Harness fault choice.** The brief's default (remove the `flip` option) is invalid for the integration's receiver (exact option set), so the #185 terrain omission was used. Laya's probabilities move little under it (about 0.01 at the first decision of the pilot), so harness episodes follow their clean partners' selections for long stretches while every request differs: {{FOLLOW_SUMMARY}}
- **Weights fault.** The alternate checkpoint selected `flip` in {{WEIGHTS_FLIP_SHARE}} of its model decisions, so weights episodes differ visibly in play as well as in probabilities.
- **Controls.** No-injection outcomes come from the seeded draws of the clean policy; a control is not a ground-truth label for chance.
- **Outcome effects.** `outcome_effect` is `not_established`; release-level score drop is untested; reference actions are unavailable.
- **Diagnosis model.** Codex CLI on a subscription: no immutable model revision and no per-call dollar cost; online replies can vary between runs.
- **Code after the collection.** After the 40 attempts were collected, the collector, adapter and helper code were changed only to resolve local locations at run time (`runtime.py`: `$VVVVVV_ASSETS`, `$VVVVVV_SDL_LIB`, `$FFMPEG`, `$EXP0_CODEX_BIN`) instead of absolute local paths, so that no published file names a local path or account; the collected bytes are unchanged, the manifests' code fingerprints refer to the collection-time files, and `pins.json` keeps the collection-time collector fingerprint. The runtime is tested on this Linux server only. A bundle built before the code is committed carries the placeholder revision `0`×40 and must be rebuilt with the real commit.
- **Not done here:** no upload, no training, no human review of decisive windows, no blind test set. {{NOT_DONE_EXTRA}}
