# Tank War Exp 0 v0: delivery checklist

Source of truth: [issue #183](https://github.com/HectorHHZ/llm-honing/issues/183),
current v0 acceptance text as checked on 2026-11-20. This file records progress;
it does not change the issue. Update each row only with a linked, verifiable
artifact. Item 1's PR must merge before item 2 starts, as agreed for this
sequence. Items 2–8 are primarily dataset, validation, release, and reporting
work; they do not each require a PR. Open another PR only if code or repository
documentation must change. Do not post to #174 until the final receipt is ready.

| Order | v0 acceptance item                                                                          | Current evidence                                                                                                                                                        | State                                               |
| ----- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| 1     | Fixed protocol, environment, versions, builders, commands                                   | [freeze.json](freeze.json), [protocol](PROTOCOL.md), [runbook](RUNBOOK.md); [PR #186](https://github.com/HectorHHZ/llm-honing/pull/186) merged 2026-11-20               | Complete                                            |
| 2     | 40 episodes, 10/10/10/10, groups, attempts, schema, hashes                                  | [Dataset card](DATASET.md); private 44 MB stage has 40 original games, 30 clean pairs, 19 seed groups, 1,330 attempts, and 587 hash-checked files                       | Private stage validated; HF delivery remains item 5 |
| 3     | Intervention and verified-component labels, separate outcome effect, provenance, exclusions | Per-case `labels.jsonl` and `index.jsonl`; 30 saved-state effects, five differing paired outcomes, 144 excluded attempts; ten checkpoint strength names resolved by SHA | Private stage validated; disclose name mismatch     |
| 4     | Replay, identity, and diagnoser label-isolation checks                                      | The offline verifier checks archived success flags and binds packaged probes to the probe-mode receipts among 320 calls; 40/40 replay, 2,151/2,151 probes, 891/891 identity calls | Checks passed; records in private stage             |
| 5     | Upload `tank-war/v0/` to `honinglab/exp0-data`; pin HF and GitHub revisions                 | No HF revision recorded                                                                                                                                                 | Not started                                         |
| 6     | Fresh download and offline score reproduction without API key                               | Requires item 5 and saved real API answers                                                                                                                              | Not started                                         |
| 7     | Real API read/probe and published replies, metrics, baselines, costs, limits                | Existing login-based CLI run is historical evidence, not an API-key run                                                                                                 | Partial                                             |
| 8     | Per-game v0 receipt and result in #174                                                      | No receipt posted                                                                                                                                                       | Not started                                         |

The existing sample and diagnostic results are in draft
[PR #187](https://github.com/HectorHHZ/llm-honing/pull/187), with private raw
evidence on ds-serv2. Do not treat its draft status or aggregate report as the HF
release required by items 5 and 6. The 40 cases reuse previously inspected
episodes, so v0 is a development/reproducibility result, not an independent
confirmatory test.

The later v1 study has larger family, score-drop, four-mode, two-model and
human-baseline targets. Those are not v0 gates. No human review of a decisive
failure window is required for v0. The harness fault used in the existing
40-game sample affects both observation and post-choice plan revalidation;
report that boundary explicitly in the dataset card and result.
