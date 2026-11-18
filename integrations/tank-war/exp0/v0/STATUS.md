# Tank War Exp 0 v0: delivery checklist

Source of truth: [issue #183](https://github.com/HectorHHZ/llm-honing/issues/183),
current v0 acceptance text as checked on 2026-11-18. This file records progress;
it does not change the issue. Update each row only with a linked, verifiable
artifact. Work on the next row starts after the current row's independent PR
merges. Do not post to #174 until the final receipt is ready.

| Order | v0 acceptance item                                                                                             | Current evidence                                                                                                          | State                                           |
| ----- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 1     | Fixed protocol, environment, game/model/harness/controller versions, fault builders, runnable commands         | [freeze.json](freeze.json), [protocol](PROTOCOL.md), [runbook](RUNBOOK.md), game-local builders and pin checks in this PR | In review; complete only after this PR merges   |
| 2     | 40 episodes, 10/10/10/10, group counts, all attempts, schema and hashes                                        | Existing 40-game sample, 19 seed blocks, private logs; all ten controls are losses                                        | Data exist; delivery and per-file checks remain |
| 3     | Intervention and verified-component labels, separate outcome effect, provenance and exclusions                 | Private same-state audits and paired outcomes                                                                             | Package and validate per-case records           |
| 4     | Replay/timing, identity and diagnoser gold-isolation checks                                                    | 40/40 deterministic replay; 2,151 decisions probed at tolerance 0.01; 320 isolated calls                                  | Publish check records with the dataset          |
| 5     | Upload reproducible `tank-war/v0/` release to `honinglab/exp0-data`, pin HF and GitHub revisions               | No HF revision is recorded                                                                                                | Not started                                     |
| 6     | Fresh-environment download and offline score reproduction without an API key                                   | Requires item 5                                                                                                           | Not started                                     |
| 7     | Real API read/probe run and published raw replies, probes, failures, metrics, baselines, costs and limitations | Prior 40-game ChatGPT-login CLI run and aggregate report; not an API-key run record                                       | Partial                                         |
| 8     | Per-game v0 receipt and result in #174                                                                         | No receipt posted                                                                                                         | Not started                                     |

The existing sample and diagnostic results are in draft
[PR #187](https://github.com/HectorHHZ/llm-honing/pull/187), with private raw
evidence on ds-serv2. That PR's old acceptance table predates the issue's v0
scope correction. Do not treat its draft status or aggregate report as the HF
release required by items 5 and 6. The 40 cases reuse previously inspected
episodes, so v0 is a development/reproducibility result, not an independent
confirmatory test.

The later v1 study has larger family, score-drop, four-mode, two-model and
human-baseline targets. Those are not v0 gates. No human review of a decisive
failure window is required for v0. The harness fault used in the existing
40-game sample affects both observation and post-choice plan revalidation;
report that boundary explicitly in the dataset card and result.
