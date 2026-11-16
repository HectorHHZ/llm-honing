# Tank War Exp 0: scoped attribution result

Related: [#183](https://github.com/HectorHHZ/llm-honing/issues/183), [#174](https://github.com/HectorHHZ/llm-honing/issues/174), and the [registered protocol](../../PROTOCOL.md).

## Decision

**No-go for reliable per-game harness/weights/controller attribution in this scoped test.** After the blind run, the user clarified that System 0 should give one component attribution per complete game, without identifying a failure-causing decision. Forty distinct-seed losses were frozen before `gpt-6-sol` produced an answer: ten each with a verified local harness, weights, or controller effect, and ten unmodified controls. The same cases received one `high` effort read diagnosis and one `high` effort diagnosis with controlled checks at every decision. The original prompt also asked for a window; the per-game result below is an exploratory rescore of its sealed cause answers, not a fresh episode-only blind run.

| Per-game class on 30 fault cases | Read | Probe |
| --- | ---: | ---: |
| Harness | 3/10 | 4/10 |
| Weights | 5/10 | 5/10 |
| Controller | 0/10 | 1/10 |
| **All three** | **8/30** | **10/30** |

The paired gain in correct classes is 2/30, or 6.7 percentage points. The class-stratified paired bootstrap 95% interval is **-16.7 to +30.0 points**, which includes zero. Each mode made one unsupported harness/controller call among ten unmodified controls. Probe accuracy was 10/30, equal to the balanced-set majority-class baseline. The result does not establish reliable three-class attribution or an improvement from probing. [`EPISODE_ATTRIBUTION.md`](EPISODE_ATTRIBUTION.md) gives the new scoring unit and technical label audit; [`episode-attribution-data-card.json`](episode-attribution-data-card.json) gives the machine-readable provenance and limitations.

All 30 fault cases have a single-layer local decision or execution difference. Only six had a better whole-game outcome in the paired clean run. The technical reference index establishes the per-game label but is not scored as a System 0 output or treated as proof that the intervention caused the loss. The original window and joint metrics remain in the frozen historical `SUMMARY.json` for audit.

## Dataset and files

The private artifact store is `/data/xinkaiz/system0/exp0/tank-war/v1-critical-window-20261115/` on ds-serv2. `inventory-final-v3/` indexes **1,232 of 1,232** unique complete losses; all indexed traces have aligned decision logs and valid recorded state hashes. The 40 blind cases also passed full deterministic replay and all-decision probe audits. Archived games remain under the sibling `v0-pilot/` directory and are referenced by the inventory; this version is not a standalone copy of every raw game. `inventory-final-v2/` is retained but marked invalid because a stale inventory script counted derived replays as original games.

`SUMMARY.json` exposes aggregate counts, paired intervals, token usage, an API-equivalent estimate at the frozen 2026-11-13 price reference, and SHA-256 hashes for the private artifacts. `protocol.json`, `collection-cutoff.json`, `prompt.txt`, and `schema.json` are exact frozen copies. Per-case gold labels, fault manifests, raw states, requests, replies, probes, and diagnosis answers stay in the private dataset root. Earlier human review records remain in its sibling `v0-pilot/` archive. Their hashes and directory locations let an authorized reviewer check the underlying evidence without publishing the answer key in this PR.

[`blind-split.json`](blind-split.json) lists exactly which 40 anonymous games entered the final test and hashes their inputs. [`RUNBOOK.md`](RUNBOOK.md) records the three-hour GPU-time ceiling, the actual diagnosis usage, and the commands used for the final analysis. The GPU-hour ceiling is derived from the pre-run three-hour and three-GPU limits; actual GPU utilization was not measured. No token or dollar cap was frozen before diagnosis.

[`REVIEW_STATUS.md`](REVIEW_STATUS.md) records the later planner-reference pass over all 1,811 recorded test decisions and the separate 30-game blind human review page. That review is no longer requested for the amended three-class task; its partial progress is preserved. The original two-reviewer criterion remains unmet.

The scorer committed with this PR was formatted for repository checks after the blind run. The original scorer remains in the private store; a separate `score-format-audit-v1/` replay of the same frozen answers produced a score file **byte-identical** to the original. The original sample freeze and score were not edited.

Final technically verified fault-loss counts are **20 harness, 20 weights, and 24 controller**. These are episodes, not independent fault families. All complete losses were retained, including untriggered or unchanged cases, but those cases were not assigned a verified injected-cause label.

## #183 deliverables and acceptance audit

| Required deliverable                                                         | Status for this PR  | Evidence or limit                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Detailed pinned protocol, splits, budgets, scoring, commands                 | Met as documentation | `PROTOCOL.md`, `blind-split.json`, and `RUNBOOK.md` now give the game/model, fault catalog, tested IDs and split, sample count, GPU-hour ceiling, statistical rules, and actual commands. The private sample/prompt/scoring freeze predates answers. The numeric GPU-hour conversion was written after the run and actual GPU occupancy was not measured; the rapid test's deviations from the original execution plan are reported separately. |
| Reset/replay checks and real-model pilot                                     | Met for scoped test | 40/40 blind full replays, 40/40 all-decision probe audits, and prior 891-decision identity gate at tolerance 0.01.                                                                                                                                                                                  |
| Fault builders, private manifests, validated score drops, exclusions         | Partial             | Game-local builders and all attempt logs are retained. Earlier 128-seed paired screens showed degradation for three _different_ fault variants; the one-turn blind variants were validated for local effect, not separately for a release-level score drop.                                         |
| Versioned dataset, schema, reference provenance, reviewer records, data card | Partial             | Versioned private store, frozen schema, technical-reference provenance, inventory, data card, and audited per-game labels exist. The later technical audit does not replace the two independent human reviewers requested in the original issue. |
| No gold access or split overlap during diagnosis                             | Met for scoped test | Blind inputs excluded seed, source path, fault labels, and gold manifests; the isolated Codex process could read only anonymous inputs and its output directory. The 40 cases used distinct seeds.                                                                                                  |
| Four modes, baselines, identity checks, separate weight probes               | Partial             | Only read and probe with Sol high were authorized. No Astra, few-shot, second API model, or separate weight-probe arm. Majority baseline and identity checks are reported.                                                                                                                          |
| Per-cause results, paired intervals, cost, failure cases, go/no-go           | Met for scoped test | The amended per-game analysis is in `EPISODE_ATTRIBUTION.md` and `episode-attribution-summary.json`; historical window metrics remain in `SUMMARY.json`. Private per-case records contain examples and raw answers. The conclusion is no-go. |

The original full benchmark in #183 remains **unmet**. This draft reports the user-amended per-game interpretation of the rapid experiment and its negative result; it does not claim completion of the full acceptance list or start method selection under #180.

## Verification and cost

The experiment used Codex CLI 0.158.0 with `gpt-6-sol`, `high` effort, and Bubblewrap isolation. Both 40-case modes finished without invalid calls or tool use. Read consumed 2,196,037 input and 47,852 output tokens; probe consumed 2,374,573 input and 50,228 output tokens. At the frozen standard-tier, uncached text-token reference, the API equivalents are **$4.87** and **$5.25**, respectively. These are estimates, not the actual ChatGPT-login bill. Internal model revision and temperature could not be pinned.

Tank War tests passed 163/163 and format checks passed. Tests over a structurally different synthetic environment and dependency boundaries passed 57/57. The new intervention and reporting code is in the Tank War adapter; no shared orchestration module was changed. The full CI suite and maintainer review remain PR gates.
