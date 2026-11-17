# Tank War Exp 0 results

[`CURRENT.json`](CURRENT.json) points to
[`v3-sustained-complete-20261117/`](v3-sustained-complete-20261117/) as the
**primary dataset**. Its three sustained faults lowered matched game scores
on independent 128-seed, two-seat screens. The private versioned index
contains 640 games: 480 fault games and 160 unmodified controls. It retains
all 450 complete fault losses; 439 have a same-state local difference that
supports a machine cause label. The other eleven remain uncertain. The
private raw games and logs remain at their referenced source paths. The
earlier `v2-sustained-20261116/` index is preserved.

[`v1-critical-window-20261115/`](v1-critical-window-20261115/) remains the
historical diagnosis and go/no-go result. Its one-decision faults did not pass
the score-drop rule. Its Sol and Astra answers belong to those **40 older
games** and cannot be scored as answers to v3. Neither the v1 sample freeze
nor the original answers have been rewritten.

## #183 acceptance status

| Deliverable | Status | Evidence or remaining work |
| --- | --- | --- |
| Detailed protocol | Complete for v1; scoped v3 follow-up frozen | [`PROTOCOL.md`](../PROTOCOL.md) and the private v3 assembly and diagnosis protocols document the variants, commands, sample cap, model budget and scoring rule. |
| Reset/replay checks and real-model pilot | Complete for scoped tests | v1 reports 40/40 deterministic replays, 40/40 all-decision probe audits and the 891-decision identity check. The new v3 follow-up subset passed 40/40 full deterministic replays. The indexer's same-state check does not replay every v3 game. |
| Fault builders, private manifests, score drops, exclusions | **Partial** | v3 has one screened fault per layer with negative paired intervals, complete 160-game collections, and all ambiguous cases retained. #183's full target is at least three faults per layer. Its harness setting also participates in post-choice plan revalidation, so that boundary needs a cleaner fault for a strict harness-only claim. |
| Versioned dataset, schema, references, reviewer records, card | **Partial** | v3 has a private versioned index, technical references, checks and a [data card](v3-sustained-complete-20261117/data-card.json). The outside human review has not yet been matched to these sample IDs or archived with reviewer and disagreement records. |
| No gold access or split overlap | Complete for v1; follow-up in progress | v1's isolated diagnosis passed its input and split audit. The new 40-game follow-up excludes seeds from the earlier rapid and blind tests; its read mode ran in Bubblewrap with only anonymous inputs. Broader exposure to development material remains a limitation. |
| Four modes, baselines, identity and separate weight probes | **Partial** | Sol high ran four modes on the v1 games; Astra medium ran read/probe on those same known games. The v3 follow-up read mode is complete; its probe mode awaits an available GPU. A separate weight-probe analysis and the full set of required baselines remain incomplete. |
| Per-cause results, intervals, cost, failures, go/no-go | Complete for scoped v1; **v3 follow-up in progress** | v1's no-go result is preserved. The v3 follow-up needs probe mode before its frozen paired comparison and scoped go/no-go can be scored. |

The full [#183 acceptance list](https://github.com/HectorHHZ/llm-honing/issues/183)
cannot yet be checked off. The v3 data fixes the **score-drop** weakness in v1;
it does not retroactively complete human review or family-transfer evidence.
