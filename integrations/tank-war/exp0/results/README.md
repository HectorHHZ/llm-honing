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

The completed [v3 diagnosis](v3-sustained-complete-20261117/DIAGNOSIS.md)
classified 28/30 verified faults with controlled probes versus 16/30 from
replay alone (paired gain 40 percentage points, 95% interval 30–50). It also
made unsupported harness/controller calls on 8/10 unmodified losses, so the
result is scoped to these constructed fault families.

[`v1-critical-window-20261115/`](v1-critical-window-20261115/) remains the
historical diagnosis and go/no-go result. Its one-decision faults did not pass
the score-drop rule. Its Sol and Astra answers belong to those **40 older
games** and cannot be scored as answers to v3. Neither the v1 sample freeze
nor the original answers have been rewritten.

## #183 acceptance status

| Deliverable | Status | Evidence or remaining work |
| --- | --- | --- |
| Detailed protocol | Complete for scoped v3 follow-up | [`PROTOCOL.md`](../PROTOCOL.md) and the private v3 assembly and diagnosis protocols document the variants, commands, sample cap, model budget and scoring rule. |
| Reset/replay checks and real-model pilot | Complete for scoped tests | v1 reports 40/40 deterministic replays, 40/40 all-decision probe audits and the 891-decision identity check. The v3 follow-up passed 40/40 full deterministic replays and audited all 1850 decisions. The indexer's same-state check does not replay every v3 game. |
| Fault builders, private manifests, score drops, exclusions | **Partial** | v3 has one screened fault per layer with negative paired intervals, complete 160-game collections, and all ambiguous cases retained. #183's full target is at least three faults per layer. Its harness setting also participates in post-choice plan revalidation, so that boundary needs a cleaner fault for a strict harness-only claim. |
| Versioned dataset, schema, references, reviewer records, card | **Partial** | v3 has a private versioned index, technical references, checks and a [data card](v3-sustained-complete-20261117/data-card.json). The outside human review has not yet been matched to these sample IDs or archived with reviewer and disagreement records. |
| No gold access or split overlap | Complete for scoped v3 follow-up | The 40-game follow-up excludes seeds from the earlier rapid and blind tests. Both modes ran in Bubblewrap on anonymous inputs with no tool calls. Broader exposure to development material remains a limitation. |
| Four modes, baselines, identity and separate weight probes | **Partial** | Sol high ran four modes on the v1 games; Astra medium ran read/probe on those same known games. The v3 follow-up completed read/probe and audited all 1850 decisions, but did not repeat all extra modes or a separate weight-probe study. |
| Per-cause results, intervals, cost, failures, go/no-go | Complete for scoped v3; full issue **partial** | The [v3 diagnosis](v3-sustained-complete-20261117/DIAGNOSIS.md) reports per-cause results, paired interval, controls, usage, errors and a scoped go. One family per class and high control false positives prevent a broad go. |

The full [#183 acceptance list](https://github.com/HectorHHZ/llm-honing/issues/183)
cannot yet be checked off. The v3 data fixes the **score-drop** weakness in v1;
it does not retroactively complete human review or family-transfer evidence.
