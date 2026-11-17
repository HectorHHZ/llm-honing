# Tank War Exp 0 results

[`CURRENT.json`](CURRENT.json) points to
[`v2-sustained-20261116/`](v2-sustained-20261116/) as the **primary dataset
for the next attribution test**. Its three sustained faults lowered matched
game scores on independent 128-seed, two-seat screens. The private versioned
index contains 587 games: 427 fault games and 160 unmodified controls. It
retains all 403 complete fault losses; 394 have a same-state local difference
that supports a machine cause label. The other nine remain uncertain. The
private raw games and logs remain at their referenced source paths.

[`v1-critical-window-20261115/`](v1-critical-window-20261115/) remains the
historical diagnosis and go/no-go result. Its one-decision faults did not pass
the score-drop rule. Its Sol and Astra answers belong to those **40 older
games** and cannot be scored as answers to v2. Neither the v1 sample freeze
nor the original answers have been rewritten.

## #183 acceptance status

| Deliverable | Status | Evidence or remaining work |
| --- | --- | --- |
| Detailed protocol | Complete for v1; **partial for v2** | [`PROTOCOL.md`](../PROTOCOL.md) and the private v2 assembly protocol document the actual variants and commands. No v2 diagnosis split, sample cap, model budget or scoring freeze exists yet. |
| Reset/replay checks and real-model pilot | Complete for the scoped v1 test | v1 reports 40/40 deterministic replays, 40/40 all-decision probe audits and the 891-decision identity check. The v2 assembler checks raw-state hashes at the first local difference; it has not replay-audited every v2 game. |
| Fault builders, private manifests, score drops, exclusions | **Partial** | v2 has one screened fault per layer with negative paired intervals and retains ambiguous cases. #183's full target is at least three faults per layer. Its harness setting also participates in post-choice plan revalidation, so that boundary needs a cleaner fault for a strict harness-only claim. |
| Versioned dataset, schema, references, reviewer records, card | **Partial** | v2 has a private versioned index, technical references, checks and a [data card](v2-sustained-20261116/data-card.json). The newly mentioned outside human review has not yet been matched to these sample IDs or archived with reviewer and disagreement records. |
| No gold access or split overlap | Complete for v1; unverified for v2 | v1's isolated diagnosis passed its input and split audit. v2 has no diagnosis freeze yet, and earlier development exposure of its collected games has not been fully audited. |
| Four modes, baselines, identity and separate weight probes | Complete only in part for v1; **not run on v2** | Sol high ran four modes on the v1 games; Astra medium ran read/probe on those same known games. A separate weight-probe analysis and all required baselines are not complete for v2. |
| Per-cause results, intervals, cost, failures, go/no-go | Complete for scoped v1; **not run on v2** | v1's no-go result is preserved. The stronger v2 faults have not yet been given to System 0, so there is no v2 attribution result or go/no-go report. |

The full [#183 acceptance list](https://github.com/HectorHHZ/llm-honing/issues/183)
cannot yet be checked off. The v2 data fixes the **score-drop** weakness in v1;
it does not retroactively complete human review, blinding or diagnosis on v2.
