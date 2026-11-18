# Tank War Exp 0 results

[`CURRENT.json`](CURRENT.json) points to
[`v4-multifamily-20261117/`](v4-multifamily-20261117/) as the current
dataset and prospective diagnosis. The private combined audit contains
**1,186 paired fault games** across **1,330 fault attempt records** and retains
160 unmodified controls. Of 1,202 complete fault-loss recordings, 1,120
(93.2%) enter the paired label table; the other 82 remain in a private
unverified index with their original logs. Of the paired losses, 1,058 have a
verified same-state local layer effect and 62 remain uncertain. The prior
[`v3-sustained-complete-20261117/`](v3-sustained-complete-20261117/) index is
preserved.

The [second prospective blind test](v4-multifamily-20261117/README.md) froze
30 verified fault losses and 10 unmodified losses before model answers. Sol
high correctly classified 25/30 with controlled probes versus 14/30 from the
same replays alone: a paired gain of 36.7 percentage points, with the
pre-registered grouped 95% interval from +19.2 to +51.5. Astra medium and
two example-assisted modes were run on the same 40 games. All 320 diagnoses
were valid, but Sol's bare probe still made unsupported harness/controller
calls on 8/10 unmodified losses. The result is a scoped go for the technical
probe comparison; autonomous attribution across fault families remains
unproven. Independent human decision-relevance review is still pending.

[`v1-critical-window-20261115/`](v1-critical-window-20261115/) remains the
historical one-decision experiment. Its faults did not pass the score-drop
rule; its model answers and sample freeze have not been rewritten.

## #183 acceptance status

| Deliverable | Status | Evidence or remaining work |
| --- | --- | --- |
| Detailed protocol | Complete for the scoped v4 test | [`PROTOCOL.md`](../PROTOCOL.md), the [frozen diagnosis protocol](v4-multifamily-20261117/diagnosis-protocol.json), and private manifests record configuration, limits and scoring. |
| Reset/replay checks and real-model pilot | Complete for the sampled blind test; full-index replay partial | The 40 sampled games passed full deterministic replay; all 2,151 decision probes and 40/40 signatures passed. The 1,186-game index was not replayed end to end. |
| Fault builders, private manifests, score drops, exclusions | Partial for the full issue | The new harness and controller instances passed separate 128-seed, two-seat score screens. The private audit preserves every attempt. The harness instance participates in post-choice revalidation, and the full independent-family target is not met. |
| Versioned dataset, schema, references, reviewer records, card | Partial | The [data card](v4-multifamily-20261117/data-card.json) and private index include technical references, attempts and hashes. Independent human reviews have not been joined to the frozen case IDs. |
| No gold access or split overlap | Complete for the scoped v4 test | The 40 cases used unexposed seed blocks, and all 320 isolated CLI calls used anonymous inputs without tool calls. |
| Four modes, baselines, identity and separate weight probes | Complete for the scoped v4 test | Sol high and Astra medium ran all four modes on the same cases. The separate [weight-probe audit](v4-multifamily-20261117/weight-probe-analysis.json) and [diagnosis summary](v4-multifamily-20261117/diagnosis-summary.json) include control identity, baselines, calibration and cost. |
| Per-cause results, intervals, cost, failures, go/no-go | Complete for the scoped technical test; full issue pending human review | The [v4 report](v4-multifamily-20261117/README.md) gives per-class counts, paired intervals, control false positives, concrete errors, usage and limits. |

The full [#183 acceptance list](https://github.com/HectorHHZ/llm-honing/issues/183)
cannot yet be checked off. The main remaining work is the independent human
decision-relevance review; a broader family-transfer claim would also require
more cleanly separated fault families. No shared pipeline module changed, and
cross-game generalization is not claimed.
