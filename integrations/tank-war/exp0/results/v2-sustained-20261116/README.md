# #183 sustained-fault replacement dataset

The one-decision variants in `v1-critical-window-20261115` changed local
actions but did not lower whole-game scores. This version uses three previously
screened **sustained** faults instead. Their complete 128-seed, two-seat screens
passed the fixed rule: mean fault-minus-clean score at most -0.05 and a paired
95% interval entirely below zero. The screen results were written before the
first independent final-test game for each variant. The game source, checkpoint,
harness, controller, and decision cadence in each screen match its later
collection. This is a retrospective assembly of retained data, not a new
preregistered screen or a new blind diagnosis run.

| Layer and sustained fault | Screen mean fault - clean | Paired 95% interval | Collected fault games | Fault losses | Losses with same-state local evidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Harness: point-blank pursuit strategy | -0.1504 | -0.2207 to -0.0801 | 107 | 98 | 89 |
| Weights: real reverse-strength-2.5 checkpoint | -0.2910 | -0.3496 to -0.2324 | 160 | 155 | 155 |
| Controller: swap selected turn | -0.2910 | -0.3516 to -0.2305 | 160 | 150 | 150 |

The separately collected `final_test` seed groups also show lower fault
scores: harness -0.0991 on 53 complete two-seat groups (95% interval
-0.1745 to -0.0283), weights -0.1813 on 80 groups (-0.2438 to -0.1250),
and controller -0.1688 on 80 groups (-0.2344 to -0.1063). This is a
**post-collection descriptive check**, not the original retention gate.
The one incomplete harness seed group is excluded from that paired statistic
but its completed game remains in the dataset. Exact values and resampling
rules are in [independent-recheck.json](independent-recheck.json).

All **403/403 complete fault losses** are retained, including nine harness
losses without an isolated model-choice difference. The private labels give a
cause only to the other **394** losses. They also retain all 24 fault wins and
160 clean controls (126 losses). The dataset has **587 games** in total. For
each fault game, the technical reference is the first decision at which the
clean and fault runs share the same raw-state hash but differ locally in the
declared layer. A clean counterpart winning is recorded separately; it is not
required for the episode-level cause label. The nine unverified harness losses
remain unlabeled, not forced into a cause class.

The private versioned store is
`/data/xinkaiz/system0/exp0/tank-war/v2-sustained-20261116/` on ds-serv2.
It contains the frozen assembly protocol, per-game private labels, control
index, aggregate summary, and data card. Raw game states, requests, replies,
events, and earlier failed attempts remain in the referenced `v0-pilot/`
directories; this version indexes them rather than copying them. The
repository includes only the [aggregate summary](summary.json) and
[data card](data-card.json), not the private answer key.

## Limits

- The harness strategy changes the model's plans and rendered input, but the
  same setting is also used during plan revalidation after model choice. Its
  functional boundary is therefore mixed. The 89 harness labels have direct
  same-state **model-choice** evidence, but this candidate alone cannot prove
  that every later effect of the setting belongs exclusively to the harness.
- There is only one fault family per layer in this assembled version. The 107
  harness games were all completed games available from a collection originally
  planned for 160; the remaining 53 were never completed. No new games were
  started while ds-serv2 reported `nvidia-smi` initialization failure and
  `torch.cuda.is_available() == False` on 2026-11-16.
- These seed groups do not overlap the later 40-game rapid blind set, but
  earlier development work may have exposed some of these games. This dataset
  is **not certified as a fresh blind test**. A local effect also does not
  establish that the fault caused the final loss. No human review or new
  System 0 call was performed for this assembly.

The assembler is
[`assemble-sustained-fault-dataset.py`](../../../scripts/assemble-sustained-fault-dataset.py).
The actual server command was:

```sh
python3 /data/xinkaiz/system0/exp0/tank-war/v2-sustained-assembler-20261116.py \
  --protocol /data/xinkaiz/system0/exp0/tank-war/v2-sustained-protocol-20261116.json \
  --out /data/xinkaiz/system0/exp0/tank-war/v2-sustained-20261116
```

Five focused label-boundary tests passed locally. The assembler changes only
the Tank War adapter; no shared pipeline module changed. Cross-game
generalization was not tested or claimed for this game-specific assembly.
