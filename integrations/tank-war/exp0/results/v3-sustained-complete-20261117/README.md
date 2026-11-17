# #183 complete sustained-fault collection

This version completes the retained `point_blank_pursuit-v1` harness collection
and keeps the 160-game weights and controller collections from v2. All three
variants previously passed separate fixed 128-seed, two-seat paired screens.
The same game build, checkpoint and fault settings were used for the later
`final_test` games. The private store is
`/data/xinkaiz/system0/exp0/tank-war/v3-sustained-complete-20261117/` on
ds-serv2. It indexes the original game files and preserves the earlier v2
index and the interrupted harness attempt.

The 53-game completion used only GPU 6, from 2026-11-17 19:26:23 to 19:55:14
UTC, or about **0.48 GPU-hours of service capacity**; actual GPU utilization
was not metered. The service exited after the collector reported 160/160 games
successful and zero failures. A preceding startup check failed before any new
game because it compared the GPU ID string with an integer; that service was
also stopped. Its first startup log was overwritten by the corrected launch,
and the reconstructed attempt record states that limit explicitly.

| Layer | Screen score difference, fault − clean | Collected games | Fault losses | Losses with same-state layer evidence |
| --- | ---: | ---: | ---: | ---: |
| Harness, point-blank pursuit | −0.1504 | 160 | 145 | 134 |
| Weights, real reverse-strength-2.5 checkpoint | −0.2910 | 160 | 155 | 155 |
| Controller, swap selected turn | −0.2910 | 160 | 150 | 150 |

All **450/450 complete fault losses** are retained. Eleven harness losses lack
an isolated same-state local difference and remain uncertain. All 160 clean
controls are retained, including 126 losses. The dataset contains **640 games**
and **439 machine-verified fault-loss labels**. A local difference supports a
component label for this constructed game; it does not prove that the change
caused the final loss.

The independent 80-seed, two-seat `final_test` recount shows lower fault scores
for harness (−0.1156; paired 95% interval −0.1906 to −0.0406), weights
(−0.1813; −0.2438 to −0.1250), and controller (−0.1688; −0.2344 to
−0.1063). This is a descriptive check after collection; the original separate
screens remain the retention gate. See [summary.json](summary.json),
[independent-recheck.json](independent-recheck.json), and
[data-card.json](data-card.json).

The previously frozen 40-game diagnosis subset came from the first 107
available harness games and the complete weights, controller, and clean
collections. Its source hashes are unchanged in v3. It excludes seeds used in
the earlier rapid and blind tests and passed 40/40 full deterministic replays.
The follow-up uses one complete-game harness/weights/controller label per
game, not a decision-window output. Diagnosis results will be recorded
separately from this dataset index.

## Limits

- One sustained fault family per layer is available, so this does not test
  transfer to unseen fault families.
- The harness strategy setting also participates in plan revalidation after
  model choice. Its functional boundary is mixed, even though the 134 labeled
  harness losses have direct same-state model-choice evidence.
- The outside human reviews have not been joined to these sample IDs or
  archived as independent reviewer records.
- Technical component labels are separate from claims that an intervention
  caused a particular loss.

The exact collection and assembly commands are retained in the private store.
The code changes in this follow-up are confined to Tank War scripts; no shared
pipeline module was changed. Cross-game generalization is not claimed.
