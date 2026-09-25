# Macro context: paired follow-up

Keep the typed checkpoint, three seed/seat pairs, candidates, option labels,
3-tick timing and inference settings from the preceding experiment. Change
only the context facts and advice in `intent-policy.mjs`. The three games
were launched together, with no tuning or extra full matches during the batch.

The preceding red games ended with zero credits and 25 tanks still queued at
44% progress. Harvesters had repeatedly received Move/ForceMove commands after
starting with a Gather order. Gather candidates were present but not selected.

The new context adds net credit change (including spending/refunds), current
miner tasks, stalled production progress and recent observed army counts. It
recommends preserving Gather/Dock orders, restoring income before adding more
unfunded orders, and replenishing an observed army decline.

| Pair | Income before / after | Combat vehicles before / after | Miner ForceMove before / after |
| --- | ---: | ---: | ---: |
| seed 1 red | 0 / 6,950 | 5 / 0 | 3,273 / 556 |
| seed 1 blue | 0 / 23,800 | 0 / 0 | 4,662 / 0 |
| seed 2 red | 0 / 6,950 | 5 / 0 | 2,423 / 556 |

All three reached 18,000 ticks without a winner. Income improved but military
production regressed: no new combat units were produced. The final normal
observations contained five refineries for each red game and eight for blue;
none had a barracks or war factory. Red repeatedly docked miners, while blue
left its automatic mining orders running and finished with 15,449 credits.
This does not validate sustained reinforcement after combat losses, because
the new games never produced the army needed to test that behavior.

Each game passed the context/action audit and independent replay hash check.
Source fingerprints and settings were compared with the paired prior games;
only the context implementation differed. Real-engine-state fixtures also
found the candidate keys, labels, hierarchy, availability and exclusions
unchanged. The normal-view input excludes opponent information and observer
income statistics. The latter are used only for offline evaluation here.

The worker diagnostic is a current-task heuristic: Move with a queued Dock
can still be classified as diverted. Current orders do not prove actual mining
income. This limitation was kept fixed across the batch. The context change
bundles several facts and instructions, so these games do not identify which
sentence caused each behavior, and three development pairs cannot estimate a
reliable win rate.

The next hypothesis to test is a phase-specific objective: when an economy is
working and credits are available, prioritize the missing barracks/war factory
and an affordable combat queue over further refinery expansion. That change
has not been implemented or tested in this batch.

See [macro-context-comparison.json](macro-context-comparison.json) for exact
results and source hashes. Full games used the frozen experimental files; the
portable example separately passed engine fixtures and context diagnostics.
