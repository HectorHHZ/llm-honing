# Final-variant paired score screen

This screen evaluates the **same one-decision fault variants** used in the 40-case blind test, independently of those test seeds. It does not select only losses or only triggered games when calculating the match score.

## Fixed rule and data

- Use the first 64 seeds in the previously generated `fault_validation` group, both seats per seed, for each of the three variants: harness `always-hold-at-clean-fire-v1`, real-checkpoint weights `reverse-strength-3p0-v1`, and controller `drop-fire-v1`.
- Score a win as 1, draw or timeout as 0.5, and loss as 0. For each seed average its two seat-level `fault − clean` score differences. Bootstrap those 64 seed blocks 10,000 times with RNG seed 183 for a paired percentile 95% interval.
- Call a variant a validated score drop only if the mean difference is at most −0.05 **and** the interval's upper bound is below zero. Missing or invalid pairs make the screen incomplete; no replacement seeds or effect-based stopping are allowed.
- These seeds are separate from the 40 frozen blind-test seeds. The attempt journals, private configurations, per-game states and hashes, and protocol JSON are retained on ds-serv2 under `/data/xinkaiz/system0/exp0/tank-war/v1-critical-window-20261115/score-drop-validation-v2/` (harness), `score-drop-validation-v2b/` (weights), and `score-drop-validation-v2e/` (controller). Earlier controller attempts remain in v2b, v2c, and v2d.

## GPU budget and run changes

The initial budget was **9 GPU-hours of capacity equivalent**. At 2026-11-16 04:10:50 UTC, before inspecting paired score effects, the user approved **18 GPU-hours**. The dated amendment is retained as `score-drop-validation-v2/BUDGET_AMENDMENT_18_GPUH.json`. The first collector launched at 03:37:08 UTC; the new hard cutoff is **09:36 UTC**, at most 17.96 GPU-hours if all three available services are counted continuously from 03:36:50. This includes the stopped v1 launch, which used less than 0.25 GPU-hours. These are shared model services, not an exclusive reservation; exact process-attributable GPU utilization is not available from the retained service counters. The final report will state observed job wall time and the resulting conservative capacity upper bound.

The weights/controller collectors were moved from v2 to v2b before any complete pair after the first even seeds routed every worker to GPU 0. Reversing the order of two identical pinned clean-checkpoint endpoints routed those arms to GPU 6. Both stopped partial roots remain archived. This changed only compute routing; seeds, variants, model hashes, score rule, and cutoff stayed fixed. No score outcomes were inspected to make this change.

GPU 5 had 38.4 GiB free, but a trial clean-checkpoint service was rejected by the existing per-user GPU lock before loading a model. The edited checkpoint service on GPU 5 remained available and unchanged. The lock was not bypassed; clean matches continue on the pinned GPU 0 and GPU 6 endpoints. No earlier exact-variant pairs on these 64 validation seeds and matching build were found outside this screen.

To avoid repeating identical clean matches, controller v2e copies the completed clean run for the same seed and seat from the weights arm. It verifies exact match source, harness, controller, and clean-checkpoint hashes against a frozen reference; it also records the source path and checks copied result and state-file SHA-256 hashes. Before enabling this, the overlapping independent clean games in weights and controller had byte-identical results, states, decisions, and executions. Complete controller pairs from earlier continuations were copied with per-file hashes; incomplete original attempts remain archived. Only the controller fault calls now use the pinned clean-checkpoint service on GPU 0 to balance load; a same-seed clean run independently sent to GPUs 0 and 6 yielded byte-identical result, state, decision, and execution files. The fixed seeds, variants, build, score rule, and cutoff did not change.

## Result

**Complete screen, negative validation.** All three arms finished 64/64 fixed seed blocks and 128/128 two-seat clean/fault pairs. The scorer found no missing pairs, invalid replay hashes, or layer-isolation failures. The predeclared criterion was a mean `fault − clean` score of at most **−0.05** with the paired 95% interval entirely below zero. All three means were positive; none is a validated release-level score drop.

| Final one-decision variant | Mean fault − clean | Paired 95% interval | Score-drop gate |
| --- | ---: | ---: | --- |
| Harness: `always-hold-at-clean-fire-v1` | +0.0234 | −0.0234 to +0.0742 | Fail |
| Real-checkpoint weights: `reverse-strength-3p0-v1` | +0.0234 | −0.0078 to +0.0586 | Fail |
| Controller: `drop-fire-v1` | +0.0156 | −0.0469 to +0.0820 | Fail |

The fault was triggered in 112/128 harness, 124/128 weights, and 112/128 controller games. It changed the local chosen or executed action in 69, 73, and 112 games respectively. The fault game lost in 82, 82, and 84 games, but a loss alone does not show that the fault caused the loss. The paired whole-game score was lower with the fault in 6, 2, and 10 games; higher with the fault in 8, 5, and 11; and equal in 114, 121, and 107. These counts include every preselected pair, including untriggered and unchanged games.

An independent recount of the 128 per-seat outcome pairs per layer confirmed the score direction and means. The positive means are small and all three confidence intervals cross zero; they are **not evidence that these faults improve performance**. A one-decision change can shift shot timing and the opponent's response in either direction. For example, in one red-seat validation pair, the clean controller game lost and the controller-fault game won. At decision 22 the model selected fire in both. The fault changed the acknowledged command from fire to hold; the clean game fired at 4.883 s, while the fault game fired at 4.967 s on the next command. The later shot triggered the blue opponent's avoidance and fire cancellation; red won at 5.167 s. The clean game instead ended with blue winning at 5.583 s. The immediate `executed_action.shot_fired` field was false in both at decision 22 because the firing command has a delay; the event stream and subsequent shot counts show the later physical difference. This example explains how a nominally harmful intervention can occasionally help in a deterministic match. It does not establish that the intervention reliably helps.

The final machine-readable result is `/data/xinkaiz/system0/exp0/tank-war/v1-critical-window-20261115/score-drop-validation-final/summary.json` (SHA-256 `bd2ecbfb2c39e40fbffcacefc97509e99d74dedb2eb2da513e9a6d06ed34bb28`). The fixed protocol and dated budget amendment are in the same directory; all original attempts, journals, configurations, manifests, states, and incomplete route-change runs remain in `score-drop-validation-v1/`, `v2/`, `v2b/`, `v2c/`, `v2d/`, and `v2e/`. The deadline-extension continuation, recorded in `v2b/CONTINUATION_ACTIVATION_18_GPUH.json`, scheduled the last frozen weights pair after its old 06:33 stop-new-work argument; the controller then used that clean run. The final scorer command was:

```bash
/data/xinkaiz/jev4ra/.venv/bin/python /data/xinkaiz/system0/exp0/tank-war/v1-critical-window-20261115/score-drop-validation-v2/score-critical-fault-validation-final.py --root /data/xinkaiz/system0/exp0/tank-war/v1-critical-window-20261115/score-drop-validation-final --out /data/xinkaiz/system0/exp0/tank-war/v1-critical-window-20261115/score-drop-validation-final/summary.json
```

The first collector launched at 03:37:08 UTC and the final summary was written at 06:44:54 UTC. Counting all three shared GPU services continuously from 03:36:50 through the summary gives a conservative **9.40 GPU-hour capacity equivalent**, within the approved 18 GPU-hour ceiling. This is an upper bound on available service capacity during the screen, not measured exclusive GPU use or an API bill. All validation collectors and their watchdog have exited; shared model services were left running.

**Deliverable 3 remains partial.** The paired screen and raw evidence are complete, but these particular final one-decision fault variants did not demonstrate a release-level performance drop. Their local effects can still support a narrow attribution test; they cannot be presented as three validated performance-harming faults. Any replacement fault design would need a new preregistered screen and must not be folded into these fixed results.
