# VVVVVV Exp 0 v0 results

**The diagnosis has not run yet.** Pending: the Codex CLI subscription used for the diagnosis answered the first call of 2026-11-26 01:12 PST, a development check outside the scored set, with "You've hit your usage limit ... try again at Nov 28th, 2026 11:57 PM"; no scored diagnosis call has been made. The inputs, probes, prompts and settings are frozen, and `release_v0.py resume` runs the 80 primary calls, the 40 variant calls and everything after them. The Hugging Face release (`vvvvvv/v0/` in `honinglab/exp0-data`) and the #174 receipt follow after the Codex run. [Protocol](PROTOCOL.md) · [Dataset card](DATASET.md)

| Metric | Read | Probe |
| --- | ---: | ---: |
| Correct on verified interventions | pending | pending |
| Macro-F1 on verified interventions | pending | pending |
| False positives on no-injection controls (any asserted component) | pending | pending |

Per-class results, confusion matrices and calibration: pending (no scored call yet).

Baselines that need no diagnosis (balanced 10/10/10 verified cases): uniform random accuracy 0.333, majority 0.333, each constant component 0.333 accuracy with 10/10 control false positives. The paired probe-minus-read difference is pending.

## Pre-registered probe variant (reported separately)

Pending, for the same reason; it runs after the 80 primary calls.

## Probe identity and weight probes (no diagnosis model involved)

Before any probe result was used, the exact logged request of the first four model decisions of every episode was replayed under the episode's own checkpoint: 40 of 40 episodes passed (160 decisions; largest difference 0.0000, tolerance 0.01, tie 0.01). `evaluation/weight-probes.json` reports the alternate-versus-reference changes on the probed requests separately from diagnosis performance: no_injection: 40/40 probed decisions above 0.01, mean largest option difference 0.561, top option changed in 3; harness: 40/40 probed decisions above 0.01, mean largest option difference 0.562, top option changed in 2; weights: 40/40 probed decisions above 0.01, mean largest option difference 0.600, top option changed in 0; controller: 40/40 probed decisions above 0.01, mean largest option difference 0.576, top option changed in 3. A reference-versus-alternate difference exists for every episode, including the controls, because the two checkpoints differ; only the logged probabilities show which checkpoint served an episode. A changed probability is not proof of a changed or repaired episode outcome.

`evaluation/scores.json` holds the confusion matrices, per-class results, baselines, calibration and paired uncertainty; `evaluation/scores-probe-variant.json` the same for the variant (its `probe` block is the variant). `tools/verify_release.py` reproduces both from the saved replies without an API key.
