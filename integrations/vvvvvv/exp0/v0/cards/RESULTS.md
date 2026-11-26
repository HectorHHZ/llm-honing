# VVVVVV Exp 0 v0 results

{{DIAGNOSIS_INTRO}} [Protocol](PROTOCOL.md) · [Dataset card](README.md)

{{RESULTS_TABLE}}

{{PER_CLASS_NARRATIVE}}

{{PAIRED_AND_BASELINES}}

## Pre-registered probe variant (reported separately)

{{VARIANT_RESULTS}}

## Probe identity and weight probes (no diagnosis model involved)

Before any probe result was used, the exact logged request of the first four model decisions of every episode was replayed under the episode's own checkpoint: {{IDENTITY_SUMMARY}} `evaluation/weight-probes.json` reports the alternate-versus-reference changes on the probed requests separately from diagnosis performance: {{WEIGHT_PROBE_SUMMARY}} A reference-versus-alternate difference exists for every episode, including the controls, because the two checkpoints differ; only the logged probabilities show which checkpoint served an episode. A changed probability is not proof of a changed or repaired episode outcome.

`evaluation/scores.json` holds the confusion matrices, per-class results, baselines, calibration and paired uncertainty; `evaluation/scores-probe-variant.json` the same for the variant (its `probe` block is the variant). `tools/verify_release.py` reproduces both from the saved replies without an API key.
