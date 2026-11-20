# Tank War Exp 0 v0: private delivery stage

This is the 40-episode development and reproducibility sample fixed in
`freeze.json`. The original recordings remain on ds-serv2. A separate,
hash-checked staging package is at
`/data/xinkaiz/system0/exp0/tank-war/v0-stage-20261120-verified/`. It has not been
uploaded to Hugging Face and is not the #183 public release.

## Contents and labels

The package contains 10 complete losses each with a verified harness, weights,
or controller change, plus 10 unmodified losses. There are 40 episodes, 19
seed/reset groups, 2,151 agent decisions, and 4,302 question records. Both
seats of a seed share its group. All selected games lost, including the
controls; the control set cannot measure behavior on wins.

`index.jsonl` identifies the game, pinned model/harness/controller hashes,
native outcome, paired-clean outcome, blind input hash, and compressed raw
recording files. The seven original recording files are included for each
selected game, with a paired-clean recording for each of the 30 altered games.
The JSONL files use deterministic gzip; `index.jsonl` records both the
uncompressed original SHA-256 and the packaged SHA-256. `file-hashes.json`
covers the package files. `sample-selection.json` retains the original frozen
sample digest and removes private server paths.

`labels.jsonl` keeps the injected layer separate from the component verified
at a saved decision. Harness requires a changed model input and changed model
choice. Weights requires the same input, a real edited checkpoint, and a changed
choice or probabilities. Controller requires the same model choice and a changed
post-selection command. The reference decision is technical evidence, not a
claim that it caused the loss. The 30 altered games have paired-clean outcomes;
five differ. That comparison is reported separately from the component label.
The ten `no_injection` controls have no asserted component label or chance
label. No human decision-window review is required for v0.

The original weight run directories and sample manifest name the two edited
checkpoint strengths in reverse: all ten `reverse_strength_3p0-v1` runs record
the frozen 4.0 checkpoint digest, and all ten `reverse_strength_4p0-v1` runs
record the frozen 3.0 digest. The package preserves the original names and
adds `verified_checkpoint_variant` from the recorded weight SHA-256, with
`source_instance_name_mismatch=true`. The `weights` component label remains
supported, but any analysis by edit strength must use the verified digest.
The harness strategy also participates in plan revalidation after model choice;
these cases have direct changed-input and changed-choice evidence, and the
functional boundary remains a stated limitation.

`attempts.jsonl` preserves all 1,330 attempted fault records with opaque run
IDs and source hashes: 1,186 labeled, 101 excluded for a build/screened-variant
mismatch, 40 failed with an invalid controller-fault manifest, and three had
no result record. The original attempt files and paths remain in the private
source store. These excluded attempts do not fill the 40-case quota.

## Checks

`checks/replay.jsonl` records 40/40 complete deterministic replay comparisons
for result, states, execution, and events. The original recordings are in the
stage; the original replay commands and duplicate replay outputs remain in the
private source store. `checks/probe-identity.jsonl` records 40/40 valid saved-state signatures
across all 2,151 selected decisions. The earlier unchanged-checkpoint identity
gate is separately retained as 891/891 calls at tolerance 0.01, with maximum
probability difference 0.0. Its input manifests were checked against the same
clean checkpoint, harness, controller, and game revision.

`checks/diagnoser-isolation.jsonl` verifies the 320 archived login-based CLI
calls: every call used the frozen anonymous input hash, expected probe/null
hash, frozen prompt/schema, one per-case Bubblewrap work mount, and no tool
call. This checks the historical diagnostic boundary only. It does not count
as the real API read/probe run required by #183 item 7.

Run the validator on ds-serv2 without a GPU:

```bash
python3 integrations/tank-war/scripts/package-exp0-v0.py \
  --out /data/xinkaiz/system0/exp0/tank-war/v0-stage-20261120-verified
```

The complete staging command is in `RUNBOOK.md`. The stage is not independent
confirmation: these 40 cases have already been diagnosed. It is intended to
make the v0 evidence reproducible, then support a later fixed-HF-revision
download and offline score check.
