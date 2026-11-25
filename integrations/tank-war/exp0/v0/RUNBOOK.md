# Tank War Exp 0 v0: pinned setup and fault builders

Run from a clone of this PR's repository root. The commands below only check
the existing release and construct candidate fault artifacts. They do not
collect episodes, run model diagnosis, publish data or use a GPU. Use Python
3.10+ and Node 22.22.0. The recorded ds-serv2 system Python was 3.10.12 and
the match manifests recorded Node 22.22.0. `package-lock.json` pins the Tank
War development dependency; the NanoJev source/model revisions are in
`integrations/ra2/pins.json` and [freeze.json](freeze.json).

## Verify the recorded code and configuration

The 40 existing games were recorded with commit
`31912a1a3378f49e542577fa383d42891a6f2dc3`, which is currently reachable
from the separate draft experiment branch. Fetch it, then check the nine exact
source hashes and clean release configuration:

```bash
git fetch origin codex/issue183-exp0
python3 integrations/tank-war/scripts/check-v0-pins.py --repo .
```

Expected output includes `source_files_verified: 9`,
`model_pins_verified: true`, and `controller_verified: true`. A missing commit,
changed source file or changed model/config pin fails. This is a recorded-build
check; the current checkout is not silently substituted for the original game
code. For a headless CPU smoke of that exact build:

```bash
EXP0_V0_PR_ROOT="$(pwd)"
EXP0_V0_SOURCE="$(mktemp -d)/recorded-source"
git worktree add --detach "$EXP0_V0_SOURCE" 31912a1a3378f49e542577fa383d42891a6f2dc3
cd "$EXP0_V0_SOURCE/integrations/tank-war"
npm ci
node scripts/match.mjs --seed=2001 --seat=blue --teacher --out="$EXP0_V0_SOURCE/teacher-smoke"
cd "$EXP0_V0_PR_ROOT"
```

That teacher game tests the engine command and does not replace the 40-case
data/replay validation in item 2. Return to the PR checkout for the commands
below.

## Build the harness and controller candidates

Use an unused output directory. The builders refuse to overwrite it. They
produce private fault manifests and candidate files; they do not assert that
each fault actually changes a decision.

```bash
EXP0_V0_BUILD="$(mktemp -d)"
python3 integrations/tank-war/scripts/build-harness-faults.py --out "$EXP0_V0_BUILD/harness"
python3 integrations/tank-war/scripts/build-controller-faults.py --out "$EXP0_V0_BUILD/controller"
node integrations/tank-war/scripts/check-v0-faults.mjs \
  --freeze=integrations/tank-war/exp0/v0/freeze.json \
  --harness-dir="$EXP0_V0_BUILD/harness/extreme_standoff" \
  --controller-fault-file="$EXP0_V0_BUILD/controller/drop_fire_after_selection.json"
```

The checker must return the clean/fault harness SHA-256 values and all three
clean, altered-harness and controller-fault controller SHA-256 values in
`freeze.json`. It checks the original builder output against the recorded
40-game release, including the exact post-selection controller manifest.
The pinned recorded game runtime, not the builder, applies that manifest.

## Rebuild the two real weights candidates

On ds-serv2, the original checksum-verified input manifests are shown below.
They are model artifacts, not uploaded dataset files. The builder uses CPU
file operations and changes only `scalar.weight`; it refuses source checkpoints
whose digests or tensor layouts differ. It needs enough free disk for two full
NanoJev checkpoints. No GPU or training is needed.
The freeze also pins every non-weight file copied into the edited checkpoint:
the run config, backbone config, tokenizer files and chat template. The builder
copies symlink targets as regular files before editing the weight file.

```bash
EXP0_V0_BUILD="$(mktemp -d)"
python3 integrations/tank-war/scripts/build-weight-faults.py \
  --freeze=integrations/tank-war/exp0/v0/freeze.json \
  --prior-manifest=/data/xinkaiz/honing-pr141-gpu4-20261103/candidate/checkpoint.json \
  --reverse-manifest=/data/xinkaiz/system0/exp0/tank-war/v0-pilot/pilot1/candidates/weights/reverse_prior_scalar_edit/checkpoint.json \
  --strength=-3.0 --strength=-4.0 --out="$EXP0_V0_BUILD/weights"
```

The source checkpoint SHA-256 values are also fixed in `freeze.json`. To
check the two generated checkpoints against the recorded 40-game weights,
first build the harness/controller candidates in the same `EXP0_V0_BUILD`,
then run:

```bash
node integrations/tank-war/scripts/check-v0-faults.mjs \
  --freeze=integrations/tank-war/exp0/v0/freeze.json \
  --harness-dir="$EXP0_V0_BUILD/harness/extreme_standoff" \
  --controller-fault-file="$EXP0_V0_BUILD/controller/drop_fire_after_selection.json" \
  --weights-root="$EXP0_V0_BUILD/weights"
```

The extra result must be `edited_checkpoints_verified: 2`. This verifies the
complete copied checkpoint file list and each non-weight file hash, as well as
the edited weights. If it fails, retain
the candidate and error for the attempt log and do not assign a weights label.
The clean release digest is checked against recorded match metadata here; a
source-side clean checkpoint manifest can also be passed with
`--clean-manifest` to verify the symmetric edit against local files.

## Local verification for this PR

```bash
python3 integrations/tank-war/scripts/check-v0-pins.py --repo .
python -m pytest -q tests/test_tank_harness_fault_builder.py tests/test_tank_weight_fault_builder.py
cd integrations/tank-war && npm test && npm run format:check
```

The next work item is the 40-episode delivery package. It starts only after
the independent item-1 PR merges. The published HF revision, fresh-download
score check, real API call record and #174 receipt have their own later
checklist rows in [STATUS.md](STATUS.md).
