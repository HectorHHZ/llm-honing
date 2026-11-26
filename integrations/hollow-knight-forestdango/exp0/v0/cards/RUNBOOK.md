# Hollow Knight (ForestDango) Exp 0 v0 runbook

Run from the repository root with Python 3.10 or newer (the release was collected with 3.13) and the `tokenizers` package (the harness counts option tokens with Laya's tokenizer). Set `HONING_LAYA_TOKENIZER` to the file `tokenizer/tokenizer.json` of `convaiinnovations/laya-typed-decisions@1a793eb568e6718f15941d08f85432581df534e3` (SHA-256 `{{TOKENIZER_SHA}}`); without it `runtime.py` fetches that file through `huggingface_hub` (with `HF_HUB_OFFLINE=1` only from a local cache), and every collector, replay and probe tool refuses to run when the package or the file is missing. Local locations are resolved at run time: plan paths are relative to the repository root, `HONING_FFMPEG` names ffmpeg for the videos, `EXP0_CODEX_BIN` the Codex CLI. The deterministic game build is not redistributable: ForestDango/Hollow-Knight-Demo has no license, so neither the build, its assets nor the modified game scripts it compiles are published. `pins.json` pins the build. The frozen plan names the build `code/hk-devmode/build/linux64_devmode_c/` and the bridge source `code/hk-devmode/bridge`, paths of the collecting workspace; here plan paths resolve against the repository root, so put your copy of the build at that path or point `release.game.player` and `release.game.player_drawn` at it, and `release.game.bridge_source` at `integrations/hollow-knight-forestdango/bridge` (the same files). Offline scoring and the exact re-render of every request work without the build; the native replay needs it. `$C` is `integrations/_shared/exp0` (`common-source/` in the bundle), `$H` is `integrations/hollow-knight-forestdango/exp0/v0` (`collector-source/` in the bundle), `$COL` the collection root (an absolute path: `batch.py` runs the collector from `$H`), `$REL` a new, empty release root.

## Capture

Start the decision services (Honing's decision server with the Laya backend on CPU, 8 threads; see `integrations/ra2/README.md`): clean Laya `convaiinnovations/laya-typed-decisions@1a793eb5` on `http://127.0.0.1:8811` and a second, identical clean service on `http://127.0.0.1:8820` (re-ask at least 10 logged requests of the first and require a largest difference of 0.0), and the alternate `jayzou3773/laya-ra2-v33@51d2f5a0` on `http://127.0.0.1:8812`. The collector checks the served weights SHA-256 before and after each episode and accepts only the plan's endpoints. Verify the starts (two boots plus the prefix must give an identical handover), freeze the plan, write the collection plan and the job lists, then run one stream per service, one game each (Xvfb displays :350–:367; the build draws nothing, the display is only the launcher's):

```bash
for s in $(python3 -c "import json; print(' '.join(json.load(open('$H/plan.json'))['starts']))"); do
  python3 -B $H/verify_starts.py check --start $s --out $COL/../starts_check/$s-boot1 --display 376
  python3 -B $H/verify_starts.py check --start $s --out $COL/../starts_check/$s-boot2 --display 377
  python3 -B $H/verify_starts.py compare $COL/../starts_check/$s-boot1/handover.json $COL/../starts_check/$s-boot2/handover.json
done
python3 -B $H/release_jobs.py --root $COL --displays 350-367      # collection.json from the frozen plan + jobs/a1_<port>.json
for port in 8811 8820 8812; do
  python3 -B $H/batch.py $COL/jobs/a1_$port.json --root $COL --parallel 1 --log $COL/batch_log_$port.jsonl &
done
wait
```

The clean-model conditions (no injection, harness, controller) of a group run on one clean service, groups alternating between 8811 and 8820; the weights condition runs on 8812. Each finished attempt prints one JSON line; an incomplete attempt keeps its `error.json` and the next attempt number is used (`release_jobs.py --attempt 2 --groups …`, cap {{ATTEMPT_CAP}} per slot). Reserve groups are collected only under the plan's selection rule (`--groups g11-r008`, then the next, …).

## Checks, audit and model inputs

```bash
python3 -B $H/release_v0.py prepare $COL
```

`prepare` runs, in order and stopping at the first failure: `validate_episode.py` (every attempt valid or incomplete), `probes.py selfcheck` (every logged request re-rendered byte for byte from its saved harness state, or the logged `clean_request` where the harness fault acted), `native_replay.py` on every attempt (a new game of the same build fed only the recorded native input; debug states, visible observations and per-frame states compared; at most four games, displays :370–:373), `audit_set.py reask` on 8811 (the weights episodes' evidence requests under the clean checkpoint), `audit_set.py audit` (condition evidence, Griddly's selection, neutral ids; it uses the native replays), `diagnosis_inputs.py` (label-free `read` inputs with its own isolation check), `probes.py capture` on 8811 and 8812 plus `probes.py compare` (identity check, `probe` inputs), and the stricter label-isolation check (`isolation_check.py`). `audit/episodes.json` maps neutral ids to attempts and holds the gold answers: only `read-inputs.jsonl` and `probes/probe-inputs.jsonl` reach the diagnoser.

## Diagnosis, scores, cards, bundle and verification: the one resume command

Real model calls run only with `EXP0_DIAGNOSIS_APPROVED=yes`, set once the diagnosis model and its subscription are approved. Check `codex login status` under the account that will launch the CLI. One command then runs, in this order and stopping at the first failure: a check that both frozen prompts are unchanged and that `prepare` has completed (identity check passed), the 80 primary calls (`read`, then `probe`), the 40 calls of the pre-registered probe variant, the offline scores (primary and variant), the cards (these documents, filled from the files), a check that the collector code holds no local path, `bundle.py release` with the given code revision (text/JSON only), the placeholders for any local path left in a record (`privacy-check.json`), a privacy gate (no absolute local path and no local account name anywhere in the bundle), `tools/verify_release.py` in a fresh venv without packages, the privacy gate again, and `upload_release.py check` (no token, no network):

```bash
EXP0_DIAGNOSIS_APPROVED=yes python3 -B $H/release_v0.py resume $COL --out $REL --code-revision <40-hex commit>
```

A rerun continues where it stopped: `diagnose.py` skips the episodes already answered (with identical frozen settings) and keeps every failed attempt. The bundle is `$REL/hollow-knight-forestdango/v0`. To rebuild it later with another code revision (e.g. after the code is committed), without new model calls:

```bash
python3 -B $H/release_v0.py bundle $COL --out <new empty dir> --code-revision $(git rev-parse HEAD)
```

Offline scores alone, without any key:

```bash
python3 -I -B $C/score.py $COL/audit/episodes.json $COL/diagnosis-read/diagnoses.jsonl $COL/diagnosis-probe/diagnoses.jsonl
python3 -I -B $C/score.py $COL/audit/episodes.json $COL/diagnosis-read/diagnoses.jsonl $COL/diagnosis-probe-variant/diagnoses.jsonl
```

## Release

The release destination is `hollow-knight-forestdango/v0/` in the private `honinglab/exp0-data`. The release holds structured text/JSON evidence and code only: no game binaries, assets, frames, screenshots or videos, and no attempt `extra/` directories. The upload and the fresh-download verification are made by the release owner only, after reviewing the locally verified release; both read the token from the file named by `HF_TOKEN_FILE` (never an argument, never printed):

```bash
python3 $C/upload_release.py plan   $REL/hollow-knight-forestdango/v0 --parent-revision $HF_PARENT_REVISION
python3 $C/upload_release.py upload $REL/hollow-knight-forestdango/v0 --parent-revision $HF_PARENT_REVISION --execute
python3 $C/upload_release.py download-verify --game-dir hollow-knight-forestdango/v0/ --revision $HF_NEW_REVISION --out $SNAPSHOT --local $REL/hollow-knight-forestdango/v0 --execute
```

Anyone with read access can repeat the verification by hand in a fresh environment without packages or an API key:

```bash
hf download honinglab/exp0-data --repo-type dataset --revision $HF_REVISION --include 'hollow-knight-forestdango/v0/*' --local-dir $SNAPSHOT
python3 -m venv --without-pip $SNAPSHOT/verify-env
$SNAPSHOT/verify-env/bin/python -I -B $SNAPSHOT/hollow-knight-forestdango/v0/tools/verify_release.py $SNAPSHOT/hollow-knight-forestdango/v0
```

The verifier checks the file set and hashes, the index schema, 40 distinct neutral episodes (10/10/10/10), the attempt log, the checkpoints against the frozen plan, split groups, the component evidence of every verified episode against the raw records, the isolation of the diagnosis inputs, every probe identity check, and that `score.py` reproduces `evaluation/scores.json` and `evaluation/scores-probe-variant.json` from the saved replies.
