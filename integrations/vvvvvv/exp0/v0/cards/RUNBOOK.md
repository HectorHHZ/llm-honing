# VVVVVV Exp 0 v0 runbook

Run from the repository root with Python 3.10 or newer (the release was collected with 3.13; standard library only for every tool here) and the game runtime that `integrations/vvvvvv/README.md` builds (the pinned build, by default `work/vvvvvv/build/VVVVVV`, SDL3, the caller's official desktop `data.zip`; see `pins.json`). Local locations are resolved at run time by `runtime.py`: set `VVVVVV_ASSETS` (the desktop `data.zip`) and `VVVVVV_SDL_LIB` (the directory of `libSDL3.so.0`) when they are not at the defaults it documents; `FFMPEG` for the videos and `EXP0_CODEX_BIN` for the Codex CLI likewise. `$C` is `integrations/_shared/exp0` (`common-source/` in the bundle), `$G` is `integrations/vvvvvv/exp0/v0` (`collector-source/` in the bundle), `$COL` the collection root, `$REL` a new, empty release root.

## Capture

Start the two decision services: clean Laya `convaiinnovations/laya-typed-decisions@1a793eb5` on `http://127.0.0.1:8813` and alternate `jayzou3773/laya-ra2-v33@51d2f5a0` on `http://127.0.0.1:8814` (Honing's decision server with the Laya backend; see `integrations/ra2/README.md`). The collector checks the served weights SHA-256 before and after each episode and refuses any other port. Freeze the plan, then run four streams, one game instance each (at most four at once; Xvfb displays :420–:423):

```bash
python3 -B $G/plans/make_plan.py $COL/collection.json --kind release --faults persistent --deaths-end no --cap 150
S=$G
python3 -B $S/collect_set.py vvvvvv $COL --stream clean --conditions no_injection --display 420 &
python3 -B $S/collect_set.py vvvvvv $COL --stream clean --conditions harness --display 421 &
python3 -B $S/collect_set.py vvvvvv $COL --stream clean --conditions controller --display 422 &
python3 -B $S/collect_set.py vvvvvv $COL --stream alternate --display 423 &
wait
```

Each finished attempt prints one JSON line; an incomplete attempt keeps its `error.json` and the next attempt number is used (cap 3 per slot). Reserve groups are collected only under the plan's `release_rules` (`--groups s11`, then `s12`, …, with the stream of the slot that needs them).

## Checks, audit and model inputs

```bash
python3 -B $G/release_v0.py prepare $COL
python3 -B $G/fault_checks.py $COL --out $COL/../checks/collection_fault_checks.json
```

`prepare` runs, in order and stopping at the first failure: `validate_episode.py` (every attempt valid or incomplete), `probes.py selfcheck` (the clean re-render reproduces every logged request, or the logged `clean_request` where the harness fault acted), `audit_set.py reask` on 8813 (the weights episodes' first requests under the clean checkpoint), `audit_set.py audit` (native replay of every attempt in a fresh game process, condition evidence, Griddly's selection, neutral ids), `diagnosis_inputs.py` (label-free `read` inputs with the isolation check), and `probes.py capture` on 8813 and 8814 plus `probes.py compare` (identity check, `probe` inputs). `fault_checks.py` re-checks each fault from decision 1 and runs the integration's own receiver (`checked_action`) on every executed reply. `audit/episodes.json` maps neutral ids to attempts and holds the gold answers: only `read-inputs.jsonl` and `probes/probe-inputs.jsonl` reach the diagnoser.

## Diagnosis, scores, cards, bundle and verification: the one resume command

Real model calls run only with `EXP0_DIAGNOSIS_APPROVED=yes`, set once the diagnosis model and its subscription are approved. Check `codex login status` under the account that will launch the CLI. One command then runs, in this order and stopping at the first failure: the 80 primary calls (`read`, then `probe`), the 40 calls of the pre-registered probe variant, the offline scores (primary and variant), the cards (these documents, filled from the files), `bundle.py release` with the given code revision (text/JSON only), the placeholders for any local path left in a record (`privacy-check.json`), a privacy gate (no absolute local path and no local account name anywhere in the bundle), `tools/verify_release.py` in a fresh venv without packages, the privacy gate again, and `upload_release.py check` (no token, no network):

```bash
EXP0_DIAGNOSIS_APPROVED=yes python3 -B $G/release_v0.py resume $COL --out $REL --code-revision <40-hex commit>
```

A rerun continues where it stopped: `diagnose.py` skips the episodes already answered (with identical frozen settings) and keeps every failed attempt. The bundle is `$REL/vvvvvv/v0`. To rebuild it later with another code revision (e.g. after the code is committed), without new model calls:

```bash
python3 -B $G/release_v0.py bundle $COL --out <new empty dir> --code-revision $(git rev-parse HEAD)
```

Offline scores alone, without any key:

```bash
python3 -I -B $C/score.py $COL/audit/episodes.json $COL/diagnosis-read/diagnoses.jsonl $COL/diagnosis-probe/diagnoses.jsonl
python3 -I -B $C/score.py $COL/audit/episodes.json $COL/diagnosis-read/diagnoses.jsonl $COL/diagnosis-probe-variant/diagnoses.jsonl
```

## Release

The release destination is `vvvvvv/v0/` in the private `honinglab/exp0-data`. The release holds structured text/JSON evidence and code only: no game binaries, assets, frames, screenshots or videos, and no attempt `extra/` directories. The upload and the fresh-download verification are made by the release owner only, after reviewing the locally verified release; both read the token from the file named by `HF_TOKEN_FILE` (never an argument, never printed):

```bash
python3 $C/upload_release.py plan   $REL/vvvvvv/v0 --parent-revision $HF_PARENT_REVISION
python3 $C/upload_release.py upload $REL/vvvvvv/v0 --parent-revision $HF_PARENT_REVISION --execute
python3 $C/upload_release.py download-verify --game-dir vvvvvv/v0/ --revision $HF_NEW_REVISION --out $SNAPSHOT --local $REL/vvvvvv/v0 --execute
```

Anyone with read access can repeat the verification by hand in a fresh environment without packages or an API key:

```bash
hf download honinglab/exp0-data --repo-type dataset --revision $HF_REVISION --include 'vvvvvv/v0/*' --local-dir $SNAPSHOT
python3 -m venv --without-pip $SNAPSHOT/verify-env
$SNAPSHOT/verify-env/bin/python -I -B $SNAPSHOT/vvvvvv/v0/tools/verify_release.py $SNAPSHOT/vvvvvv/v0
```

The verifier checks the file set and hashes, the index schema, 40 distinct neutral episodes (10/10/10/10), the attempt log, the checkpoints against the frozen plan, split groups, the component evidence of every verified episode against the raw records, the isolation of the diagnosis inputs, every probe identity check, and that `score.py` reproduces `evaluation/scores.json` and `evaluation/scores-probe-variant.json` from the saved replies.
