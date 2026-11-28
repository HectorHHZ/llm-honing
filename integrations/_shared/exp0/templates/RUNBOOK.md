# {{GAME_NAME}} Exp 0 v0 runbook

Run from `{{REPO_ROOT}}` with {{ENVIRONMENT}}. `$C` is the common tools directory (`integrations/_shared/exp0` here, `common-source/` in the bundle), `$COL` an empty or resumable collection root, `$ADAPTER` the game adapter (`{{ADAPTER_SPEC}}`).

## Capture and audit

Start the clean and alternate decision services ({{SERVICES}}). {{COLLECT_COMMANDS}}

```bash
python3 $C/validate_episode.py $COL                       # every attempt valid or incomplete
python3 $C/probes.py selfcheck $COL --adapter $ADAPTER    # the clean re-render reproduces logged requests
python3 $C/audit_set.py reask $COL $COL/weights-reask.jsonl --endpoint {{CLEAN_URL}}   # only if weights attempts never share an identical request with their clean pair
python3 $C/audit_set.py audit $COL $COL/audit --reask $COL/weights-reask.jsonl --adapter $ADAPTER
```

Audit into a new directory when incorporating more attempts. Keep all evidence, including incomplete attempts and `error.json` records. `audit/episodes.json` maps neutral ids to attempts and holds the gold answers: restrict the diagnosis path to the outputs below.

## Prepare model inputs

```bash
python3 $C/diagnosis_inputs.py $COL $COL/audit/episodes.json $COL/read-inputs.jsonl
python3 $C/probes.py capture $COL $COL/audit/episodes.json $COL/probe-reference.jsonl --endpoint {{CLEAN_URL}} --checkpoint-role clean --adapter $ADAPTER
python3 $C/probes.py capture $COL $COL/audit/episodes.json $COL/probe-alternate.jsonl --endpoint {{ALTERNATE_URL}} --checkpoint-role alternate --adapter $ADAPTER
python3 $C/probes.py compare $COL $COL/audit/episodes.json $COL/probes --reference-capture $COL/probe-reference.jsonl --alternate-capture $COL/probe-alternate.jsonl --adapter $ADAPTER
```

## Run the diagnosis comparison (real model calls: only once the diagnosis model and its subscription are approved)

Use one model id for both modes. Check `codex login status` under the account that will launch the CLI, and verify one development example first. The runner passes only the [prompt](diagnosis_prompt.txt), [schema](diagnosis_schema.json) and mode-specific inputs to each new session.

```bash
export EXP0_DIAGNOSIS_APPROVED=yes      # only after approval
python3 $C/diagnose.py $COL/read-inputs.jsonl $COL/diagnosis-read  --mode read  --backend codex --model {{DIAGNOSIS_MODEL}}
python3 $C/diagnose.py $COL/read-inputs.jsonl $COL/diagnosis-probe --mode probe --probes $COL/probes/probe-inputs.jsonl --backend codex --model {{DIAGNOSIS_MODEL}}
```

## Offline scoring and release

```bash
python3 -I -B $C/score.py $COL/audit/episodes.json $COL/diagnosis-read/diagnoses.jsonl $COL/diagnosis-probe/diagnoses.jsonl
python3 $C/bundle.py release $COL $COL/audit {{RELEASE_ROOT}}/{{GAME_ID}}/v0 --cards {{CARDS_DIR}} --code-revision {{CODE_REVISION}} {{SOURCE_FLAGS}}
python3 -I -B {{RELEASE_ROOT}}/{{GAME_ID}}/v0/tools/verify_release.py {{RELEASE_ROOT}}/{{GAME_ID}}/v0
```

The release destination is `{{GAME_ID}}/v0/` in the private `honinglab/exp0-data`. The release holds structured text/JSON evidence and code only (no game binaries, assets, frames, screenshots or videos). Check it and print the upload plan (no token, no network):

```bash
python3 $C/upload_release.py check {{RELEASE_ROOT}}/{{GAME_ID}}/v0
python3 $C/upload_release.py plan  {{RELEASE_ROOT}}/{{GAME_ID}}/v0 --parent-revision $HF_PARENT_REVISION
```

The upload and the fresh-download verification are made by the release owner only, after reviewing the locally verified release. Both read the token from the file named by `HF_TOKEN_FILE` (never an argument, never printed):

```bash
python3 $C/upload_release.py upload {{RELEASE_ROOT}}/{{GAME_ID}}/v0 --parent-revision $HF_PARENT_REVISION --execute
python3 $C/upload_release.py download-verify --game-dir {{GAME_ID}}/v0/ --revision $HF_NEW_REVISION --out $SNAPSHOT --local {{RELEASE_ROOT}}/{{GAME_ID}}/v0 --execute
```

Anyone with read access can repeat the verification by hand in a fresh environment without packages or an API key:

```bash
hf download honinglab/exp0-data --repo-type dataset --revision $HF_REVISION --include '{{GAME_ID}}/v0/*' --local-dir $SNAPSHOT
python3 -m venv --without-pip $SNAPSHOT/verify-env
$SNAPSHOT/verify-env/bin/python -I -B $SNAPSHOT/{{GAME_ID}}/v0/tools/verify_release.py $SNAPSHOT/{{GAME_ID}}/v0
```
