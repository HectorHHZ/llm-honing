# RA2 integration

Plays one Red Alert 2 match between a Laya decision model and the Supalosa bot on the headless `ra2-game-env` engine, then audits it. The runner records the context, every model request, the command receipts, and the replay, and the audit rebuilds each decision from those records.

This directory is the engine side of bringing RA2 into Honing (#50). It runs as its own process: nothing under `honing/` imports it. Later pieces put it behind Honing's interfaces: the decision server (#52), the episode runner and grader (#53), and the harness (#54).

It is based on the RA2 experiment in #49.

## External inputs

None of these are in the repository. Each is pinned or supplied separately.

| Input | Where it comes from | License or terms |
|---|---|---|
| `ra2-game-env` engine and SDK | supplied privately, set `RA2_SDK_DIR` | restricted to the people it was sent to; do not redistribute the package or the binary |
| Red Alert 2 game files | a complete retail installation, set `RA2_GAME_DIR` | EA's retail license; the engine reads `ra2.mix`, `language.mix`, rules, maps, and theaters from it |
| Supalosa bot | cloned at the commit in `pins.json` | the upstream repository has no license file, so it is cloned at setup and never copied here |
| Laya source | cloned at the commit in `pins.json` | Apache-2.0 |
| Laya weights (`english`, `typed-decisions`) | Hugging Face, revisions in `pins.json` | Apache-2.0 |

Supalosa's own dependencies currently report 11 advisories from `npm audit`, one of them critical. They run only at setup and inside the match process.

## Setup

Linux, Node.js 22 or newer, Python 3.10 or newer, and an NVIDIA GPU for inference.

```bash
cd integrations/ra2
npm ci
node scripts/setup-dependencies.mjs          # clones Supalosa and Laya at their pins, builds Supalosa
cp .env.example .env && $EDITOR .env && source .env
"$LAYA_PYTHON" scripts/setup-model.py --model typed-decisions
```

`LAYA_PYTHON` is a separate environment with a CUDA build of PyTorch and the Laya revision in `pins.json`. The base Honing install gains no model or GPU dependency.

## Run and audit

```bash
npm run match -- --gpu=0 --seat=red --seed=1 --ticks=18000 --out=runs/seed1-red
npm run audit -- runs/seed1-red
```

`--out` must not exist yet. Each run directory holds the manifest, `decisions.jsonl`, `actions.jsonl`, `result.json`, an observer-only `viewer.json`, `match.rplx`, the source snapshot, and the exact controller and question files used.

## Controller settings and question text

Two files control how the model is asked. Both default to the files in `config/`, which reproduce the controller's previous built-in behavior exactly.

- `--controller=<file>` holds `decisionMode`, `candidateMode`, `decisionInterval`, and `cadence`. A flag such as `--decision-interval=15` overrides the file for one run.
- `--questions=<file>` holds the question and recommendation text, with `{name}` placeholders filled at run time. A replacement file must define every key in `config/questions.json`, or the run stops before it starts.

The manifest records the path and SHA-256 of both files, and the run directory keeps copies. Per-function parameter labels are still in code; #54 moves the rest of the prompt text and turns these files into a Honing `HarnessSpec`.

## Tasks and grading in Honing

`tasks/suite.jsonl` holds 412 tasks built from `tasks/protocol.json`, #49's league protocol:

- `train`: 256 tasks (8 maps, 16 seeds, both seats);
- `val`: #49's development matrix of 24 tasks, which the gate uses;
- `heldout`: 132 tasks (the qualification and sealed cases), which the gate never uses.

Rebuild the file with `python -m honing.backends.ra2.scenarios tasks/protocol.json tasks/suite.jsonl`.

`honing.backends.ra2.RA2EpisodeRunner` plays a task by running this directory's match as a separate process, with the policy served by a decision server. `RA2Grader` then runs the audit and `scripts/verify_replay.py`, which replays the match with no policy. It scores a win the engine ended 1, a draw, tick-limit truncation, or unended match 0.5, and a loss 0, counting wins as #49 did. A failed match, audit, or replay check raises instead of scoring. The grade's feedback summarizes income, units built, killed, and lost, and command counts.

## Training a decision model

`honing.backends.ra2.RA2DemonstrationProvider` reads labelled matches (teacher demonstrations and DAgger corrections, in #49's directory format) as one `typed_decisions` batch. It keeps every check #49's trainer made: the match must be a train-split scenario, labelled data, not a model evaluation, with unchanged source files, and every row must name the same scenario as its match. `honing.backends.jev.trainer.LocalJevTrainer` then trains Laya on that batch and writes a checkpoint manifest the decision server can load. `parity/` compares both with #49's code.

## Tests

```bash
npm test                  # controller, staging, logs, question text, HTTP client; no engine or GPU
npm run test:engine       # needs the engine, game files, and a GPU
```

The model code the worker runs is `honing/backends/jev/`, tested with the rest of Honing (`pytest`).

## Using a decision server

`--decision-server=http://host:port` sends every model request to a running Honing decision server instead of starting a worker for this match. Start one with:

```bash
"$LAYA_PYTHON" -m honing.backends.jev.server --pins pins.json --model typed-decisions \
  --model-config-dir .cache/models --laya-source "$LAYA_SOURCE_DIR" --gpu 0 --port 8801
```

It answers one request at a time, so several matches can share one loaded model. The worker and the server run the same model code.
