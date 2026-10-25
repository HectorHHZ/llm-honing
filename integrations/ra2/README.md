# Red Alert 2

Decision models need training.

https://github.com/user-attachments/assets/b092c09b-2bfc-461c-8721-ad8f8c7aaec4

We gave NanoJev, an open-source decision model, a game of Red Alert 2 against the Supalosa bot.
Early on it sold its own construction yard, and it lost at tick 140.

After supervised training on a teacher's matches, the same model builds a base, fights, and wins
that match-up. Laya, another family trained the same way, fields Kirov airships and bombs the
enemy base. The game and the opponent are the same in both halves of the video; the model's weights
are what changed.

This directory runs those matches. It has a headless Red Alert 2 engine, the Supalosa bot as the
opponent, four decision-model families behind one HTTP decision server, a match runner that records
every request and command, and an audit that rebuilds each decision from the records. Honing's
gate, trainers, and harness methods run on top of it.

| | |
| --- | --- |
| Models that play | [NanoJev](https://huggingface.co/C-Tianyu/NanoJev), Laya, [Kev](https://github.com/jaredpalmer/kev), [StartLux-Decision](https://github.com/StartLuxLabs/Startlux-Decision), and DiffusionGemma through [djev](../djev/README.md) |
| What a model gets | the game state as text, a question, and a short list of options; it returns a probability for every option |
| What improves it | SFT on a teacher's matches, GRPO on its own matches, and GEPA on the question text |
| What decides a release | the candidate plays the validation tasks next to the current release and must win more than it loses |

## Contents

1. [How the model learns](#1-how-the-model-learns): SFT, GRPO, GEPA, and the release check
2. [Quick start](#2-quick-start)
3. [Reference](#3-reference): external inputs, controller settings, tasks and grading, the harness, duels, training code, the model families, the campaign end to end

## 1. How the model learns

The model plays full matches against Supalosa, a scripted AI. A win scores 1, a draw or timeout
0.5, and a loss 0. Every few ticks the controller shows the model the state and a question, such as
which goal to pursue next, with a list of options. The model returns a probability for each option
and the controller carries out the chosen one.

Training changes the weights in two stages, and the same steps work for the three trainable
families (Laya, Kev, and NanoJev). GEPA leaves the weights alone and rewrites the questions.

| | SFT | GRPO | GEPA |
| --- | --- | --- | --- |
| Changes | weights | weights | question text |
| Learns from | the teacher's pick at each decision | the match result | the match result and the grader's notes |
| Plays on | training maps only | training tasks, several times each | a training minibatch, then the validation set |

### Learn from a teacher (SFT)

The teacher is codex. At every decision point it sees the same state and options the model gets and
picks one. We record the teacher's own matches, and we also let the model play while the teacher
writes down what it would have done in each situation the model reached (DAgger). The model is then
trained to give the teacher's pick the highest probability.

Only training maps are used. Every recorded match is checked against its saved hashes before it
reaches the trainer, so a validation match or an edited record is refused
(`honing/backends/ra2/demonstrations.py`). The trainer (`honing/backends/jev/trainer.py`) is the
procedure from the original RA2 experiment, and with deterministic kernels it writes byte-identical
weights ([`parity/`](parity/README.md)).

### Learn from its own games (GRPO)

The trained model plays the training tasks several times, and every decision in a match gets that
match's score as its reward. Repeats of one task form a group: a decision's advantage is its reward
minus the group average, divided by the group spread, so decisions from won matches go up and
decisions from lost ones go down. A group whose matches all ended the same way has no spread and is
skipped. Each update is clipped so the model cannot move too far in one step
(`honing/backends/jev/policy_gradient.py`).

The rollouts have to be sampled from the model's probabilities, because argmax play repeats exactly
and gives a group no spread. The RA2 match controller does not sample its choices yet; the trainer
and the rollout provider (`honing/backends/ra2/rollouts.py`) are in place for when it does.

### Evolve the question text (GEPA)

GEPA keeps the model fixed and rewrites the question text in
[`config/questions.json`](config/questions.json): the wording of each question, its options, and the
hints. The controller settings, such as how often the model is asked, stay as they are. One round
(`honing/harness/methods/gepa/`) goes like this:

1. GEPA keeps an archive of every question text it has tried, with its scores. It picks one as the
   parent and plays it on a small batch of training tasks.
2. The batch results go to a language model together with the grader's notes on each match (who won,
   how it ended, what the model built and lost). The language model rewrites one part of the
   question text.
3. The rewrite plays the same batch. If its total score is not higher than the parent's, it is
   dropped.
4. A rewrite that passes plays the full validation set and is released only if its average score is
   higher than the current version's. Its scores go into the archive either way, so later rounds can
   start from it.

Before any match, the harness check refuses question text the model cannot read: every option,
instruction, and state entry has to fit the family's token limits.

### Check before releasing

New weights and new question text go through the same gate. The candidate plays the 24 validation
tasks (two maps, six seeds, both seats) next to the current release and is released only if it wins
more of them than it loses. A release records the weights and the question text together, so a
rollback restores both. The gate never plays the 132 held-out tasks.

## 2. Quick start

You need Linux, Node.js 22 or newer, Python 3.10 or newer, and an NVIDIA GPU for inference. You
also need the `ra2-game-env` engine SDK, which is supplied privately, and the files of a retail Red
Alert 2 installation. Neither is in the repository; [External inputs](#external-inputs) says where
each comes from. Without them, [Tank Duel](../tank-duel/README.md) is the quickest way to try a
decision model: it needs only Node.js.

```bash
cd integrations/ra2
npm ci
node scripts/setup-dependencies.mjs          # clones Supalosa and Laya at their pins, builds Supalosa
cp .env.example .env && $EDITOR .env && source .env
"$LAYA_PYTHON" scripts/setup-model.py --model typed-decisions
```

`LAYA_PYTHON` is a separate environment with a CUDA build of PyTorch and the Laya revision in
`pins.json`, so the base Honing install gains no model or GPU dependency.

Play one match and audit it:

```bash
npm run match -- --gpu=0 --seat=red --seed=1 --ticks=18000 --out=runs/seed1-red
npm run audit -- runs/seed1-red
```

`--out` must not exist yet. The run directory holds the manifest, `decisions.jsonl`,
`actions.jsonl`, `result.json`, an observer-only `viewer.json`, `match.rplx`, the source snapshot,
and the controller and question files the match used.

To let several matches share one loaded model, start a decision server and point the matches at it
([Using a decision server](#using-a-decision-server)). The whole loop runs as one campaign with no
game, Node, or GPU, which is what CI runs:

```bash
python scripts/e2e/ra2_campaign.py --config configs/e2e/ra2_dry_run.json
```

The other families are in [Using Kev](#using-kev), [Using NanoJev](#using-nanojev),
[Using StartLux-Decision](#using-startlux-decision), and [DiffusionGemma / djev](#diffusiongemma--djev).
Training by hand and gating a candidate are in [End to end](#end-to-end).

## 3. Reference

### This directory

The match runs as its own process, and nothing under `honing/` imports it. Honing reaches it through
the decision server (#52), the episode runner and grader (#53), and the harness (#54). It is based on
the RA2 experiment in #49.

### External inputs

None of these are in the repository. Each is pinned or supplied separately.

| Input | Where it comes from | License or terms |
|---|---|---|
| `ra2-game-env` engine and SDK | supplied privately, set `RA2_SDK_DIR` | restricted to the people it was sent to; do not redistribute the package or the binary |
| Red Alert 2 game files | a complete retail installation, set `RA2_GAME_DIR` | EA's retail license; the engine reads `ra2.mix`, `language.mix`, rules, maps, and theaters from it |
| Supalosa bot | cloned at the commit in `pins.json` | the upstream repository has no license file, so it is cloned at setup and never copied here |
| Laya source | cloned at the commit in `pins.json` | Apache-2.0 |
| Laya weights (`english`, `typed-decisions`) | Hugging Face, revisions in `pins.json` | Apache-2.0 |
| Kev source | cloned at the commit in `pins.json` (`kevSource`) | Apache-2.0 |
| Kev weights (`kev-0.8b`) and its Qwen3.5-0.8B-Base base | Hugging Face, revisions in `pins.json` (`kevModels`) | Apache-2.0 |
| NanoJev source | cloned at the commit in `pins.json` (`nanojevSource`), the one #49 pinned | MIT |
| NanoJev weights (`nanojev`, full Qwen3-0.6B) | Hugging Face, revision in `pins.json` (`nanojevModels`) | the card names no separate weight license: its source is MIT and the Qwen3-0.6B base keeps Apache-2.0 |
| StartLux-Decision source | cloned at the commit in `pins.json` (`startluxSource`) | Apache-2.0 |
| StartLux-Decision weights (`startlux-0.8b`, `startlux-4b`, `startlux-27b`) | Hugging Face, revisions in `pins.json` (`startluxModels`) | CC BY-NC 4.0: non-commercial use only, and commercial use needs a license from StartLux Labs; the tokenizer files they keep from Qwen3.5 stay Apache-2.0 |

Supalosa's own dependencies currently report 11 advisories from `npm audit`, one of them critical. They run only at setup and inside the match process.

### Controller settings and question text

Two files control how the model is asked. Both default to the files in `config/`.

- `--controller=<file>` holds `decisionMode`, `candidateMode`, `decisionInterval`, and `cadence`. A flag such as `--decision-interval=15` overrides the file for one run.
- `--questions=<file>` holds the question and recommendation text, with `{name}` placeholders filled at run time. A replacement file must define every key in `config/questions.json`, or the run stops before it starts.

The manifest records the path and SHA-256 of both files, and the run directory keeps copies. Per-function parameter labels are still in code; #54 moves the rest of the prompt text and turns these files into a Honing `HarnessSpec`.

### Tasks and grading in Honing

`tasks/suite.jsonl` holds 412 tasks built from `tasks/protocol.json`, #49's league protocol:

- `train`: 256 tasks (8 maps, 16 seeds, both seats);
- `val`: #49's development matrix of 24 tasks, which the gate uses;
- `heldout`: 132 tasks (the qualification and sealed cases), which the gate never uses.

Rebuild the file with `python -m honing.backends.ra2.scenarios tasks/protocol.json tasks/suite.jsonl`.

`honing.backends.ra2.RA2EpisodeRunner` plays a task by running this directory's match as a separate process, with the policy served by a decision server. `RA2Grader` then runs the audit and `scripts/verify_replay.py`, which replays the match with no policy. It scores a win the engine ended 1, a draw, tick-limit truncation, or unended match 0.5, and a loss 0, counting wins as #49 did. A failed match, audit, or replay check raises instead of scoring. The grade's feedback summarizes income, units built, killed, and lost, and command counts.

### The harness

In Honing, the two files above are the harness an evolution method rewrites. `honing.backends.ra2.seed_harness` builds a `HarnessSpec` for agent `ra2_decision` from `config/`:

- `ra2/controller.json` (kind `settings`);
- `ra2/questions.json` (kind `instructions`).

`RA2EpisodeRunner` passes them to the match as `--controller` and `--questions`.

Before any match, `LayaQuestionCheck` refuses a harness the model cannot read (rule `questions_fit_model`). It fills placeholders with their longest values, encodes each entry with the model's tokenizer, and checks three limits:
- every option fits Laya's 48-token limit;
- instructions stay within 256 tokens, half the question budget;
- each state entry stays within 512 tokens.

It also refuses missing question keys and invalid controller settings. The cadence belongs to the task, as in #49's gates: the runner always passes the scenario's `--cadence`, so it refuses a harness whose `cadence` differs (rule `cadence_fixed_by_task`) rather than ignore it. With the pinned `typed-decisions` tokenizer, the seed harness's longest option is 14 tokens, its instruction 12, and its longest state entry 105.

Text built from game content can still overrun during a match. The worker then refuses the request rather than truncating it, and `RA2Grader` scores that match 0, with feedback naming the question and its token counts. Any other failure mid-match is still an invalid episode.

### Duels

`integration/models/duel.mjs`, from #49, plays two decision models against each other:
- both seats see normal views frozen before either prediction;
- both have the same cadence and a one-command budget;
- the engine advances only after both choices resolve.

A seat can use a running decision server (`--red-decision-server=URL`, `--blue-decision-server=URL`). `--questions` sets the question text for both seats.

`honing.backends.ra2.DuelEvaluator` is a gate `Evaluator` that plays the candidate against the incumbent on each task, once in each seat. The candidate scores its result and the incumbent one minus it. Both must use the same question text and the default controller settings, which the duel process uses for both seats, so it judges weight changes; a win counts only when the engine ended the match, as in #49; harness changes are judged against the bot.

### Training a decision model

`honing.backends.ra2.RA2DemonstrationProvider` reads labelled matches (teacher demonstrations and DAgger corrections, in #49's directory format) as one `typed_decisions` batch. It keeps every check #49's trainer made: the match must be a train-split scenario, labelled data, not a model evaluation, with unchanged source files, and every row must name the same scenario as its match. `honing.backends.jev.trainer.LocalJevTrainer` then trains Laya on that batch and writes a checkpoint manifest the decision server can load. `parity/` compares both with #49's code.

### Training on game outcomes

`honing.backends.ra2.RA2RolloutProvider` plays tasks with the policy being trained, and turns each model call in the decision logs into a decision. Each decision carries the model's probability for every option, the match's score as its reward, the task as its group, and the release and harness it was played under.

`LocalJevTrainer` then runs `grpo` on those decisions:
- each decision's advantage is its reward against its group, in standard deviations, and groups whose rewards all agree are skipped;
- the update maximizes a clipped probability ratio;
- the starting probabilities are recomputed from the model, because inference may apply temperature calibration that training does not.

The matches must be played with sampled choices. Argmax play repeats exactly, gives no reward spread within a group, and so teaches nothing. The controller cannot sample yet.

### Tests

```bash
npm test                  # controller, staging, logs, question text, HTTP client; no engine or GPU
npm run test:engine       # needs the engine, game files, and a GPU
```

The model code the worker runs is `honing/backends/jev/`, tested with the rest of Honing (`pytest`).

### Using a decision server

`--decision-server=http://host:port` sends every model request to a running Honing decision server instead of starting a worker for this match. Start one with:

```bash
"$LAYA_PYTHON" -m honing.backends.jev.server --pins pins.json --model typed-decisions \
  --model-config-dir .cache/models --laya-source "$LAYA_SOURCE_DIR" --gpu 0 --port 8801
```

It answers one request at a time, so several matches can share one loaded model. The worker and the server run the same model code.

### Using Kev

[Kev](https://github.com/jaredpalmer/kev) is a second decision-model family: a LoRA adapter and pointer head on a Qwen3.5 base. Two sizes are pinned: `kev-0.8b` (the default) and `kev-4b` (select it with `--model kev-4b` on the server, `--model=kev-4b` on a match, or `KEV_MODEL=kev-4b`). Both answer the same typed questions as Laya with the same response fields. `honing.backends.jev.kev` runs them through Kev's own code at the commit in `pins.json`; the two checkpoints record identical Kev source files, so one commit serves both. It needs its own environment (Python 3.12, torch 2.8, transformers 5.x). A kev-4b server held about 13 GB of GPU memory in bf16 on an H100:

```bash
scripts/setup-kev.sh                  # every pinned Kev model; KEV_MODELS=kev-4b for one. Prints the exports
"$KEV_PYTHON" -m honing.backends.jev.server --family kev --pins pins.json --kev-source "$KEV_SOURCE_DIR" --gpu 0 --port 8802
npm run match -- --family=kev --decision-server=http://127.0.0.1:8802 --seat=red --seed=1 --out=runs/kev-seed1-red
npm run match -- --family=kev --gpu=0 --seat=red --seed=1 --out=runs/kev-seed1-red-worker   # or a stdio worker
```

Everything Laya has works with `family=kev`:

| Laya | Kev |
|---|---|
| `LayaDecisionModel` | `KevDecisionModel`: the same four state sections, trimmed by whole entries to fit Kev's context (8,192 state tokens) instead of Laya's 2,048; an overrun is refused with the same wording, so `RA2Grader` scores it 0 |
| decision server, stdio worker | `--family kev`; the worker reads `DECISION_FAMILY=kev` and the `KEV_*` settings, and `KEV_PYTHON` starts it |
| match, duel, audit | `--family=kev` (or `--red-family=kev` in a duel); the manifest records the family, and a server that loaded another family is refused |
| `LayaQuestionCheck` | `KevQuestionCheck`, with Kev's tokenizer and limits (`KEV_LIMITS`) |
| `LocalJevTrainer` (`sft`, `grpo`) | `KevTrainer`: continues the adapter and pointer head as `kev.train --init_from` does, and saves a Kev checkpoint with a `trained-kev` manifest the server loads with `--checkpoint-manifest` |

Each request runs on its own, with no state cached across requests, so an answer does not depend on what came before. `parity/kev_serving.sh` and `parity/kev_training.py` check both against Kev's own code.

### Using NanoJev

[NanoJev](https://huggingface.co/C-Tianyu/NanoJev) is a third family: a full Qwen3-0.6B backbone that scores every option on its own candidate path, with a scalar head and a set attention head. `honing.backends.jev.nanojev` runs NanoJev's own `DecisionPredictor` at the source and model revisions #49 pinned, and handles requests as #49's NanoJev adapter did. Its environment follows NanoJev's own requirements (torch 2.14, transformers 5.17):

```bash
scripts/setup-nanojev.sh              # prints the NANOJEV_PYTHON and NANOJEV_SOURCE_DIR exports
"$NANOJEV_PYTHON" -m honing.backends.jev.server --family nanojev --pins pins.json --nanojev-source "$NANOJEV_SOURCE_DIR" --gpu 0 --port 8803
npm run match -- --family=nanojev --decision-server=http://127.0.0.1:8803 --seat=red --seed=1 --out=runs/nanojev-seed1-red
```

As for Kev, everything Laya has works with `family=nanojev`: `NanoJevDecisionModel`, the server and the stdio worker (`DECISION_FAMILY=nanojev`, `NANOJEV_*` settings, started with `NANOJEV_PYTHON`), match, duel and audit, `NanoJevQuestionCheck` (`NANOJEV_LIMITS`), and `NanoJevTrainer` (`sft`, `grpo`; full weights, or `param_subset` for the heads). The model files are checked against the checksums the model repository publishes, and a candidate path is limited to 2,048 tokens, as #49 ran it; the state is trimmed by whole entries until every path fits. `parity/nanojev_serving.py` compares serving with #49's worker and `parity/nanojev_training.py` checks the trainer.

### Using StartLux-Decision

[StartLux-Decision](https://github.com/StartLuxLabs/Startlux-Decision) is a fourth family: dense Qwen3.5 models that render each question as one prompt with lettered options and read the answer from the letters' next-token logits after one forward pass. Three sizes are pinned: `startlux-0.8b`, `startlux-4b` (the default) and `startlux-27b`. `honing.backends.jev.startlux` runs StartLux's own `StartLuxDecision` at the source commit in `pins.json`, with its prompts, temperatures and CUDA graphs. The environment follows what StartLux was tested with (torch 2.11, transformers 5.8.1, flash-linear-attention and causal-conv1d, which StartLux requires on a GPU and which `setup-startlux.sh` builds with `nvcc`):

```bash
HF_HOME=/disk/with/room scripts/setup-startlux.sh   # 3 GB for 0.8B and 4B, 54 GB more for 27B; prints the exports
"$STARTLUX_PYTHON" -m honing.backends.jev.server --family startlux --pins pins.json --startlux-source "$STARTLUX_SOURCE_DIR" --model startlux-27b --gpu 0 --port 8804
npm run match -- --family=startlux --decision-server=http://127.0.0.1:8804 --seat=red --seed=1 --out=runs/startlux-seed1-red
```

The server, the stdio worker (`DECISION_FAMILY=startlux`, `STARTLUX_MODEL`, `STARTLUX_MAX_LEN`, started with `STARTLUX_PYTHON`), match, duel, audit and `StartLuxQuestionCheck` (`STARTLUX_LIMITS`) work as they do for the other families. There is no StartLux trainer yet, so `--checkpoint-manifest` is refused. On an H100 in bf16 the servers held 3.9 GB (0.8B), 11.5 GB (4B) and 55.7 GB (27B).

Each weight shard is checked against the SHA-256 the model repository published for it. The state is trimmed by whole entries until every question's prompt fits 8,192 tokens (`--max-len`, up to 65,536). StartLux's shared-prefix path is turned off: at the pinned commit it reads the option letters in bf16 when two or more questions share 4,096 or more prompt tokens, so its probabilities do not sum to one. Those requests run as one eager batch instead, which on the 27B takes 2 to 3 seconds for three questions of 6,500 to 9,500 tokens.

The first request at each new prompt length compiles StartLux's linear-attention kernels. Triton caches them on disk, but on a new machine a 27B server's first requests can take several seconds, and the tank duel's 5-second limit failed once for that reason. Warm a fresh server with a few requests of the lengths it will see before a timed match. `parity/startlux_serving.py` compares serving with StartLux's own server.

### End to end

This is the whole loop on RA2 today: serve a model, play and grade tasks, gate a candidate, train, and duel.

`scripts/e2e/ra2_campaign.py` runs the loop as one campaign from a JSON config that names every component by import path. The dry-run config needs no game, Node, or GPU:

```bash
python scripts/e2e/ra2_campaign.py --config configs/e2e/ra2_dry_run.json
```

It plays 4 val tasks through the real runner and grader with stand-in match and check processes, and fake model, trainer, data, and harness method. A reworded harness ties its gate and is rolled back; a train move wins and is published as `r0001`. The release chain, every episode's run directory, and `summary.json` go to `runs/e2e-dry-run`. A config for real matches with the Jev model is the next part of #70; until then, the steps below run each piece by hand. New trainers (Slime, #68) and agents (OpenHands, #69) plug in at the interfaces listed at the end.

CI runs two end-to-end checks on every pull request (the `e2e` job), and both run locally the same way:

```bash
python scripts/e2e/ra2_campaign.py --config configs/e2e/ra2_dry_run.json --work-dir /tmp/campaign
python scripts/e2e/check_campaign.py /tmp/campaign --expect harness="rolled back" --expect train=published
scripts/e2e/serving.sh /tmp/serving    # a decision server process and the match's Node client stack
```

`check_campaign.py` reads what the run wrote: a result for every step, the release chain's records and head, and a result on disk for every gated episode. `serving.sh` starts Honing's decision server with a stand-in model whose answers a client can compute, then sends 25 requests through `decisionClientOptions`, `createDecisionClient`, the permutation client (one binary target question, asked twice), and the HTTP client, and checks every answer, probability, and state text. Neither needs a GPU, the game, or model weights; GPU and in-game runs are reported in the pull request.

#### Environments

| Part | Environment |
|---|---|
| Honing (tasks, runner, grader, gate, trainers' Python side) | repository `.venv`, Python 3.10+, no GPU |
| Match process | Node.js 22, `npm ci`, `RA2_SDK_DIR`, `RA2_GAME_DIR`, `SUPALOSA_DIR` (see [Quick start](#2-quick-start)) |
| Decision server and `LocalJevTrainer` | `LAYA_PYTHON`: CUDA PyTorch, the Laya revision in `pins.json`, and Honing installed (`pip install -e .`) |
| Replay check | `RA2_REPLAY_PYTHON`: a Python with the SDK's `ra2_game_env` package |

The game files and the engine SDK cannot be redistributed; get them as the External inputs table says. Shell commands run from `integrations/ra2` and Python from the repository root, both with `.env` sourced. The Python steps continue one session.

#### 1. Serve the model

```bash
"$LAYA_PYTHON" -m honing.backends.jev.server --pins pins.json --model typed-decisions \
  --model-config-dir .cache/models --laya-source "$LAYA_SOURCE_DIR" --gpu 0 --port 8801
```

Add `--checkpoint-manifest <train-dir>/checkpoint.json` to serve a trained checkpoint. Serve each checkpoint you compare on its own GPU and port.

#### 2. Play one match by hand

```bash
npm run match -- --decision-server=http://127.0.0.1:8801 --seat=red --seed=1 --ticks=18000 --out=runs/seed1-red
npm run audit -- runs/seed1-red
"$RA2_REPLAY_PYTHON" scripts/verify_replay.py --game-dir "$RA2_GAME_DIR" --match runs/seed1-red
```

`viewer.json` in the run directory is an observer-only record of the match.

#### 3. Play and grade tasks through Honing

```python
import asyncio
import json
import os
from pathlib import Path

from honing.backends.jev import JevEndpoints
from honing.backends.jev.laya import laya_snapshot
from honing.backends.ra2 import RA2EpisodeRunner, RA2Grader, load_suite, seed_harness
from honing.core import PolicyEndpoint, PolicyRef, run_graded

ra2 = Path("integrations/ra2").resolve()  # the match process runs in this directory, so paths are absolute
suite = load_suite(ra2 / "tasks/suite.jsonl")
base = laya_snapshot(Path("<path from .cache/models/typed-decisions.json>"))
endpoints = JevEndpoints(
    {base.checkpoint.checksum: PolicyEndpoint("http://127.0.0.1:8801", "tokenizer", "laya", runtime="jev_decision")}
)
runner = RA2EpisodeRunner(suite=suite, endpoints=endpoints, integration_dir=ra2, runs_dir=ra2 / "runs/honing")
replay = [os.environ["RA2_REPLAY_PYTHON"], "scripts/verify_replay.py", "--game-dir", os.environ["RA2_GAME_DIR"]]
grader = RA2Grader(suite=suite, integration_dir=ra2, replay_command=replay)
policy = PolicyRef(base.checkpoint, base.architecture, "master")

record = asyncio.run(run_graded(runner, grader, seed_harness(ra2), suite.tasks("val")[0], policy))
print(record.score, record.feedback)
```

A win the engine ended scores 1, a draw or truncation 0.5, a loss 0.

#### 4. Gate a candidate against the incumbent

Serve both checkpoints (step 1), map both checksums in `JevEndpoints`, then:

```python
from honing.core import HoneRelease, UpdateCandidate
from honing.release import EpisodeEvaluator, Gate, WinLossSelector

incumbent = HoneRelease("r0000", UpdateCandidate("init", None, base, harness=seed_harness(ra2)), None)
candidate = UpdateCandidate("train", "r0000", trained, harness=seed_harness(ra2))  # trained = laya_snapshot(...)
gate = Gate(EpisodeEvaluator(runner, grader, suite.tasks("val", limit=4)), default_selector=WinLossSelector())
evaluation, decision = gate.judge(candidate, incumbent)
```

Use `DuelEvaluator` in place of `EpisodeEvaluator` to play the two checkpoints against each other instead of against the bot.

#### 5. Train

Supervised, on labelled matches in #49's directory format (`parity/make_demos.py` writes a small synthetic set):

```python
from honing.backends.jev.trainer import JevTrainerSettings, LocalJevTrainer
from honing.backends.ra2 import RA2DemonstrationProvider
from honing.core import TrainJobSpec, requirements_for

job = TrainJobSpec("sft", steps=1, contract="typed_decisions")
settings = JevTrainerSettings(
    laya_source=ra2 / "deps/laya", pins_path=ra2 / "pins.json",
    model_record=ra2 / ".cache/models/typed-decisions.json", work_dir=ra2 / "runs/train", gpu="1",
)
protocol = json.loads((ra2 / "tasks/protocol.json").read_text())
batch = RA2DemonstrationProvider(match_dirs, protocol).next_batch(requirements_for(job).data_contract())
result = LocalJevTrainer(settings).train(job, base, batch)
```

`parity/train_honing.py` is the same code as a runnable script. On game outcomes, `RA2RolloutProvider(runner, grader).collect(...)` returns a rollout batch for `TrainJobSpec("grpo", ...)`. It needs sampled play, which the controller does not do yet.

#### Where new components plug in

| Step | Interface | RA2 default | Planned |
|---|---|---|---|
| agent that plays | `EpisodeRunner` | `RA2EpisodeRunner` with the Jev controller | OpenHands playing through MCP tools (#69) |
| model serving | `PolicyEndpoints` | `JevEndpoints` and the decision server | an SGLang endpoint for an LLM agent (#69, #68) |
| grading | `TaskGrader` | `RA2Grader` | reused by #69 |
| training | `Trainer` | `LocalJevTrainer` (`sft`, `grpo` on `typed_decisions`) | `SlimeTrainer` on language-model contracts (#68) |
| training data | `DataProvider`, `RolloutProvider` | `RA2DemonstrationProvider`, `RA2RolloutProvider` | token-level rollouts from the agent's sessions (#68) |
| comparison | `Evaluator` | `EpisodeEvaluator` against the bot, or `DuelEvaluator` | unchanged |

To check a new component, point its slot in a copy of `configs/e2e/ra2_dry_run.json` at your factory (`package.module:function`, called with the run's context and the entry's `args`) and run the script; `tests/test_ra2_e2e.py` does this for a trainer and a runner. With real matches, swap it into step 3 or 4 in place of the default.

### DiffusionGemma / djev

`--family=djev --decision-server=<Honing bridge URL>` uses an existing djev structured-read deployment. See [the bridge setup and validation boundary](../djev/README.md). The adapter preserves upstream probabilities and total-prompt token accounting; it does not load weights or provide a trainer. Match and duel records count structured decision requests rather than GPU forwards.
