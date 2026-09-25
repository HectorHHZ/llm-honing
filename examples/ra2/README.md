# RA2: Laya versus Supalosa

Run a pinned Laya decision model against the Supalosa bot through the delivered
`ra2-game-env` headless engine. The JavaScript adapter translates the bot's game
API calls into engine actions. A local Python worker scores Laya's choices;
the runner records the exact context, choices, receipts, and replay.

This is a standalone experiment example. It does not implement Honing's
`EpisodeRunner` or change core protocols. The optional [learning loop](league/README.md)
trains local Laya weights and supervises development matches; it remains separate
from the installed Honing campaign API.
The engine package, RA2 assets, upstream bot, and model weights are external
inputs. They are not redistributed here. Supalosa's strategy is unchanged.

## Setup

Use Linux, Node.js 22 or newer, Python 3.10 or newer, and an NVIDIA GPU for model
inference. The CPU tests need no GPU, checkpoint, engine, or game assets.

```bash
cd examples/ra2
npm ci
node scripts/setup-dependencies.mjs
```

The setup script clones the public Supalosa and Laya repositories at the
commits in `pins.json`, then installs and builds Supalosa using its lockfile.
Existing checkouts must already match the pinned commits. You can instead
point `SUPALOSA_DIR` and `LAYA_SOURCE_DIR` at existing pinned checkouts.

Create a separate model environment with your host's CUDA PyTorch build, then
install `requirements-model.txt`. Set `LAYA_PYTHON` to that environment's Python.
The base `llm-honing` installation gains no model or GPU dependencies.

Copy `.env.example` to `.env`, fill in the paths, and source it. `RA2_GAME_DIR`
is the complete asset directory; `RA2_SDK_DIR` is the delivered JavaScript SDK
package directory containing `dist/index.js` and its engine binary. The tested
engine commit is recorded in `pins.json`. The engine's build metadata is saved
in every run manifest. Game assets alone do not provide a renderer or UI.

```bash
source .env
"$LAYA_PYTHON" scripts/setup-model.py --model english
"$LAYA_PYTHON" scripts/setup-model.py --model typed-decisions
```

The two commands download the exact revisions in `pins.json` and save local
path records under `.cache/models`. Inference is offline and rejects model
configuration records or source checkouts that do not match the pins.

## Run and audit

```bash
npm run match -- --model=typed-decisions --gpu=0 --seat=red --seed=1 \
  --decision-interval=3 --cadence=3 --ticks=18000 --out=runs/typed-seed1-red
npm run audit -- runs/typed-seed1-red

npm run match -- --model=english --gpu=0 --seat=red --seed=1 \
  --decision-interval=3 --cadence=3 --ticks=18000 --out=runs/english-seed1-red
```

Run each command from this directory. `--out` is relative to this directory or
an absolute path and must not exist. The default mode is staged intent, the
checkpoint is typed-decisions, and both cadences are 3 ticks. Use
`--decision-interval=15` to reproduce the slower Laya decision schedule.
A decision interval must be a multiple of the engine cadence.

For parallel matches, select separate idle GPUs and output directories. The
worker locks the GPU UUID for this user's workers and refuses a GPU with an
existing compute process. That lock is not a cluster scheduler reservation.
All model failures stop the run; no fallback bot chooses Laya's actions.

## Observation and action boundary

Both seats receive their normal observation, public map/rules, and their own
identity/start. Laya does not receive the opponent's assigned start, resources,
production, commands, observer statistics, or replay hashes. Camera masking
is disabled for both; fog of war remains active. Remembered enemies have an
age and unknown current status. Only currently visible enemies can be direct
attack targets.

Laya chooses one intent, then a function, actor(s), and target. Every follow-up
call carries the previous choices from the same frozen observation. The
simulation advances after all parameters are resolved. A single remaining
candidate needs no additional call; selling and resigning require model
confirmation. Existing orders continue between decisions.

Context includes own economy, queues, construction progress, visible enemies,
recent own effects, and a deterministic strategic recommendation. Candidate
construction uses public rules and observations. It also imposes declared
policy restrictions: harvesters do not receive distant military orders,
no-effect construction-yard Deploy bindings are excluded, and camera moves
are merged with wait. Individual and same-type squad attacks are offered.
Parameter shortlists and omissions are logged. This is an engineered policy,
not unrestricted model control or a learned RTS strategy.

Both sides normally decide every 3 ticks (0.2 game seconds). Laya submits at
most one command per decision; Supalosa retains a 16-command cap. Equal vision
and timing do not imply equal action budgets. Model inference pauses the
simulation for both sides. The local model uses bf16 and deterministic argmax,
with 2048 total tokens and a 512-token question budget. Optional history is
pruned before essential state; option or essential-state truncation is an error.

The macro context also summarizes net credit change, current harvester orders,
production progress stalls and recent observed army counts. Net credits include
spending/refunds and are never labeled mining income. Advice prioritizes restoring
idle or diverted miners, preserving Gather/Dock orders, funding useful queues,
and replenishing an observed army decline. Incomplete unit observations suppress
the replenishment inference. This advice changes model input only; it adds no
automatic mining command and removes no candidate action. Miner diagnosis uses
the current task as a heuristic; it can miss a queued Dock task behind Move
and is not evidence of actual income.

The compatibility facade approximates some native bot API queries, including
placement/connectivity and weapon state. It is not a native Chrono Divide
match. The runner preserves these limitations rather than changing the
upstream strategy to compensate for them.

## Results and verification

Each run saves a source snapshot and hashes, manifest, `decisions.jsonl`,
`actions.jsonl`, `result.json`, an observer-only `viewer.json`, and `match.rplx`.
The audit reconstructs staged decisions and checks action ownership, visible
targets, argmax choices, token preservation, timing, and command receipts.
An accepted receipt means the command was queued, not that it had an effect.

An engine-ended match can yield a win or defeat. Reaching `max_ticks` is
truncation with no winner; privileged scores do not break a tie. With short
game enabled, losing all significant buildings without a base vehicle causes
defeat. The engine also has a stalemate rule, which can defeat both seats.

The paired checkpoint experiment is documented in [results](results/README.md).
The records are development observations, not a reliable win-rate estimate.
Use the engine package's separately supplied `replay_minimap.py` for an offline
minimap; `.rplx` is not an original RA2 client replay.

## Tests

```bash
# No assets or model required:
npm test
npm run format:check
# From the repository root:
python -m pytest tests/test_ra2_example.py

# With RA2_GAME_DIR, RA2_SDK_DIR and SUPALOSA_DIR configured, CPU only:
npm run test:engine
```

The engine suite checks normal-view isolation, staging, parameter coverage,
production controls, ownership/visibility checks, and observed state changes.
Its scripted choices are explicitly test fixtures, not model match results.

## Compressed decision logs

Pass `--decision-log=gzip` to `npm run match -- ...` to write
`decisions.jsonl.gz` directly with bounded stream buffering. The default remains
plain JSONL. The compressed stream must finish successfully before `result.json`
is written; `decision-log.json` records its decoded byte count, row count and
SHA-256. Existing action audits, replay verification and offline labeling read
both formats. This changes log storage only, not observations or model choices.

Staged protocol v1.5 describes paged menus using the parameter being selected.
Actor pages name the first and last owned-unit groups on that page; they no
longer repeat an enemy-target label that can make several pages look identical.
Leaf actions, page ordering and bindings are preserved. Because the model-facing
text changes, compare checkpoints with the exact recorded source snapshot.

Attack target choices also include current normal-view HP, position and the
straight-line distance range from the selected owned units. Every offered
target carries these facts, even when the compact context lists only a few
enemies. Hidden, radar-only and remembered targets contribute no new facts;
geometric distance is not path length or a claim about weapon range. This
changes input descriptions, not available commands or the chosen target.
