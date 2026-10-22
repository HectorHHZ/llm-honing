# Laya RA2 v33 harness v1

This experiment compares a harness candidate with the frozen Laya RA2 v33 policy, using the **same model weights on both sides**. H1 changes when the candidate requests inference; H2 adds a short, grounded advisory to its input. There is no training or weight update. Whether either change improves play must be established by completed matches; this directory does not claim a win over v33.

This is a standalone experiment. It does not change the main RA2 runner, implement a Honing `HarnessSpec`, or plug into the current `DuelEvaluator`. The latter evaluates weight changes under a shared harness; this experiment deliberately gives only one seat the candidate harness.

## Harness changes

| ID | Implementation | Change and boundary | Hypothesis |
| --- | --- | --- | --- |
| **H1: inference scheduling while waiting** | `harness/scheduler.mjs`, `run.mjs` | Extend a wait only after the model explicitly chooses the root Wait option, production or deployment is busy, and there are no military units, visible enemies, power problems, stalled production, or useful offered actions. Inspect the seat's normal observation every 3 ticks. Queue changes, completion, damage, and new opportunities can wake the model; a heartbeat bounds the wait at 150 ticks by default. | Reduce repeated inference during unchanged production without missing useful opportunities. |
| **H2: advice about current opportunities** | `harness/context.mjs` | Append at most one English note of 320 characters, separating `Observed` facts from `Advice`. It can suggest responding to nearby visible enemies, placing a ready building, resuming production, repairing a key building, establishing vehicle production, recruiting in parallel, or using idle combat troops. Evidence must come from the current normal observation and offered actions. | Improve production and unit use while preserving the model's choice. |

H1 event wakes may happen before the baseline's next fixed decision. That timing difference is part of the treatment. It replaces the generic `Wait 0.2 seconds; ...` label with `Wait until the scheduler rechecks; existing orders continue`, so the label does not promise an earlier reevaluation. Production-specific wait labels retain their original meaning.

H2 does not remove original context, add candidates, change action bindings, choose a unit type, or issue commands. The frozen staged model still selects the action, actor, and target, including the option to wait. Both sides retain the release's binary target permutation ensemble and command validation. No other model is called, and neither harness reads the opponent's hidden units.

## Comparison contract

- Model: [`jayzou3773/laya-ra2-v33`](https://huggingface.co/jayzou3773/laya-ra2-v33), revision `51d2f5a0b36c0a9921a450f14a5afb7cb6d30685`.
- Weight SHA-256: `2ed46ac94c366db2dc68751d0eabfe945ceaeaf430e9fea1ec9e6c60c0207925`. The unchanged release loader checks this at model startup. Preflight also checks the pinned release checksum manifest and non-weight files, including the controller and tokenizer. `LAYA_MODEL_PATH` must resolve to the same release directory; a different tokenizer or configuration is not a valid fixed-model comparison.
- Laya source: revision `970dc8c5f63d7b886a68409493f37d569424f933`. Keep that external checkout unmodified; setup rejects tracked changes, while the frozen worker checks its Git revision at startup.
- Baseline: the release's frozen `IntentPolicy`, with H1 and H2 disabled. It is not the current controller under `integration/laya/` and is not Supalosa.
- Supalosa source: revision `165b77a71d0cf5ebd27c65b19d0486bcbae78d0f`. Its dependencies supply the game API and map metadata; its bot does not control either player.
- Default profile: both seats have a **30-tick base decision interval**, or 2 game seconds. This is the **v33 CPU d30 comparison**, not the published d3 cadence. Use `--decision-interval 3` to compare at the published cadence, at substantially greater CPU cost.
- Both players are Russians, with 10,000 starting credits, no starting army, normal observations, and the same map, game seed, and tick limit. `--seat` identifies the candidate; the other seat is the baseline.
- The engine advances 3 ticks at a time. The adaptive candidate generates candidates on that grid to support event wakes; H1 controls whether it actually calls the model.
- The seats have independent policy history and exploration memory. One CPU worker serves both, using a model with no cross-request state. Both observations are frozen at the same game tick, predictions run sequentially, and the engine advances only after both resolve. Wall-clock inference latency gives neither side additional game time.

## Prepare the environment

Real matches require **x86_64 Linux**, Node.js 22 or newer, Python 3.12 with virtual-environment support, Git, and the external inputs below. An Intel Mac can run the experiment inside an x86_64 Linux VM. No GPU is required. Help, saved-result reporting, and offline JavaScript tests can run on the host with Node.js 22 or newer.

| Input | Configuration | Source and handling |
| --- | --- | --- |
| Engine and JavaScript SDK 0.1.1 | `RA2_SDK_DIR`, optionally `RA2_ENGINE_BIN` | Privately supplied `ra2-game-env`; obtain separately. Do not commit or redistribute the SDK or binary. |
| Compatible game assets | `RA2_GAME_DIR` | Supply a prepared retail asset directory containing `ra2.mix`, `language.mix`, and the maps, rules, and theaters required by the engine. Setup does not obtain or convert game assets. |
| Frozen v33 release and weights | `LAYA_RELEASE_DIR` | Downloaded at the pinned Hugging Face revision by the setup script. `LAYA_MODEL_PATH` must point to this same directory. |
| Laya source and CPU Python environment | `LAYA_SOURCE_DIR`, `LAYA_PYTHON` | Prepared separately from the main Honing environment by the setup script. |
| Supalosa and its built dependencies | `SUPALOSA_DIR` | Prepare an external clone at the pin above; setup does not install it, and it is never vendored into this experiment. |
| Node.js | `RA2_NODE_BIN` | Explicit executable override, otherwise the portable installation under the lab directory, then `node` on `PATH`. |

From the repository root:

```bash
cd integrations/ra2/experiments/laya-v33-harness-v1
cp .env.example .env
$EDITOR .env
export RA2_ENV_FILE="$PWD/.env"
```

`RA2_ENV_FILE` is an explicit path to a trusted shell configuration. The scripts do not automatically load `.env`. Alternatively, export the settings directly. `.env` is local configuration and must not be committed.

The default external runtime directory is `$HOME/ra2-lab`; set `RA2_LAB_DIR` to use another location. The default SDK path is the ignored `engine/ra2-game-env-0.1.1/js/package` directory under this repository's RA2 integration. Override it if the privately supplied SDK is elsewhere. Existing compatible external installations can be reused; the original sibling experiments are not required.

For a new runtime, install Node.js 22 or newer and provide the private SDK and prepared game assets first. With Node and npm on `PATH`, prepare a new external Supalosa checkout at the required revision:

```bash
export RA2_LAB_DIR="${RA2_LAB_DIR:-$HOME/ra2-lab}"
export SUPALOSA_DIR="${SUPALOSA_DIR:-$RA2_LAB_DIR/supalosa-165b77a}"
mkdir -p "$RA2_LAB_DIR"
git clone --no-checkout https://github.com/Supalosa/supalosa-chronodivide-bot.git "$SUPALOSA_DIR"
git -C "$SUPALOSA_DIR" checkout --detach 165b77a71d0cf5ebd27c65b19d0486bcbae78d0f
(
  cd "$SUPALOSA_DIR"
  npm ci --workspace @supalosa/chronodivide-bot --include-workspace-root --include=dev
  npm run build --workspace @supalosa/chronodivide-bot
)
```

This recipe is for a new directory. If a compatible checkout already exists, reuse it rather than resetting local files. Ensure the paths in the explicitly selected environment file match these installations.

Inspect setup, then opt in to preparing the pinned Laya release, source, and CPU Python environment:

```bash
bash setup-linux.sh --help
bash setup-linux.sh --install
bash run.sh --check
```

Setup requires Git, Python 3.12 with `venv`/pip support, and network access. It does not install Node, Supalosa, the private SDK, or game assets; it does not train, load model weights for inference, or start a game. An already prepared runtime needs only `bash run.sh --check`, not reinstallation. `--check` checks installed files, release identity, and module imports without loading a model or creating a game. The full weight hash is checked later by the frozen loader at model startup.

Running `bash run.sh` or `bash run.sh --help` only displays usage. No match starts without explicit `--run`, and the Python executable is selected by the launcher, so manual virtual-environment activation is unnecessary.

## Run a comparison

First check a short match and the replay path. This is 100 game seconds, generally too short to evaluate strength:

```bash
bash run.sh --run --seat red --ticks 1500 --map mp03t4.map --seed 1
```

For a full comparison, run the same seed once in each seat:

```bash
bash run.sh --run --seat red --ticks 27000 --map mp03t4.map --seed 1
bash run.sh --run --seat blue --ticks 27000 --map mp03t4.map --seed 1
```

Run these sequentially. Both players need CPU inference, so this duel is slower than a model playing a rules bot. There are 15 ticks per game second: 27,000 ticks is a 30-minute **game-time cap**, not 30 minutes of wall time. Progress is printed every 150 ticks; inference can take considerable time between messages.

Repeat both seats with seeds 2 and 3 before drawing an initial conclusion. A single win establishes the outcome of that match, not a stable win rate. Keep the map, tick cap, decision interval, and code snapshot fixed within a comparison. Compare game outcomes alongside inference calls and time; lower inference cost with comparable playing strength is also a useful result.

## Isolate H1 and H2

Each command below **starts a match**. Select the configuration needed for an ablation, then repeat it with the same seeds and seat reversals. The default enables both changes.

```bash
# H1 only: adaptive wait scheduling.
bash run.sh --run --ticks 27000 --scheduler adaptive --advice off

# H2 only: advice with the same fixed decision cadence on both sides.
bash run.sh --run --ticks 27000 --scheduler fixed --advice on

# Neither: frozen policy against itself, useful for checking seat effects.
bash run.sh --run --ticks 27000 --scheduler fixed --advice off

# Both changes, compared against the published d3 decision cadence.
bash run.sh --run --ticks 27000 --decision-interval 3
```

`--max-wait-ticks 150` sets H1's maximum wait and `--stall-ticks 150` sets its production plateau threshold. Each must be a positive multiple of 3, and maximum wait must be at least the base decision interval. These settings are recorded and kept in separate report groups.

## Results and replay

Every match reserves a new directory under this experiment's `runs/`, with seat, map, seed, interval, UTC timestamp, and a random suffix in its name. `--out NAME` uses a chosen directory name under `runs/` and refuses to overwrite an existing one.

| File | Contents |
| --- | --- |
| `match.rplx` | Replay for the existing browser viewer: Replays > Import, then Load. |
| `result.json` | Engine result, `candidateResult`, per-seat inference/action metrics, and final state summary. |
| `manifest.json` | Model identity, H1/H2 settings, comparison conditions, source checksums, seat assignment, and environment. |
| `source/` | Local snapshots of experiment, release, Supalosa, and SDK code used by this run; no model weights. |
| `decisions.jsonl.gz` | Context, advisory evidence, candidates, staged model outputs, and final choice for each actual decision. |
| `scheduler.jsonl` | H1 inference/skip reasons and events at each observation. |
| `actions.jsonl`, `filtered.jsonl` | Command receipts and filtering reasons. |
| `replay.json` | SDK replay status and lost-data markers. |
| `progress.json` | Most recent progress snapshot. |

Keep `runs/`, `checks/`, local environment files, weights, game assets, virtual environments, and downloaded dependencies out of Git. In particular, `source/` snapshots can contain privately supplied SDK code and upstream Supalosa code; the entire run directory is local evidence, not a distributable artifact. Replays are runtime artifacts, not harness source.

The candidate result is `win`, `loss`, `draw`, `unresolved`, or `invalid`. A win requires a real engine terminal state (`ended=true`) with candidate `+1` and opponent `-1`. An engine-ended `0/0` is a draw. A tick-limit cutoff without victory is **unresolved**, never a win. `done=true` alone does not establish a winner. Partial replays or replay data loss invalidate the run.

```bash
bash run.sh --report
```

The report groups compatible results by code snapshot and experiment configuration. It separates wins, losses, draws, unresolved matches, invalid runs, and unfinished directories; reports inference calls and time; and identifies paired, unpaired, or repeated seeds. It does not discard unresolved matches from the denominator or count duplicate seed/seat runs as new pairs. Short matches, long matches, and different ablation settings remain separate. Grouping assumes unchanged external engine binaries and game assets; source/configuration hashes are not a complete asset fingerprint.

`Ctrl+C` interrupts a run. Its directory may lack a complete replay or result file, so it must not be used as a valid outcome. Model or engine failures invalidate the run; the runner does not silently fall back to a rules policy.

## Source and offline validation

| File | Responsibility |
| --- | --- |
| `run.mjs` | Duel loop, seat assignment, H1/H2 integration, logs, replay handling, and outcome classification. |
| `runtime.mjs` | Frozen module imports, dependency checks, unique output directories, and source snapshots. |
| `release-integrity.mjs` | Published manifest and non-weight file validation before loading frozen release code. |
| `cpu-client.mjs` | CPU transport for the frozen release worker; it does not implement a game strategy. |
| `harness/scheduler.mjs` | H1 wait scheduling state machine. |
| `harness/context.mjs` | H2 bounded, evidence-based advisory rules. |
| `env.sh`, `.env.example`, `run.sh` | Portable external runtime configuration and explicit launch modes. |
| `setup-linux.sh`, `download-release.py` | Pinned external dependency preparation. |
| `check.mjs`, `report.mjs`, `tests/` | Read-only preflight, result aggregation, and offline tests. |

Run offline tests with Node.js 22 or newer:

```bash
node --test tests/*.test.mjs
python3 -m unittest discover -s tests -p 'test_*.py'
```

From `integrations/ra2`, `npm run test:harness-v1` runs the JavaScript tests; the normal `npm test` suite also includes them. CI additionally runs the Python downloader tests, shell syntax checks, and formatting checks.

These tests use stand-ins and do not start the real engine, load model weights, or require a GPU. The original v1 implementation passed 68 offline tests before repository migration; the test command above also covers migration checks added here. Passing tests establishes the tested software contracts, not playing strength.

An earlier tokenizer/serializer audit used 2,000 recorded ticks from local d3/d30 runs and reconstructed 694 staged requests plus 8 target-permutation requests. H2 added a note to 60 requests; none exceeded the frozen serializer budget or trimmed optional context, and the largest input used 1,288 of 2,048 state tokens. This was a historical-input compatibility check without model inference or a new match. Its private inputs are not committed here, and it does not guarantee future decisions or win rate. Runtime requests retain the release's strict token-budget checks.

Keep this version available when introducing a new harness treatment. A future `v2` should be a separate, identifiable experiment so that existing results and source snapshots retain their meaning.
