# Craftax

Full [Craftax](https://github.com/MichaelTMatthews/Craftax) as a Honing game
scenario. The upstream game supplies the rules, procedural worlds and pixel
art. This integration supplies public observations, action execution, typed
model decisions, recordings, native action replay, task splits, an
`EpisodeRunner` and a `TaskGrader`.

The full game has 43 actions covering movement, combat, crafting, equipment,
magic and dungeon traversal. The committed task suite starts with three
crafting achievements in that full game. Completing those tasks is not a claim
of full-game victory or general roguelike competence.

## Setup

Use a separate Python 3.12 environment. From the repository root:

```bash
python3.12 -m venv integrations/craftax/.venv
integrations/craftax/.venv/bin/python -m pip install -r integrations/craftax/requirements.txt
```

Craftax/JAX are optional and are not dependencies of the base Honing package.
An existing external environment can be used by substituting its Python for
`integrations/craftax/.venv/bin/python`. The supplied pins also support Intel
macOS; newer JAX releases no longer supply Intel macOS wheels.

## Run and watch

Run a seeded random exploration demo, recording the official pixel renderer:

```bash
integrations/craftax/.venv/bin/python integrations/craftax/scripts/match.py \
  --policy random --seed 1 --policy-seed 1 --max-steps 300
```

The last JSON line names the new run directory under `integrations/craftax/runs`.
Open its `replay.mp4` in a video player to watch the game, pause and seek. It
includes the upstream character, terrain, enemies and inventory HUD. No custom
graphics replace the game. Rendering and recording work on a headless Linux VM;
the video can be viewed on the host through a shared directory. The first run
builds a texture cache and compiles JAX functions, so startup takes longer.

`--seed` controls world/game randomness. `--policy-seed` controls only the random
controller. The same pair reproduces the same run; change only `--policy-seed`
to test different random actions in the same initial world. Random demos are
scripted checks, not model evaluation scores.

`--out DIR` must name a new directory. Omitting it creates a unique name. Previous
runs are never overwritten. `--video-fps 10` changes playback speed, not game
rules or inference cadence. Use `--no-video` for headless batches that do not
need visual recordings.

## Choose the actions yourself

Prescribe a JSON list using the action names from the recorded catalogue:

```bash
integrations/craftax/.venv/bin/python integrations/craftax/scripts/match.py \
  --policy actions --actions integrations/craftax/config/example-actions.json --seed 1
```

Without an explicit step budget, a prescribed-action run uses the list's length.
With `--max-steps`, the list must cover that many steps. Native death or task
completion still ends the episode immediately. This is a precise action
sequence, not a human-controlled GUI session.

For manual keyboard play in a graphical desktop session:

```bash
integrations/craftax/.venv/bin/python integrations/craftax/scripts/play.py --save_trajectories
```

This starts the upstream Pygame window: WASD moves, Space interacts/attacks,
Tab sleeps, comma/period changes floors, and number keys craft tools. The
upstream program prints the full controls and saves its own human trajectories
on exit. A terminal-only Multipass guest has no graphical display; use the
recording command there, or run human play in a graphical session. The human
entry point's files are separate from this integration's evaluation records.

## Connect a model

Start an existing Honing decision server as described in
[RA2's model setup](../ra2/README.md#using-a-decision-server), then run:

```bash
integrations/craftax/.venv/bin/python integrations/craftax/scripts/match.py \
  --policy model --decision-server http://127.0.0.1:8080 \
  --seed 2001 --max-steps 300
```

The server must be reachable from the machine running the engine. The game
checks `GET /v1/health`, then posts one typed `action` choice per game step to
`POST /v1/decide`. All 43 native options are presented, including ones whose
prerequisites are not met. The engine applies the chosen action under its own
rules. It does not pre-solve navigation or expose a hidden-state legality mask.

The request uses the existing `id`, `parts`, `questions` shape. The response
must echo the ID and full `stateText`, report no trimmed/truncated context,
and return finite probabilities for every option that sum to one, plus an
argmax choice. Failed requests and invalid replies end the run; no random
or scripted fallback replaces the model. `HONING_TOKEN` supplies an optional
Bearer token without recording it. No weights or training are started here.

The observed 9x11 map is decoded from the native visibility-masked symbolic
observation. Unknown tiles stay `null`. Public character attributes, inventory
and observed entity classes are included; hidden world maps, enemy health and
unidentified potion mappings are not model inputs. Native state hashes are
recorded for replay integrity and are not shown to the model.

## Tasks, harness and grading

`tasks/suite.jsonl` separates train, validation and heldout world seeds. Each
task fixes its world seed, maximum steps and required achievement names.
`--goals COLLECT_WOOD,PLACE_TABLE,MAKE_WOOD_PICKAXE` selects the same introductory
goal set for standalone runs. Any achievement from the pinned full game may
be used in a new task, including dungeon traversal and combat achievements.

The `craftax_decision` harness follows the other games' file delivery:

| Harness path | Kind | Role |
| --- | --- | --- |
| `craftax/controller.json` | settings | Number of recent actions/outcomes in context, from 0 to 32 |
| `craftax/questions.json` | instructions | Text of the `action` question |
| Other settings/instructions files | either | Operating notes shown to the model |

`seed_harness()` reads `config/`. Task seeds, step budgets, goals, game rules,
and the action catalogue cannot be changed by the harness. `CraftaxEpisodeRunner`
resolves the policy's existing `jev_decision` endpoint and starts the separate
match process. Pass an explicit `command` to use an external Python environment;
the default is this integration's `.venv`. `record_video=True` enables native
video for runner-created episodes; the standalone CLI records video by default.

`CraftaxGrader` audits the trace and checks the task scenario, model controller,
and harness/policy identity. The score is **completed required goals / required goals**
in `[0,1]`; partial progress counts even if the player later dies. Engine reward,
survival duration, and full-game victory are separate diagnostics. Random and
prescribed-action demos are refused as model evaluation results.

These two classes satisfy the existing runner/grader protocols and can be used
with `run_graded` and `EpisodeEvaluator`. Craftax-specific model training and a
full train/evolve campaign configuration are not included in this integration.

`honing.backends.craftax.e2e` exposes the `tasks`, `harness_seed`,
`episode_runner` and `grader` factories for the existing JSON component loader.
The surrounding deployment supplies `endpoints`; the runner factory accepts
`python`, `concurrency` and `record_video` arguments. A custom harness can use
the same runner with the existing evaluator and gate.

## Recordings and replay

| File | Contents |
| --- | --- |
| `manifest.json` | Scenario, runtime/platform, controller, model health metadata, harness hashes, clock and video settings |
| `steps.jsonl` | Initial frame and every executed action/result, public observations and native/frame hashes |
| `decisions.jsonl` | Pre-action frame reference, exact model request/reply, applied action and latency |
| `result.json` | Outcome, health, achievements, accumulated engine reward and final frame hash |
| `replay.mp4` | Official pixel video when enabled |
| `first-frame.png`, `last-frame.png` | Native rendered stills when video is enabled |
| `error.json` | Failure or interruption; such a run has no completed result |

The game is turn-based: it waits during inference and advances once per selected
action. Decision latency is measured separately. Reaching task goals can end the
adapter episode before the native game ends; the frame distinguishes native
termination from adapter truncation. Reaching the step cap is not success.

Check artifact consistency without the game runtime:

```bash
python3 integrations/craftax/scripts/verify_replay.py RUN_DIR --audit-only
```

Verify the complete trajectory by executing its actions in a fresh native game:

```bash
integrations/craftax/.venv/bin/python integrations/craftax/scripts/verify_replay.py RUN_DIR
```

Native verification requires the recorded game/JAX/NumPy versions and platform.
It compares every frame and the native state/RNG hashes, calls no model, and
does not modify the recording. Video playback and log auditing alone do not
establish native replay equivalence.

## Checks

```bash
python -m pytest tests/test_craftax.py
```

Contract tests run without game dependencies. Real-engine tests run when Craftax
is installed and otherwise skip. CI has a dedicated optional-runtime job for
the real game and official rendering. No real model win rate is implied by
the HTTP stand-in or random smoke tests.

Upstream Craftax code and assets remain installed from PyPI under their MIT
license; they are not vendored into this repository.
