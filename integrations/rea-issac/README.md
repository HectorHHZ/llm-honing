# REA-Issac

REA-Issac is an independent Isaac-inspired room-combat scenario for Honing.
It provides a deterministic CPU simulation, a real-time browser UI, action
replay, a public state/action interface, seeded tasks, and a model episode
runner using the existing typed decision service. No original game installation,
REA/Ghidra runtime, game assets, GPU or third-party Python packages are needed
for the simulation and UI. A model server is needed only for model play.

This is not the original Isaac engine. REA helped inspect the installed game's
resource extractor; a few factual XML values inform the simulation. Room
layouts, combat behavior, physics, timing, items and all graphics are independent
implementations. See [ANALYSIS.md](ANALYSIS.md) for the source and scope of that work.

## Play with the UI

Run from the repository root on macOS or Linux (Python 3.10+):

```bash
python3 integrations/rea-issac/scripts/play.py --port 4183 --seed 7
```

Open <http://127.0.0.1:4183/>. Use **WASD** to move, **arrow keys** to shoot,
**Space** for bombs and **P** to pause. Movement and shooting are independent.
Select **Watch baseline** to watch the built-in controller. The speed control
changes playback/baseline speed. **New descent** saves the previous run first.
Only one unpaused browser tab should drive a session's clock.

The server binds to loopback. When running in a Linux VM, forward port 4183
through VS Code Remote SSH, then open the forwarded address on the host.

## Run and replay a baseline

```bash
python3 integrations/rea-issac/scripts/match.py --policy baseline --seed 7 --max-ticks 6000
```

The last line prints the new run directory under `integrations/rea-issac/runs/`.
In the UI, select **Open replay** and choose that directory's `replay.json`.
The UI verifies the final state hash before importing it, then reproduces the
recorded actions. No video encoder is needed. To verify the full recording:

```bash
python3 integrations/rea-issac/scripts/verify_replay.py /path/to/run-directory
```

Every run uses a unique directory; explicit `--out` refuses an existing path.
A headless run records `manifest.json`, `harness/`, `steps.jsonl`,
`decisions.jsonl`, `result.json` and `replay.json`. The public state, executed
actions, model requests/replies, latencies and full simulation hashes are
recorded. Ctrl+C stops a CLI run with a partial replay and `error.json`;
incomplete runs do not produce a completed evaluation result. UI runs record
an action log, periodic public states, replay and a result with its save reason.
UI recordings are demonstrations, not graded model trajectories.

## Connect a decision model

Serve a supported model with Honing's existing decision service (see the
[RA2 guide](../ra2/README.md#using-a-decision-server)), then run:

```bash
python3 integrations/rea-issac/scripts/match.py --policy model \
  --decision-server http://127.0.0.1:8080 --seed 7 \
  --max-ticks 6000 --decision-frames 3
```

`GET /v1/health` must report `ready: true` and model metadata.
`POST /v1/decide` receives the standard `id`, `parts`, and `questions` object.
The three choice questions are `move`, `shoot` (NONE/N/NE/E/SE/S/SW/W/NW), and
`bomb` (NO/YES). A reply includes a normalized probability for every option,
its maximizing choice, the matching ID, the complete `stateText`, `trimmed: []`
and `tokenAudit`. The adapter rejects missing options and truncated context;
model errors stop the run rather than substitute a rule action. If the server
requires authentication, set `HONING_TOKEN`; it is not written to the run files.

Each decision holds its action for `--decision-frames` ticks (default 3,
maximum 30). There are 30 simulation ticks per game second. The game clock
pauses during model inference, so inference latency changes wall time, not the
simulated action duration. The model receives only the current room and discovered
map, not hidden rooms or RNG state. Bomb placement is edge-triggered: choose
NO before a later YES to place another bomb.

The model connection is tested with a local protocol stand-in and the actual
simulation. No real model win rate or gameplay capability is claimed yet.

## Use as a Honing scenario

Implementation: `honing/backends/rea_issac/`. The integration follows the
existing `HarnessSpec`, `EpisodeRunner`, `PolicyEndpoints` and `TaskGrader`
contracts without a new core protocol. Component factories are available at
`honing.backends.rea_issac.e2e:{tasks,harness_seed,episode_runner,grader}`.
The application supplies the `endpoints` component with runtime `jev_decision`.

`tasks/suite.jsonl` has disjoint train, validation and held-out seeds. Each task
fixes its seed, tick budget and decision cadence. The score is 1 for defeating
the Warden and 0 for death or timeout. Kills, rooms, health and latency are
reported separately. The grader re-simulates the trajectory and checks model
answers against the actual public states and saved harness. It refuses rule
baseline recordings as model evaluations. Local logs are not signed model-server
attestations; a server operator is responsible for checkpoint identity.

Harness experiments may edit `config/questions.json`, history length in
`config/controller.json`, or add instruction files through `HarnessSpec`.
They cannot change the task budget, action options, physics or scoring. Keep
seeds, cadence and budgets fixed when comparing models or harness versions.
The initial six seeds are a smoke suite, not a benchmark-quality evaluation.

## Public state and actions

Python: `Engine(seed, max_ticks)`, `observe()`, `step(action, frames)`,
`result()` and `digest()`. Start a new engine instance to reset. Terminal states
are `won`, `lost` and `truncated`; subsequent steps do not advance the clock.

The local UI server also exposes:

| Route | Behavior |
| --- | --- |
| `GET /api/state` | Read public state; does not advance time |
| `POST /api/step` | Apply an action and step 1..300 ticks |
| `POST /api/reset` | Start a new seeded run and save the old one |
| `POST /api/controller` | Choose human or rule baseline |
| `GET /api/export` | Save and return a replay |
| `POST /api/replay` | Validate a replay, then enter playback mode |

Example step body:

```json
{"action": {"move_x": 1, "move_y": 0, "shoot_x": 0, "shoot_y": -1, "bomb": false}, "frames": 3}
```

Axes accept only integers -1/0/1; diagonal vectors are normalized. The HTTP
interface is for local interactive tools. Headless model evaluations use the
isolated runner, so a UI tab cannot change their game clock.

## Checks

```bash
python3 -m pytest tests/test_rea_issac*.py
```

Checks exercise real combat/collision rules, seeded determinism, strict action
validation, UI import/export, local HTTP model decisions, completed and failed
runs, harness delivery, component factories, replay tampering and grading.
Replay verification requires matching recorded simulation source/configuration;
the manifest stores their hashes. Retain a code revision with each experiment.
