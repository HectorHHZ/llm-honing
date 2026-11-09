# Griddly Spiders

Game code is in this directory. Honing runners, graders and model adapters are
in `honing/backends/griddly/`. See [the directory layout](../README.md) for
imports, installation and recording compatibility.

The native Spiders game from Griddly 1.6.7 as a Honing task environment. Move
the gnome to the gem while avoiding moving spiders. The upstream engine owns
movement, collision, random enemies, rewards and termination. This integration
adds structured observations, legal actions, model decisions, replay, task
grading and typed training data.

The scenario is `spiders-full-state-v1`: the model sees the global object map.
The original game's rotating local 7x7 observer is not used as model input.
Relative controls and all game mechanics remain native. This information
condition must be named when reporting results.

## Setup

Use a separate Python 3.10 or 3.11 environment on Windows or Linux x86-64.
The published Griddly version has no Python 3.12 wheel. From the repository
root, on Linux:

The native extension also needs the Vulkan loader (for example Ubuntu's
`libvulkan1` package). Symbolic headless play needs no display or GPU.

```bash
python3.11 -m venv integrations/griddly/.venv
integrations/griddly/.venv/bin/python -m pip install -r integrations/griddly/requirements.txt
```

On Windows, use the Python 3.11 launcher (or substitute an installed Python
3.11 executable) to create that same venv, then run its `Scripts/python.exe`:

```powershell
py -3.11 -m venv integrations\griddly\.venv
& integrations\griddly\.venv\Scripts\python.exe -m pip install -r integrations\griddly\requirements.txt
```

The remaining examples use the Linux interpreter path. Substitute
`integrations\griddly\.venv\Scripts\python.exe` on Windows. Neither the base
Honing package nor the model server needs these game dependencies. The scripts
find the Honing checkout without installing it into the game environment.

`requirements.txt` pins the game and its transitive runtime dependencies.
`pins.json` identifies the native GDY and its SHA-256 after normalizing CRLF to
LF; the Windows and Linux wheels package its line endings differently.
The adapter refuses another game version, changed rules, or changed action
mapping. It implements the 1.6.7 four-return-value step API, not develop's API.

## Python engine API

Import `GriddlyEngine` from `integrations.griddly.engine` and
`GriddlyScenario` from `honing.backends.griddly.scenarios`. Run this API in the
separate game environment. Construction loads the native engine but does not
start an episode. Call `reset()` before reading or advancing it. One instance
owns one native environment; use separate instances for concurrent episodes.

### Construction and lifecycle

`GriddlyEngine(scenario: GriddlyScenario, *, render: bool = False)` requires a
scenario. `GriddlyScenario()` has these defaults and validation rules:

| Argument | Type and default | Meaning and accepted values |
|---|---|---|
| `seed` | `int = 1` | Native game RNG seed, `0 <= seed < 2**32` |
| `level_id` | `int = 0` | Upstream Spiders layout, `0..4` |
| `max_steps` | `int = 100` | Episode budget in native ticks, `1..10000` |
| `observation` | `str = "spiders-full-state-v1"` | Only this global public-object condition is supported |

Boolean values are refused for integer arguments. Invalid scenario values
raise `honing.core.ContractError`. The constructor raises `RuntimeError` for
missing or mismatched pinned dependencies, a changed rule hash or incompatible
action mapping. Native loader, filesystem and engine exceptions propagate.
No game package is imported merely by importing the adapter module.

| Method | Arguments and return value | Preconditions, effects and errors |
|---|---|---|
| `reset()` | No arguments; returns a frame `dict[str, Any]` described below | Starts a seeded episode at the configured level, with step/tick 0 and reward 0. Reseeds and clears prior episode history. May be called again during or after an episode. A closed engine raises `RuntimeError`. |
| `observe()` | No arguments; returns a JSON deep copy of the latest frame | Requires a reset, open engine; otherwise `RuntimeError`. Reads without advancing ticks or RNG. |
| `step(action: str)` | One exact action name; returns the resulting frame | Requires a reset, open, nonterminal engine; otherwise `RuntimeError`. `action` must occur in the current `legal_actions`, or `ValueError` is raised without advancing. Each accepted action advances exactly one native tick, including WAIT. |
| `pixels()` | No arguments; returns a copied NumPy `ndarray[uint8]` with shape `(height_px, width_px, 3)`, RGB channels `0..255` | Requires construction with `render=True`, a reset and an open engine; otherwise `RuntimeError`. Uses the upstream global Vector2RGB renderer. Reading does not advance time and is allowed on the final frame. Pixel dimensions depend on the grid and renderer. |
| `close()` | No arguments; returns `None` | Releases the native environment. Repeated calls are harmless. The instance cannot be reopened. |

The engine also supports `with GriddlyEngine(...) as engine:`; leaving the
context calls `close()` even if the body raises. It does not call `reset()`
automatically. After any non-running status, `step()` fails until a new
`reset()`. There is no automatic reset or continuation from a saved frame.
Treat the frames returned by `reset()` and `step()` as read-only; use
`observe()` when a separate mutable copy is needed.

### Frame schema and units

`reset()`, `observe()` and `step()` expose the same fields:

| Field | Type | Meaning |
|---|---|---|
| `step` | `int` | Number of actions applied since reset; starts at 0 |
| `observation` | `dict[str, Any]` | Global public object state, detailed below |
| `legal_actions` | `list[str]` | Current native legal action names in the fixed order WAIT, TURN_LEFT, FORWARD, TURN_RIGHT, BACKWARD; empty after any non-running status |
| `reward` | `float` | Native reward from this action, not accumulated reward; reset gives `0.0` |
| `status` | `str` enum | `running`, `won`, `lost`, `native_terminal` or `step_limit` |
| `terminated` | `bool` | The native Griddly `done` flag |
| `truncated` | `bool` | True only when the adapter stops at `max_steps` with status `step_limit` |
| `native_state` | `dict[str, Any]` | JSON snapshot returned by upstream `get_state()`, for audit/replay; includes internal objects, hashes and queued actions. Its nested schema belongs to the pinned Griddly runtime and it must not be sent to a decision model. |
| `events` | `list[dict[str, Any]]` | The action's native `info["History"]`, or an empty list; nested event fields belong to the upstream runtime |
| `frame_hash` | `str` | SHA-256 hex digest of canonical JSON for all other frame fields |

`observation` contains:

| Field | Type | Meaning |
|---|---|---|
| `grid` | `dict[str, int]` with `Width`, `Height` | Grid dimensions in cells |
| `objects` | `list[dict[str, Any]]` | Public objects sorted by canonical JSON; native names starting with `_` are excluded |
| `tick` | `int` | Native `GameTicks`; ticks, not milliseconds |
| `npc_period_ticks` | `int` | `3` for the pinned spiders' random-motion schedule |
| `coordinates` | `str` | Declares `[x,y]`, right/down positive, absolute orientation and relative actions |

Each public object has `Name: str`, `Location: list[int]` of length 2,
`Orientation: str` (`NONE`, `UP`, `RIGHT`, `DOWN`, `LEFT`), `PlayerId: int` and
`Variables: dict[str, int]`. Coordinates are grid cells from the upper-left
origin, with x increasing right and y increasing down. `NONE` uses the native
UP default for relative actions. The model request uses public observations
and legal actions, excluding `native_state` and internal event information.
There is no wall-clock simulation during inference or observation calls.

`won` means native termination with a surviving gnome and no gem; `lost`
means the gnome is absent. `native_terminal` covers other native endings.
`step_limit` means the configured action budget was exhausted without one of
those outcomes. Otherwise the status is `running`. Native win/loss/end
classification takes priority over reaching the budget on the same action.
The two flags are deliberately distinct: a normal budget stop has
`terminated=False, truncated=True`; a native terminal at the budget keeps its
native outcome and has `truncated=False`. End loops on `status != "running"`
and retain both flags in recordings.

### Minimum successful and error sequences

This complete sequence initializes, observes, advances and closes a game with
no model server. The rule controller selects from public state; the reported
outcome remains a native game outcome, not a learned-model result.

```python
from integrations.griddly.engine import GriddlyEngine
from honing.backends.griddly.model import baseline
from honing.backends.griddly.scenarios import GriddlyScenario

with GriddlyEngine(GriddlyScenario(seed=7), render=True) as engine:
    frame = engine.reset()
    assert engine.observe() == frame
    while frame["status"] == "running":
        frame = engine.step(baseline(frame))
    rgb = engine.pixels()
    print(frame["status"], frame["step"], rgb.shape)
```

This error sequence catches the expected exceptions, then resets and closes
normally. It shows that a budget ending does not allow another action:

```python
from integrations.griddly.engine import GriddlyEngine
from honing.backends.griddly.scenarios import GriddlyScenario

engine = GriddlyEngine(GriddlyScenario(seed=7, max_steps=1))
try:
    try:
        engine.observe()
    except RuntimeError:
        pass  # reset is required first
    engine.reset()
    try:
        engine.step("INVALID_ACTION")
    except ValueError:
        pass  # no native tick was applied
    frame = engine.step("WAIT")
    assert frame["truncated"] and not frame["terminated"]
    try:
        engine.step("WAIT")
    except RuntimeError:
        pass  # the episode has ended
    assert engine.reset()["step"] == 0
finally:
    engine.close()
    engine.close()  # idempotent
```

## Run an episode

The project-local Windows launcher runs a game and opens an English HTML
replay after it ends:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File integrations\griddly\run.ps1 -Mode demo -Open
```

Run these commands from the repository root. The launcher also supports
`server`, `model` and `replay` modes. Game output is written to a fresh
`integrations/griddly/runs` directory.

```bash
integrations/griddly/.venv/bin/python integrations/griddly/scripts/match.py \
  --policy baseline --level 0 --seed 7 --max-steps 100
```

The last JSON line gives a new directory under `integrations/griddly/runs/`.
`--out DIR` can name another new directory; existing directories are refused.
Controllers are:

| Policy | Behavior |
|---|---|
| `random` | Uniformly sample the currently legal actions |
| `baseline` | Shortest-path heuristic using only the public map and current enemy cells |
| `teacher` | The same rule heuristic, with explicit supervised labels and exact questions |
| `actions` | Execute a JSON list supplied with `--actions FILE` |
| `model` | Use an existing Honing typed decision server |

The rule teacher is not an LLM teacher or an oracle. It does not simulate future
enemy actions and can fail. Baseline and scripted runs are not model grades.

The native action names are `WAIT`, `TURN_LEFT`, `FORWARD`, `TURN_RIGHT` and
`BACKWARD`. Forward/backward are relative to facing; initial `NONE` facing uses
the native UP default. `WAIT` advances time, including enemy actions. An action
list must cover `--max-steps`; native victory or death can stop it earlier.
Illegal actions are refused rather than silently replaced.

`--seed` controls the game RNG; `--policy-seed` controls random or sampled model
choices. Game time advances exactly once per action and pauses during inference.
Latency is recorded separately. The step budget is a task property.

## Watch a recording

For a labeled, step-by-step replay with player/goal/enemy symbols, action
descriptions and recorded model probabilities, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File integrations\griddly\run.ps1 `
  -Mode replay -RunDir RUN_DIR -Open
```

This audits the saved records and writes a standalone `replay.html` inside the
run directory. It renders saved public object positions and executed actions;
it does not simulate gameplay. It needs no running model server. The Windows
`demo` and `model` modes also generate this page after an episode completes.

Install the optional encoder, then request a video:

```bash
integrations/griddly/.venv/bin/python -m pip install -r integrations/griddly/requirements-render.txt
integrations/griddly/.venv/bin/python integrations/griddly/scripts/match.py \
  --policy baseline --seed 7 --record-video
```

Open `replay.mp4` in the reported run directory. It uses Griddly's own
`Vector2RGB` visualization of the global object channels, not sprite graphics.
Its 10 FPS playback rate does not change game steps or decisions.

The 1.6.7 native Vulkan Sprite2D renderer produced an access violation at
process shutdown on the tested Windows runtime, even after writing a playable
video. This integration therefore uses the upstream vector renderer for
recording. Symbolic simulation and action replay need no Vulkan display.

## Connect a decision model

Start a supported model behind Honing's existing decision server, as described
in [RA2's model setup](../ra2/README.md#using-a-decision-server), then run:

```bash
integrations/griddly/.venv/bin/python integrations/griddly/scripts/match.py \
  --policy model --decision-server http://127.0.0.1:8080 --seed 7
```

The integration checks `GET /v1/health` and posts one `action` question to
`POST /v1/decide`, using the existing `id`, `parts`, `questions` format.
The criteria contain exactly the current native legal actions. All option
probabilities must be finite, lie in `[0, 1]` and cover every legal option.
The reply must preserve
the request ID and complete state text and report no context trimming.
`tokenAudit` must cover exactly the submitted questions. Each entry must report
Boolean `stateTruncated: false` and an integer `options` count equal to the
offered choices. `stateTokens`, `stateRoom` and `inputTokens` must be present;
explicit `null` counts are permitted for unavailable accounting, including DJEV.
Available counts, including optional `maxInputTokens`, must be nonnegative
integers, not booleans. For nonempty state, reported zero counts are rejected.
State tokens must fit the reported room and input length; reported input length
and state room must fit an available maximum. Missing, malformed or contradictory
audits fail before native execution. Replay and training behavior-probability
export use the same validation. These checks verify the server's report; they
do not independently tokenize the model input.
The pinned Laya source rounds each probability to four decimal places. The
adapter accepts a sum error of at most `N * 0.00005 + 1e-12`, where `N` is
the number of legal options; the last term covers floating-point comparison.
A larger error is refused. Accepted values are divided by their sum (computed
with `math.fsum`) in native legal-action order. Evaluation preserves the
server's maximizing choice, including ties. Sampling, audit reconstruction
and exported training `behavior_probs` all use this normalized distribution.
The raw response probabilities are retained unchanged in `decisions.jsonl`;
they are not claimed to be the model's pre-rounding softmax values.

Model/network failures stop the run; there is no scripted fallback.
When configured, `HONING_TOKEN` is forwarded as a Bearer header and is not recorded.
Configure authentication at the serving deployment.

### Use the repository's NanoJev model

NanoJev is already supported by this repository. Reuse
`honing.backends.jev.nanojev.NanoJevDecisionModel` and the model/source pins in
`integrations/ra2/pins.json`: `C-Tianyu/NanoJev`, revision
`047b927b30882a1138fc504821b82ac145a4b81a`, on Qwen3-0.6B. Its trainer is
`honing.backends.jev.nanojev_trainer.NanoJevTrainer`, with SFT and GRPO support.
The RA2 directory holds shared model setup; Griddly does not require the RA2
game, Node match process, SDK or retail assets.

Run the model setup and service in Linux or a suitable WSL GPU environment.
The existing model loader imports `fcntl` and uses POSIX GPU locking, so it
does not load on native Windows. The game process can run on Windows and
connect to that service over HTTP. From the repository root in a Linux
checkout (shell scripts must have LF line endings):

```bash
bash integrations/ra2/scripts/setup-nanojev.sh
export NANOJEV_PYTHON="$PWD/integrations/ra2/.cache/nanojev-env/bin/python"
export NANOJEV_SOURCE_DIR="$PWD/integrations/ra2/deps/nanojev"
"$NANOJEV_PYTHON" -m honing.backends.jev.server \
  --family nanojev --pins integrations/ra2/pins.json \
  --nanojev-source "$NANOJEV_SOURCE_DIR" \
  --nanojev-inference reference --gpu 0 --max-len 8192 --port 8803
```

Then, in another terminal with the game environment, run:

```bash
integrations/griddly/.venv/bin/python integrations/griddly/scripts/match.py \
  --policy model --decision-server http://127.0.0.1:8803 \
  --level 0 --seed 1001 --max-steps 100
```

The project-local service wrapper accepts explicit Linux paths:

```bash
bash integrations/griddly/scripts/serve_nanojev.sh \
  --python /path/to/model-env/bin/python --source /path/to/NanoJev
```

It also reads `NANOJEV_PYTHON` and `NANOJEV_SOURCE_DIR` from its environment,
including an existing `/etc/profile.d/honing.sh` when present. It does not
require a particular model installation directory.

On Windows, open two terminals at the repository root. Start the server in
the first terminal, using paths inside the selected WSL distribution:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File integrations\griddly\run.ps1 -Mode server -WslDistribution Ubuntu -NanoJevPython /path/to/model-env/bin/python -NanoJevSource /path/to/NanoJev
```

Keep that terminal open after `"ready": true`. The model-path options may be
omitted when already configured in the WSL environment. `Ubuntu` is the
default distribution and can be changed with `-WslDistribution`. In the
second terminal, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File integrations\griddly\run.ps1 -Mode model -Seed 1001 -Open
```

Use the reachable service URL when the game and model run in different
environments. The service defaults to loopback; configure `--host` and
`HONING_TOKEN` when making it available across hosts.

The 8,192-token setting is a starting configuration, not a measured Griddly
capacity result. The global map can be large. Measure every native candidate
path with the pinned tokenizer before comparing scores or training. A request
that exceeds the limit must fail without removing objects or legal options.
The setup downloads model weights into a separate model environment. A local
Ubuntu/WSL environment has now loaded the pinned NanoJev and served one native
Windows game: level 0, seed 1001, 100 steps. The model chose WAIT throughout
and reached the step limit. Replay passed. This checks real model integration;
it is not a fixed-suite baseline or training result.

### Observation and action contract

The state text contains global object classes, positions, orientations,
public object variables, tick and recent outcomes. RNG, internal future
`DelayedActions`, native hashes and synthetic boundary objects are excluded.
This is a global object observation, not a complete simulator checkpoint or
a guarantee of Markov observability. Legal actions can still be dangerous.
There is no path hint, safety projection or automatic dodge in model play.

Evaluation uses the server's argmax choice. Add `--sample --policy-seed N` for
training rollouts sampled from the returned distribution. Logs keep the
server reply and sampled/executed choice separately. A singleton legal action
is executed and logged as forced, without a model request or training sample.

## Honing tasks and scoring

`tasks/suite.jsonl` fixes native level, seed, budget and observation condition:

| Split | Native levels | Tasks |
|---|---|---|
| train | 0 and 1 | 8 |
| val | 2 and 3 | 8 |
| heldout | 4 | 4 |

Levels and seeds are disjoint between splits. These are only five native
layouts of different sizes, so this is an integration suite, not a broad
generalization benchmark. A new suite must keep the native rule pin and name
any different observation condition explicitly.

`GriddlyEpisodeRunner` resolves a `jev_decision` endpoint and starts a separate
game process. Supply `command` to select another game Python. The default
chooses `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` elsewhere.
`GriddlyGrader` audits the records, then invokes replay in the game environment.
Only reaching the gem scores 1; death or the step limit scores 0. Engine reward,
latency and steps are diagnostics. Scripted/teacher runs cannot become model
evaluation grades. Checkpoint identity is supplied by the serving deployment;
local logs are not signed server attestations.

The backend implements the existing `HarnessSpec`, `EpisodeRunner`,
`PolicyEndpoints`, `TaskGrader`, `DataProvider` and `RolloutProvider` contracts.
No core protocol changes or new model server are required. Factories are in
`honing.backends.griddly.e2e` for `tasks`, `harness_seed`, `episode_runner`,
`grader`, `demonstrations`, `rollouts` and `delivery_probe`.

The harness contains `griddly/controller.json` with history length (0..8),
`griddly/questions.json` with the action instructions, and optional operating
notes. It cannot change task budget, rules, observation condition or legal
action construction. GEPA may evolve instructions through the existing
framework; a Griddly GEPA experiment has not been run.

## Replay and records

The integration-local `griddly-match-v1` format is accepted in
[RFC #148](https://github.com/HectorHHZ/llm-honing/issues/148#issuecomment-6623228995).
It does not establish the
shared Reef record/data format discussed in
[#137](https://github.com/HectorHHZ/llm-honing/issues/137).

Every completed run contains:

- `manifest.json`: runtime/platform, native rule and adapter hashes, scenario,
  controller, seeds, model metadata and harness identity;
- `source/` and `harness/`: the adapter source and delivered harness files;
- `steps.jsonl`: reset and every action's resulting public/native state,
  legal actions, reward, termination and events;
- `decisions.jsonl`: exact requests/replies, pre-action frame, chosen/applied
  action, forced-action marker and latency;
- `result.json`: outcome and accumulated native reward;
- `replay.mp4` when requested, or `error.json` on failure/interruption.

Audit without importing the optional game, or verify with native re-execution:

```bash
python integrations/griddly/scripts/verify_replay.py RUN_DIR --audit-only
integrations/griddly/.venv/bin/python integrations/griddly/scripts/verify_replay.py RUN_DIR
```

The audit reconstructs every model question from public state, legal actions
and saved harness. Replay resets a fresh native engine with the same level
and seed and compares every state, reward and termination. Recorded sampling
is checked against the saved probabilities and policy seed. Verification
requires the original adapter source, runtime versions and platform. Preserve
the source snapshot or experiment commit; cross-platform bitwise equivalence
and mid-episode snapshot continuation are not promised.

## Training data

`GriddlyDemonstrationProvider` reads explicit `--policy teacher` runs and
native-verifies them before returning `typed_decisions`. Each scenario must
match exactly one train task. It refuses model evaluations and val/heldout
teacher runs. Generate teacher runs using the exact seed, level and budget
from the training suite, rather than the CLI's unrelated default seed.

`GriddlyRolloutProvider(runner, grader)` samples each train task at least twice,
grades each replay, and returns `DecisionSample` records with reward, behavior
probabilities, task group, source release and harness. It can feed the existing
NanoJev/Laya/Kev GRPO trainers. Argmax evaluation runs are not rollout data.

Real model training has not been run for this integration. Set the training
backend's internal validation-group pattern explicitly: task groups are named
`spiders-train-l0-s1001`, not RA2's seed-101 format. Keep outer val/heldout out
of training. The existing trainer recomputes starting probabilities and uses
decision-weighted group statistics; neither training behavior is changed here.

## Checks

```bash
integrations/griddly/.venv/bin/python -m pip install pytest
integrations/griddly/.venv/bin/python -m pytest tests/test_griddly.py
integrations/griddly/.venv/bin/python scripts/e2e/check_runner.py \
  --config configs/e2e/griddly_runner_check.json
```

Contract tests run without Griddly; native tests opt in when it is installed.
The runner check uses the real native game and decision HTTP server with a
public-state rule stand-in. It checks interfaces and delivery, not learned
model capability. The optional recording path is included in the native
Linux CI job.


### Local verification

An isolated candidate containing the repository baseline plus only this
integration's changes was checked on Ubuntu 24.04 with Python 3.11.18:

- 55 Griddly tests and 55 core contract tests passed, including native vector video
  and rounded HTTP replies through sampling, replay and training conversion.
- The runner check passed support, episodes, delivery, concurrency, repeated
  event loops and Gate behavior.
- Black, isort, Ruff and mypy checks passed for the Griddly backend and its
  associated Python scripts/tests.
- A rule-policy episode reached the gem in 8 steps and passed native replay.

The Windows launcher, English replay controls, timeline and recorded model
probabilities were also checked. Explicit WSL model-path options loaded the
pinned NanoJev and completed a three-step connection check. These checks use
a separate game environment and do not imply a learned-policy benchmark.
The pull request records repository-wide test and hosted CI results.

The upstream engine and resources remain installed from PyPI; they are not
vendored into this checkout. Upstream code is MIT-licensed. Use the upstream
resource notices when redistributing game artwork.
