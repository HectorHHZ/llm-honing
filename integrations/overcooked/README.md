# Overcooked-AI

Run native [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai)
through Honing's existing typed decision interface. Two chefs collect
ingredients, fill pots, cook, plate soup and serve it for a shared score.
The pinned upstream engine owns movement, collisions, interactions, recipes,
rewards and rendering. The GUI displays its official PNG sprites.
All integration code lives here; the root wheel and core protocols are unchanged.

## Why this game and what comes next

This adds cooperative spatial coordination and multi-stage resource handling
alongside the repository's adversarial combat, navigation and survival games.
Narrow passages, shared stations, held objects and cooking delays make useful
coordination depend on the map and both chefs' states. This can test transfer
to another task category. Integration success alone proves no generalization.

The first post-training objective is:

> With chef 1 fixed to the autonomous greedy (`solo`) rule policy, can NanoJev
> control chef 0 from the map and current state to maximize team score over
> 400 native ticks?

Only chef 0 is model-controlled. Chef 1 is an independent, fixed rule teammate.
One game step applies **both chefs' actions together**. Inference consumes no
game ticks. Use manual cooking (`old_dynamics=false`) and compare candidates
under the same teammate, budget, layout and rules. Report raw team scores and
a matched chef-0 `STAY` control: the teammate can score alone. Compare a
rule-controlled chef 0 separately; do not attribute all team reward to the model.

First improve this fixed-teammate setting through post-training. Then extend
to other teammate strategies, seats and layouts. Keep training, validation
selection and heldout evaluation separate. The three-layout suite here is an
integration smoke suite, not a sufficient generalization benchmark.

GEPA and Meta-Harness code, protocols and results remain on
`experiment/overcook_gepa` and `experiment/overcook_meta-harness`. Earlier pilot
and SFT artifacts also remain in those branches and are excluded here.
This feature PR makes no training-improvement claim.

## Setup and play

The verified game runtime is Windows x86-64 with Python 3.10, the version
declared by the pinned upstream revision. From the repository root:

```powershell
.\integrations\overcooked\setup.ps1 -Conda 'D:\anaconda3\Scripts\conda.exe'
.\integrations\overcooked\run.ps1 -Mode play -Open
```

Substitute your conda executable. Setup creates an isolated prefix at
`integrations/overcooked/.venv`, installs `requirements-lock.txt`, downloads the
Git revision in `pins.json`, and checks its native code/resource hash. The
native package is editable with `--no-deps --no-build-isolation`. Environments,
upstream downloads, caches and runs stay in this directory and are ignored.
Existing conda environments are unchanged. `requirements.txt` lists direct
dependencies; the lock records the tested Windows environment. Linux game
setup has not been validated. This is a source-checkout integration.

Open <http://127.0.0.1:4185/>. Defaults: `cramped_room`, chef 0, `solo` teammate,
400 ticks, manual cooking. Select **Rule baseline** and **Start** for scripted
play; select **Human control** for WASD/arrows and Space to interact. Use
**Pause**, **Single step**, the state/map/rules tabs and **Export recording**
to inspect play. Scenario settings take effect on **New game**. Ctrl+C stops
the foreground service. Use `-Port 4186` if needed. Rule play is a runtime
check, not a learned-model result.

Five layouts are supported: `cramped_room`, `asymmetric_advantages`,
`coordination_ring`, `forced_coordination`, `counter_circuit_o_1order`.

## Python observation and control API

Import `Engine` and `Scenario` from `integrations.overcooked.engine` in the game
environment. Module import needs only the standard library; construction loads
native dependencies. `Engine()` uses `Scenario()` and resets immediately to
tick 0. Each instance owns one episode. `with Engine(...) as game:` closes it
on exit. A finished engine can reset; a closed instance cannot reopen.

| Scenario argument | Default | Accepted values |
| --- | --- | --- |
| `layout` | `"cramped_room"` | One of the five names above |
| `max_steps` | `400` | Integer `1..10000` |
| `old_dynamics` | `False` | Boolean; false requires manual start, true starts at three ingredients |
| `controlled_player` | `0` | Integer `0` or `1` |
| `partner` | `"solo"` | `solo`, `role`, `stay`, `solo_slow` |
| `target_deliveries` | `2` | Integer `1..100`; diagnostic success target, no early stop |
| `score_scale` | `40.0` | Finite positive number; fixed task grading scale |

Booleans are refused for integer/number settings. Invalid settings raise
`ValueError`. Missing dependencies propagate import errors. Changed native
code/resources raise `RuntimeError` with a setup instruction.

| Method | Return and effects | Errors |
| --- | --- | --- |
| `reset()` | Frame dictionary at tick 0; clears score/events | Closed: `RuntimeError` |
| `observe()` | Detached JSON-compatible frame dictionary; no tick advancement | Closed: `RuntimeError` |
| `game_map()` | Detached map dictionary; read-only, including after close | Native errors propagate |
| `rules()` | Resolved rules dictionary; read-only, including after close | Native errors propagate |
| `step(actions)` | List/tuple of exactly two action names in chef-0, chef-1 order; advances one native tick and returns its frame | Bad pair/action: `ValueError` without stepping; closed/finished: `RuntimeError` |
| `png(frame=None)` | Official PNG `bytes` for live or supplied historical frame; no tick advancement | Live read after close: `RuntimeError`; invalid historical data/rendering errors propagate |
| `close()` | `None`; marks the instance closed; repeated calls are harmless | None on repeated close |

Supply only unmodified frames from the same engine/layout to historical
rendering. Observation, rendering and model inference never advance time.

```python
from integrations.overcooked.engine import Engine, Scenario
from integrations.overcooked.policy import joint_baseline

with Engine(Scenario(partner="solo", max_steps=400)) as game:
    frame = game.observe()
    kitchen, rules = game.game_map(), game.rules()
    assert game.observe() == frame
    while frame["status"] == "running":
        frame = game.step(joint_baseline(frame, partner="solo"))
    print(frame["step"], frame["score"], len(game.png()))
```

Minimum error/lifecycle sequence:

```python
from integrations.overcooked.engine import Engine, Scenario

game = Engine(Scenario(max_steps=1))
try:
    before = game.observe()
    try:
        game.step(["INVALID", "STAY"])
    except ValueError:
        assert game.observe() == before
    game.step(["STAY", "STAY"])
    try:
        game.step(["STAY", "STAY"])
    except RuntimeError:
        pass
    assert game.reset()["step"] == 0
finally:
    game.close()
    game.close()
```

### Observation fields and units

The condition is `overcooked-full-kitchen-v1`: the entire public kitchen is
visible; the teammate's pending action is hidden. Coordinates are integer
cells `[x,y]`, origin upper-left, positive right/down; `terrain[y][x]`.
Time is in native ticks, not seconds. Every frame has:

| Field | Type and meaning |
| --- | --- |
| `protocol` | String `overcooked-full-kitchen-v1` |
| `map` | Dictionary: layout string, integer width/height, terrain character lists, string legend, coordinate description, walkable/start `[x,y]` lists |
| `rules` | Dictionary: Boolean dynamics flag, integer max ingredients, recipes (ingredients, integer cook ticks, numeric values), order bonus, six actions and English instructions |
| `state` | Native JSON snapshot: `players`, `objects`, `all_orders`, `bonus_orders`, integer `timestep` |
| `state.players` | Two dictionaries: integer position/facing pairs and held-object dictionary or null |
| `state.objects` | Object dictionaries with name/position; soup includes `_ingredients`, `_cooking_tick`, `cook_time`, `is_idle`, `is_cooking`, `is_ready` |
| `pots` | Native category mapping to lists of pot `[x,y]` positions |
| `score`, `reward` | Numbers: cumulative team sparse reward and latest joint reward |
| `deliveries`, `delivery_attempts` | Integers: rewarded soups and all native deliveries, including zero-reward disallowed recipes |
| `events` | Latest native event-name mapping to two per-chef Booleans; diagnostic |
| `step`, `max_steps` | Integers: elapsed joint ticks and episode budget |
| `status` | String `running` or `step_limit` |
| `terminated`, `truncated` | Both false while running; false/true at the budget |
| `legal_actions` | Six action-name strings while running; empty at the budget |
| `frame_hash` | SHA-256 hex string of canonical JSON for all other fields |

There is no native win/lose ending in these fixed-horizon kitchens. Budget
expiry is truncation, not victory. Two deliveries never end an episode early.
Nested native object and order details belong to the pinned upstream schema.

Actions: `NORTH`, `SOUTH`, `EAST`, `WEST`, `STAY`, `INTERACT`. Directions move
one floor cell and set facing; blocked moves can still turn. `INTERACT` affects
the adjacent tile being faced. Ineffective interactions are legal no-ops.
There is no dynamic action mask or fallback for model failures.

### HTTP and GUI interfaces

`ui.py` serves a loopback-only observer. These GET routes are read-only:

| Route | Response |
| --- | --- |
| `/api/state` | Envelope with full `frame`, scenario, layout choices, replay position and model availability |
| `/api/map` | Same dictionary as `game_map()` |
| `/api/rules` | Same dictionary as `rules()` |
| `/frame.png?hash=HASH` | Official PNG tied to a retained frame hash |

POST `/api/step` accepts `{"policy":"human","action":"NORTH",
"frame_hash":"CURRENT_HASH"}`; stale actions are refused. `policy` also accepts
`baseline` or `model`. POST `/api/reset` accepts scenario settings. POST
`/api/export` freezes a recording copy and returns its `out` directory. Replay
accepts `/api/step` with integer `index` to seek. Cross-origin browser commands
are refused. Reads remain at the pre-action frame during model inference.
Interactive recordings are labelled `interactive` even with a selected model;
the grader refuses them as completed evaluations.

## Model input, actions and teammates

NanoJev receives **text, not the GUI image**: controlled chef index; full map
and legend; both chefs' positions, facing and held objects; counter/pot objects,
ingredients and cooking progress; allowed orders and resolved rules; score,
deliveries and tick budget; recent joint actions/rewards (four by default).
`decisions.make_request()` puts these in `parts.essential`. One `action`
question offers all six native primitives. The model returns probabilities
and an argmax; the controller validates and applies its choice for chef 0.
The native engine changes the game state. Chef 1 chooses independently from
the same pre-action frame.

Replies must echo the ID and state, trim nothing, audit exactly `action`,
explicitly set `stateTruncated=false`, cover six options and supply
`stateTokens`, `stateRoom`, `inputTokens`. Explicit null counts are allowed;
available counts must be positive integers and agree with state room, input
size and optional `maxInputTokens`. Probabilities must cover exactly the
options, be finite in `[0,1]`, and sum within `6 * 0.00005 + 1e-12` of one
for Laya's four-decimal output. Divide accepted values by their `math.fsum`
total in action order for sampling/audit. Evaluation keeps the reported
argmax, including ties. Invalid replies stop before stepping; replay uses the
same checks.

Partner names select **policies**, not extra native actions:

| Policy | Behavior |
| --- | --- |
| `solo` | Autonomous greedy: supplies onions, starts full idle pots, takes dishes for ready soup and serves; current-state shortest paths treat the other chef as an obstacle |
| `role` | Role heuristic: chef 0 primarily supplies; chef 1 prioritizes serving, can fetch a dish while cooking, otherwise parks near its start; can stall |
| `stay` | Always `STAY`, occupying its cell |
| `solo_slow` | Diagnostic CLI variant: `solo` only on ticks divisible by three, otherwise `STAY`; not a GUI option |

The GUI offers the first three. Every policy emits one of the same six actions
per tick. These are local heuristics, not upstream trained or optimal agents;
they consult no future simulator. Baseline/teacher mode uses `role` for the
controlled chef; model mode replaces that action with the validated model
choice, without rescue.

Connect an existing compatible decision server:

```powershell
.\integrations\overcooked\run.ps1 -Mode play -DecisionServer 'http://127.0.0.1:8803' -Open
.\integrations\overcooked\run.ps1 -Mode model -DecisionServer 'http://127.0.0.1:8803'
```

`model.ps1 -Mode inspect` checks an existing WSL NanoJev cache; `-Mode serve`
uses Honing's shared loader/server and `integrations/ra2/pins.json`.
`-Distribution`/`-LinuxPython` select the WSL environment; `model_env.py --source`
selects the NanoJev checkout. Defaults under `/opt/honing/` are local conveniences,
not bundled dependencies. See [RA2 model setup](../ra2/README.md) for source,
weights and GPU requirements. No RA2 game assets are needed. Model inference
is separate from the Windows Python 3.10 game process.

## Recordings, grading and Honing

```powershell
.\integrations\overcooked\run.ps1 -Mode demo
.\integrations\overcooked\run.ps1 -Mode verify -RunDir 'D:\path\to\run'
.\integrations\overcooked\run.ps1 -Mode replay -RunDir 'D:\path\to\run' -Open
```

CLI matches write unique ignored directories under `runs/`; explicit outputs
must be new. `manifest.json` binds scenario, source/runtime, harness and policy;
`steps.jsonl` records reset and every joint action/frame; `decisions.jsonl`
retains exact requests/replies/choices/latencies; `harness/files.json` retains
delivered files; `result.json` reports score, deliveries, completion and grade.
Optional first/last PNGs are visual aids. Failed runs have `error.json` and
cannot pass verification.

Replay reconstructs requests, partner actions and every native state/reward,
validates model replies/choices and checks the final result. It makes no model
calls or recording changes. Retain original code/runtime for old runs: changed
play/audit source or runtime identities are refused. Cross-platform bitwise
replay and mid-episode resume are not promised. Interactive or sampled rollout
records cannot masquerade as evaluation grades. Records are not signed
model-server attestations.

The local format is `overcooked-match-v2`, not a shared Reef schema.
[RFC #148](https://github.com/HectorHHZ/llm-honing/issues/148) accepts Griddly's
format only. This independent PR documents the Overcooked-local format for maintainer
review under `CONTRIBUTING.md`; it has no dependency on the Griddly PR.

`adapter.py` implements existing runner/grader contracts using isolated game
subprocesses. Grading requires native replay and a complete model evaluation
for the exact task, harness and checkpoint. Grade is
`raw_score / (raw_score + score_scale)`, default scale 40. It keeps increasing
beyond two soups; `success` at two deliveries is diagnostic. Raw 400-tick team
score is the primary research metric.

`e2e.py` supplies existing JSON component-loader factories. Harness files are
`overcooked/controller.json` (`history_steps` only) and
`overcooked/questions.json` (`action` instructions). Extra settings/instruction
files become model-visible notes. Harness edits cannot change map, teammate,
scoring, action options or budget. Executable harness files are refused.

## Verification

```powershell
.\integrations\overcooked\run.ps1 -Mode test
& integrations\overcooked\.venv\python.exe scripts\e2e\check_runner.py --config integrations\overcooked\config\runner-check.json --work-dir integrations\overcooked\.cache\runner-check
npm install --prefix integrations/overcooked/.cache/browser playwright
node integrations/overcooked/tests/browser_check.cjs
```

Runner conformance uses the real HTTP `DecisionServer` with an explicitly
labelled rule stand-in, not model weights. It checks support, harness delivery,
isolated/concurrent episodes, repeated event loops and the release gate.
Browser checks launch their own ephemeral native service and test PNGs,
delivery, pause, keyboard input, state/map/rules, export, layout/dynamics,
mobile layout and verified replay seeking. Set `OVERCOOKED_BROWSER` to a
Chromium executable if Edge is unavailable. Receipts go under `.cache/`.
The committed [acceptance receipt](validation/acceptance.json) reports the
checks that actually ran. English screenshots show
[scripted delivery](validation/gui-delivery.png),
[real model control](validation/gui-model.png) and
[mobile layout](validation/gui-mobile.png).
The real NanoJev connectivity smoke completed 400 ticks with zero score;
it demonstrates no capability improvement.

Upstream source and sprites retain their MIT license in the separately
downloaded `upstream/LICENSE`. No commercial Overcooked installation is needed.
