# Soul Knight native interface

This experimental integration controls the pinned author release of
[`huyanjueluowei/SoulKnight`](https://github.com/huyanjueluowei/SoulKnight).
The original Unity player owns gameplay, physics and progression. The adapter
adds private JSON control, snapshots and native camera readback. It does not
register an `EpisodeRunner`, grader, task suite, reward or training protocol in
Honing.

## Prerequisites and a complete control example

Supply the original game release and a local Wine/Mono/Cecil toolchain as
described in [README.md](README.md). The preparation command checks the pinned
original inputs before writing a separate instrumented copy. Keep the game,
patched copy, model files and run output outside this checkout. The repository
contains the adapter and dependency identities, not those external inputs.

From the repository root, set these variables to caller-owned locations:

```bash
export SOUL_KNIGHT_GAME=/private/path/to/original/Game
export SOUL_KNIGHT_PATCHED=/private/path/to/patched/Game
export SOUL_KNIGHT_RUN=/private/path/to/a/new/run
export SOUL_KNIGHT_WINE=/private/path/to/wine
export SOUL_KNIGHT_MONO=/private/path/to/mono
export SOUL_KNIGHT_MCS=/private/path/to/mcs.exe
export SOUL_KNIGHT_CECIL=/private/path/to/Mono.Cecil.dll
export SOUL_KNIGHT_MONO_ROOT=/private/path/to/extracted/mono/root
export PYTHONPATH="$PWD/integrations/soul-knight${PYTHONPATH:+:$PYTHONPATH}"

python -m soul_knight_bridge prepare \
  --game-dir "$SOUL_KNIGHT_GAME" --output "$SOUL_KNIGHT_PATCHED" \
  --mono "$SOUL_KNIGHT_MONO" --mcs "$SOUL_KNIGHT_MCS" \
  --cecil "$SOUL_KNIGHT_CECIL" --mono-root "$SOUL_KNIGHT_MONO_ROOT"
python -m soul_knight_bridge launch \
  --game-dir "$SOUL_KNIGHT_PATCHED" --run-dir "$SOUL_KNIGHT_RUN" \
  --wine "$SOUL_KNIGHT_WINE"

printf '%s\n' \
  '{"op":"reset","seed":7}' \
  '{"op":"step","action":{"move":[1,0],"aim":[-35,1],"attack":true},"frames":30}' \
  '{"op":"state"}' |
  python -m soul_knight_bridge client --ipc-dir "$SOUL_KNIGHT_RUN/ipc"

python -m soul_knight_bridge stop \
  --launch-receipt "$SOUL_KNIGHT_RUN/launch.json"
```

The absolute `PYTHONPATH` remains usable after changing working directory.
Omit `--mono-root` when using a normally installed Mono toolchain.
`python /absolute/checkout/integrations/soul-knight/soul_knight.py --help` is an
alternative entry point. A successful launch prints its own process/display,
IPC, rendering and log locations. Use a fresh run directory; the launcher does
not attach to an existing game. Stop only a process set identified by that
run's launch receipt. Preparation outputs and run artifacts must be outside
this Honing checkout; the adapter rejects paths inside it.

## Requests and responses

The JSONL client reads one object per line and writes one result per line:

```json
{"op":"state"}
{"op":"reset","seed":7}
{"op":"step","action":{"move":[1,0],"aim":[-35,1],"attack":true},"frames":30}
```

Successful results have `{"ok":true,"state":{...}}`; errors have
`{"ok":false,"error":"message"}`. The Python client returns the snapshot
directly and raises an exception on failure. Its `last_request` and
`last_response` retain the actual native IPC envelopes, including matching IDs.

```python
from soul_knight_bridge import NativeClient

client = NativeClient("/private/path/to/a/new/run/ipc", timeout_seconds=30)
initial = client.state()
state = client.reset(seed=7)
state = client.step({"move": [1, 0], "aim": [-35, 1], "attack": True}, frames=30)
```

Offline schemas describe [requests](schemas/request.schema.json),
[actions](schemas/action.schema.json) and [snapshots](schemas/state.schema.json).

| Operation | Arguments | Effect |
| --- | --- | --- |
| `state` | None | Reads the current paused scene; does not advance gameplay. A newly launched player starts in scene 0 and can have `player:null`. |
| `reset` | `seed`, a signed 32-bit integer; default `7` | Seeds Unity's RNG, reloads build scene 1 (`SampleScene`), sets `bridge_frame` to zero and writes frame 0. Removes this run's older native PNGs. |
| `step` | `action`, plus integer `frames` in `1..250`; default `50` | Applies the action for the requested rendered-frame batch, then pauses before replying. |

Reset skips the source menu; it does not configure arbitrary maps, encounters,
weapons or player attributes. Unity's global `Time.time` does not reset to zero,
and equal seeds do not establish deterministic physics or replay. A terminal
snapshot is a reported condition, not a guarantee that later `step` calls are
rejected or have no effect. The model driver stops issuing actions on player
death or a reported terminal state.

### Action fields

Omitted fields use the defaults below. Unknown fields, wrong types, nonfinite
coordinates and invalid movement components are rejected by the Python client
before it submits a request.

| Field | Type and default | Native meaning |
| --- | --- | --- |
| `move` | Two components from `{-1,0,1}`; `[0,0]` | Held horizontal/vertical axes throughout the batch. `+x` is east/right and `+y` north/up. The source does not normalize diagonal axes. |
| `aim` | Two finite numbers, each within `[-1000000,1000000]`; `[1,0]` | Absolute world position `[x,y]`, converted through the original camera to the source mouse-position API. It is not a screen pixel or direction vector. |
| `attack` | Boolean; `false` | One left-mouse-button-down event on the first advanced frame. |
| `interact` | Boolean; `false` | One Space-key-down event on the first advanced frame. |
| `skill` | Boolean; `false` | One right-mouse-button-down event on the first advanced frame. |
| `switch_weapon` | Boolean; `false` | One positive mouse-wheel event on the first advanced frame. |

The native interface permits several button flags in one action. Each true flag
is an edge for that request, so repeat the flag on a later `step` to press again;
`frames=100` does not add 100 shots. Movement and aiming are independent. Normal
game rules still determine whether a press fires, consumes energy or has an
effect. Nearby pickups and portals also depend on the source trigger callbacks.
An edge on a rendered frame is not a promise that a physics callback observes
it; interaction controls need task-specific acceptance before relying on them.

Coordinates and velocities use the author's world units and Unity time.
Collider shapes and navigable geometry are not exposed, and landmark origins
are not guaranteed to be useful navigation targets.

### Transport, errors and timing

Each game has one private IPC directory. The client serializes cooperating
callers with a file lock, atomically publishes `request.json`, and waits for a
matching-ID `response.json`. Unity consumes the request file. Preserve the
client's in-memory envelopes rather than rereading that consumed file.
`HONING_SOUL_KNIGHT_IPC` identifies the directory inside Wine; the launcher sets
it. This local file protocol has no authentication or network listener and is
not a hardened multi-user service. Do not write IPC files directly or share the
directory with unrelated processes.

Python validation errors occur before submission. Native errors carry
`ok:false`; absent replies or an unresponsive process raise a client timeout.
A timeout does not roll back an already submitted native action. Inspect the
owned game's state and logs before deciding whether to retry. The interface
does not promise exactly-once recovery after a crash or timeout.
The JSONL/Python client's reply timeout defaults to 30 seconds and starts after
the file lock is acquired and the request is published; lock acquisition has no
separate timeout. `run` defaults to a 90-second native reply timeout, with
`--timeout-seconds` overriding it. Its HTTP calls also have a 90-second timeout.

The bridge guards source update loops and freezes simulation between requests.
An action batch counts rendered frames, not physics ticks. Its default target
is 50 rendered frames per second with a 0.02-second fixed delta; the first frame
after resuming can report zero `delta_time`. Read `sim_time` to measure observed
simulation advancement. A verified 30-frame native move advanced 0.58 seconds,
not exactly 0.60 seconds. Model inference latency is wall-clock time and remains
separate from both counts.

## Snapshot fields

The snapshot schema identifier is `honing.soul-knight.native.v1`. Snapshots are
full scene debug state. They include actors outside the current room and do
not define a research observation or visibility restriction.

| Field | Meaning |
| --- | --- |
| `schema` | Schema identifier above. |
| `bridge_frame` | Advanced render frames since reset. Native PNG index, starting at zero. |
| `unity_frame` | Unity's global rendered-frame counter, including paused engine frames. |
| `advanced_frames` | Requested batch size in a step reply; zero for state/reset replies. |
| `sim_time`, `fixed_delta_time`, `delta_time` | Unity-reported timing in seconds. |
| `paused` | Whether the bridge has stopped advancing gameplay. Normal synchronous replies are paused. |
| `scene` | `{name,index}` from the active Unity build scene. |
| `player` | Actor object below, or `null` where no player exists. |
| `enemies` | Scene enemy/Boss actor objects, including inactive elite objects. |
| `projectiles` | Active native bullets: `{id,name,tag,x,y,vx,vy}`. |
| `landmarks` | Name-filtered door, water and reward transforms: `{name,tag,x,y,active}`. This is not a map or collision mesh. |
| `progress` | Source GameManager flags and enemy-list counts below. |
| `executed_action` | The normalized action retained by the bridge. On `state`, it describes the previous submitted action, not a newly executed one. |
| `native_frame_path` | Native PNG path reported inside Wine. Resolve on the host through the launch receipt's `render_dir` and frame index. |
| `terminal`, `won` | Derived smoke conditions described below. They are not grades. |

Actor fields are `id`, `name`, `tag`, `x`, `y`, `vx`, `vy`, `hp`, `max_hp`,
`defence`, `energy`, `dead`, `active`, `elite`, `speed` and `weapon`.
`defence` is the source's current defence/shield field. `speed` comes from the
player controller and is not an enemy movement-speed measurement. Instance IDs
identify objects within a scene lifetime; do not use them as stable IDs across
reset. Inactive elites can have zero current HP before their source `Awake`
initializes them. Their inactive status does not prove they were defeated.

`progress` has `first_stage_end`, `game_over`, `player_in_main_room`,
`player_in_boss_room`, `enemies_one` and `enemies_two`. The counts reflect source
GameManager lists; they are not the length of `enemies`.

### Boss death and final scene

The pinned source contains an ordinary wave, an elite wave and a Boss named
`大黄蜂`. The Boss room is part of `SampleScene`; it is not a separately loaded
Boss level. After the ordinary enemy list empties, the source activates the elite
objects. Empty ordinary and elite lists set `first_stage_end` and enable reward
chests. Entering the Boss-room trigger sets `player_in_boss_room`.

Boss death sets the source's `GameManager.isGameOver`. The bridge exposes this
as `progress.game_over` and `won:true`; `terminal` is player death, that Boss
death flag or the final scene. The GameManager then activates the transition
portal. Overlapping
the portal and pressing Space loads build scene 2 (`Congratulation`). Therefore
`won:true` covers the Boss-death milestone and the final scene; use
`scene.index == 2` to identify the final-scene transition. The final scene has
no gameplay GameManager, so its `progress` fields can have default values even
though `won` and `terminal` remain true. A client that stops on `terminal` will
not perform
the later portal interaction automatically.

These statements come from the pinned source's `GameManager`, `BossStats`,
`BossDoorOpen`, `TransitionDoor` and build-scene configuration. Source presence,
an encountered runtime Boss and a completed legal-action route are distinct
evidence levels; the model smoke recording did not reach the Boss.
The enabled scenes are `MainMenu`, `SampleScene` and `Congratulation`. The author
release was not independently rebuilt from the inspected source, so source
control flow alone does not verify every runtime guard or callback.

## Honing model HTTP adapter

The `run` command requires an already running real Honing decision server:

```bash
export SOUL_KNIGHT_DECISION_SERVER=http://127.0.0.1:8808
export SOUL_KNIGHT_MODEL_RUN=/private/path/to/new/model-output
python -m soul_knight_bridge run \
  --ipc-dir "$SOUL_KNIGHT_RUN/ipc" \
  --decision-server "$SOUL_KNIGHT_DECISION_SERVER" \
  --output "$SOUL_KNIGHT_MODEL_RUN" \
  --decisions 1 --frames 5 --reset --seed 7
```

Start the model service using Honing's existing serving workflow. This
integration does not download a checkpoint, choose a GPU or bundle a separate
CPU model server. It reads `GET /v1/health` and calls `POST /v1/decide` using
Honing's established `{id,parts,questions}` envelope. The temporary driver asks
three choice questions against one paused native snapshot:

| Question | Offered choices | Native mapping |
| --- | --- | --- |
| `move` | `stay`, `east`, `west`, `north`, `south`, `north_east`, `north_west`, `south_east`, `south_west` | Corresponding axis pair; e.g. `north_east` becomes `[1,1]`. |
| `button` | `none`, `attack`, `interact`, `skill`, `switch_weapon` | Exactly the chosen Boolean is true, or all are false for `none`. |
| `aim` | Four cardinal points five world units from the current player, plus up to six active living scene enemies | The selected absolute point or that enemy's current world position. No nearest-enemy selection or automatic aiming policy is added. |

Before any native `step`, the driver checks the echoed request ID, exact
question/option sets, and each returned `choice` against a maximum probability
(ties are allowed). Probabilities must be JSON numbers, excluding booleans,
finite and in `[0,1]`. Their sum must differ from 1 by at most
`option_count * 0.00005 + 1e-12`. The pinned Laya source
(`970dc8c5f63d7b886a68409493f37d569424f933`, `laya/agent.py`,
`_decode_answers`) rounds each probability to four decimal places: each can
contribute at most `0.00005` rounding error; `1e-12` allows floating-point
summation error. This contract covers that serving format, not arbitrary
decision servers. For example, three `0.3333` values plus zeros total `0.9999`
and can pass; all-zero, all-two and all-`true` maps fail. No normalization or
fallback action is applied.

The smoke controller requires every submitted `essential` and `details` entry,
including player/progress and entity information. Only optional history in
`logs` and `memory` may be removed: whole logs oldest first, then memory newest
first, matching Honing's token-budget order. The driver reconstructs retained
text from the submitted parts and reported `trimmed` entries and requires exact
equality with `stateText` (newline join in `essential`, `details`, `memory`,
`logs` order). Unreported changes, fabricated/out-of-order removals and any
required-section removal are rejected. Honing's server already protects
`essential`; this receiver also protects the submitted `details`.

`tokenAudit` must cover exactly the submitted questions. Each audit must contain
nonnegative integer `stateTokens`, `stateRoom`, `inputTokens` and `options`,
with the offered option count intact and `stateTruncated` exactly `false`.
State must fit its reported room; counts must be coherent and agree across
questions on state size and the implied total budget. These checks validate
the server's report; the adapter does not independently tokenize text or prove
model comprehension. Whole optional-entry removal is distinct from token
clipping. The latter is rejected. Oversized required context stops the run;
reducing the current observation is a separate research choice.

The driver preserves the complete request, raw model response, choices, mapped
action, native request/response and before/after snapshots. Exact response
bytes are saved before JSON parsing, ID/context checks or choice validation;
any rejected response stops the run before native execution and retains its
request, decision ID and explicit reason locally.
Float32 aim serialization uses a bounded comparison tolerance; discrete
movement and button controls remain exact.

The following payloads illustrate the wire format; they are not recorded model
outputs. This small example offers fewer choices than the driver:

```json
{
  "id": "example:0",
  "parts": {
    "essential": ["Player is at world (0,0); choose a movement."],
    "details": [], "memory": [], "logs": []
  },
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Choose a direction.",
      "criteria": {"east":"Move right", "north":"Move up"}
    }
  }
}
```

```json
{
  "id": "example:0",
  "stateText": "Player is at world (0,0); choose a movement.",
  "trimmed": [],
  "tokenAudit": {
    "move": {
      "stateTokens":12, "stateRoom":2000, "inputTokens":60,
      "options":2, "stateTruncated":false
    }
  },
  "answers": {
    "move": {
      "type":"choice", "choice":"east",
      "probabilities":{"east":0.7,"north":0.3}
    }
  }
}
```

The token counts above illustrate the audit shape, not measured tokenization.
The service can return additional model metadata and latency fields; retain
its complete response rather than reducing it to this example.
The driver formats full debug state into text, filters inactive enemies, limits
the displayed entity lists and retains a short action history. Candidate order,
text precision, argmax readout, frame duration and full-state access are smoke
defaults, not accepted task-collection research settings.

The fresh `--output` directory contains:

| Artifact | Contents |
| --- | --- |
| `model_health.json`, `model_health.raw.json` | Parsed health and exact response bytes. A ready endpoint alone is not evidence of its model's identity; inspect its metadata. |
| `run_manifest.json` | Run ID, endpoint/health identity, controller hash, frame count, reset seed, observation defaults and response-validation policy. |
| `reset.json` | Reset snapshot when `--reset` is requested. |
| `decisions.jsonl` | One completed decision per row: HTTP request/raw response, hashes, selected choices/action, execution UTC times and latency, native IPC envelopes and before/after snapshots. |
| `current_decision.json` | Latest completed row. |
| `current_decision.response.raw` | Exact latest decision HTTP body, written before decoding/validation; also preserves malformed JSON or invalid UTF-8. |
| `current_decision.pending.json` | Request, decision ID, raw-body reference/hash and snapshot saved before JSON parsing. Updated after validation; retained on rejection/execution failure and removed after a completed checked row. |
| `current_decision.rejected.json` | On validation failure, the pending exchange plus parsed response when available, `response_rejected`, explicit `rejection_reason`, and `execution_pending=false`. No action is submitted. |
| `status.json` | Current phase, completed decisions, last frame/action and stop/error details. |

Model IDs use `run_id:decision_id`; native IPC operations have separate generated
IDs. The driver checks each model reply's ID and each native request/response
pair, retaining the actual submitted and executed actions. The artifacts record
conditions and actions; they do not contain a reward or score.

## Rendering and evidence limits

The supported Linux route uses the original Unity camera's
`Camera.Render -> RenderTexture -> ReadPixels -> PNG` in batch mode. It renders
the author's assets and adapts the original overlay canvas to that camera. Each
advanced frame is atomically written under `render_dir` as
`frame_<bridge_frame:06d>.png`; reset writes frame zero. Reset can initially
show scene-serialized HUD text before the first source update refreshes it.

Windowed Wine attempts stalled before a normal source update on the tested
host. Native camera readback is the verified route; a usable interactive
desktop window is not established by these tests. Encoding the PNG sequence
at 50 frames per second omits inference waits, so a short video is not a
measurement of real-time model speed.

The evidence available before this portable packaging pass has separate scopes:

| Evidence | Established | Not established |
| --- | --- | --- |
| Pinned source, scene assets and native debug snapshot | A later Boss exists in the same gameplay scene; its native snapshot reported 300 HP. Source defines Boss death, portal activation and final-scene transition. | A player entered that room or completed the route. |
| Native control preflight | Movement, shooting, energy consumption, projectile creation and paused-state stability. | Learned-policy combat or complete source/binary equivalence. |
| Bounded legal-input script | Cleared three ordinary enemies, activated and damaged both elites; player remained alive at the action cap. | Boss-room entry, Boss kill, portal use or a model-controlled route. |
| 12-decision real-Laya loop | Actual model outputs drove native movement; the player also hit walls. | Attacks, enemy damage, combat success or Boss completion. |

Private assets, snapshots, videos, model files and run logs are not distributed
in this integration.
Commands and results for the portable implementation belong in the PR's
verification record; source inspection alone does not substitute for that run.
