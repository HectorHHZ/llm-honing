# Hurrican local control API

This is an integration-local stdin/stdout interface inside the pinned native
game. It is not a new Honing core protocol. Use only trusted local processes;
there is no listener or remote authentication surface.

## Launch and lifecycle

`HONING_HURRICAN_SESSION` enables the hooks and names an existing writable
local directory. Without it, Hurrican follows its ordinary UI and input path.
The hook uses the game's own initialization to start one player, medium
skill, stage 1 (`jungle.map`), skipping menus and intro. It does not enable god
mode, grant weapons, reposition the player or change enemy logic. One neutral
startup heartbeat releases the shoot key before the first observation because
the original weapon arms only after seeing shoot released. This setup update
is outside the controlled `tick` counter and captured demo.

`HONING_HURRICAN_CAPTURE=1` enables PPM capture. Any present value enables
capture; unset it to disable. `HONING_HURRICAN_IDLE_SECONDS` is an integer 1–3600,
default 240. Invalid values stop the game before the first observation. A
single absolute deadline covers waiting for the next valid step: invalid,
stale, observe and partial lines do not renew it. EOF, overlong input (more
than 256 bytes before newline) or idle expiry exits through normal game cleanup.

There is no in-process reset or arbitrary-state restore. Close the game and
launch a fresh process with a new save/output directory. This reinitializes
stage 1 and player defaults; the RNG and every hidden state are not guaranteed
to match another run. The client sends no automatic retries after a transport
error or ambiguous step. It never attaches to or kills an existing game.

`NativeClient.close()` closes stdin and waits five wall seconds, then sends
SIGTERM to its own newly-created process group, followed by SIGKILL after
another five seconds if required. It records the PID, exit code and whether
the owned process remained alive. The caller owns Xvfb and the model service.

## Commands and scheduling

Send ASCII lines, terminated by newline, through the child's stdin:

| Command | Behavior |
| --- | --- |
| `observe ID` | Return the current paused state, without advancing or renewing the deadline. |
| `step ID FRAMES ACTION` | Hold one offered input combination for 1–120 updates; return the next paused state. |
| `stop ID` | Exit normally. EOF also exits. |

`ID` must exactly match the current observation's integer `id`. A valid step
increments it once before execution. `FRAMES` and `ACTION` must be integers;
extra tokens, unsupported verbs, stale IDs and out-of-range values return an
observation with an `error` string, preserving `id` and `tick`. A repeated
accepted step's old ID is stale and cannot advance again. These error replies
do not acknowledge a new step. The Python client treats any error or unexpected
ID/tick as a failure; it does not retry.

The loop blocks before `Heartbeat()` between steps. Each accepted update runs
one ordinary game heartbeat, native input/physics and render. Engine elapsed
time is fixed at 1/60 second and speed factor is `MoveSpeed/60` (normally 10/60),
independent of model latency or graphics speed. Game logic can subdivide an
update when its speed factor exceeds 0.3; at the normal speed one update is one
physics chunk. `tick` counts completed heartbeats, not wall time. Internal
menus, deaths and level transitions remain game-controlled. A heartbeat in a
non-gameplay mode is not evidence of player physics; inspect `mode` and state.

The game remains stopped during inference, error handling and observation.
Trace `wallSeconds` measures each decision plus validation and its native step;
it includes rendering/capture and is not pure inference latency.
There is no background movement, controller auto-fire, dodging or pathfinding.
Some actions explicitly combine shoot with movement/jump; Hurrican decides
whether a held key can fire, charge, jump or repeat. Repeated identical actions
keep that key held across the boundary; the integration adds no release pulse.

| ACTION | Model choice | Held inputs |
| --- | --- | --- |
| 0 | `wait` | none |
| 1 | `left` | left |
| 2 | `right` | right |
| 3 | `shoot` | shoot in current direction |
| 4 | `left_shoot` | left and shoot |
| 5 | `right_shoot` | right and shoot |
| 6 | `jump` | jump |
| 7 | `jump_left_shoot` | left, jump and shoot |
| 8 | `jump_right_shoot` | right, jump and shoot |
| 9 | `crouch_shoot` | crouch and shoot |

## Observation and visibility

Normal game text may appear on stdout. Only a line prefixed `HONING ` carries
one JSON observation. At startup it has `id=0,tick=0`; each step returns after
its final heartbeat. Native startup can take up to the client's90-second
receive timeout. The client does not use screenshots for model input.

| Field | Meaning and units |
| --- | --- |
| `id` | Accepted-step counter within this process. |
| `tick` | Completed controlled native updates. `tick/60` is simulated seconds. |
| `stage` | Native stage index, initially 1. |
| `mode` | Native enum: 0 cracktro, 1 prelude, 2 intro, 3 menu, 4 gameplay, 5 outro. |
| `primaryProjectiles` | Count of existing native primary-weapon projectiles (types 0–15) owned by player 0, including offscreen. |
| `timeLimit` | Native level clock units, difficulty-dependent; not seconds. |
| `camera` | World-pixel x/y camera offsets. |
| `player.x/y` | Native player sprite world position; positive x right, positive y down. |
| `player.vx/vy` | Engine nominal velocities; normally multiply by 10/60 to obtain pixels/update. |
| `player.energy` | Native health, initially 100. |
| `player.shotDelay`, `autoFireCount` | Native weapon cooldown and remaining auto-fire counter, in engine units. |
| `player.lives`, `score` | Native remaining lives and score. |
| `player.canJump` | Native jump-permission Boolean. |
| `player.action` | Internal PlayerActionEnum ordinal; see pinned Player.hpp. |
| `player.facing` | -1 left or 1 right. |
| `enemies` | Up to six nearest active, destroyable native objects passing `IsOnScreen()`. |
| `enemies[].type/x/y/energy` | Native type code, world-pixel position and internal health. |
| `error` | Present only when a command was rejected without advancing. |

This is privileged structured-state access: exact health, jump permission and
world coordinates are available even when the artwork does not show them.
Enemy filtering uses native screen bounds, not pixel occlusion; a geometrically
on-screen object can be hidden by scenery. No full offscreen enemy list, map,
projectile list, inventory, exit coordinates or hidden RNG state is exposed.
The nearest-six cap can omit threats; observations are not full-game state.

Capture reads the rendered native backbuffer immediately before swap on every
other controlled update. `frame000000.ppm` follows the first update, then
updates 3,5,…; frames do not include model waits. Encoding at 30 fps therefore
shows game-time pacing. No audio is recorded.

## Decision response acceptance

The client uses existing `GET /v1/health` and `POST /v1/decide`. The request ID
is the native observation ID. It submits a single `action` choice question and
ten criteria in the table above. Instructions and the complete serialized
observation are `parts.essential`; there is no optional history or removable
context in this integration.

Before `NativeClient.step`, the receiver requires the exact integer response
ID, exact question and option sets, a supported choice and its argmax (ties
accepted). Each probability must be a finite numeric non-Boolean value in
[0,1]. Four-decimal Laya rounding permits a total-sum error of
`10*0.00005+1e-8`; the receiver never renormalizes malformed output.

`stateText` must equal both required entries joined with newline, `trimmed`
must be an explicit empty list, and `tokenAudit` must cover exactly the
submitted question. `stateTruncated` must be false and `options` exactly 10.
Required count fields `stateTokens`, `stateRoom`, `inputTokens` must be present;
nullable DJEV counts are accepted, available counts must be nonnegative
integers, fit their reported budget and contradict neither retained state nor
available input limits. Optional `maxInputTokens` is checked when available.
No required text may be removed or clipped.

Every attempt writes the request and raw HTTP bytes before JSON decoding or
validation can fail. A rejected attempt writes a separate reason artifact,
raises and sends no native command. Attempts have unique local names, including
retries initiated manually, so rejected bytes are not overwritten. HTTP errors
also retain their body. Connection failures can have no response bytes; the
reason artifact records that failure. Raw diagnostics remain local.

The CLI defaults to 12 decisions, 40 updates/input and 180 wall seconds per model
request; `--decisions` accepts 1–10000. A timeout, malformed reply or audit failure
aborts that run and closes its owned game. A model request timeout greater than
the native idle deadline can let the native watchdog exit first; raise the
explicit native idle setting if a legitimately slower server requires it.
