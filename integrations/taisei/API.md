# Private Taisei stage adapter

This opt-in interface is local to this standalone integration. It adds no Honing
core protocol, server endpoint, campaign runner, grader or persisted release
format. The existing `/v1/health` and `/v1/decide` contract is unchanged.

## Native lifecycle and transport

`build.py` accepts only Taisei commit
`5bfc9b842fe6d9bb8d1610f76f442ba07c72c4cf`. It patches three hooks in `stage.c`
and four GCC 11 compatibility declarations, copies two integration headers, and
builds the original game. Without `HONING_SESSION_DIR`, control and the random
seed retain upstream behavior. In bridge mode, each new process begins Stage 1,
Easy, Youmu A, game RNG seed 7, normal practice starting resources. No arbitrary
savestate or model-controlled reset is implemented. Reset means a new process
and fresh storage/cache directory. Identical full trajectories and visual RNG
replay have not been established.

Only the controller writes the private session directory. It atomically renames
`action.tmp` to `action.json`; the game consumes the file once. At every decision
boundary the native main thread writes `observation.tmp`, renames it to
`observation.json`, and blocks before any simulation tick. Physics, gameplay
coroutines, rendering and input remain paused during model inference. The native
wall-clock watchdog exits with code 74 after 240 seconds awaiting a command.
The controller owns and terminates only its game and Xvfb child processes; it
never starts, stops or reconfigures the supplied decision service. Each run
uses a fresh directory and refuses an occupied display. No TCP native port is
opened.

## Observations

Each JSON object contains:

| Field | Meaning |
| --- | --- |
| `id` | Increasing decision-boundary sequence, starting at 1 |
| `frame` | Completed gameplay simulation ticks, starting at 0 |
| `fps` | 60 ticks per simulation second |
| `stage` | Upstream stage ID, 1 for the demonstrated segment |
| `gameover` | Upstream game-over state integer; 0 during ordinary gameplay |
| `viewport` | `[480,560]` gameplay coordinate extent in pixels |
| `player` | Current `x,y`, spare `lives`, `bombs`, and integer `score` |
| `visibleEnemyBullets` | Count of current on-screen enemy projectiles |
| `nearestBullets` | Up to eight nearest on-screen enemy bullets, sorted by current distance to player, each `[x,y,vx,vy]` |

Origin is the gameplay viewport's top-left; x increases right, y down. Bullet
velocity is the current native `MoveParams.velocity` parameter in pixels/tick;
scripted movement may differ from this parameter. Off-screen bullets,
future trajectories/scripts, RNG state, enemies, items, collision radii, lasers
and dialog text are omitted. This is an explicitly partial object sensor; it
is neither vision-only input nor a complete hazard representation. The full
screen video includes the upstream UI. Observation rounding is three decimal
places for player position and two for bullet positions/deltas.

## Commands

The native parser deliberately accepts only the controller's canonical compact
JSON with exact field order, no duplicate/extra fields and integer values:

```json
{"id":1,"action":"up_left","frames":60}
```

Trailing whitespace is permitted. `id` must equal the outstanding observation;
`frames` is 1..120 inclusive. Unknown, stale, malformed, overlong or embedded-NUL
commands cannot advance simulation; the native game writes
`rejected-command.json` and retained raw command bytes, then waits. Overlong
commands retain their first 255 bytes. Errors do not consume the observation ID.
`stop` exits the current stage loop without a simulation tick. There is no
network retry protocol; use one controller and one outstanding command.

| Action | Held flags |
| --- | --- |
| `stay` | shot, focus |
| `left`, `right`, `up`, `down` | named movement, shot, focus |
| `up_left`, `up_right`, `down_left`, `down_right` | named diagonal movement, shot, focus |
| `stop` | end stage; no movement or tick |

Movement uses native Taisei rules, including normalized diagonal speed. Focus
slows movement and reveals the hitbox; shot is held continuously by every
movement action. This shot/focus assistance is fixed controller behavior;
every movement direction comes from the model's returned argmax. Bomb and
power-surge buttons are excluded. Inputs are native held flag transitions,
recorded through `player_event(EV_INFLAGS)`, rather than synthetic OS key presses.
Each command advances exactly the requested gameplay ticks, unless the native
stage independently terminates. Natural terminal exits are reported as exited
processes by the controller; this adapter does not grade or restart them.

## Decision acceptance

Every request submits a bounded observation and rules as `parts.essential`.
There is no optional context or history. The controller therefore requires
`trimmed: []`, exact retained `stateText`, and per-question `tokenAudit` covering
all submitted questions with `stateTruncated: false`, the correct option count,
and consistent available nonnegative integer token counts. Optional removal is
not applicable to this request policy. No current sensor field may be silently
clipped or omitted by the model service.

Before publishing a native command, the controller checks the echoed integer
ID, exact answer-question and probability-option sets, offered choice and
argmax agreement. Probabilities must be numeric, non-Boolean, finite and in
`[0,1]`. For Laya's four-decimal probabilities, nine choices permit a total
rounding error of `9 * 0.00005 + 1e-8`; invalid distributions are never
renormalized. Tied maxima are accepted. Raw bytes are saved before JSON parsing
or acceptance can fail, and request, response, context/token audit and rejection
reason are retained locally. Rejection makes zero native step calls.

## Timing and evidence

One gameplay tick is 1/60 simulation second. The rendering loop produces one
frame per gameplay tick under this configuration. `TAISEI_FRAMEDUMP` captures
native PNGs; `encode.py` encodes at 60 fps and fully decodes the result to check
it. Inference waits are omitted from the encoded clip. The video is not evidence
of real-time model throughput. `result.json` reports simulation and wall time
separately. Frames outside the gameplay segment, if any, remain in the frame
dump and its reported video duration. Audio is disabled.

`decisions.jsonl` retains requests, raw-response paths, decoded responses and
accepted actions. `actions.jsonl` pairs each action with before/after native
observations. `verify_native.py` labels its injected invalid responses as
synthetic faults, checks real paused-frame stability and zero step calls, tests
stale and embedded-NUL native commands, then verifies a valid 30-tick action.
`--watchdog-only` runs the actual 240-second idle deadline under repeated stale
commands, checking that malformed traffic does not renew the deadline.
Portable fixtures, native fault injection, real model smoke, deterministic
replay and model task proficiency are separate evidence.
