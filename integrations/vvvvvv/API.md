# VVVVVV standalone bridge API

This local experimental interface does not change Honing core contracts.
It applies only to the pinned original source in [README.md](README.md).
The model uses the existing Honing health/decision routes. The native game uses
files in a private caller-supplied `HONING_SESSION_DIR`; it opens no network port.

## Pause, step and input

The game publishes `observation.json` by atomic same-directory rename and
waits without calling the original fixed/render loop. Physics, gameplay,
deaths, timers and scripted state do not advance during inference. Window/event
processing also waits; this bridge is intended for an isolated game session.
The bridge explicitly keeps the original loop focused when its isolated window
loses OS focus, so an accepted tick still advances gameplay.
Each accepted gameplay command advances the original main loop once per native
tick, including original rendering and game input/logic. The startup reset runs
165 original ticks to finish countdown/fade before publishing tick zero.

`command.txt` contains one whitespace-separated command:

```text
<current observation integer id> <action> <ticks>
```

Publish with a temporary file and rename, as `NativeClient.command` does.
Exactly one command may be outstanding. Six actions are offered to the model:

| Action | Native input |
| --- | --- |
| `wait` | Release movement and flip |
| `left`, `right` | Hold that direction for the full tick window |
| `flip` | Release movement; one flip edge on the second tick |
| `flip_left`, `flip_right` | Hold movement and issue that same flip edge |

Ticks must be integers 2–120. A first release tick permits consecutive flip
commands to create separate edges. Original `gameinput` still decides whether
the player is grounded/on the roof and whether a flip is legal; issuing flip
does not force gravity in midair. Native acceleration, friction and collision
remain active, so `wait` does not instantly zero velocity. There is no aiming,
attack, menu interaction or synthetic teleport command. A dedicated display
should have no concurrent human keyboard/controller input.

Success publishes the next `id`, with `tick` increased by exactly the requested
count; `NativeClient.step` checks both. Stale/duplicate ids, unknown actions,
out-of-range ticks, malformed/extra fields are consumed and appended to
`native_error.txt` without advancing the game. A malformed native command has
no new observation acknowledgement; inspect the error file rather than waiting
for a new id. IDs are monotonically increasing within the process.

At normal speed a native tick is 34 ms in this source. Death also uses 34 ms;
slowdown settings can change the value. `simulationMilliseconds` sums actual
native timestep values for accepted action ticks since reset. Rendering is one
frame per action tick; neither model wall-clock time nor the startup countdown
is included. This is the existing VVVVVV fixed-loop cadence, not Unity physics
time or an independently chosen render-frame duration.

## Observation

| Field | Meaning |
| --- | --- |
| `id` | Current command sequence, starts at 1 |
| `epoch`, `tick` | Reset count and accepted action ticks since reset |
| `engineFrame` | Original engine frame counter; each accepted gameplay tick advances it once |
| `simulationMilliseconds` | Sum of accepted action tick durations since reset |
| `tickMilliseconds` | Current original engine timestep |
| `lastAction`, `lastActionTicks` | Native echo of the last accepted command and requested tick window |
| `room` | Original internal room coordinates, e.g. `[113,105]` |
| `player` | `x,y` sprite origin in room pixels; `vx,vy` pixels per tick; collision `width,height`; native contact counters `onGround,onRoof` (>0 means contact) |
| `gravity` | 0 downward, 1 upward |
| `deathSequence` | Original countdown; -1 means no death sequence |
| `deaths`, `flips` | Original trial counters |
| `hasControl` | Original player-control flag |
| `checkpoint` | `[roomX,roomY,x,y,gravity]` for the current native death-respawn checkpoint |
| `localOriginTiles` | Top-left tile of the exported 9×9 neighborhood (8 px/tile) |
| `localSolidRows` | Nine 9-character rows: `#` colliding solid, `!` spike tiles detected by the original invincibility collision variant, `.` other, `?` outside the 40×29 room grid |
| `entities` | All non-invisible valid entities within current room pixel bounds, except player: `[original type,x,y,width,height,harmful 0/1]` |

Coordinates increase rightward/downward. Player x/y are sprite origins, not
collision-box centers. Contact counters may be negative while airborne. The
local tile field excludes terrain outside this 9×9 area; entities include the
whole current room, including objects farther than the local tile field. It
does not export future rooms, hidden scripts, RNG state or campaign solution
data. The entity type is the pinned source's `EntityType` enum. The spike
classification is a collision-derived hint, not a complete future hazard model;
moving hazards remain entities. Full snapshots are retained in local artifacts.

## Reset, stop and lifecycle

`<id> reset 0` calls the original Space Station 1 trial start, runs its countdown
again, increases epoch and publishes a new id with tick/time zero. It restarts
the fixed initial checkpoint and trial counters. Deaths during a normal action
use the game's own current checkpoint respawn. Neither is an arbitrary
runtime-entity/RNG savestate or a deterministic paired replay guarantee. The
source hard reset reseeds RNG from its original clock/frame setting.

`<id> stop 0` runs the original cleanup and exits zero without another snapshot.
The launcher waits for it, and on exceptions terminates only its own subprocess.
The game exits nonzero on capture/write failure or after 600 seconds without
a valid command. Caller-owned assets, model service and display are untouched.
`HONING_IDLE_TIMEOUT_MS` can shorten the watchdog to 100–600000 ms for native
lifecycle tests; its default is 600000. Invalid commands never refresh it.

## Model acceptance boundary

`request_for` submits only essential prompt/current observation; no optional
history is sent by the baseline. `checked_action` also supports removal of
whole optional `details`, `memory` or `logs` entries when explicitly supplied.
It rejects essential removal, unknown removal and any altered retained text.
All question answers and token audits must match the offered questions/options.
The audit must report `stateTruncated=false`; available token counts must be
nonnegative, consistent with capacity and non-Boolean. Null counts are allowed
for the existing DJEV path, without inventing unavailable counts.

Before `NativeClient.step`, the client checks integer request id, exact option
set, offered choice, finite non-Boolean numeric probabilities in [0,1], total
probability, and agreement with a maximum (ties allowed). Pinned Laya rounds to
four decimals, so six options allow `6×0.00005 + 1e-8` total rounding error.
Invalid distributions are never renormalized. Responses and HTTP error bytes
are written before JSON parsing or semantic validation; rejection artifacts
retain the request, reason, raw filename, paused state and native call count.
Portable negative tests prove zero native calls, with a separate real native
paused-state rejection probe. Neither fixture is model-performance evidence.
