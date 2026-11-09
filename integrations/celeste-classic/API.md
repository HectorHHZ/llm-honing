# Celeste Classic observation, input and lifecycle

This opt-in Linux file bridge is active only when `HONING_SESSION_DIR` is set.
Without it, the copied player keeps upstream interactive behavior. The client
creates a fresh output directory and owns its Xvfb/player child processes.
`Xvfb -displayfd` allocates an unused display, with no fallback to an existing
session. Only those child process handles are stopped/reaped. The shared HTTP
model is neither launched nor stopped by the client.

## Startup and timing

Startup restores the engine's pre-init globals, seeds RNG with `HONING_SEED`
(default 7, integer 0..1,000,000,000), initializes the existing engine and enters
its original first room through `begin_game`. Sixty neutral update/draw pairs
settle the existing spawn animation. This is fixed setup, excluded from captured
model gameplay and reported separately as `setupTicks=60`. It is not a model
policy. Room skip/debug commands are not exposed.

A simulation tick is one call of original `Celeste_P8_update`, followed by one
original `Celeste_P8_draw`. The game uses 30 ticks/second and velocities in pixels
per tick. The bridge counts calls after they return and checks the original
modulo-30 clock as well. Freeze/hit-stop ticks are valid engine ticks even when
position does not move. There is no separate physics thread.

**Draw mutates game state**, including orb/flag logic. While paused, neither
function runs: SDL only presents its cached surface and polls window-close events.
Physical keyboard/controller/focus events cannot change model inputs or skip
engine work in bridge mode. Steps are not wall-clock paced. BMP capture occurs
once after each update/draw pair; encoding requires all consecutive frames and
verifies full decode, duration and frame count.

## Native observation

`observation.json` is published with a same-directory temporary-file rename after
setup, every accepted command and every rejected command. It contains:

| Field | Meaning |
| --- | --- |
| `id` | Expected command sequence, starts at 1; accepted step/reset/stop increments it |
| `status` | `paused`, `stopped`, `watchdog` or `window_closed` |
| `ticks` | Accepted gameplay update/draw pairs over the process lifetime; excludes setup |
| `episodeTicks` | Same count since last reset |
| `setupTicks` | 60 initial neutral pairs |
| `nativeUpdates`, `nativeDraws` | Completed original engine calls including setup; monotonic across reset |
| `clockFrame` | Original engine modulo-30 frame variable, restored on reset |
| `lastMask`, `lastFrames` | Last accepted command's button mask and duration; reset/stop set both to 0 |
| `resets`, `rejected` | Reset and invalid-command counts |
| `room` | Original `room.x % 8 + room.y * 8` |
| `deaths`, `freezeTicks`, `restartPending` | Native death total, remaining freeze and room-restart flag |
| `player` | Null during absence/spawn/death; otherwise x/y, vx/vy, remaining `dashes`, `grounded` |
| `summitFlagTouched` | Active native summit flag's `show` field; not a reward/grader |
| `tiles` | 16 strings of 16 cells for the current room, eight pixels/cell |
| `entities` | Current room's active interactive/spawn objects with native type and x/y |

Coordinates have origin at the room's top left; x increases right and y down.
The player's x/y is its sprite reference, not a bounding rectangle. `grounded`
queries the native solid test one pixel below. Tile symbols are `#` for the
native solid flag, `^/v/</>` for spike tiles 17/27/43/59, and `.` for everything
else; dynamic fall floors/platforms/walls are separate entities. Entity records
exclude the player, smoke, room-title text and lifeup text. They include offscreen
or occluded objects in the current room without line-of-sight filtering. The
bridge omits future rooms, future random outcomes, decorative particles, entity
hitboxes and many internal timers. It is structured scene access, not vision-only
observation or a complete serialized engine state.

## Native commands

The client atomically writes `command.txt` containing four whitespace-separated
ASCII fields: `operation id mask frames`. It waits for the next observation
before submitting another command. This temporary format is local working state,
not a supported public replay/save or network protocol.

- `step <current-id> <mask> <frames>` holds the mask for 1..30 ticks.
- `reset <current-id> 0 0` restores an in-memory snapshot captured after setup.
- `stop <current-id> 0 0` publishes `stopped` and exits the player.

All numeric fields must be unsigned decimal digits <=1,000,000,000; the mask
range is 0..63. Opposing horizontal or vertical buttons are rejected. Negative,
stale, future, malformed, overlong, embedded-NUL, extra-field and unknown commands
are discarded without changing sequence, input mask or engine calls. A rejection
publishes the same paused state with an increased `rejected` count and retains a
local raw prefix (up to 255 bytes) plus reason/counter receipt. No rejected command
renews the watchdog.

The six mask bits are left=1, right=2, up=4, down=8, jump=16, dash=32. Inputs remain
held across adjacent steps with the same bit set. Native `p_jump/p_dash` implement
press edges: repeating jump/dash without a release does not trigger again. The
original engine decides cooldowns, collision, dash availability and deaths.

| Model option | Mask |
| --- | --- |
| wait / left / right | 0 / 1 / 2 |
| jump / jump_left / jump_right | 16 / 17 / 18 |
| dash_left / dash_right / dash_up / dash_down | 33 / 34 / 36 / 40 |
| dash_up_left / dash_up_right | 37 / 38 |
| dash_down_left / dash_down_right | 41 / 42 |

Reset restores room/player/objects/RNG/timers from the same process's upstream
save/load functions. Playing audio, cached render pixels and bridge counters are
excluded. Reset performs no draw; the cached image changes on the next step.
This is not arbitrary state loading, cross-build portability or validated
bit-exact replay. Stop, window close and watchdog do no gameplay work. Allocation,
observation/capture I/O errors terminate the player with a nonzero code. The client
fails on unexpected exit, timeout, nonpaused state or mismatched action/counters.

`HONING_WATCHDOG_SECONDS` is integer 1..3600 (default 240); the normal runner uses
model timeout +30 seconds, capped at 3600. Every poll checks the deadline before
reading any command. Only a syntactically valid, current command renews it. A
continuous stale/invalid stream cannot bypass it. Python cleans up its children
on model/bridge exceptions too. The CLI model timeout defaults to 180 seconds.

## Model response acceptance

The client uses existing Honing health/decide routes, with one 14-way `action`
question. The entire prompt and observation are `essential`; no baseline context
is optional. If an operator adds `parts.logs`, only audited oldest-first whole-log
removals are permitted. Essential, details and memory entries cannot be removed.

Before sending input it requires strict nonboolean integer request ID equality,
exact answer/question and probability-option sets, offered choice at a maximum
probability (ties allowed), finite numeric nonboolean values within [0,1], and a
sum within `14 * 0.00005 + 1e-8` of one. This bound follows the pinned Laya's
four-decimal rounding; invalid distributions are never normalized.

`stateText` must equal exactly the retained submitted sections in serving order.
`trimmed` is required and checked against permitted removals. Every question must
have `tokenAudit` fields stateTokens/stateRoom/inputTokens/options/stateTruncated;
option count must match and stateTruncated must be false. Present counts must be
nonnegative integers, state must fit room/input totals and any maxInputTokens.
DJEV's explicit null counts are allowed; missing fields are rejected.

Raw response bytes, including HTTP failures and malformed/duplicate-key/nonfinite
JSON, are saved before decoding/validation. Every attempted decision logs its
request, raw-file reference, decoded response when serializable, accepted input
or rejection reason, and native before/after states. Rejection sends zero native
commands. Portable regressions use synthetic responses; native gate checks inject
a deliberately bad fixture while the actual game is paused. They remain separate
from the real Laya run.

Local artifacts include model/build/process identity, response bytes,
`decisions.jsonl`, numbered BMPs when captured, result/video metadata, player/Xvfb
logs, native rejection diagnostics and owned-process cleanup receipts. These stay
local. A completed capture proves neither a room clear nor independent replay.
