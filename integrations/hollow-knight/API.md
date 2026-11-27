# Hollow Knight input demo contract

This contract describes `HoningBridge.cs` and `run.py`, using the game revision in `pins.json`.
It is a native input demonstration, with no reset endpoint, reward, grader or deterministic
replay guarantee. Build prerequisites and model setup are in [README.md](README.md).

## Transport and ownership

`run.py` creates a fresh session, starts its own Unity player, and sets `HONING_SESSION_DIR`.
The bridge is inactive when that variable is absent. One controller owns one session; files
are a single command slot, not a queue. Do not share the directory between players/controllers.

The bridge publishes UTF-8 `observation.json` and appends `observations.jsonl`. The controller
waits for a new observation ID, calls `GET /v1/health` then `POST /v1/decide` on an operator-supplied
Honing server, validates its answer, and publishes `action.json` with a same-directory temporary
file and rename. Readers tolerate the bridge's short delete/rename gap while publishing state.
The bridge does not produce a separate command acknowledgement or structured error envelope;
the next observation acknowledges an accepted window by incrementing its ID and frame count.

## Observation

Every field below is emitted on every observation. There are no client-side substitutions for
missing fields. Values come from the live player/scene, not default template values; empty
enemy/exit arrays are `[]`. Coordinates and distances are Unity world units, not pixels;
`+x` is right and `+y` is up. Floats are finite JSON numbers in the supported game build.

| Field | JSON type | Meaning, units and initial/default behavior |
| --- | --- | --- |
| `id` | integer | Observation sequence starting at 1; increments once per published snapshot. Persists across scene changes; a new player/session restarts it. Commands must echo the current ID. |
| `scene` | string | Current Unity scene name. CLI default is `Level1`; supported CLI choices are `Spawn`, `Level1`, `Level2`. Direct bridge startup defaults to `Spawn` if `HONING_SCENE` is absent. |
| `frame` | integer | Completed action render/update frames since bridge startup, initially 0; increases by accepted `frames`. Waiting/initialization frames are excluded. It is not a physics tick or screenshot of the initial state. |
| `fixedSteps` | integer | Count of `FixedUpdate` calls while an action is active, initially 0. Multiple physics steps can occur per action frame. It is not total physics work during startup. |
| `simulationTime` | number | Unity scaled `Time.time`, in seconds since player startup. Includes startup; can be nonzero at the first snapshot. It is not inference wall time and does not reset on scene changes. |
| `x`, `y` | number | Player transform position in world units. Starting values depend on the selected scene. |
| `vx`, `vy` | number | Player `Rigidbody2D.velocity`, in world units per simulation second. |
| `health` | integer | Current player hit points. The tested scene starts with 5; no client default or healing is applied. Zero alone is not a graded episode result. |
| `jumps` | integer | Controller's remaining jump count (`jumpLeft`), normally 0–2. Grounding/climbing updates it under the game's rules; it can be 0 in an airborne initial snapshot. |
| `grounded` | boolean | Bridge probe: downward circle cast of radius 0.2 for 0.5 units against `Platform`. False means that probe missed; this is not a promise that every controller action is permitted. |
| `floorLeft`, `floorRight` | boolean | Downward ray cast for 2 units, starting 1.2 units left/right of the player. Tests `Platform`, `Wall`, `AirWall`; false means no such terrain within the probe. |
| `wallLeft`, `wallRight` | number | Horizontal ray-hit distance in world units, range 0–2, testing the same three terrain layers. No hit returns 2; a hit at the maximum distance is indistinguishable from no hit. |
| `enemies` | array of enemy objects | At most six enemies, selected as described below. Empty when no matching objects are found. |
| `exits` | array of exit objects | Bounds of scene `NextLevel` collider objects; empty when none are found. No nearest-exit or visibility filter. |

Enemy fields are all present; there are no missing-field defaults:

| Field | JSON type | Meaning / units |
| --- | --- | --- |
| `name` | string | Unity object name. Not a stable or unique entity ID. |
| `x`, `y` | number | Enemy transform position, world units. |
| `health` | integer | Current enemy hit points; this is not player damage or a reward. |

Exit fields are all present; there are no missing-field defaults:

| Field | JSON type | Meaning / units |
| --- | --- | --- |
| `x`, `y` | number | Center of the collider's world-space axis-aligned bounds. |
| `width`, `height` | number | Full size of those bounds in world units, not half extents. |

### Visibility and input boundary

The bridge uses Unity object discovery, not image perception. It sorts found `EnemyController`
objects by Euclidean distance from the player and takes up to six. Equal-distance ordering and
enemy names are not identity guarantees. It exposes found `NextLevel` collider bounds from the
scene. Neither list is filtered by camera frustum, screen visibility, occlusion or line of sight;
off-screen enemies/exits can be provided to the model. It does not reveal inactive objects not
returned by Unity's `FindObjectsOfType` or the game's entire internal state.

Ground/wall probes are the explicit local ray/circle queries above, not a map or pathfinder.
The adapter replaces seven player input reads; it does not bypass movement, health, collisions,
attack/dash cooldowns or coroutines. An accepted command means the input was delivered, not that
the character necessarily moved, jumped, dealt damage or crossed a scene boundary.

## Visible observation profile

Setting `HONING_OBSERVATION_PROFILE=visible` in the player's environment replaces `observation.json` with what a person
sees on the screen. Any other value, or no value, keeps the scene-state observation above and the behavior of the rest
of this document. `run.py` does not set the variable. The [harness v8 experiment](experiments/harness-v8/README.md) does.
This profile is the one that experiment used. It is not a settled observation format for other games.

| File | Content with the profile on |
| --- | --- |
| `observation.json` | The `honing.visible.v1` envelope below; the only file the harness reads |
| `visible_observations.jsonl` | One envelope per observation |
| `debug_observation.json`, `observations.jsonl` | The scene-state observation, kept for evaluation |
| `scene-inspection.txt` | Cameras, renderers and colliders of the first observed scene, for people checking the adapter |

The envelope is published after the last frame of a window has been rendered. Game time stops at the same point as
without the profile. Commands, `id` sequencing and window lengths work as described in this document, and `frame.id` is
the `id` a command must echo.

| Field | JSON type | Meaning |
| --- | --- | --- |
| `schema`, `adapter`, `profile`, `game` | string | `honing.visible.v1`, `hollow-knight-visible-v1`, `visible`, `hollow-knight-imitation` |
| `frame.id`, `frame.tick`, `frame.simulation_time` | integer, integer, number | Observation id, completed action frames, Unity scaled time in seconds |
| `view.screen_rect` | integer array | `[0, 0, 960, 540]`; screen pixels, `+x` right and `+y` down |
| `view.world_from_screen` | number array | `[a, b, c, d, e, f]`: world x = `a*sx + c*sy + e`, world y = `b*sx + d*sy + f` for screen point `(sx, sy)` |
| `terrain.cell_size`, `terrain.rows` | integer array, string array | Cell size in pixels and the grid, top row first |
| `terrain.legend` | object | Descriptions of the classes below, except scenery |
| `objects[]` | array of `{id, kind, bounds}` | Every thing with at least one visible pixel. `bounds` is `[x, y, width, height]` of those pixels in screen pixels |
| `self.object_id` | string | The `objects` id of the player sprite; empty while no pixel of it is drawn, for example when it blinks after a hit |
| `hud.health.value` | integer | Full hearts when all five heart images are drawn unobstructed. Otherwise 0, because Unity's JSON writer cannot omit the field: the first picture after a level exit can read 0 at full health |
| `rules`, `capabilities`, `disclosure` | strings, booleans | Fixed text describing the game's rules and this observation's limits |

The default grid has 48 columns and 27 rows of 20 pixel cells. `HONING_VISIBLE_CELL_PIXELS` selects another size between 5
and 120 that divides both 960 and 540. Each character of `terrain.rows` is `U+0100 + mask`, where `mask` is the union of
the classes of every pixel in that cell:

| Bit | Class | Source |
| --- | --- | --- |
| 1 | background | Nothing was drawn at that pixel |
| 2 | wall | Sprite on the `Wall` or `AirWall` layer |
| 4 | platform | Sprite on the `Platform` layer |
| 8 | hazard | Sprite on the `Trap` or `Projectile` layer |
| 16 | actor | The player or an enemy |
| 32 | interactable | Sprite on the `Switch` layer or a level exit |
| 64 | unknown | Translucent or unsupported sprite pixels, UI rectangles, and the caption box |
| 128 | scenery | Decoration and the player's own equipment |

`objects[].kind` is one of `player`, `visible_equipment`, `enemy`, `visible_hazard`, `platform`, `wall`, `interactable` or
`visible_scenery`. An `id` stays the same while the object exists.

Classes come from a second render of the real camera in which each sprite is drawn in a flat label color, so sprite
order and occlusion match the picture and anything hidden behind another sprite is not reported. Terrain classes describe
what a sprite looks like, not its collider. An unknown cell is never evidence of free space. With `HONING_CAPTURE=1` the
caption box the bridge draws into the recording (x 8 to 667, y 8 to 55) is unknown as well, so a captured run and an
uncaptured run can see slightly different grids. `HONING_VISIBLE_MASK_CAPTURE=1` also saves each label image as
`semantic-<id>.png` for people to inspect; label images are never model input.

Withheld on purpose: anything off screen including exits, enemy health, the player's velocity, grounded state and
remaining jumps, colliders without a sprite, and what a switch or secret does. A camera that is not an axis-aligned
orthographic 960x540 view, a non-sprite renderer in view, or a sprite with an unsupported shader or mask raises an error
in the bridge rather than publishing a partial observation, and the controller's observation timeout then ends the run.
The `disclosure.resolution` text mentions a compact glyph adapter; none is part of this repository, and `terrain.rows`
is always in the encoding described here.

## Commands

| Field | JSON type | Required / default / interpretation |
| --- | --- | --- |
| `id` | integer | Required, exactly the current observation ID. Missing deserializes to 0 and is rejected. |
| `action` | string | Required, one of the 11 inputs below or `stop`. Missing/null/unknown actions are rejected. |
| `frames` | integer | Required for inputs, inclusive range **1..120**. No implicit bridge default: missing deserializes to 0 and is rejected. CLI `--frames` defaults to 12. Ignored for `stop`. |
| `model` | string | Optional display-only model label, default null/empty. Does not choose a policy. |
| `probability` | number | Optional display-only normalized model score, default 0. The HTTP client validates it as part of the distribution before generating this file; it is not a win probability. |

| Input | Held axes during the window | Single button press at window start |
| --- | --- | --- |
| `wait` | horizontal 0, vertical 0 | none; gravity and existing coroutines still run |
| `left`, `right` | horizontal −1 / +1 | none |
| `jump_left`, `jump_right` | horizontal −1 / +1 | Jump, subject to the game's remaining jumps/climbing rules |
| `dash_left`, `dash_right` | horizontal −1 / +1 | K, subject to dash availability/cooldown |
| `attack_left`, `attack_right` | horizontal −1 / +1 | J; the horizontal input also requests movement |
| `attack_up`, `attack_down` | horizontal 0, vertical +1 / −1 | J; downward attacks depend on the game's airborne rules |

Unspecified axes are zero. The adapter emits no Jump-release edge. A held window does not
repeatedly press Jump/J/K; another command can generate another press. `stop` with the current
ID requests `Application.Quit` immediately while waiting and does not advance an action window.

### Stale, duplicate, invalid and delayed commands

- Only a waiting bridge consumes a command. While an action is active it does not poll the slot.
  Writers must wait for the next observation; overwriting the slot early is unsupported.
- Each accepted input advances `id` exactly once after its window. Leaving/repeating its old
  command file does not repeat the input; it becomes stale. Older and future IDs are ignored.
- Invalid frame counts, missing required fields, unsupported actions and JSON parse errors
  leave the bridge waiting; no structured error reply is emitted. Replace the file with a valid
  current-ID command to recover. Do not use repeated commands as a retry queue.
- Invalid/stale files do not restart or bypass the watchdog. No valid input before the deadline
  logs `Honing bridge timed out waiting for a model response.` and exits the player. Its process
  exit code alone is not proof that an episode completed successfully.
- `run.py` does not retry HTTP decisions. HTTP/network, validation and observation errors abort
  the session and terminate its owned child, escalating to kill after 5 seconds if necessary.
  It does not stop unrelated games, model servers or Wine sessions.

The default HTTP timeout is 180 wall seconds; `--timeout` changes it. `run.py` supplies that
value plus 30 seconds as `HONING_MODEL_TIMEOUT_SECONDS`. Direct bridge use without the variable
defaults to 180 seconds. `--observation-timeout` defaults to 60 wall seconds; normal final stop
must exit cleanly within 15 seconds. All timeouts are distinct from Unity simulation time.

## Required model response

The generated request places both the rules and the complete serialized observation in
`parts.essential`, with one `questions.action` choice containing all 11 inputs. All submitted
state is required; no optional history is sent. Before any `action.json` write the client requires:

1. An integer `id` exactly matching the original request.
2. `trimmed: []` and `stateText` exactly equal to the original state entries joined with newlines
   in Honing section order (`essential`, `details`, `memory`, `logs`). Missing/changed echoes fail.
3. A `tokenAudit` object covering exactly every submitted question. Each audit must explicitly
   report boolean `stateTruncated: false` and the correct integer option count (11 for `action`).
   Available token counts must be nonnegative integers and must not contradict the reported
   state room, input length or maximum length. Null/unavailable counts are permitted for DJEV;
   they never substitute for an explicit untruncated audit or the exact state echo.
4. All 11 probabilities, finite non-boolean numbers in `[0,1]`, summing to 1 within Laya's
   four-decimal rounding tolerance `11 * 0.00005 + 1e-8`. The selected action must be an argmax;
   ties accept any maximal option. No scripted fallback is used.

These are consistency checks on the serving response, not independent access to model internals.

## Examples and artifacts

A complete example observation (numbers are illustrative):

```json
{"id":1,"frame":0,"fixedSteps":0,"simulationTime":0.0667,"scene":"Level1","x":-6.56,"y":4.66,"vx":0.0,"vy":-0.59,"health":5,"jumps":0,"grounded":false,"floorLeft":false,"floorRight":false,"wallLeft":2.0,"wallRight":2.0,"enemies":[{"name":"Enemy (1)","x":24.37,"y":2.18,"health":3}],"exits":[{"x":113.71,"y":11.06,"width":0.45,"height":4.45}]}
```

An accepted command and its successful acknowledgement:

```json
{"id":1,"action":"right","frames":12,"model":"Honing decision server","probability":0.42}
```

The next observation has `id: 2`, `frame: 12`, updated `fixedSteps`/`simulationTime` and the
remaining fields above. Exact physics/position changes depend on the game. A following
`{"id":2,"action":"stop"}` exits. Resending the first command while ID 2 is current is stale.
`{"id":2,"action":"right","frames":121}` is invalid and leaves the bridge waiting.

For the exact full HTTP request/response (including all probabilities, state echo and audit),
see each line's `request` and `response` in [the recorded trace](validation/laya-cpu-trace.jsonl).
The complete success schema also permits model metadata/latency fields used by Honing.
For example, a valid action audit can be:

```json
{"action":{"stateTokens":361,"stateRoom":1914,"inputTokens":495,"options":11,"stateTruncated":false}}
```

Error example: a response containing `trimmed: ["essential"]`, changed `stateText`, or
`tokenAudit.action.stateTruncated: true` is rejected even if its ID/probabilities are valid.
`decisions.jsonl` retains `request`, `response` when JSON decoding succeeds, `responseRawFile`
(e.g. `responses/000001.raw`) and an `error` reason. No game command is written for that response,
and no successful `result.json` is created. On later-decision failure the previous, now-stale
command can remain in the slot; it is not a new action.

HTTP 503 with body `{"error":"model unavailable"}`, malformed JSON, invalid encoding and
nonstandard numeric constants (`NaN`/`Infinity`) and numeric overflow (`1e309`) also abort. Exact response body bytes are saved
before JSON decoding/validation, including HTTP error bodies. A network failure before any
body arrives has a request/error record but no raw-body file. There is no automatic retry.

Successful sessions contain `model.json`, `observations.jsonl`, `decisions.jsonl`, `responses/`,
player/stdout logs and `result.json`; optional `frames/000000.png` onward are game-render exports.
`result.json` says how many decisions completed and whether capture/headless/fallback was used;
completion means the requested input windows and clean stop finished, not level completion.

## Timing, terminal behavior and reproducibility

Unity sets `Time.timeScale = 0` between inputs. During a window it resumes scaled time,
targets 30 fps, and uses `Time.captureFramerate = 30`. A resume frame may advance the render
counter without advancing simulation time. Render frames, `FixedUpdate` counts, simulation
seconds, encoded-video duration and inference wall time are therefore different measurements.
Video encodes action frames only; the initial and waiting state are not extra video frames.

The bridge seeds Unity random state with 42 on startup, but does not reset all engine/persistent
state or promise deterministic replay. Scene changes remain the original game's behavior;
counters survive because the bridge persists. There is no `done`/`reward` field. Health 0,
level exits and process exit are not benchmark grades. A missing player or stalled scene can
lead to the controller's observation timeout and owned-child cleanup.

Run `verify_native.py` as described in README for the fixed scripted sequence, duplicate/invalid
input checks and native watchdog receipts. Run the real-model command separately. Author-side
native receipts, portable tests/CI and an independent reviewer's native rerun are separate evidence.

## Shared decision-response receiver

The adapter imports the [shared stdlib-only receiver](../_shared/README.md).
It requires `answers.action.type: "choice"` and explicit `stateTokens`,
`stateRoom` and `inputTokens` audit keys; null counts are permitted, omitted
keys are rejected. It rejects duplicate JSON keys at every nesting level as
well as nonfinite numbers. Raw response bytes are saved before decoding,
including HTTP error bodies. The game's required/optional context policy
above remains in effect; invalid decisions cannot reach native input.
