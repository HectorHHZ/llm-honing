# hallowdale bridge contract

This contract describes `honing.ml`, `honing_input.ml`, `honing_command.ml` and `run.py` at the game
revision pinned in [build.py](build.py). The integration is a native play-and-record demo: it has no
reset endpoint, reward, grader or campaign runner. Build and model setup are in [README.md](README.md).

## Transport and ownership

`run.py` creates a new session directory, starts its own Xvfb and game process and sets
`HONING_SESSION_DIR`. Without that variable the game behaves like the original. One controller owns one
session. `action.json` is a single command slot, not a queue.

| File in the session | Written by | Content |
| --- | --- | --- |
| `observation.json` | bridge | Latest human-visible observation; the only game state a model may receive |
| `visible_observations.jsonl` | bridge | Every visible observation, one per line |
| `debug_observation.json`, `observations.jsonl` | bridge | Global state for plumbing and evaluation; never sent to a model |
| `action.json` | client | One command, published with a same-directory rename |
| `ack.json`, `acks.jsonl` | bridge | What the bridge applied or refused, for every command it read |
| `bridge.log` | bridge | Start, stop, refusals and timeouts |
| `frames/%06d.png` | bridge | One picture per bridge frame when `HONING_CAPTURE=1`; never sent to a model |
| `saves/` | game | Save files of this session |

The game is paused between windows. Nothing ticks until a valid command with the id of the latest
observation arrives.

## Visible observation

The observation gives a model what a human player can see on the real game window, as a grid plus a list
of the things in view. It holds no pixels.

The bridge renders the scene a second time with the game's own camera and draw order, painting every
drawable in a flat colour that encodes its class and object. Occlusion is therefore what the player sees:

- Only the real window is reported. Cells beyond it are `unknown` and no object lies outside it.
- Anything covered by terrain, shadow layers, the HUD, a text box or a menu is not reported.
- Cells covered by the HUD or by other opaque UI are `unknown`. The health and soul values that the HUD
  prints are reported as numbers in `hud`.
- While the screen is faded (alpha 200 or more) every cell is `unknown`, and there are no objects or HUD values.
- Map files and global game state do not enter this file. `debug_observation.json` is separate.

| Field | JSON type | Meaning |
| --- | --- | --- |
| `schema`, `adapter`, `profile`, `game` | string | `honing.visible.v1`, `hallowdale-visible-v1`, `visible`, `hallowdale` |
| `frame.id` | integer | Observation sequence from 1. Commands must echo the current value |
| `frame.tick` | integer | Bridge frames played in action windows since boot, from 0 |
| `frame.simulation_time` | number | Game clock in seconds. It starts at 10 and includes the boot sequence |
| `view.screen_rect` | `[0, 0, w, h]` | Screen in pixels, 960 x 523 by default |
| `view.world_from_screen` | 6 numbers | `[a, b, c, d, e, f]`: world x = a·sx + c·sy + e, world y = b·sx + d·sy + f |
| `view.window` | object | `size`, `rect` of the real window on this screen, `player_centred` (false by default) |
| `view.world_units` | string | 1 unit is one map tile, x right, y up, origin at the room's top-left corner |
| `terrain.cell_size` | `[c, c]` | Cell size in screen pixels, `HONING_VISIBLE_CELL_PIXELS`, default 20 |
| `terrain.rows` | list of strings | The screen as cells, top row first. See below |
| `terrain.legend` | object | Text meaning of each class |
| `objects[].id` | string | Stable name inside one room, `<room>/<name>` |
| `objects[].kind` | string | `player`, `enemy`, `wall`, `platform`, `interactable`, `visible_hazard` or `visible_scenery` |
| `objects[].bounds` | `[x, y, w, h]` | Visible part of the object in screen pixels, x right, y down |
| `objects[].look` | string | What it looks like, for example `terrain`, `door`, `jar`, `lever`, `spikes`, `acid`, `PENGUIN` |
| `self` | object | `object_id`, `screen_box` `[x, y, w, h]` and `facing`; only `object_id: null` when the player is not in view |
| `hud.health`, `hud.health_max` | object | `value` as drawn on the HUD; absent in Steel Sole mode and while faded |
| `hud.soul` | object | `fraction` of the gauge, 0 to 1; absent while faded |
| `hud.text` | list | On-screen text: `kind` (`text`, `dialogue`, `menu`, `corner`, `floating`, `label`), `blocking`, `content`, plus `speaker` for dialogue and `bounds` for a label |
| `rules`, `capabilities`, `disclosure` | list, object | Fixed description of the game and of this observation |

Each character of `terrain.rows` is U+0100 plus the sum of the classes seen in that cell:

| Bit | Class | Meaning |
| --- | --- | --- |
| 1 | background | Visible background; not a collision guarantee |
| 2 | solid | Solid-looking terrain and solid doors |
| 4 | platform | Floating platform |
| 8 | hazard | Spikes, acid or a projectile |
| 16 | actor | Player or enemy sprite |
| 32 | interactable | Something that reacts to a nail strike: door, jar or lever |
| 64 | unknown | Outside the window, behind the HUD or a text box, or the screen is faded |
| 128 | scenery | Scenery, debris, water, a dead enemy or the player's own strike |

Texels below 0.9 alpha count as not drawn. A shadow layer that hides a secret room looks like terrain and is
reported as solid, as it looks to a player.

`HONING_VIEW_ANCHOR`, `HONING_VIEW_ZOOM` and `HONING_VIEW_SIZE` can cut a player-centred part out of the
window. They can only remove information, are off by default, and `run.py` never sets them.

### Debug state

`debug_observation.json` holds the player's room coordinates, velocity, health, soul, abilities, death count
and every enemy in the room, including those off screen. It exists for evaluation and for the checks in
`verify_native.py`. `run.py` does not read it.

## Commands

| Field | JSON type | Required and meaning |
| --- | --- | --- |
| `id` | integer | Required, the current `frame.id` |
| `action` | string | Required: a named action below, `keys`, or `stop` |
| `frames` | integer | Required except for `stop`: 1..600 bridge frames (`run.py` uses 1..120) |
| `jump_frames` | integer | Optional, 1..`frames`, only on a command that presses JUMP. See "Jump height" |
| `hold` | list of keys | Optional: more keys held for the whole window |
| `press` | list of keys | Optional: more keys pressed at the start of the window |
| `first` | list of direction keys | Optional: directions down in the first tick instead of the held directions |
| `model`, `probability` | any | Optional labels, ignored |

Key names are `JUMP`, `NAIL`, `DASH`, `FOCUS`, `C_DASH`, `D_NAIL`, `CAST`, `INTERACT`, `UP`, `DOWN`, `LEFT`, `RIGHT`.

### Named actions

The list comes from the game's own control table (`all_game_actions` in `src/controls.ml`). "Available" says when
the game lets the action take effect. A locked ability behaves as in the game: the press does nothing. The bridge
unlocks nothing. A new game starts with no abilities, 5 health and a full soul gauge.

| Action | Held | Pressed | Available |
| --- | --- | --- | --- |
| `wait` | | | Always |
| `left`, `right` | `LEFT` / `RIGHT` | | Always |
| `up`, `down` | `UP` / `DOWN` | | Always; they aim strikes and spells |
| `jump` | | `JUMP` | On the ground; in the air with `monarch_wings`; on a wall with `mantis_claw`; in water |
| `jump_left`, `jump_right` | `LEFT` / `RIGHT` | `JUMP` | As `jump` |
| `attack` | | `NAIL` | Always; strikes the way the player faces |
| `attack_left`, `attack_right` | `LEFT` / `RIGHT` | `NAIL` | Always |
| `attack_up` | `UP` | `NAIL` | Always |
| `attack_down` | `DOWN` | `NAIL` | In the air; on the ground the strike goes sideways |
| `dash` | | `DASH` | `mothwing_cloak` |
| `dash_left`, `dash_right` | `LEFT` / `RIGHT` | `DASH` | `mothwing_cloak` |
| `interact` | | `INTERACT` | At something readable or at someone to talk to; advances open text |
| `focus` | `FOCUS` | | On the ground, soul of 33 or more, health below maximum. Heals 1 after 1.5 s of holding |
| `cast` | | `CAST` | `vengeful_spirit` and soul of 33 or more |
| `cast_left`, `cast_right` | `LEFT` / `RIGHT` | `CAST` | As `cast` |
| `cast_up` | `UP` | `CAST` | `howling_wraiths` and soul of 33 or more |
| `cast_down` | `DOWN` | `CAST` | `desolate_dive` and soul of 33 or more |
| `c_dash` | `C_DASH` | | `crystal_heart`, on a floor or wall. Launches on release after 0.8 s of charge |
| `dream_nail` | `D_NAIL` | | `dream_nail`, on the ground. Swings after about 0.95 s of holding |
| `keys` | `hold` | `press` | Any combination of the keys above, for example walking right while striking up |

`focus`, `c_dash` and `dream_nail` are holds. Repeating one keeps the key down, so charging continues across
windows. Any other action releases the key, and that release is what launches a charged c-dash. Pressing
`jump` or `c_dash` during a c-dash ends it. Holding `C_DASH` during a skippable cutscene skips it, as on a keyboard.

Controls that are left out, and why:

| Control | Why |
| --- | --- |
| `PAUSE` | Opens the pause menu, which stops the game and offers settings, key rebinding and quitting |
| `OPEN_MAP` | Opens the world map, which stops the game |
| `DEBUG_*` | Have no key outside the game's development build; the bridge runs as a release build |

A command that names one of these is refused. The title and save-file menus are passed by a fixed boot sequence
before the first observation.

### Key timing inside a window

One bridge frame is 1/30 s and plays 5 game ticks of 1/150 s.

- A pressed key goes down at tick 0 and stays down for one bridge frame. JUMP follows "Jump height" below, and in
  `sticky` mode it always goes down at tick 1.
- If a pressed key was still down at the end of the previous window, it is let go for tick 0 and goes down at
  tick 1. A repeated jump, strike, dash or cast therefore always has a new press edge.
- A held key is down in every tick. A held key that continues from the previous window gets no release.
- `first` replaces the held directions in tick 0 only.
- A press that lands while the game freezes the player follows the game's own input buffer, as on a keyboard.

### Jump height

The game cuts a jump when the jump key is released while the player is rising, so height follows how long the key
is held. A command without `jump_frames` keeps the existing meaning, set by `HONING_JUMP_MODE`:

| `HONING_JUMP_MODE` | JUMP stays down | Used by |
| --- | --- | --- |
| `window` (bridge default) | To the end of the window | Direct clients |
| `sticky` | Until the next jump press, which first releases it for one tick. Every jump is a full jump | `run.py` |

With `jump_frames: n`, JUMP stays down for `n` bridge frames from the press and is then released, in either mode,
and nothing stays latched afterwards. Measured heights from the same floor, in tiles:

| `jump_frames` | 1 | 3 | 6 | 12 | not set (`sticky`) |
| --- | --- | --- | --- | --- | --- |
| Height | 1.413 | 3.139 | 5.313 | 8.183 | 9.093 |

## Acknowledgement

The bridge publishes one acknowledgement per command it reads, and publishes it before the next observation. Every
acknowledgement has a running number `ack`. An applied window reports what reached the game:

```json
{"ack":10,"id":10,"status":"applied","action":"jump_left","frames":12,"jump_frames":3,
 "keys":{"JUMP":{"first_tick":1,"last_tick":15,"ticks":15},"LEFT":{"first_tick":0,"last_tick":59,"ticks":60}},
 "ticks_per_frame":5,"frame_before":108,"frame_after":120,"observation_id":11,"started":[],"children":[]}
```

| Field | Meaning |
| --- | --- |
| `keys` | For every key that was down: first tick, last tick and number of ticks, counted from the start of the window |
| `frame_before`, `frame_after` | `frame.tick` before and after the window |
| `observation_id` | Id of the observation that follows this window |
| `started` | Actions the game recorded in its own history during the window: `NAIL`, `DASH`, `SHADE_DASH`, `FLAP`, `WALL_KICK`, `FOCUS`, `DREAM_NAIL`, `C_DASH_CHARGE`, `C_DASH`, `C_DASH_COOLDOWN`, `CAST_VENGEFUL_SPIRIT`, `CAST_HOWLING_WRAITHS`, `CAST_DESOLATE_DIVE`, `DIVE_HOP`, `TAKE_DAMAGE`, `TAKE_DAMAGE_AND_RESPAWN`, `DIE`, `HARDFALL`. A plain jump is not recorded there |
| `children` | What is attached to the player when the window ends, for example `NAIL_UP`, `FOCUS`, `WRAITHS`, `DREAM_NAIL`, `C_DASH_CHARGE_CRYSTALS` |

`started` and `children` read game state, so they are plumbing for checks and are not given to a model.

A refused command plays nothing:

```json
{"ack":13,"id":1,"status":"refused","reason":"key \"F12\" cannot be used","observation_id":1}
```

`{"id":<current>,"action":"stop"}` is acknowledged with `status: "stopped"`, and the game exits.

## Stale, duplicate and invalid commands

- A command runs only when its `id` equals the current `frame.id`. Each applied window advances the id, so an old
  command left in the slot cannot run again.
- Every other command is refused before anything is played: malformed JSON, duplicate keys, a non-integer `id`,
  a stale or future `id`, missing or out-of-range `frames`, an unknown action or key, a non-direction key in `first`,
  or an invalid `jump_frames`.
- The bridge reads a given file content once. To retry, write a new command.
- The idle watchdog is checked on every pass of the wait loop. Refused and stale commands do not extend it. Without
  a valid command for `HONING_MODEL_TIMEOUT_SECONDS` the bridge logs `timed out waiting for an action` and exits.
  `run.py` sets this to the model timeout plus 30 seconds.
- If the player dies on the last tick of a window, the bridge plays the game's one reload tick before it publishes
  the observation. That tick takes no input, is counted in the debug `fixedSteps` and not in `frame.tick`.

## Timing

| Measure | Value |
| --- | --- |
| Bridge frame | 1/30 s of game time, `frame.tick` counts these |
| Game tick | 1/150 s, fixed step, close to the original's 144 fps; debug `fixedSteps` counts these and includes boot |
| Model waiting time | Not played; the game is paused |
| Video from `encode.py` | One picture per bridge frame at 30 fps; playback time, not inference time |

The boot sequence selects a new game in Classic mode, advances the opening text with `INTERACT` and waits until the
player has control. It is scripted by the bridge, runs before the first observation and is not a model decision. The
random seed is fixed (`HONING_SEED`, 42 by default). The same commands gave byte-identical observations in the
regression replay described in the README. This does not promise identical results on another machine or renderer.

## Request sent by `run.py`

`run.py` sends the observation's changing fields as required context and offers 28 choices in one question.
`screen`, `rows`, `objects`, `self` and `hud` come from `observation.json`, with the room name removed from object
ids. The fixed rules and the cell legend are in the prompt. The grid uses 40 pixel cells by default (`--cell-pixels`),
so that a request fits a 2048 token Laya context.

The choices are the named actions above, except that `keys` is not offered and three low hops are added:
`short_jump`, `short_jump_left` and `short_jump_right` send `jump`, `jump_left` or `jump_right` with `jump_frames: 3`.

Before any command is written, the [shared receiver](../_shared/README.md) checks the response id, the complete
retained state, the token audit, every probability and the argmax. No context may be removed. The raw response bytes
are saved before decoding. A rejected response writes no command, and there is no scripted fallback. After each
window `run.py` compares the acknowledgement with the command it sent and stops on any difference.

## Session files of `run.py`

`model.json`, `launch.json`, `decisions.jsonl` (request, response, `responseRawFile`, choice, probability,
acknowledgement or error), `responses/<id>.raw`, `stdout.log`, `xvfb.log`, `cleanup.json`, the bridge files above, and
`result.json` after a run that completed. These names follow the Hollow Knight integration. They are not a shared
trace format.

## Bridge settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `HONING_SESSION_DIR` | unset | Turns the bridge on |
| `HONING_MODEL_TIMEOUT_SECONDS` | 180 | Idle watchdog in seconds; positive and finite |
| `HONING_JUMP_MODE` | `window` | `window` or `sticky` |
| `HONING_VISIBLE_CELL_PIXELS` | 20 | Cell size of the grid |
| `HONING_SCENE` | `new` | `new`, or `save:<file>` for a debug start from a save file |
| `HONING_SEED` | 42 | Random seed |
| `HONING_CAPTURE` | 0 | 1 saves one picture per bridge frame |
| `HONING_RENDER` | `all` | Which ticks are drawn: `all`, `frame` or `window`; observations are the same |
| `HONING_OBSERVATION_PROFILE` | `visible` | `global_debug` publishes the debug state as `observation.json`; `run.py` refuses it |
| `HONING_FRAME_HZ`, `HONING_TICKS_PER_FRAME` | 30, 5 | Bridge frame rate and ticks per frame |
| `HONING_WINDOW_SCALE` | 20/33 | Window scale; the default gives a 960 pixel wide window |

Invalid numbers and unknown mode names stop the game at startup. `run.py` passes no inherited `HONING_*` variable to the game.
