# Brogue CE

[Brogue Community Edition](https://github.com/tmewett/BrogueCE) is a turn-based
dungeon crawler with procedurally generated floors, stealth, combat, and item
identification. This scenario runs the upstream rules and saves the engine's
own `.broguerec` recordings. The official Tiles client plays them back.

## Setup

From the Honing repository root, build a separate WebBrogue executable. The
script requires Git, Make, and a C compiler on macOS or Linux.

```bash
bash integrations/brogue_ce/setup.sh /path/to/brogue-ce-engine
```

The script checks out [Brogue CE 1.15.1](https://github.com/tmewett/BrogueCE/releases/tag/v1.15.1)
at `1ba4240b7a928ddf0ffb772717bf1d433cd63804`. Its patch makes the upstream
WebBrogue driver compile, translates the display glyphs, and keeps display
packets from being dropped under local socket backpressure. The source and
compiled executable stay in the engine directory.

For a window with the original sprites, also download the graphical build of
the same release. On Intel macOS, choose `BrogueCE-1.15.1-macos-x86_64.zip`.
The executable is inside the bundle at `Brogue CE.app/Contents/MacOS/brogue`.
Brogue CE is AGPL-3.0. The original tile credits and CC BY-SA 4.0 notice ship
with the upstream assets.

## Run and watch the demo

```bash
python3 -m integrations.brogue_ce demo \
  --engine /path/to/brogue-ce-engine/bin/brogue \
  --out integrations/brogue_ce/runs/seed7

python3 -m integrations.brogue_ce verify \
  --engine /path/to/brogue-ce-engine/bin/brogue \
  integrations/brogue_ce/runs/seed7/match.broguerec

python3 -m integrations.brogue_ce replay \
  --tiles "/path/to/Brogue CE.app/Contents/MacOS/brogue" \
  integrations/brogue_ce/runs/seed7/match.broguerec
```

Seed 7 explores the opening floor, picks up items, and fights jackals, rats, and
kobolds. The driver presses native auto-explore and then confirms quit to finish
the recording. One auto-explore input can advance several turns. The output
directory must be new. It contains `trajectory.jsonl` with each public
observation and input, `result.json` with the run outcome, and
`match.broguerec`. Use Space to pause and the arrow keys to change speed or
seek in the native replay. The `play` command with `--tiles` and `--seed 7`
opens a new game for keyboard play.

## Drive the engine

Use `agent` for a headless controller in any language. Send one JSON object
per line on stdin and read one reply on stdout. `observe` reads the current
state and `step` sends a key before returning the next observation. Each reply
has `ok: true` and `observation`, or `ok: false` and `error` for an invalid
request. Close stdin to end the session.

```bash
printf '%s\n' '{"op":"observe"}' '{"op":"step","action":"h"}' |
  python3 -m integrations.brogue_ce agent --engine /path/to/brogue-ce-engine/bin/brogue --seed 7
```

### Observation fields

`schema_version` is `1`.

- `map.rows`, the 79 × 29 rendered glyph grid. Index it as `rows[y][x]`
  from the top left. A space is unshown, `@` marks the player, `#` groups
  walls and other obstacles, `·` is floor-like terrain, and `<` / `>` are
  stairs. Items, creatures and terrain may share glyphs. The map reflects
  the player's visibility and remembered terrain.
- `player.position`, the single `@` position when identifiable, and
  `player.bars`, the visible health and nutrition bars. Their `background`
  values are the 20 rendered RGB hex cells in left-to-right order.
- `depth`, `deepest_level`, `gold`, `sidebar`, `messages`, and
  `location_hint`. The sidebar contains the stats and descriptions shown
  for visible creatures, items, and terrain.
- `action_space`, with common key bindings during play. At a `prompt`, use
  the `display` rows to select the next key, for example a letter in the
  inventory, `y` or `n` at confirmation, or `\u001b` to send Escape. The
  map and position are unavailable while a prompt obscures the map.
- `clock.value`, the number of keys submitted. One key can open a menu or
  advance several turns. `done` and `outcome` report game over and `seed`
  is a decimal string to preserve large native seeds.

In Python, construct `BrogueSession(engine, seed)` from
`integrations.brogue_ce.engine`, call `observe_for_ai()`, and use `step(key)`
to advance. `observe()` still returns the original 100 × 34 display.
Pass `--actions` to `demo` to run a JSON list of keys and record it. Complete
any menu prompts before the final quit. Use `save_recording(path)` from
Python after the terminal event to retain the native recording.

The adapter uses Python's standard library. Headless play and verification
require no graphical session. The replay viewer requires a graphical desktop.
