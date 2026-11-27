# Ember Vault

An original top-down action roguelike in the browser. Clear the Intake and
Smelter, choose a module after each vault, defeat the Furnace, and drive
through the final green gate. A lost rover ends the run. A new seed changes
the cover layout, enemy placements, and upgrade choices.

Open `web/index.html` in a desktop browser. **Start run** uses the selected
map seed. **Watch demo** plays a full run on seed 2 with the rule controller.
You can pause the run, take control from the demo, or save its inputs as JSON.

| Control | Action |
| --- | --- |
| WASD or arrows | Move |
| Mouse and left click | Aim and fire |
| Space | Fire at the closest hostile |
| Shift | Dash and briefly avoid damage |
| P | Pause or resume |

Sentries mark their aim before shooting. Use the cover blocks and the dash
meter to get around shots. Green repair packs restore hull. The right-hand
gate lights up after the last hostile falls.

The engine, graphics, and browser controls are included here. Opening the
HTML from disk is enough. Use the bundler to share a single offline file.

```bash
node integrations/ember-vault/scripts/bundle.cjs /tmp/ember-vault.html
```

## Research scope

Primary category: **Roguelikes**. Capability tags: combat, resource use, upgrade choice.

Clear the vaults, choose upgrades, defeat the Furnace and reach the final gate.

| Component | Responsibility in this configuration |
| --- | --- |
| Model | A future model client can choose movement, aim, fire, dash and upgrades through the control API. |
| Harness | A client must map model output to API actions and define decision timing. The supplied demo uses a rule controller. |
| Engine and other policies | Apply combat and upgrades. When aim coordinates are omitted, the engine API aims at the nearest hostile. |

The current rule demo and deterministic replay establish a control path. They
do not establish a decision-model run.

Suggested evaluation: Measure gate completion, damage and upgrade choices. Use
separate configuration names for explicit model aim and automatic aim.

See the [game research index](../README.md#research-categories) for the
classification method and the shared model/harness responsibility matrix.

## Drive the engine

Use `agent` for a headless controller in any language. Send one JSON object
per line on stdin and read one reply on stdout. `observe` reads the current
state and `step` applies an action before returning the next observation.
Each reply has `ok: true` and `observation`, or `ok: false` and `error` for
an invalid request. Close stdin to end the session.

```bash
printf '%s\n' '{"op":"observe"}' '{"op":"step","action":{"dx":1,"fire":true}}' |
  node integrations/ember-vault/scripts/run.cjs agent 2
```

### Observation fields

`schema_version` is `1`.

- `map` contains the arena's pixel dimensions, collision bounds and cover
  rectangles. An object's `x` points right and `y` points down from the top
  left. Walls use top-left coordinates with `w` and `h`. Actors use their
  center position and collision radius `r`.
- `player` includes hull (`hp`, `maxHp`), damage, and timers for firing and
  dashing. `enemies` includes positions, hull and attack windup `wind` in
  ticks. `projectiles` includes `vx` and `vy` in pixels per tick, allegiance
  (`friendly`), damage and remaining `life` in ticks.
- `gate.open` becomes true when the floor is cleared. Move within 33 pixels
  of the gate to advance. `pickups` includes available repairs, `events`
  describes the last step, and `done` is true in `won` or `lost` states.
- `action_space` describes the inputs for the current state. During play,
  `dx` and `dy` are optional movement axes between −1 and 1. Specify both
  `aimX` and `aimY` to aim at a pixel, or omit both to auto-aim at the
  nearest hostile when firing. `fire` and `dash` are booleans. The readiness
  flags tell you whether the next tick can fire or dash. One action advances
  one tick at 30 ticks per simulation second. Use `{}` to advance without
  input. Choose an offered `{ "upgrade": "<id>" }` at the upgrade screen to
  enter the next floor without advancing the tick counter.

The same API is available directly from `web/engine.js` in Node. The
browser and replay runner also use this engine.

```javascript
const { createRun, observeForAI, stepForAI } = require("./integrations/ember-vault/web/engine.js");
const run = createRun(2);
console.log(observeForAI(run).map);
console.log(stepForAI(run, { dx: 1, fire: true }).player);
```

Replay a saved browser trace to reproduce its result without drawing the game.

```bash
node integrations/ember-vault/scripts/run.cjs demo 2 /tmp/ember-trace.json
node integrations/ember-vault/scripts/run.cjs replay /tmp/ember-trace.json
node --test integrations/ember-vault/tests/*.test.cjs
```
