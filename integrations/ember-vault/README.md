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

`web/engine.js` exports a dependency-free Node API. Each call to `step`
advances one 30 Hz tick during play. A choice from `run.upgrades` advances
the vault while an upgrade is pending. `observe` gives the caller a copy of
the state, including hostile positions and attack warnings.

```javascript
const { createRun, observe, step } = require("./integrations/ember-vault/web/engine.js");
const run = createRun(2);
step(run, { dx: 1, dy: 0, aimX: 700, aimY: 270, fire: true });
console.log(observe(run).player);
```

The browser and the command-line demo use the same engine. Replay a saved
browser trace to reproduce its result without drawing the game.

```bash
node integrations/ember-vault/scripts/run.cjs demo 2 /tmp/ember-trace.json
node integrations/ember-vault/scripts/run.cjs replay /tmp/ember-trace.json
node --test integrations/ember-vault/tests/*.test.cjs
```
