# Frost Circuit

An original turn-based tactical roguelike. Recover a core from each of three
seeded stations and take it to the airlock. The final core is sealed until you
disable the Warden. Pick a module after each extraction. A lost rover ends the
run and a fresh seed changes the rooms, paths, drones, and supplies.

Open `web/index.html` in a browser. **Start run** uses the selected map seed.
**Watch demo** plays the entire seed 1729 run using the rule controller. Pause
the demo or take over from the current turn.

- Click an adjacent tile, or use WASD or arrows, to move. Move into a drone
  to attack it.
- Red tiles mark attacks that will resolve after your next action. Leave a
  marked tile or interrupt the attacker with **Pulse**.
- **Space** spends a pulse charge. Pulse hits drones within two tiles of
  movement distance measured on the grid, ignoring walls. It cancels their
  current warnings. **.** waits a turn.
- The cyan diamond is the core. The airlock turns green once you carry it.
  Green medkits repair hull, and blue cells restore pulse charge.

The engine and its drawn graphics are included. Use the bundler to make a
single offline HTML file.

```bash
node integrations/frost-circuit/scripts/bundle.cjs /tmp/frost-circuit.html
```

## Research scope

Primary category: **Roguelikes**. Capability tags: route planning, tactical timing, resource use.

Recover station cores, avoid warned attacks, disable the Warden and extract
through the airlock.

| Component | Responsibility in this configuration |
| --- | --- |
| Model | A future model client can choose movement, wait, Pulse and upgrades through the control API. |
| Harness | A client must construct model requests and map choices to API actions. The supplied demo uses a rule controller. |
| Engine and other policies | Expose attack intentions and danger cells. Apply turn transitions, attacks, Pulse costs and extraction rules. |

The engine supplies attack warnings directly. Using them does not establish
independent prediction of enemy attacks. The rule demo is not a decision-model
result.

Suggested evaluation: Measure extraction success, damage and Pulse use. An
experiment that removes warnings needs a separate observation profile and
validation.

See the [game research index](../README.md#research-categories) for the
classification method and the shared model/harness responsibility matrix.

## Drive the engine

Use `agent` for a headless controller in any language. Send one JSON object
per line on stdin and read one reply on stdout. `observe` reads the current
state and `step` applies an action before returning the next observation.
Each reply has `ok: true` and `observation`, or `ok: false` and `error` for
an invalid request. Close stdin to end the session.

```bash
printf '%s\n' '{"op":"observe"}' '{"op":"step","action":"wait"}' |
  node integrations/frost-circuit/scripts/run.cjs agent 1729
```

### Observation fields

`schema_version` is `1`.

- `map.tiles[y][x]` contains `0` for walls and `1` for walkable cells. The
  17 × 11 grid starts at the top left. A walkable cell may contain an item
  or enemy. Entities use the same zero-based coordinates.
- `player` includes hull (`hp`, `maxHp`), melee damage and Pulse charges
  (`charge`, `maxCharge`). `enemies` contains each drone's hull, position,
  attack damage and `intent` cells. `danger_cells` is the union of the
  marked cells. A warning resolves after the player's next
  action if the enemy survives it and the attack is still pending.
- `core` shows whether the core is collected and whether a Warden locks
  access. `airlock.open` indicates that the player can exit with the core.
  `items` contains available medkits and cells. `events` describes the
  latest turn or floor transition.
- `action_space.choices` gives all accepted actions for the current state.
  Direction actions move into an empty tile or attack an adjacent drone.
  `wait` and `pulse` also advance one turn. Pulse costs a charge and deals
  2 damage to enemies within Manhattan distance 2 while interrupting their
  pending attacks. When choosing an upgrade, the action is the upgrade ID. This
  starts the next floor without advancing the turn counter. `done` is true
  in `won` or `lost` states.

The same API is available directly from `web/engine.js` in Node.

```javascript
const { createRun, observeForAI, stepForAI } = require("./integrations/frost-circuit/web/engine.js");
const run = createRun(1729);
console.log(observeForAI(run).action_space.choices);
console.log(stepForAI(run, "wait").danger_cells);
```

The browser's **Save action trace** button exports the seed and actions.
The headless runner can reproduce that run or generate the demo trace.

```bash
node integrations/frost-circuit/scripts/run.cjs demo 1729 /tmp/frost-trace.json
node integrations/frost-circuit/scripts/run.cjs replay /tmp/frost-trace.json
node --test integrations/frost-circuit/tests/*.test.cjs
```
