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

The pure Node engine in `web/engine.js` supplies `createRun`, `observe`,
`legalActions` and `step`. An observation contains the grid, current
attacks, core and airlock positions, and the rover's status. Each accepted
move, attack, pulse, or wait advances one turn. Upgrade choices use the same
action interface and start the next station.

```javascript
const { createRun, observe, step } = require("./integrations/frost-circuit/web/engine.js");
const run = createRun(1729);
step(run, "east");
console.log(observe(run).legalActions);
```

The browser's **Save action trace** button exports the seed and actions.
The headless runner can reproduce that run or generate the demo trace.

```bash
node integrations/frost-circuit/scripts/run.cjs demo 1729 /tmp/frost-trace.json
node integrations/frost-circuit/scripts/run.cjs replay /tmp/frost-trace.json
node --test integrations/frost-circuit/tests/*.test.cjs
```
