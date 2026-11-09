"use strict";

const fs = require("node:fs");
const game = require("../web/engine.js");
const format = "frost-circuit-v1";
const [mode, argument, output] = process.argv.slice(2);

let run;
if (mode === "demo") {
  run = game.createRun(argument === undefined ? undefined : Number(argument));
  const actions = [];
  while (actions.length < 450 && ["playing", "upgrade"].includes(run.status)) {
    const action = game.demoAction(run);
    actions.push(action);
    game.step(run, action);
  }
  if (output) fs.writeFileSync(output, JSON.stringify({ format, seed: run.seed, actions }) + "\n", { flag: "wx" });
} else if (mode === "replay" && argument) {
  const trace = JSON.parse(fs.readFileSync(argument, "utf8"));
  if (trace?.format !== format || !Array.isArray(trace.actions) ||
      !trace.actions.every(action => typeof action === "string")) {
    throw new Error("Expected a Frost Circuit action trace");
  }
  run = game.createRun(trace.seed);
  for (const action of trace.actions) {
    if (!game.legalActions(run).includes(action)) throw new Error("Illegal replay action: " + action);
    game.step(run, action);
  }
} else {
  console.error("Usage: node scripts/run.cjs demo [seed] [new-trace.json] | replay trace.json");
  process.exit(2);
}

console.log(JSON.stringify({
  seed: run.seed, status: run.status, floor: run.floor, turns: run.turn,
  kills: run.kills, hull: run.player.hp, coreRecovered: run.hasCore,
}, null, 2));
