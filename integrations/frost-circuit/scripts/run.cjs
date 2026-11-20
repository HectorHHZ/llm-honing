"use strict";

const fs = require("node:fs");
const { createInterface } = require("node:readline");
const game = require("../web/engine.js");
const format = "frost-circuit-v1";
const [mode, argument, output] = process.argv.slice(2);

async function serve(run) {
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    let reply;
    try {
      const request = JSON.parse(line);
      if (request?.op === "observe") {
        reply = { ok: true, observation: game.observeForAI(run) };
      } else if (request?.op === "step") {
        reply = { ok: true, observation: game.stepForAI(run, request.action) };
      } else {
        throw new RangeError("expected an object with op 'observe' or 'step'");
      }
    } catch (error) {
      if (!(error instanceof SyntaxError || error instanceof RangeError || error instanceof TypeError)) throw error;
      reply = { ok: false, error: error.message };
    }
    console.log(JSON.stringify(reply));
  }
}

if (mode === "agent" && output === undefined) {
  serve(game.createRun(argument === undefined ? undefined : Number(argument))).catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
  return;
}

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
  console.error("Usage: node scripts/run.cjs agent [seed] | demo [seed] [new-trace.json] | replay trace.json");
  process.exit(2);
}

console.log(JSON.stringify({
  seed: run.seed, status: run.status, floor: run.floor, turns: run.turn,
  kills: run.kills, hull: run.player.hp, coreRecovered: run.hasCore,
}, null, 2));
