"use strict";

const fs = require("node:fs");
const { createInterface } = require("node:readline");
const game = require("../web/engine.js");
const format = "ember-vault-v1";
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
  const inputs = [];
  while (inputs.length < 9000 && ["playing", "upgrade"].includes(run.status)) {
    const input = game.demoInput(run);
    inputs.push(input);
    game.step(run, input);
  }
  if (output) fs.writeFileSync(output, JSON.stringify({ format, seed: run.seed, inputs }) + "\n", { flag: "wx" });
} else if (mode === "replay" && argument) {
  const trace = JSON.parse(fs.readFileSync(argument, "utf8"));
  if (trace?.format !== format || !Array.isArray(trace.inputs) ||
      !trace.inputs.every(input => input && typeof input === "object" && !Array.isArray(input))) {
    throw new Error("Expected an Ember Vault action trace");
  }
  run = game.createRun(trace.seed);
  for (const input of trace.inputs) game.step(run, input);
} else {
  console.error("Usage: node scripts/run.cjs agent [seed] | demo [seed] [new-trace.json] | replay trace.json");
  process.exit(2);
}

console.log(JSON.stringify({
  seed: run.seed, status: run.status, floor: run.floor, ticks: run.tick,
  kills: run.kills, hull: run.player.hp,
}, null, 2));
