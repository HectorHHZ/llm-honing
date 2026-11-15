"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const path = require("node:path");
const game = require("../web/engine.js");

test("the showcase clears all vaults and its recorded inputs reproduce the run", () => {
  const first = game.createRun();
  const inputs = [];
  for (let i = 0; i < 3000 && ["playing", "upgrade"].includes(first.status); i += 1) {
    const input = game.demoInput(first);
    inputs.push(input);
    game.step(first, input);
  }
  assert.equal(first.status, "won");
  assert.equal(first.floor, 3);
  assert.equal(first.kills, 16);
  assert.equal(game.observeForAI(first).done, true);
  assert.equal(game.observeForAI(first).action_space, null);

  const replay = game.createRun(first.seed);
  for (const input of inputs) game.step(replay, input);
  assert.deepEqual(game.observe(replay), game.observe(first));

  const view = game.observe(first);
  view.player.hp = 0;
  assert.ok(first.player.hp > 0);
});

test("cover blocks player dashes and shots", () => {
  const run = game.createRun();
  run.player.x = 200;
  run.player.y = 270;
  run.walls = [{ x: 240, y: 200, w: 42, h: 135 }];
  run.enemies = [{
    id: 0, kind: "sentry", x: 335, y: 270, hp: 3, maxHp: 3, r: 17,
    speed: 0, cooldown: 500, wind: 0, angle: Math.PI, flash: 0,
  }];
  for (let i = 0; i < 34; i += 1) {
    game.step(run, { dx: 1, aimX: 335, aimY: 270, fire: true, dash: true });
  }
  assert.ok(run.player.x + run.player.r < run.walls[0].x);
  assert.equal(run.enemies[0].hp, 3);
  assert.equal(run.gate.open, false);
});

test("only an offered upgrade opens the next vault", () => {
  const run = game.createRun(123);
  run.enemies = [];
  Object.assign(run.player, { x: run.gate.x, y: run.gate.y });
  game.step(run);
  assert.equal(run.status, "upgrade");
  game.step(run, { upgrade: "unknown" });
  assert.equal(run.floor, 1);
  const choices = game.observeForAI(run).action_space.choices;
  assert.deepEqual(choices.map(({ id }) => id), run.upgrades);
  assert.throws(() => game.stepForAI(run, { upgrade: "unknown" }), RangeError);
  const next = game.stepForAI(run, { upgrade: choices[0].id });
  assert.equal(next.clock.value, 1);
  assert.equal(run.floor, 2);
  assert.equal(run.status, "playing");
  assert.ok(run.enemies.length > 0);
  assert.equal(run.gate.open, false);
});

test("AI snapshots expose arena geometry and the next tick's shot state", () => {
  const run = game.createRun(2);
  run.player.fireCooldown = 1;
  run.player.dashCooldown = 2;
  const start = game.observeForAI(run);
  assert.deepEqual(start.map.walls, run.walls);
  assert.deepEqual(start.enemies, run.enemies);
  assert.equal(start.map.units, "pixels");
  assert.equal(start.map.width, 960);
  assert.equal(start.action_space.fire_ready, true);
  assert.equal(start.action_space.dash_ready, false);
  start.map.walls[0].x = -1;
  start.player.hp = 0;
  assert.ok(run.walls[0].x > 0);
  assert.ok(run.player.hp > 0);

  const next = game.stepForAI(run, { dx: 1, aimX: 700, aimY: 270, fire: true });
  assert.deepEqual(next.clock, { unit: "tick", value: 1, ticks_per_second: 30 });
  assert.ok(next.player.x > start.player.x);
  assert.equal(next.projectiles[0].friendly, true);
  assert.equal(next.projectiles[0].vx, 13.2);
  assert.equal(next.action_space.fire_ready, false);
  assert.equal(next.action_space.dash_ready, true);
});

test("JSON Lines agent accepts valid inputs and leaves the state intact after invalid requests", () => {
  const input = [
    '{"op":"observe"}',
    "{",
    '{"op":"step","action":{"dx":2}}',
    '{"op":"step","action":{"aimX":120}}',
    '{"op":"observe"}',
    '{"op":"step","action":{"dx":1,"fire":true}}',
  ].join("\n") + "\n";
  const runner = path.join(__dirname, "../scripts/run.cjs");
  const result = spawnSync(process.execPath, [runner, "agent", "2"], { input, encoding: "utf8", timeout: 5000 });
  assert.equal(result.status, 0, result.stderr);
  const [start, badJson, badMove, badAim, again, next] = result.stdout.trim().split("\n").map(JSON.parse);
  assert.equal(start.ok, true);
  assert.equal(start.observation.seed, "2");
  assert.equal(badJson.ok, false);
  assert.equal(badMove.ok, false);
  assert.equal(badAim.ok, false);
  assert.deepEqual(again, start);
  assert.equal(next.ok, true);
  assert.equal(next.observation.clock.value, 1);
});
