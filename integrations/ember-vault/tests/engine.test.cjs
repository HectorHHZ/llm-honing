"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
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
  game.step(run, { upgrade: run.upgrades[0] });
  assert.equal(run.floor, 2);
  assert.equal(run.status, "playing");
  assert.ok(run.enemies.length > 0);
  assert.equal(run.gate.open, false);
});
