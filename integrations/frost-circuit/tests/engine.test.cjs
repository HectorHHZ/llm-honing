"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const path = require("node:path");
const game = require("../web/engine.js");

test("the showcase recovers all cores and replays exactly", () => {
  const run = game.createRun();
  const actions = [];
  for (let i = 0; i < 250 && ["playing", "upgrade"].includes(run.status); i += 1) {
    const action = game.demoAction(run);
    assert.ok(game.legalActions(run).includes(action));
    actions.push(action);
    game.step(run, action);
  }
  assert.equal(run.status, "won");
  assert.equal(run.floor, 3);
  assert.ok(run.hasCore);
  assert.ok(run.player.hp > 0);
  const finished = game.observeForAI(run);
  assert.equal(finished.done, true);
  assert.equal(finished.action_space, null);
  assert.throws(() => game.stepForAI(run, "wait"), RangeError);

  const replay = game.createRun(run.seed);
  for (const action of actions) game.step(replay, action);
  assert.deepEqual(game.observe(replay), game.observe(run));
});

test("sentry warnings can be dodged or canceled before the shot resolves", () => {
  const run = game.createRun();
  run.tiles.fill(1);
  Object.assign(run.player, { x: 3, y: 4 });
  run.enemies = [{ id: 0, kind: "sentry", x: 5, y: 4, hp: 3, maxHp: 3, cooldown: 0, intent: [] }];
  const warning = game.stepForAI(run, "wait");
  assert.equal(run.player.hp, run.player.maxHp);
  assert.ok(game.danger(run).has(4 * game.COLS + 3));
  assert.deepEqual(warning.danger_cells, run.enemies[0].intent);
  assert.equal(warning.enemies[0].attack_damage, 2);

  const dodge = structuredClone(run);
  game.step(dodge, "north");
  assert.equal(dodge.player.hp, dodge.player.maxHp);

  const hold = structuredClone(run);
  game.step(hold, "wait");
  assert.equal(hold.player.hp, hold.player.maxHp - 2);

  const next = game.stepForAI(run, "pulse");
  assert.equal(run.player.hp, run.player.maxHp);
  assert.equal(run.player.charge, run.player.maxCharge - 1);
  assert.equal(run.enemies[0].hp, 1);
  assert.deepEqual(run.enemies[0].intent, []);
  assert.deepEqual(next.danger_cells, []);
});

test("AI snapshots describe the current grid, core, airlock and available actions", () => {
  const run = game.createRun(1729);
  const state = game.observeForAI(run);
  assert.equal(state.map.height, 11);
  assert.equal(state.map.width, 17);
  assert.deepEqual(state.map.tiles.flat(), run.tiles);
  assert.equal(state.map.tiles[state.player.y][state.player.x], 1);
  assert.equal(state.core.collected, false);
  assert.equal(state.airlock.open, false);
  assert.deepEqual(state.action_space.choices, game.legalActions(run));
  state.player.hp = 0;
  state.map.tiles[0][0] = 1;
  assert.ok(run.player.hp > 0);
  assert.equal(run.tiles[0], 0);

  run.status = "upgrade";
  const upgrades = game.observeForAI(run).action_space;
  assert.deepEqual(upgrades.upgrades.map(({ id }) => id), upgrades.choices);
  assert.equal(game.stepForAI(run, upgrades.choices[0]).clock.value, 0);
  run.status = "upgrade";
  const finalFloor = game.stepForAI(run, "blade");
  assert.equal(finalFloor.floor, 3);
  assert.equal(finalFloor.core.locked, true);
});

test("JSON Lines agent returns an observation for each accepted turn", () => {
  const input = [
    '{"op":"observe"}',
    "{",
    '{"op":"step","action":"unknown"}',
    '{"op":"observe"}',
    '{"op":"step","action":"wait"}',
  ].join("\n") + "\n";
  const runner = path.join(__dirname, "../scripts/run.cjs");
  const result = spawnSync(process.execPath, [runner, "agent", "1729"], { input, encoding: "utf8", timeout: 5000 });
  assert.equal(result.status, 0, result.stderr);
  const [start, badJson, badAction, again, next] = result.stdout.trim().split("\n").map(JSON.parse);
  assert.equal(start.ok, true);
  assert.equal(start.observation.game, "frost-circuit");
  assert.equal(badJson.ok, false);
  assert.equal(badAction.ok, false);
  assert.deepEqual(again, start);
  assert.equal(next.ok, true);
  assert.deepEqual(next.observation.clock, { unit: "turn", value: 1 });
});

test("generated layouts connect the start, core and airlock on all floors", () => {
  for (let seed = 0; seed < 32; seed += 1) {
    const run = game.createRun(seed);
    for (let floor = 1; floor <= 3; floor += 1) {
      const index = ({ x, y }) => y * game.COLS + x;
      const pending = [{ x: run.player.x, y: run.player.y }];
      const reached = new Set([index(pending[0])]);
      for (let cursor = 0; cursor < pending.length; cursor += 1) {
        const { x, y } = pending[cursor];
        for (const [dx, dy] of Object.values(game.DIRECTIONS)) {
          const to = { x: x + dx, y: y + dy };
          const cell = index(to);
          if (!run.tiles[cell] || reached.has(cell)) continue;
          reached.add(cell);
          pending.push(to);
        }
      }
      assert.ok(reached.has(index(run.core)));
      assert.ok(reached.has(index(run.exit)));
      if (floor < 3) {
        run.status = "upgrade";
        game.step(run, "blade");
      }
    }
  }
});
