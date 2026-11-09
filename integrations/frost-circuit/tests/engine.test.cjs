"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
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

  const replay = game.createRun(run.seed);
  for (const action of actions) game.step(replay, action);
  assert.deepEqual(game.observe(replay), game.observe(run));
});

test("sentry warnings can be dodged or canceled before the shot resolves", () => {
  const run = game.createRun();
  run.tiles.fill(1);
  Object.assign(run.player, { x: 3, y: 4 });
  run.enemies = [{ id: 0, kind: "sentry", x: 5, y: 4, hp: 3, maxHp: 3, cooldown: 0, intent: [] }];
  game.step(run, "wait");
  assert.equal(run.player.hp, run.player.maxHp);
  assert.ok(game.danger(run).has(4 * game.COLS + 3));

  const dodge = structuredClone(run);
  game.step(dodge, "north");
  assert.equal(dodge.player.hp, dodge.player.maxHp);

  const hold = structuredClone(run);
  game.step(hold, "wait");
  assert.equal(hold.player.hp, hold.player.maxHp - 2);

  game.step(run, "pulse");
  assert.equal(run.player.hp, run.player.maxHp);
  assert.equal(run.player.charge, run.player.maxCharge - 1);
  assert.equal(run.enemies[0].hp, 1);
  assert.deepEqual(run.enemies[0].intent, []);
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
