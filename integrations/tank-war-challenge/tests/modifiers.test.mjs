// The human challenge's modifier panel (game.mods[0]) changes only the player's rules; the engine runs headless.
import { test } from "node:test";
import assert from "node:assert/strict";
import { adapter } from "../web/classic/adapter.js";
import { RADIUS } from "../web/classic/engine.js";

const still = { drive: 0, turn: 0, fire: false };
function running(mods) {
    const game = new adapter.Game(2001);
    adapter.initialize(game);
    game.start();
    if (mods) game.mods = { 0: mods };
    return game;
}
// A shell already in flight, sitting on the given tank.
const shellOn = (game, tank, owner, r) =>
    game.bullets.push({ x: tank.x, y: tank.y, vx: 0.01, vy: 0, age: 1, bounces: 1, owner, trail: [], id: `t-${game.bullets.length}`, ...(r ? { r } : {}) });
const step = (game, manual = { 0: still, 1: still }) => game.step(1 / 60, manual);
const shells = (game, owner) => game.bullets.filter((b) => b.owner === owner).length;

test("without modifiers the cooldown allows one shell per 0.3 s", () => {
    const game = running();
    assert.equal(game.fire(game.tanks[0]), true);
    assert.equal(game.fire(game.tanks[0]), false);
});

test("no fire cooldown fires every call, up to the shells-at-once limit", () => {
    const game = running({ noCooldown: true, maxShells: 8 });
    for (let i = 0; i < 12; i++) game.fire(game.tanks[0]);
    assert.equal(shells(game, 0), 8);
    const capped = running({ noCooldown: true });
    for (let i = 0; i < 12; i++) capped.fire(capped.tanks[0]);
    assert.equal(shells(capped, 0), 5);
});

test("shell size sets the player's shells' radius, and hits reach that far", () => {
    const game = running({ noCooldown: true, shellRadius: 12 });
    game.fire(game.tanks[0]);
    assert.equal(game.bullets.at(-1).r, 12);
    const enemy = game.tanks[1];
    game.bullets.push({ x: enemy.x + RADIUS + 10, y: enemy.y, vx: 0.01, vy: 0, age: 1, bounces: 1, owner: 0, trail: [], id: "big", r: 12 });
    step(game);
    assert.equal(enemy.alive, false);
    const small = running();
    const other = small.tanks[1];
    small.bullets.push({ x: other.x + RADIUS + 10, y: other.y, vx: 0.01, vy: 0, age: 1, bounces: 1, owner: 0, trail: [], id: "small" });
    step(small);
    assert.equal(other.alive, true);
});

test("invincible absorbs enemy shells", () => {
    const game = running({ invincible: true });
    shellOn(game, game.tanks[0], 1);
    step(game);
    assert.equal(game.tanks[0].alive, true);
    assert.equal(shells(game, 1), 0);
    assert.equal(game.phase, "running");
});

test("own shells still kill their tank", () => {
    const plain = running({ invincible: false });
    shellOn(plain, plain.tanks[0], 0);
    step(plain);
    assert.equal(plain.tanks[0].alive, false);
});

test("hit points: the tank survives hp - 1 hits and dies on the last", () => {
    const game = running({ hp: 3 });
    for (const expected of [1, 2]) {
        shellOn(game, game.tanks[0], 1);
        step(game);
        assert.equal(game.tanks[0].alive, true);
        assert.equal(game.tanks[0].hits, expected);
    }
    shellOn(game, game.tanks[0], 1);
    step(game);
    assert.equal(game.tanks[0].alive, false);
});

test("modifiers belong to the player only: the opponent keeps one hit and the cooldown", () => {
    const game = running({ invincible: true, noCooldown: true, hp: 9 });
    assert.equal(game.fire(game.tanks[1]), true);
    assert.equal(game.fire(game.tanks[1]), false);
    shellOn(game, game.tanks[1], 0);
    step(game);
    assert.equal(game.tanks[1].alive, false);
});

test("W/S drive along the hull: forward input moves the tank along its heading", () => {
    const game = running();
    const t = game.tanks[0], x0 = t.x, y0 = t.y;
    for (let i = 0; i < 30; i++) step(game, { 0: { drive: 1, turn: 0, fire: false, fastHumanTurn: true, responsiveTurn: true }, 1: still });
    const moved = (t.x - x0) * Math.cos(t.angle) + (t.y - y0) * Math.sin(t.angle);
    assert.ok(moved > 30, `moved ${moved.toFixed(1)} px forward`);
});

test("shell range: shells last their own lifetime, Infinity keeps them", () => {
    const game = running({ shellLife: Infinity });
    game.bullets.push({ x: 450, y: 315, vx: 0.01, vy: 0, age: 600, bounces: 1, owner: 0, trail: [], id: "old", life: Infinity });
    game.bullets.push({ x: 450, y: 315, vx: 0.01, vy: 0, age: 4.6, bounces: 1, owner: 0, trail: [], id: "stock" });
    step(game);
    assert.deepEqual(game.bullets.map((b) => b.id), ["old"]);
    game.fire(game.tanks[0]);
    assert.equal(game.bullets.at(-1).life, Infinity);
});

test("a shell that grazes a tank and leaves is a near miss", () => {
    const game = running();
    const t = game.tanks[0];
    game.bullets.push({ x: t.x - 60, y: t.y + RADIUS + 3 + 8, vx: 290, vy: 0, age: 1, bounces: 1, owner: 1, trail: [], id: "graze" });
    for (let i = 0; i < 40; i++) step(game);
    assert.equal(t.alive, true);
    assert.equal(game.events.filter((e) => e.type === "near_miss" && e.victim === 0).length, 1);
});

test("Super Tank crushes the interior wall it drives into and links the cells; the edge stays", () => {
    const game = running({ superTank: true });
    const t = game.tanks[0], walls = game.maze.walls.length;
    // Face the nearest interior wall and drive into it.
    const inner = game.maze.walls.filter((w) => w.w < 900 && w.h < 630);
    const target = inner.reduce((a, w) => {
        const d = Math.hypot(w.x + w.w / 2 - t.x, w.y + w.h / 2 - t.y);
        return d < a.d ? { w, d } : a;
    }, { d: Infinity }).w;
    t.angle = Math.atan2(target.y + target.h / 2 - t.y, target.x + target.w / 2 - t.x);
    for (let i = 0; i < 180 && game.maze.walls.includes(target); i++) step(game, { 0: { drive: 1, turn: 0, fire: false }, 1: still });
    assert.equal(game.maze.walls.includes(target), false);
    assert.ok(game.maze.walls.length < walls);
    assert.equal(game.maze.walls.filter((w) => w.w >= 900 || w.h >= 630).length, 4);
    assert.ok(game.events.some((e) => e.type === "wall_break" && e.player === 0));
    const plain = running();
    const before = plain.maze.walls.length;
    for (let i = 0; i < 180; i++) step(plain, { 0: { drive: 1, turn: 0, fire: false }, 1: still });
    assert.equal(plain.maze.walls.length, before);
});

test("Super Tank runs the opponent over and wins the round; without it the tanks just touch", () => {
    for (const superTank of [true, false]) {
        const game = running(superTank ? { superTank: true } : undefined);
        const [me, them] = game.tanks;
        // Put the opponent just ahead in open ground, then drive straight at it.
        game.maze.walls = game.maze.walls.filter((w) => w.w >= 900 || w.h >= 630);
        me.x = 300; me.y = 315; me.angle = 0;
        them.x = 360; them.y = 315;
        for (let i = 0; i < 60 && game.phase === "running"; i++) step(game, { 0: { drive: 1, turn: 0, fire: false }, 1: still });
        assert.equal(them.alive, !superTank);
        if (superTank) {
            assert.equal(game.result, 0);
            assert.ok(game.events.some((e) => e.type === "kill" && e.crush && e.victim === 1));
        } else assert.ok(Math.hypot(me.x - them.x, me.y - them.y) >= 44 - 1e-6);
    }
});

test("firing with the magazine empty reports a dry fire; cooldown does not", () => {
    const game = running({ noCooldown: true, maxShells: 2 });
    const dry = [];
    game.onDryFire = (t) => dry.push(t.id);
    for (let i = 0; i < 4; i++) game.fire(game.tanks[0]);
    assert.equal(shells(game, 0), 2);
    assert.deepEqual(dry, [0, 0]);
    const cooling = running();
    const none = [];
    cooling.onDryFire = (t) => none.push(t.id);
    cooling.fire(cooling.tanks[0]);
    cooling.fire(cooling.tanks[0]);
    assert.deepEqual(none, []);
});
