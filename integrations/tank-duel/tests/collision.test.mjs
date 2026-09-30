import test from "node:test";
import assert from "node:assert/strict";
import * as classic from "../web/classic/engine.js";
import * as auto from "../web/auto-aim/physics.js";
import {
    Game as AutoGame,
    buildObservation as autoObservation,
} from "../web/auto-aim/engine.js";
import { buildPlans } from "../web/classic/planner.js";
import { TANK_RADIUS, circleTouchesWall } from "../web/shared/collision.js";

for (const [name, engine, Game] of [
    ["classic", classic, classic.Game],
    ["auto-aim", auto, AutoGame],
]) {
    test(`${name}: a long frame cannot carry a tank through a thin wall`, () => {
        const wall = { x: 100, y: 0, w: 1, h: 400 },
            maze = { walls: [wall] },
            tank = { x: 50, y: 100 };
        engine.movePosition(maze, tank, 1, 0, 2);
        assert.ok(tank.x <= 100 - TANK_RADIUS);
        assert.ok(tank.x > 75);
        assert.ok(!circleTouchesWall(tank.x, tank.y, TANK_RADIUS, wall));
    });
    test(`${name}: sliding along a corner preserves full-body clearance`, () => {
        const wall = { x: 100, y: 0, w: 8, h: 200 },
            maze = { walls: [wall] },
            tank = { x: 78, y: 100 };
        engine.movePosition(maze, tank, 1, 1, 0.5);
        assert.ok(tank.y > 130);
        assert.ok(tank.x <= 78);
        for (let k = 0; k < 100; k++) {
            engine.movePosition(maze, tank, 1, 1, 1 / 60);
            assert.ok(!circleTouchesWall(tank.x, tank.y, TANK_RADIUS, wall));
        }
        assert.ok(tank.x > 108 + TANK_RADIUS);
    });
    test(`${name}: a tank cannot pass through another tank`, () => {
        const tank = { x: 50, y: 100 },
            other = { alive: true, x: 150, y: 100 };
        engine.movePosition({ walls: [] }, tank, 1, 0, 3, other);
        assert.ok(tank.x <= 150 - 2 * TANK_RADIUS);
        assert.ok(tank.x > 100);
    });
    test(`${name}: fast shells ricochet instead of skipping a one-pixel wall`, () => {
        const wall = { x: 100, y: 0, w: 1, h: 400 },
            b = { x: 50, y: 100, vx: 290, vy: 0, age: 0, bounces: 0 };
        engine.advanceBullet(b, 0.4, [wall]);
        assert.ok(b.vx < 0);
        assert.ok(b.x < 97);
        assert.equal(b.bounces, 1);
        assert.ok(Math.abs(b.age - 0.4) < 1e-9);
    });
    test(`${name}: movement stress across generated mazes never overlaps a wall or live tank`, () => {
        const random = classic.rng(71213);
        for (let seed = 1; seed <= 12; seed++) {
            const g = new Game(seed);
            g.start();
            for (let frame = 0; frame < 600; frame++) {
                const manual = Object.fromEntries(
                    [0, 1].map((id) => [
                        id,
                        name === "classic"
                            ? {
                                  drive: Math.floor(random() * 3) - 1,
                                  turn: Math.floor(random() * 3) - 1,
                                  fire: false,
                              }
                            : {
                                  move: ["up", "down", "left", "right", "stay"][
                                      Math.floor(random() * 5)
                                  ],
                                  fire: false,
                              },
                    ]),
                );
                g.step(1 / 60, manual);
                for (const t of g.tanks)
                    assert.ok(
                        g.maze.walls.every(
                            (w) => !circleTouchesWall(t.x, t.y, TANK_RADIUS, w),
                        ),
                        `seed ${seed} frame ${frame}`,
                    );
                assert.ok(
                    engine.distance(...g.tanks) >= 2 * TANK_RADIUS - 1e-8,
                );
            }
        }
    });
}
test("classic rotates beside a wall without moving its collision footprint", () => {
    const maze = { walls: [{ x: 100, y: 0, w: 5, h: 300 }] },
        tank = { x: 78, y: 100, angle: 0 };
    for (let i = 0; i < 500; i++) {
        classic.tankMotion(maze, tank, 0, 1, 1 / 60);
        assert.equal(tank.x, 78);
        assert.equal(tank.y, 100);
        assert.ok(!classic.blocked(maze, tank.x, tank.y));
    }
});
test("both model contexts disclose the full tank radius used by their forecasts", () => {
    const g = new classic.Game();
    assert.match(classic.buildObservation(g, 0).state, /tank radius 22/);
    assert.match(buildPlans(g, 0).state, /tank radius 22/);
    assert.match(autoObservation(new AutoGame(), 0).state, /tank radius 22/);
});
