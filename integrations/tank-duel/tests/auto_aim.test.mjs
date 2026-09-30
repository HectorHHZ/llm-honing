import test from "node:test";
import assert from "node:assert/strict";
import {
    Game,
    buildObservation,
    bestShot,
    forecastMovement,
} from "../web/auto-aim/engine.js";
import { bulletHits, traceShot } from "../web/auto-aim/physics.js";
function open() {
    const g = new Game();
    g.maze.walls = [];
    Object.assign(g.tanks[0], {
        x: 350,
        y: 315,
        angle: Math.PI / 2,
        turretAngle: 0,
    });
    Object.assign(g.tanks[1], { x: 650, y: 315 });
    g.start();
    return g;
}
test("screen-relative movement and independently aimed firing happen in the same physics step", () => {
    const g = open();
    g.setControls(0, { move: "up", fire: true });
    g.step(1 / 60);
    assert.equal(g.tanks[0].x, 350);
    assert.ok(g.tanks[0].y < 315);
    assert.equal(g.shots[0], 1);
    assert.ok(g.bullets[0].vx > 280);
    assert.ok(Math.abs(g.tanks[0].angle - g.tanks[0].turretAngle) > 1);
});
test("auto-aim aligns its independent turret and hits the enemy without rotating the hull", () => {
    const g = open();
    g.tanks[0].turretAngle = Math.PI;
    g.setControls(0, { move: "stay", fire: true }, 3);
    for (let k = 0; k < 180 && g.phase === "running"; k++) g.step(1 / 60);
    assert.ok(
        g.events.some(
            (e) => e.type === "kill" && e.player === 0 && e.victim === 1,
        ),
    );
    assert.equal(g.tanks[0].angle, Math.PI / 2);
});
test("held fire cannot launch through a blocked firing lane", () => {
    const g = open();
    g.maze.walls = [{ x: 420, y: 0, w: 8, h: 630 }];
    g.setControls(0, { move: "stay", fire: true }, 1);
    for (let k = 0; k < 50; k++) g.step(1 / 60);
    assert.equal(g.shots[0], 0);
    assert.equal(g.tanks[0].alive, true);
});
test("a verified one-bounce lane is automatically selected", () => {
    const g = open();
    g.maze.walls = [
        { x: 0, y: 0, w: 900, h: 7 },
        { x: 490, y: 240, w: 8, h: 150 },
    ];
    const aim = bestShot(g, 0);
    assert.ok(aim);
    assert.equal(aim.bounces, 1);
    const shot = traceShot(g.maze, g.tanks[0], aim.angle, 2.8, g.tanks[1]);
    assert.equal(shot.hit, true);
    assert.equal(shot.selfHit, false);
});
test("all ten bullets from both players remain in context", () => {
    const g = open();
    g.bullets = Array.from({ length: 10 }, (_, i) => ({
        id: `shell-${i}`,
        owner: i % 2,
        x: 100 + i * 40,
        y: 100,
        vx: 290,
        vy: 0,
        age: 1,
        bounces: 1,
        trail: [],
    }));
    for (const id of [0, 1]) {
        const o = buildObservation(g, id);
        assert.equal(o.bullets.length, 10);
        assert.equal(o.bullets.filter((b) => b.owner === "self").length, 5);
        assert.ok(o.state.includes("All 10 active bullets, none omitted"));
    }
});
test("own ricochet kills immediately after a bounce, including before 150 ms", () => {
    const g = open();
    const t = g.tanks[0];
    assert.equal(
        bulletHits({ x: t.x, y: t.y, age: 0.05, owner: 0, bounces: 1 }, t),
        true,
    );
    g.bullets = [
        {
            id: "self",
            owner: 0,
            x: t.x,
            y: t.y,
            vx: 0,
            vy: 0,
            age: 0.05,
            bounces: 1,
            trail: [],
        },
    ];
    g.step(1 / 60);
    assert.equal(t.alive, false);
    assert.ok(g.events.some((e) => e.type === "kill" && e.self));
});
test("expired controls stop translation and held firing", () => {
    const g = open();
    g.setControls(0, { move: "up", fire: false }, 0.1);
    for (let k = 0; k < 30; k++) g.step(1 / 60);
    const y = g.tanks[0].y;
    g.step(1 / 60);
    assert.equal(g.tanks[0].y, y);
    assert.equal(g.tanks[0].executedMove, "stay");
});
test("old three-head controls cannot accidentally enter this edition", () => {
    const g = open();
    assert.throws(() =>
        g.setControls(0, { drive: "forward", turn: "left", fire: true }),
    );
});
