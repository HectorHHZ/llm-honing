import test from "node:test";
import assert from "node:assert/strict";
import {
    Game,
    findAimDirections,
    traceShot,
    buildObservation,
    angleDelta,
} from "../web/classic/engine.js";
import { buildPlans } from "../web/classic/planner.js";
function fixture(bank = false) {
    const g = new Game();
    Object.assign(g.tanks[0], { x: 200, y: 300, angle: 0 });
    Object.assign(g.tanks[1], { x: 600, y: 300 });
    g.maze.walls = bank
        ? [
              { x: 0, y: 90, w: 900, h: 8 },
              { x: 396, y: 210, w: 8, h: 200 },
          ]
        : [];
    g.start();
    return g;
}
test("direct aim exposes absolute heading, signed rotation and hit time without mutating the tank", () => {
    const g = fixture();
    g.tanks[0].angle = Math.PI / 2;
    const before = JSON.stringify(g.tanks),
        a = findAimDirections(g.maze, ...g.tanks);
    assert.equal(a.best.heading_deg, 0);
    assert.equal(a.best.delta_deg, -90);
    assert.equal(a.best.turn, "left");
    assert.equal(a.best.bounces, 0);
    assert.ok(a.best.hit_ms > 1000);
    assert.equal(JSON.stringify(g.tanks), before);
});
test("a hidden enemy gets a verified one-bounce firing direction distinct from its bearing", () => {
    const g = fixture(true),
        a = findAimDirections(g.maze, ...g.tanks);
    assert.ok(a.best);
    assert.equal(a.best.kind, "ricochet");
    assert.equal(a.best.bounces, 1);
    assert.ok(a.best.heading_deg < -40 && a.best.heading_deg > -50);
    assert.equal(a.best.turn, "left");
    const shot = traceShot(g.maze, g.tanks[0], a.best.angle, 3, g.tanks[1]);
    assert.equal(shot.hit, true);
    assert.equal(shot.selfHit, false);
});
test("blocked and returning shots are never advertised as hit directions", () => {
    const g = fixture();
    g.maze.walls = [{ x: 230, y: 0, w: 8, h: 630 }];
    const a = findAimDirections(g.maze, ...g.tanks);
    assert.equal(a.best, null);
    assert.deepEqual(a.options, []);
});
test("rotation wraps at ±180 degrees using the shorter direction", () => {
    const g = fixture();
    g.tanks[0].angle = (179 * Math.PI) / 180;
    Object.assign(g.tanks[1], { x: 100, y: 298.25 });
    const a = findAimDirections(g.maze, ...g.tanks);
    assert.equal(a.best.turn, "right");
    assert.ok(a.best.delta_deg > 0 && a.best.delta_deg < 3);
});
test("planner and both model modes see verified angles, and bank shots guide hull rotation", () => {
    const g = fixture(true),
        plan = buildPlans(g, 0, { reactionDelay: 0 }),
        raw = buildObservation(g, 0);
    assert.equal(plan.enemyVisible, false);
    assert.equal(plan.aim.best.bounces, 1);
    assert.ok(plan.state.includes("verified firing direction"));
    assert.ok(raw.state.includes("verified hit directions"));
    assert.ok(plan.admissible.some((k) => k.endsWith("_left")));
    assert.ok(!plan.admissible.some((k) => k.endsWith("_right")));
    const candidate = plan.actions.find((p) => p.id === "stop_left");
    assert.ok(candidate.allowFire);
    g.setPlan(0, {
        ...candidate,
        motionId: candidate.id,
        controls: { ...candidate.controls, fire: true },
    });
    for (let k = 0; k < 180 && g.phase === "running"; k++) g.step(1 / 60);
    assert.ok(g.events.some((e) => e.type === "kill" && !e.self));
    assert.ok(!g.events.some((e) => e.type === "kill" && e.self));
});
test("current and delayed aiming solutions use their own positions", () => {
    const g = fixture();
    g.setControls(0, { drive: "forward", turn: "right", fire: false }, 0.6);
    g.step(1 / 60);
    const p = buildPlans(g, 0, { reactionDelay: 0.3 });
    assert.notDeepEqual(p.aim.origin, p.arrivalAim.origin);
    assert.equal(p.aim.origin.x, g.tanks[0].x);
    assert.ok(p.state.includes("AFTER 300ms"));
    assert.ok(p.request.weapon_state.includes("new-shot forecast"));
    assert.ok(p.request.weapon_state.length < p.state.length);
});
test("all ten shells remain in both full and compact contexts", () => {
    const g = fixture();
    for (let owner = 0; owner < 2; owner++)
        for (let n = 0; n < 5; n++)
            g.bullets.push({
                id: `shell-${owner}-${n}`,
                owner,
                x: 450,
                y: 200 + n * 10,
                vx: 290,
                vy: 0,
                age: 1,
                bounces: 1,
            });
    const p = buildPlans(g, 0);
    for (const b of g.bullets) {
        assert.ok(p.state.includes(b.id));
        assert.ok(p.request.weapon_state.includes(b.id));
    }
});
