import test from "node:test";
import assert from "node:assert/strict";
import {
    Game,
    TURN_SPEED,
    forecastShot,
    distance,
} from "../web/classic/engine.js";
import { buildPlans, applyPlanDecision } from "../web/classic/planner.js";
const empty = () => {
    const g = new Game();
    g.maze.walls = [];
    Object.assign(g.tanks[0], { x: 300, y: 300, angle: 0 });
    Object.assign(g.tanks[1], { x: 500, y: 500 });
    g.start();
    return g;
};
test("planned turn uses physical angular speed, stops at its deadline and fires only afterward", () => {
    const g = empty(),
        turn = 0.4;
    Object.assign(g.tanks[1], {
        x: 300 + 200 * Math.cos(turn * TURN_SPEED),
        y: 300 + 200 * Math.sin(turn * TURN_SPEED),
    });
    g.setPlan(0, {
        motionId: "stop_right",
        controls: { drive: "stop", turn: "right", fire: true },
        duration: 0.7,
        turnDuration: turn,
        fireDelay: turn,
    });
    for (let k = 0; k < 24; k++) {
        g.step(1 / 60);
        assert.equal(g.shots[0], 0);
        assert.ok(
            Math.abs(g.tanks[0].angle - ((k + 1) * TURN_SPEED) / 60) < 1e-8,
        );
    }
    g.step(1 / 60);
    assert.equal(g.shots[0], 1);
    assert.ok(Math.abs(g.tanks[0].angle - 1) < 1e-8);
    assert.equal(g.tanks[0].x, 300);
    assert.equal(g.tanks[0].y, 300);
    assert.ok(!g.events.some((e) => e.type === "assist"));
    for (let k = 0; k < 60 && g.phase === "running"; k++) g.step(1 / 60);
    assert.ok(g.events.some((e) => e.type === "kill" && !e.self));
});
test("latest-state shot check vetoes a wall introduced while the tank was turning", () => {
    const g = empty();
    Object.assign(g.tanks[1], { x: 500, y: 300 });
    g.setPlan(0, {
        motionId: "stop_none",
        controls: { drive: "stop", turn: "none", fire: true },
        duration: 0.6,
        turnDuration: 0,
        fireDelay: 0.2,
    });
    for (let k = 0; k < 12; k++) g.step(1 / 60);
    g.maze.walls = [{ x: 340, y: 0, w: 8, h: 630 }];
    g.step(1 / 60);
    assert.equal(g.shots[0], 0);
    assert.ok(
        g.events.some((e) => e.type === "plan_veto" && e.reason === "self_hit"),
    );
    assert.equal(g.tanks[0].alive, true);
});
test("planning state contains all own/enemy shells and joint candidate outcomes", () => {
    const g = empty();
    for (let owner = 0; owner < 2; owner++)
        for (let n = 0; n < 5; n++)
            g.bullets.push({
                id: `b${owner}-${n}`,
                owner,
                x: 450 + n * 10,
                y: 200,
                vx: 290,
                vy: 0,
                age: 1,
                bounces: 1,
            });
    const o = buildPlans(g, 0, { reactionDelay: 0.8 });
    assert.equal(o.bullets.length, 10);
    assert.equal(o.actions.length, 9);
    assert.ok(o.state.includes("800ms"));
    for (const b of g.bullets) assert.ok(o.state.includes(b.id));
    assert.ok(
        o.request.motions.every((m) => typeof m.shot.can_fire === "boolean"),
    );
});
test("spawn facing a nearby wall has no permitted opening shot", () => {
    const g = new Game(),
        o = buildPlans(g, 0, { reactionDelay: 0 });
    assert.ok(o.actions.every((p) => !p.allowFire));
});
test("replacing a stale motion cancels the weapon decision conditioned on that motion", () => {
    const g = empty();
    Object.assign(g.tanks[1], { x: 500, y: 300 });
    const o = buildPlans(g, 0, { reactionDelay: 0 });
    const denied = o.actions.find((a) => !o.admissible.includes(a.id));
    assert.ok(denied);
    const r = applyPlanDecision(g, 0, {
        motion_id: denied.id,
        controls: { fire: true },
        probabilities: {
            motion: Object.fromEntries(o.actions.map((a) => [a.id, 1 / 9])),
        },
    });
    assert.notEqual(r.motion_id, denied.id);
    assert.equal(r.controls.fire, false);
    assert.equal(g.tanks[0].pendingFire, false);
    assert.ok(g.events.some((e) => e.type === "plan_revalidate"));
});
test("bounded motion forecast agrees with actual plan expiry", () => {
    const g = empty(),
        o = buildPlans(g, 0, { reactionDelay: 0 }),
        p = o.actions.find((a) => a.id === "forward_right");
    g.setPlan(0, {
        ...p,
        motionId: p.id,
        controls: { ...p.controls, fire: false },
    });
    for (let k = 0; k < 60; k++) g.step(1 / 60);
    assert.ok(distance(g.tanks[0], p.movement) < 3);
    assert.ok(Math.abs(g.tanks[0].angle - p.movement.angle) < 0.025);
});
test("planning does not give immunity: own ricochets remain lethal", () => {
    const g = empty();
    g.maze.walls = [{ x: 340, y: 0, w: 8, h: 630 }];
    g.setPlan(0, {
        motionId: "stop_none",
        controls: { drive: "stop", turn: "none", fire: false },
        duration: 0.6,
        turnDuration: 0,
        fireDelay: 0,
    });
    g.fire(g.tanks[0]);
    for (let k = 0; k < 60 && g.phase === "running"; k++) g.step(1 / 60);
    assert.equal(g.tanks[0].alive, false);
    assert.ok(g.events.some((e) => e.type === "kill" && e.self));
});
