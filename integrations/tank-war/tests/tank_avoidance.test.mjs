import test from "node:test";
import assert from "node:assert/strict";
import {
    Game,
    TURN_SPEED,
    blocked,
    buildAvoidance,
    buildObservation,
    forecastCurrentControls,
} from "../web/game/engine.js";
import { buildPlans } from "../web/game/planner.js";
function arena({
    direction = 0,
    owner = 1,
    angle = 0,
    wall = false,
    enabled = true,
} = {}) {
    const g = new Game();
    g.setAssistProfile("strong");
    g.maze.walls = [];
    g.setAssistEnabled(enabled);
    Object.assign(g.tanks[0], { x: 450, y: 315, angle });
    Object.assign(g.tanks[1], { x: 780, y: 530 });
    const a = (direction * Math.PI) / 2;
    g.bullets = [
        {
            id: "incoming",
            owner,
            x: 450 - Math.cos(a) * 200,
            y: 315 - Math.sin(a) * 200,
            vx: Math.cos(a) * 290,
            vy: Math.sin(a) * 290,
            age: 1,
            bounces: owner === 0 ? 1 : 0,
            trail: [],
        },
    ];
    if (wall)
        g.maze.walls =
            direction % 2 === 0
                ? [{ x: 400, y: 343, w: 100, h: 8 }]
                : [{ x: 478, y: 260, w: 8, h: 100 }];
    g.start();
    return g;
}
const reports = [];
for (const owner of [0, 1])
    for (const direction of [0, 1, 2, 3])
        for (const wall of [false, true])
            for (const angle of [0, Math.PI / 2]) {
                test(`avoidance survives incoming shell: owner=${owner}, direction=${direction}, wall=${wall}, heading=${angle}`, () => {
                    const opts = { owner, direction, wall, angle },
                        g = arena(opts),
                        baseline = arena({ ...opts, enabled: false });
                    // No live model command: this specifically covers inference stalls and expiry.
                    for (let k = 0; k < 72; k++) {
                        const t = g.tanks[0],
                            before = { x: t.x, y: t.y, angle: t.angle };
                        g.step(1 / 60);
                        baseline.step(1 / 60);
                        assert.ok(
                            Math.hypot(t.x - before.x, t.y - before.y) <=
                                125 / 60 + 1e-8,
                        );
                        assert.ok(
                            Math.abs(
                                Math.atan2(
                                    Math.sin(t.angle - before.angle),
                                    Math.cos(t.angle - before.angle),
                                ),
                            ) <=
                                TURN_SPEED / 60 + 1e-8,
                        );
                        assert.equal(blocked(g.maze, t.x, t.y), false);
                    }
                    reports.push({
                        ...opts,
                        survived: g.tanks[0].alive,
                        baseline_survived: baseline.tanks[0].alive,
                        events: g.events,
                    });
                    assert.equal(baseline.tanks[0].alive, false);
                    assert.equal(g.tanks[0].alive, true);
                    assert.ok(
                        g.events.some(
                            (e) =>
                                e.type === "assist" &&
                                e.threat.id === "incoming" &&
                                e.threat.owner === owner,
                        ),
                    );
                });
            }
test("changes to a plan cancel a future conditioned shot before its scheduled time", () => {
    const g = arena(),
        t = g.tanks[0];
    g.setPlan(0, {
        motionId: "stop_none",
        controls: { drive: "stop", turn: "none", fire: true },
        duration: 0.65,
        turnDuration: 0,
        fireDelay: 0.5,
    });
    g.step(1 / 60);
    assert.equal(t.assistActive, true);
    assert.equal(t.pendingFire, false);
    assert.equal(t.firing, true);
    assert.equal(t.fireBlocked, true);
    assert.equal(t.drive, "stop");
    assert.equal(t.turn, "none");
    for (let k = 0; k < 65; k++) g.step(1 / 60);
    assert.equal(g.shots[0], 0);
    assert.equal(
        g.events.filter((e) => e.type === "assist_fire_cancel").length,
        1,
    );
});
test("new model responses cannot sneak a pending shot through an active override", () => {
    const g = arena();
    g.step(1 / 60);
    assert.equal(g.tanks[0].assistActive, true);
    g.setControls(0, { drive: "stop", turn: "none", fire: true }, 0.6);
    g.step(1 / 60);
    assert.equal(g.tanks[0].pendingFire, false);
    assert.equal(g.shots[0], 0);
    assert.ok(g.events.some((e) => e.type === "assist_fire_cancel"));
});
test("turn deadline and command expiry are included in the original-path forecast", () => {
    const g = arena();
    g.setPlan(0, {
        motionId: "forward_right",
        controls: { drive: "forward", turn: "right", fire: false },
        duration: 0.2,
        turnDuration: 0.1,
        fireDelay: 0,
    });
    const f = buildAvoidance(g, 0).proposed;
    g.setAssistEnabled(false);
    for (let k = 0; k < 54 && g.phase === "running"; k++) g.step(1 / 60);
    assert.equal(f.hitTime !== null, !g.tanks[0].alive);
    // Survivor comparison avoids expecting simulation to move after an actual death.
    if (f.hitTime === null) {
        assert.ok(Math.hypot(f.x - g.tanks[0].x, f.y - g.tanks[0].y) < 3);
        assert.ok(
            Math.abs(f.angle - g.tanks[0].angle) < TURN_SPEED / 60 + 1e-8,
        );
    }
});
test("all nine forecast outcomes agree with physical replay including wall ricochets", () => {
    for (const owner of [0, 1]) {
        const g = arena({ owner });
        g.maze.walls = [{ x: 670, y: 0, w: 8, h: 630 }];
        g.bullets[0].x = 600;
        g.bullets[0].vx = 290;
        const obs = buildAvoidance(g, 0);
        for (const f of obs.options) {
            const replay = arena({ owner, enabled: false });
            replay.maze = g.maze;
            replay.bullets = g.bullets.map((b) => ({ ...b, trail: [] }));
            replay.setControls(0, { ...f.controls, fire: false }, 2);
            for (let k = 0; k < 54 && replay.phase === "running"; k++)
                replay.step(1 / 60);
            const hit = replay.events.find(
                (e) => e.type === "kill" && e.victim === 0,
            );
            assert.equal(f.hitTime !== null, !!hit, JSON.stringify(f.controls));
            if (hit) assert.ok(Math.abs(f.hitTime - hit.time) <= 1 / 60 + 1e-8);
        }
    }
});
test("model context reports the real override, and response extrapolation uses it after expiry", () => {
    const g = arena();
    g.step(1 / 60);
    const t = g.tanks[0];
    assert.equal(t.assistActive, true);
    assert.deepEqual(forecastCurrentControls(g, t, 0.3), t.avoidance.controls);
    for (const s of [
        buildObservation(g, 0).state,
        buildPlans(g, 0).state,
        buildPlans(g, 0).request.weapon_state,
    ]) {
        assert.ok(s.includes("Local bullet avoidance=ON; active=true"));
        assert.ok(s.includes(`actual=${t.executedDrive}/${t.executedTurn}`));
        assert.ok(s.includes("incoming"));
    }
});
test("disable clears the override immediately and no bullet-free scene triggers avoidance", () => {
    const g = arena();
    g.step(1 / 60);
    assert.equal(g.tanks[0].assistActive, true);
    g.setAssistEnabled(false);
    g.step(1 / 60);
    assert.equal(g.tanks[0].assistActive, false);
    assert.equal(g.tanks[0].executedDrive, "stop");
    g.bullets = [];
    g.setAssistEnabled(true);
    g.setControls(0, { drive: "forward", turn: "right", fire: false });
    g.step(1 / 60);
    assert.equal(g.tanks[0].assistActive, false);
    assert.equal(g.tanks[0].executedDrive, "forward");
    assert.equal(g.tanks[0].executedTurn, "right");
});
test("keyboard players retain their own driving, turning and shooting decisions", () => {
    const g = arena();
    g.step(1 / 60, { 0: { drive: 0, turn: 1, fire: true } });
    assert.equal(g.tanks[0].assistActive, false);
    assert.equal(g.tanks[0].x, 450);
    assert.equal(g.tanks[0].executedTurn, "right");
    assert.equal(g.shots[0], 1);
});
test("avoidance releases after danger has passed instead of controlling indefinitely", () => {
    const g = arena();
    for (let k = 0; k < 90; k++) g.step(1 / 60);
    assert.equal(g.tanks[0].alive, true);
    assert.equal(g.tanks[0].assistActive, false);
    assert.ok(g.events.some((e) => e.type === "assist_end"));
});
test("a new projectile invalidates the cached check immediately", () => {
    const g = arena();
    g.bullets = [];
    g.step(1 / 60);
    g.tanks[0].safetyUntil = 10;
    g.bullets = arena().bullets;
    g.step(1 / 60);
    assert.equal(g.tanks[0].assistActive, true);
});
test("the tenth shell is considered, including a returning own shell", () => {
    const g = arena({ owner: 0 }),
        incoming = g.bullets[0];
    g.bullets = Array.from({ length: 9 }, (_, n) => ({
        id: "far-" + n,
        owner: n % 2,
        x: 100 + n * 5,
        y: 50,
        vx: 290,
        vy: 0,
        age: 7.98,
        bounces: 0,
        trail: [],
    })).concat(incoming);
    g.step(1 / 60);
    assert.equal(g.tanks[0].assistActive, true);
    assert.equal(g.tanks[0].lastAvoidance.threat.id, "incoming");
});
test("expired shells do not initiate an override", () => {
    const g = arena();
    g.bullets[0].age = 7.99;
    g.step(1 / 60);
    assert.equal(g.tanks[0].assistActive, false);
});
test("an unavoidable point-blank own shell still kills with assistance enabled", () => {
    const g = arena({ owner: 0 });
    Object.assign(g.bullets[0], {
        x: 450,
        y: 315,
        vx: 0,
        vy: 0,
        age: 0.05,
        bounces: 1,
    });
    g.step(1 / 60);
    assert.equal(g.tanks[0].alive, false);
    assert.ok(g.events.some((e) => e.type === "kill" && e.self));
});
