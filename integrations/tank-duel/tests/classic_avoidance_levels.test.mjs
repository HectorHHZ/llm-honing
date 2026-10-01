import test from "node:test";
import assert from "node:assert/strict";
import {
    Game,
    AVOIDANCE_PROFILES,
    buildAvoidance,
    avoidanceContext,
    forecastCurrentControls,
} from "../web/classic/engine.js";
function incoming(level = "light", gap = 200) {
    const g = new Game();
    g.maze.walls = [];
    g.setAssistProfile(level);
    g.setAssistEnabled(true);
    Object.assign(g.tanks[0], { x: 450, y: 315, angle: Math.PI / 2 });
    Object.assign(g.tanks[1], { x: 780, y: 530 });
    g.bullets = [
        {
            id: "incoming",
            owner: 1,
            x: 450 - gap,
            y: 315,
            vx: 290,
            vy: 0,
            age: 1,
            bounces: 0,
            trail: [],
        },
    ];
    g.start();
    return g;
}
function untilActive(g) {
    for (let k = 0; k < 60 && g.phase === "running"; k++) {
        g.step(1 / 60);
        if (g.tanks[0].assistActive) return;
    }
    throw Error("expected a dodge");
}
test("default is light and does not preempt a distant threat that the old level dodges", () => {
    assert.equal(new Game().assistProfile, "light");
    const light = incoming(),
        strong = incoming("strong");
    light.step(1 / 60);
    strong.step(1 / 60);
    assert.equal(light.tanks[0].assistActive, false);
    assert.equal(strong.tanks[0].assistActive, true);
    untilActive(light);
    const event = light.events.find(
        (e) => e.type === "assist" && e.player === 0,
    );
    assert.ok(event.hit_ms <= 480);
    assert.ok(event.time >= 0.1);
});
test("model responses cannot extend a light burst past 400ms", () => {
    const g = incoming("light", 100);
    untilActive(g);
    const began = g.tanks[0].avoidance.startedAt;
    let count = 1;
    while (g.tanks[0].assistActive && g.elapsed - began < 1) {
        g.setControls(0, { drive: "stop", turn: "none", fire: false }, 2);
        g.step(1 / 60);
        if (g.tanks[0].assistActive) count++;
    }
    assert.ok(count <= 24);
    assert.ok(g.elapsed - began <= 0.4 + 1 / 60);
    assert.ok(
        g.events.some(
            (e) =>
                e.type === "assist_end" &&
                e.player === 0 &&
                ["burst_limit", "safe_path"].includes(e.reason),
        ),
    );
    assert.ok(g.tanks[0].assistCooldownUntil - g.elapsed >= 0.27);
});
test("new shells and model commands do not bypass the 280ms cooldown", () => {
    const g = incoming("light", 100);
    untilActive(g);
    while (g.tanks[0].assistActive) g.step(1 / 60);
    const t = g.tanks[0],
        deadline = t.assistCooldownUntil;
    g.bullets = [
        {
            id: "second",
            owner: 1,
            x: t.x - 80,
            y: t.y,
            vx: 290,
            vy: 0,
            age: 1,
            bounces: 0,
            trail: [],
        },
    ];
    for (let k = 0; k < 30 && t.alive; k++) {
        g.setControls(0, { drive: "stop", turn: "none", fire: false }, 2);
        g.step(1 / 60);
        assert.equal(t.assistActive, false);
        assert.equal(t.assistCooldownUntil, deadline);
    }
    assert.equal(
        t.alive,
        false,
        "a second attack during cooldown is allowed to hit",
    );
});
test("short-burst predictions stop the dodge at its deadline", () => {
    const g = incoming(),
        obs = buildAvoidance(g, 0);
    g.bullets = [];
    for (const f of obs.options) {
        const replay = incoming();
        replay.bullets = [];
        replay.setAssistEnabled(false);
        replay.setControls(0, { ...f.controls, fire: false }, 0.4);
        for (let k = 0; k < 39; k++) replay.step(1 / 60);
        assert.ok(
            Math.hypot(f.x - replay.tanks[0].x, f.y - replay.tanks[0].y) < 0.01,
        );
        assert.ok(Math.abs(f.angle - replay.tanks[0].angle) < 1e-8);
    }
});
test("response-time forecast returns to original controls after a burst, including while still active now", () => {
    const g = incoming("light", 100);
    g.setControls(0, { drive: "stop", turn: "none", fire: false }, 2);
    untilActive(g);
    const t = g.tanks[0];
    assert.deepEqual(forecastCurrentControls(g, t, 0.05), t.avoidance.controls);
    assert.deepEqual(forecastCurrentControls(g, t, 0.45), {
        drive: "stop",
        turn: "none",
    });
});
test("light does not recheck immediately on a new-shell signature inside the polling interval", () => {
    const g = incoming("light", 200);
    g.step(1 / 60);
    const next = g.tanks[0].safetyUntil;
    g.bullets[0].id = "replacement";
    g.bullets[0].x = 350;
    g.step(1 / 60);
    assert.equal(g.tanks[0].safetyUntil, next);
    assert.equal(g.tanks[0].assistActive, false);
});
test("normal shooting is available in cooldown; a cancelled shot is never queued for later", () => {
    const g = incoming("light", 100);
    g.setControls(0, { drive: "stop", turn: "none", fire: true }, 2);
    untilActive(g);
    assert.equal(g.shots[0], 0);
    while (g.tanks[0].assistActive) g.step(1 / 60);
    g.step(1 / 60);
    assert.equal(g.shots[0], 0);
    g.setControls(0, { drive: "stop", turn: "none", fire: true }, 2);
    g.step(1 / 60);
    assert.equal(g.shots[0], 1);
    assert.equal(g.tanks[0].assistActive, false);
});
test("current strength, real cadence and cooldown are visible in context", () => {
    for (const level of Object.keys(AVOIDANCE_PROFILES)) {
        const g = incoming(level),
            p = AVOIDANCE_PROFILES[level],
            state = avoidanceContext(g, 0);
        assert.ok(state.includes("Level=" + level));
        assert.ok(
            state.includes(`checks every ${Math.round(p.interval * 1000)}ms`),
        );
        assert.ok(!state.includes("nonems"));
    }
    const g = incoming("light", 100);
    untilActive(g);
    while (g.tanks[0].assistActive) g.step(1 / 60);
    assert.ok(avoidanceContext(g, 0).includes("remaining cooldown=280ms"));
});
test("changing level clears an existing override and round reset preserves the selection", () => {
    const g = incoming("strong");
    untilActive(g);
    g.setAssistProfile("light");
    assert.equal(g.tanks[0].assistActive, false);
    assert.equal(g.tanks[0].avoidance, null);
    assert.equal(g.assistEnabled, true);
    g.reset();
    assert.equal(g.assistProfile, "light");
    assert.throws(() => g.setAssistProfile("unlimited"));
});
