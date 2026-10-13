import test from "node:test";
import assert from "node:assert/strict";
import { AdaptiveCadence } from "../web/shared/cadence.js";
import { validProbabilities } from "../web/shared/model-client.js";
import {
    Game,
    SHELL_LIFETIME,
    FIRE_COOLDOWN,
    forecastShot,
} from "../web/classic/engine.js";
import {
    STRATEGIES,
    rangeCost,
    publishedStrategies,
    strategyFrom,
} from "../web/classic/strategies.js";
import { buildPlans } from "../web/classic/planner.js";
test("cadence learns each seat separately, bounds history and recovers from busy", () => {
    const a = new AdaptiveCadence(),
        b = new AdaptiveCadence();
    [300, 310, 320, 330, 900].forEach((x) => a.observe(x));
    assert.equal(a.interval, 300);
    assert.equal(a.prediction, 330);
    assert.equal(b.interval, 100);
    a.busy();
    assert.equal(a.interval, 400);
    a.observe(315);
    assert.equal(a.interval, 360);
    for (let i = 0; i < 50; i++) a.observe(200);
    assert.equal(a.samples.length, 40);
    assert.equal(a.interval, 200);
    a.observe(NaN);
    a.observe(-1);
    assert.equal(a.samples.length, 40);
    a.reset();
    assert.equal(a.stats.samples, 0);
    assert.equal(a.interval, 100);
});
test("native four-decimal probabilities are accepted without renormalization", () => {
    const p = { a: 0.3333, b: 0.3333, c: 0.3333 };
    assert.equal(validProbabilities(p, Object.keys(p)), true);
    assert.equal(p.a, 0.3333);
    for (const bad of [
        { a: 0.3, b: 0.3, c: 0.3 },
        { a: NaN, b: 0.5, c: 0.5 },
        { a: -0.1, b: 0.5, c: 0.6 },
        { a: 1, b: 0 },
    ])
        assert.ok(!validProbabilities(bad, ["a", "b", "c"]));
});
test("ammo forecast agrees with shortened lifetime, both spawns face inward", () => {
    const g = new Game();
    g.maze.walls = [];
    g.start();
    g.bullets = Array.from({ length: 5 }, (_, i) => ({
        id: String(i),
        owner: 0,
        x: 700,
        y: 50 + i * 15,
        vx: 0,
        vy: 0,
        age: SHELL_LIFETIME - 0.05,
        bounces: 0,
        trail: [],
    }));
    assert.equal(
        forecastShot(
            g,
            0,
            { drive: "stop", turn: "none" },
            { reactionDelay: 0.1 },
        ).canFire,
        true,
    );
    for (let i = 0; i < 5; i++) g.step(1 / 60);
    assert.equal(g.bullets.length, 0);
    assert.equal(FIRE_COOLDOWN, 0.3);
    for (const seed of [1, 20, 99]) {
        const h = new Game(seed);
        for (const t of h.tanks) assert.equal(t.angle, t.x < 450 ? 0 : Math.PI);
    }
});
const STANDOFF = {
    id: "standoff-v1",
    description:
        "Compare firing angles while maintaining a 220-360 px engagement range.",
    allAnglePrompt: true,
    multiAim: true,
    spacingWeight: 1.6,
    minRange: 220,
    maxRange: 360,
};
test("only the two baselines are published; a harness strategy is checked", () => {
    assert.deepEqual(
        publishedStrategies().map((s) => s.id),
        ["baseline", "raw"],
    );
    assert.equal(strategyFrom("baseline"), STRATEGIES.baseline);
    assert.throws(() => strategyFrom("raw"), /Unknown planned strategy/);
    assert.throws(
        () => strategyFrom({ ...STANDOFF, aimBot: true }),
        /Unknown strategy parameter aimBot/,
    );
    assert.throws(
        () => strategyFrom({ ...STANDOFF, spacingWeight: "1" }),
        /must be a number/,
    );
    assert.throws(
        () => strategyFrom({ ...STANDOFF, maxRange: 100 }),
        /minRange <= maxRange/,
    );
    assert.throws(
        () => strategyFrom({ multiAim: true }),
        /id and a description/,
    );
    const s = strategyFrom(STANDOFF);
    assert.equal(s.mode, "plans");
    assert.equal(s.clearanceWeight, undefined);
});
test("a harness strategy's spacing improves toward its preferred range", () => {
    const s = strategyFrom(STANDOFF);
    assert.ok(rangeCost(100, s) > rangeCost(200, s));
    assert.equal(rangeCost(300, s), 0);
    assert.ok(rangeCost(600, s) > 0);
    const g = new Game();
    g.maze.walls = [];
    Object.assign(g.tanks[0], { x: 300, y: 300, angle: 0 });
    Object.assign(g.tanks[1], { x: 400, y: 300 });
    const plans = buildPlans(g, 0, { reactionDelay: 0, strategy: s });
    assert.equal(plans.actions.length, 9);
    assert.ok(
        plans.actions.find((x) => x.id === "reverse_none").spacingProgress > 0,
    );
    assert.ok(
        plans.actions.find((x) => x.id === "forward_none").spacingProgress < 0,
    );
    assert.ok(plans.state.includes("standoff-v1"));
    assert.ok(plans.actions.every((x) => Number.isFinite(x.aimHeading)));
});

test("revalidateSlack preserves a still-safe stationary shot without weakening the fresh shot veto", async () => {
    const { applyPlanDecision } = await import("../web/classic/planner.js");
    const fixture = () => {
        const g = new Game();
        g.maze.walls = [];
        Object.assign(g.tanks[0], { x: 300, y: 300, angle: 0 });
        Object.assign(g.tanks[1], { x: 500, y: 300 });
        g.start();
        return g;
    };
    const reply = {
        motion_id: "stop_none",
        controls: { drive: "stop", turn: "none", fire: true },
        probabilities: {
            motion: Object.fromEntries(
                buildPlans(fixture(), 0).actions.map((p) => [
                    p.id,
                    p.id === "stop_none" ? 1 : 0,
                ]),
            ),
        },
    };
    const ordinary = fixture(),
        steady = fixture();
    assert.equal(applyPlanDecision(ordinary, 0, reply).controls.fire, false);
    const applied = applyPlanDecision(
        steady,
        0,
        reply,
        strategyFrom({
            id: "steady-aim-v1",
            description: "Retain a still-safe selected motion.",
            revalidateSlack: 60,
        }),
    );
    assert.equal(applied.motion_id, "stop_none");
    assert.equal(applied.controls.fire, true);
    assert.ok(steady.events.some((e) => e.type === "plan_intent_retained"));
    steady.maze.walls = [{ x: 340, y: 0, w: 8, h: 630 }];
    steady.step(1 / 60);
    assert.equal(steady.shots[0], 0);
    assert.ok(steady.events.some((e) => e.type === "plan_veto"));
});
