import test from "node:test";
import assert from "node:assert/strict";
import {
    Game,
    makeMaze,
    route,
    blocked,
    advanceBullet,
    center,
    W,
    H,
    COLS,
    ROWS,
    buildObservation,
    bulletForecast,
    distance,
    forecastMovement,
    forecastShot,
    traceShot,
    TURN_SPEED,
    responseDelay,
} from "../web/classic/engine.js";

test("arenas are reproducible, symmetric, connected and have clear alternative routes", () => {
    for (let seed = 0; seed < 30; seed++) {
        const maze = makeMaze(seed);
        assert.deepEqual(maze, makeMaze(seed));
        for (const w of maze.walls.slice(4))
            assert.ok(
                maze.walls.some(
                    (other) =>
                        other.x === W - w.x - w.w &&
                        other.y === H - w.y - w.h &&
                        other.w === w.w &&
                        other.h === w.h,
                ),
                `${maze.name}: asymmetric cover`,
            );
        for (const spawn of [11, 58]) {
            const p = center(spawn),
                direction = spawn === 11 ? 1 : -1;
            for (let d = 0; d <= 150; d += 3)
                assert.equal(blocked(maze, p.x + direction * d, p.y), false);
        }
        for (let i = 0; i < COLS * ROWS; i++) {
            if (i)
                assert.ok(
                    route(maze, 0, i).length > 0,
                    `disconnected ${seed}/${i}`,
                );
            const p = center(i);
            assert.equal(blocked(maze, p.x, p.y), false);
            assert.ok(maze.links[i].length >= 2, `${maze.name}: dead end ${i}`);
            // A tank occupying a single cell must not cut off the only route.
            if (i !== 11 && i !== 58)
                assert.ok(route(maze, 11, 58, i).length > 0);
            // Waypoints must use openings that fit the complete tank body.
            for (const j of maze.links[i]) {
                const end = center(j);
                for (let step = 0; step <= 30; step++)
                    assert.equal(
                        blocked(
                            maze,
                            p.x + ((end.x - p.x) * step) / 30,
                            p.y + ((end.y - p.y) * step) / 30,
                        ),
                        false,
                    );
            }
        }
    }
});
test("a ricochet reverses its wall-normal velocity and keeps its tangential velocity", () => {
    const b = { x: 11, y: 30, vx: -100, vy: 20, age: 0, bounces: 0 };
    advanceBullet(b, 0.05, [{ x: 0, y: 0, w: 7, h: 100 }]);
    assert.equal(b.vx, 100);
    assert.equal(b.vy, 20);
    assert.equal(b.bounces, 1);
    assert.ok(b.x >= 10);
});
test("tank movement cannot cross walls even across thousands of steps", () => {
    const g = new Game(81);
    g.start();
    for (let i = 0; i < 1200; i++) {
        g.step(1 / 60, {
            0: {
                drive: Math.sign(Math.sin(i / 40)),
                turn: Math.sign(Math.cos(i / 40)),
                fire: false,
            },
            1: { drive: -1, turn: 1, fire: false },
        });
        for (const t of g.tanks) assert.equal(blocked(g.maze, t.x, t.y), false);
    }
});
test("ammo is limited to five live shells per tank", () => {
    const g = new Game(8),
        t = g.tanks[0];
    g.maze.walls = [];
    for (let i = 0; i < 5; i++) {
        t.cooldown = 0;
        assert.equal(g.fire(t), true);
    }
    t.cooldown = 0;
    assert.equal(g.fire(t), false);
    g.bullets.shift();
    assert.equal(g.fire(t), true);
});
test("enemy hit scores the surviving tank and changes the round", () => {
    const g = new Game(9);
    g.maze.walls = [];
    g.tanks[0].x = 100;
    g.tanks[0].y = 100;
    g.tanks[1].x = 200;
    g.tanks[1].y = 100;
    g.start();
    g.fire(g.tanks[0]);
    for (let i = 0; i < 60; i++) g.step(1 / 60);
    assert.equal(g.phase, "roundover");
    assert.deepEqual(g.scores, [1, 0]);
    assert.equal(g.tanks[1].alive, false);
    for (let i = 0; i < 200; i++) g.step(1 / 60);
    assert.equal(g.round, 2);
    assert.equal(g.phase, "running");
});
test("own ricochet can destroy its shooter and scores the opponent", () => {
    const g = new Game(3);
    g.maze.walls = [{ x: 250, y: 0, w: 7, h: H }];
    Object.assign(g.tanks[0], { x: 180, y: 100, angle: 0 });
    Object.assign(g.tanks[1], { x: 500, y: 400 });
    g.start();
    g.fire(g.tanks[0]);
    for (let i = 0; i < 120 && g.phase === "running"; i++) g.step(1 / 60);
    assert.equal(g.tanks[0].alive, false);
    assert.deepEqual(g.scores, [0, 1]);
    assert.ok(g.events.some((e) => e.type === "kill" && e.self));
});
test("expired commands stop driving, turning and firing; round IDs never repeat", () => {
    const g = new Game(11);
    g.maze.walls = [];
    g.start();
    g.setControls(0, { drive: "forward", turn: "right", fire: true }, 0.1);
    for (let i = 0; i < 30; i++) g.step(1 / 60);
    const t = g.tanks[0],
        before = { x: t.x, y: t.y, angle: t.angle },
        shots = g.shots[0];
    for (let i = 0; i < 30; i++) g.step(1 / 60);
    assert.deepEqual({ x: t.x, y: t.y, angle: t.angle }, before);
    assert.equal(g.shots[0], shots);
    assert.equal(t.firing, false);
    const id = g.roundSerial;
    g.reset();
    assert.ok(g.roundSerial > id);
});
test("models receive three independent heads and all nine driving/turning forecasts", () => {
    const g = new Game();
    for (const id of [0, 1]) {
        const obs = buildObservation(g, id);
        assert.deepEqual(
            obs.candidates.drive.map((c) => c.id),
            ["forward", "reverse", "stop"],
        );
        assert.deepEqual(
            obs.candidates.turn.map((c) => c.id),
            ["left", "right", "none"],
        );
        assert.deepEqual(
            obs.candidates.weapon.map((c) => c.id),
            ["fire", "hold"],
        );
        assert.equal(obs.forecasts.length, 9);
        assert.ok(obs.state.includes("NO AUTO-AIM"));
        assert.ok(obs.state.includes("hull/barrel"));
    }
});
test("legacy strafing commands are rejected instead of silently mapped", () => {
    const g = new Game();
    assert.throws(
        () => g.setControls(0, { move: "up", fire: true }),
        /Invalid tank controls/,
    );
});
test("all ten shells, including friendly shells, reach both models without filtering", () => {
    const g = new Game(17);
    g.maze.walls = [];
    for (let owner = 0; owner < 2; owner++)
        for (let n = 0; n < 5; n++)
            g.bullets.push({
                id: `shell-${owner}-${n}`,
                owner,
                x: 75 + n * 80,
                y: 120 + owner * 200,
                vx: owner ? -290 : 290,
                vy: n * 7,
                age: n * 0.3,
                bounces: n,
                trail: [],
            });
    for (const id of [0, 1]) {
        const obs = buildObservation(g, id);
        assert.equal(obs.bullets.length, 10);
        assert.equal(obs.bullets.filter((b) => b.owner === "self").length, 5);
        assert.equal(obs.bullets.filter((b) => b.owner === "enemy").length, 5);
        const rows = JSON.parse(obs.state.split("hit_s]:\n")[1].split("\n")[0]);
        assert.equal(rows.length, 10);
        for (const b of g.bullets) {
            const row = rows.find((r) => r[0] === b.id);
            assert.ok(row);
            assert.deepEqual(row.slice(2, 7), [
                b.x,
                b.y,
                b.vx,
                b.vy,
                b.bounces,
            ]);
        }
    }
});
test("forecasts treat own and enemy shells as lethal and follow reflections", () => {
    const g = new Game(3);
    g.maze.walls = [{ x: 250, y: 0, w: 7, h: H }];
    const t = { id: 0, x: 180, y: 100 };
    const shell = {
        id: "returning",
        x: 205,
        y: 100,
        vx: 290,
        vy: 0,
        age: 0.2,
        bounces: 0,
    };
    const own = bulletForecast(g, { ...shell, owner: 0 }, t),
        enemy = bulletForecast(g, { ...shell, owner: 1 }, t);
    assert.ok(own.hitTime > 0 && own.hitTime < 0.8);
    assert.deepEqual(own, enemy);
    Object.assign(g.tanks[0], t);
    g.bullets = [{ ...shell, owner: 0, trail: [] }];
    assert.ok(
        buildObservation(g, 0).forecasts.find(
            (f) => f.drive === "stop" && f.turn === "none",
        ).hitTime !== null,
    );
});
test("expired bullets do not generate future threats", () => {
    const g = new Game();
    g.maze.walls = [];
    const f = bulletForecast(
        g,
        { x: 30, y: 100, vx: 290, vy: 0, age: 7.99, bounces: 0, owner: 1 },
        { id: 0, x: 200, y: 100 },
    );
    assert.equal(f.hitTime, null);
    assert.ok(f.min > 100);
});
const emptyGame = () => {
    const g = new Game();
    g.maze.walls = [];
    Object.assign(g.tanks[0], { x: 300, y: 300, angle: 0 });
    Object.assign(g.tanks[1], { x: 700, y: 500 });
    g.start();
    return g;
};
test("forward follows hull heading; it never strafes toward screen up", () => {
    const g = emptyGame(),
        t = g.tanks[0];
    t.angle = Math.PI / 4;
    g.setControls(0, { drive: "forward", turn: "none", fire: false });
    g.step(1 / 60);
    assert.ok(t.x > 300 && t.y > 300);
    assert.ok(Math.abs(t.x - 300 - (t.y - 300)) < 1e-6);
    assert.equal(t.angle, Math.PI / 4);
});
test("turn-only rotates in place and fires along the rotated hull", () => {
    const g = emptyGame(),
        t = g.tanks[0];
    g.setControls(0, { drive: "stop", turn: "right", fire: true });
    g.step(1 / 60);
    assert.equal(t.x, 300);
    assert.equal(t.y, 300);
    assert.ok(t.angle > 0);
    assert.equal(g.shots[0], 1);
    assert.ok(
        Math.abs(Math.atan2(g.bullets[0].vy, g.bullets[0].vx) - t.angle) < 1e-8,
    );
});
test("driving, turning and shooting execute together on one physics step", () => {
    const g = emptyGame(),
        t = g.tanks[0];
    g.setControls(0, { drive: "forward", turn: "left", fire: true });
    g.step(1 / 60);
    assert.ok(t.x > 300 && t.y < 300);
    assert.ok(t.angle < 0);
    assert.equal(g.shots[0], 1);
    assert.ok(
        Math.abs(Math.atan2(g.bullets[0].vy, g.bullets[0].vx) - t.angle) < 1e-8,
    );
});
test("reverse backs away while barrel keeps its forward heading", () => {
    const g = emptyGame(),
        t = g.tanks[0];
    g.setControls(0, { drive: "reverse", turn: "none", fire: true });
    g.step(1 / 60);
    assert.ok(t.x < 300);
    assert.equal(t.y, 300);
    assert.equal(t.angle, 0);
    assert.equal(g.shots[0], 1);
    assert.equal(g.bullets[0].vx, 290);
});
test("fire is not gated on a predicted hit and never steers toward the enemy", () => {
    const g = emptyGame(),
        t = g.tanks[0];
    t.angle = -Math.PI / 2;
    g.setControls(0, { drive: "stop", turn: "none", fire: true });
    g.step(1 / 60);
    assert.equal(g.shots[0], 1);
    assert.ok(g.bullets[0].vy < -289);
    assert.equal(t.angle, -Math.PI / 2);
    g.setControls(0, { drive: "stop", turn: "none", fire: false }, 2);
    Object.assign(g.tanks[1], { x: 100, y: 200 });
    for (let i = 0; i < 30; i++) g.step(1 / 60);
    assert.equal(t.angle, -Math.PI / 2);
});
test("keyboard controls use the same body-relative drive and rotation", () => {
    const g = emptyGame(),
        t = g.tanks[0];
    g.step(1 / 60, { 0: { drive: 0, turn: -1, fire: false } });
    assert.equal(t.x, 300);
    assert.equal(t.y, 300);
    assert.ok(t.angle < 0);
    const angle = t.angle;
    g.step(1 / 60, { 0: { drive: -1, turn: 0, fire: true } });
    assert.ok(t.x < 300 && t.y > 300);
    assert.equal(t.angle, angle);
    assert.equal(g.shots[0], 1);
});
test("held turn has correct angular speed; fire-only leaves heading unchanged", () => {
    const g = emptyGame(),
        t = g.tanks[0];
    g.setControls(0, { drive: "stop", turn: "right", fire: false }, 1);
    for (let i = 0; i < 30; i++) g.step(1 / 60);
    assert.ok(Math.abs(t.angle - TURN_SPEED * 0.5) < 1e-8);
    const angle = t.angle;
    g.setControls(0, { drive: "stop", turn: "none", fire: true });
    g.step(1 / 60);
    assert.equal(t.angle, angle);
});
test("trajectory forecast follows the curved driving path and final heading", () => {
    const g = emptyGame(),
        controls = { drive: "forward", turn: "left" },
        f = forecastMovement(g, 0, controls, {
            reactionDelay: 0,
            horizon: 0.5,
        });
    g.setControls(0, { ...controls, fire: false }, 1);
    for (let i = 0; i < 30; i++) g.step(1 / 60);
    assert.ok(distance(f, g.tanks[0]) < 1);
    assert.ok(Math.abs(f.angle - g.tanks[0].angle) < 1e-8);
});
test("path forecast catches a collision during travel even when endpoint is safe", () => {
    const g = emptyGame();
    Object.assign(g.tanks[0], { x: 450, y: 315 });
    Object.assign(g.tanks[1], { x: 800, y: 500 });
    const bullet = {
        id: "crossing",
        x: 500,
        y: 215,
        vx: 0,
        vy: 250,
        age: 1,
        bounces: 0,
        owner: 1,
        trail: [],
    };
    g.bullets = [bullet];
    const moving = forecastMovement(
        g,
        0,
        { drive: "forward", turn: "none" },
        { reactionDelay: 0 },
    );
    assert.ok(moving.hitTime !== null);
    assert.equal(
        bulletForecast(g, bullet, { id: 0, x: moving.x, y: moving.y }, 0.9)
            .hitTime,
        null,
    );
});
test("body-facing driving escapes own or enemy shells with 120 ms delay", () => {
    for (const owner of [0, 1]) {
        const make = () => {
            const g = emptyGame();
            Object.assign(g.tanks[0], { x: 450, y: 315, angle: -Math.PI / 2 });
            g.bullets = [
                {
                    id: "incoming",
                    x: 250,
                    y: 315,
                    vx: 290,
                    vy: 0,
                    age: 1,
                    bounces: 0,
                    owner,
                    trail: [],
                },
            ];
            return g;
        };
        const still = make(),
            moving = make();
        assert.ok(
            forecastMovement(still, 0, { drive: "stop", turn: "none" })
                .hitTime !== null,
        );
        assert.equal(
            forecastMovement(moving, 0, { drive: "forward", turn: "none" })
                .hitTime,
            null,
        );
        for (let i = 0; i < 7; i++) moving.step(1 / 60);
        moving.setControls(
            0,
            { drive: "forward", turn: "none", fire: true },
            0.9,
        );
        for (let i = 0; i < 54; i++) {
            still.step(1 / 60);
            moving.step(1 / 60);
        }
        assert.equal(still.tanks[0].alive, false);
        assert.equal(moving.tanks[0].alive, true);
        assert.ok(moving.shots[0] > 0);
    }
});
test("optional avoidance keeps raw drive/turn/fire and reports corrections", () => {
    const g = emptyGame();
    g.setAssistProfile("strong");
    g.assistEnabled = true;
    Object.assign(g.tanks[0], { x: 450, y: 315, angle: -Math.PI / 2 });
    Object.assign(g.tanks[1], { x: 750, y: 200 });
    g.bullets = [
        {
            id: "incoming",
            owner: 1,
            x: 450,
            y: 115,
            vx: 0,
            vy: 290,
            age: 1,
            bounces: 0,
            trail: [],
        },
    ];
    g.setControls(0, { drive: "forward", turn: "none", fire: true }, 1);
    for (let n = 0; n < 54; n++) g.step(1 / 60);
    assert.equal(g.tanks[0].drive, "forward");
    assert.equal(g.tanks[0].turn, "none");
    assert.equal(g.tanks[0].firing, true);
    assert.equal(g.tanks[0].alive, true);
    assert.ok(
        g.events.some(
            (e) => e.type === "assist" && e.requested.drive === "forward",
        ),
    );
});
test("very early ricochets hit in both the observation and the real physics", () => {
    const g = emptyGame(),
        t = g.tanks[0];
    g.maze.walls = [{ x: 330, y: 0, w: 8, h: H }];
    const trace = traceShot(g.maze, t, t.angle);
    assert.equal(trace.selfHit, true);
    assert.ok(trace.time < 0.15);
    assert.ok(trace.firstBounceTime < trace.time);
    g.fire(t);
    while (g.phase === "running" && g.elapsed < 0.3) g.step(1 / 60);
    assert.equal(t.alive, false);
    assert.ok(
        Math.abs(g.events.find((e) => e.type === "kill").time - trace.time) <=
            1 / 60,
    );
});
test("blocked muzzle is unavailable, not a fictitious self-hit", () => {
    const g = emptyGame(),
        t = g.tanks[0];
    g.maze.walls = [{ x: 320, y: 0, w: 8, h: H }];
    const trace = traceShot(g.maze, t, t.angle);
    assert.equal(trace.muzzleBlocked, true);
    assert.equal(trace.selfHit, false);
    assert.equal(g.fire(t), false);
});
test("new-shot predictions agree with actual moving and rotating trajectories", () => {
    for (const drive of ["forward", "reverse", "stop"])
        for (const turn of ["left", "right", "none"]) {
            const g = emptyGame();
            g.maze.walls = [{ x: 345, y: 0, w: 8, h: H }];
            const c = { drive, turn };
            const f = forecastShot(g, 0, c, { reactionDelay: 0, horizon: 1.2 });
            g.setControls(0, { ...c, fire: true }, 1.3);
            while (g.elapsed < 1.2 && g.phase === "running") g.step(1 / 60);
            const death = g.events.find((e) => e.type === "kill" && e.self);
            assert.equal(f.selfHitTime !== null, !!death, drive + "/" + turn);
            if (death)
                assert.ok(
                    Math.abs(f.selfHitTime - death.time) <= 1 / 60 + 1e-8,
                );
        }
});
test("proposed shot is launched after inference delay and uses the updated barrel angle", () => {
    const g = emptyGame();
    g.setControls(0, { drive: "forward", turn: "right", fire: false }, 2);
    g.step(1 / 60);
    const f = forecastShot(
        g,
        0,
        { drive: "reverse", turn: "left" },
        { reactionDelay: 0.12, horizon: 0.8 },
    );
    for (let i = 0; i < 8; i++) g.step(1 / 60);
    g.setControls(0, { drive: "reverse", turn: "left", fire: true });
    g.step(1 / 60);
    const b = g.bullets[0];
    assert.ok(b);
    assert.ok(Math.abs(f.launchAngle - Math.atan2(b.vy, b.vx)) < 1e-8);
});
test("one model fire decision triggers only once while movement remains held", () => {
    const g = emptyGame(),
        t = g.tanks[0];
    g.setControls(0, { drive: "forward", turn: "none", fire: true }, 1.5);
    for (let i = 0; i < 65; i++) g.step(1 / 60);
    assert.equal(g.shots[0], 1);
    assert.ok(t.x > 400);
    g.setControls(0, { drive: "stop", turn: "none", fire: true });
    g.step(1 / 60);
    assert.equal(g.shots[0], 2);
});
test("a trigger during cooldown is discarded rather than firing later into a new heading", () => {
    const g = emptyGame();
    g.tanks[0].cooldown = 0.2;
    g.setControls(0, { drive: "stop", turn: "right", fire: true }, 1);
    for (let i = 0; i < 40; i++) g.step(1 / 60);
    assert.equal(g.shots[0], 0);
});
test("a human can still hold the fire key to shoot repeatedly", () => {
    const g = emptyGame();
    for (let i = 0; i < 65; i++)
        g.step(1 / 60, { 0: { drive: 0, turn: 0, fire: true } });
    assert.equal(g.shots[0], 4);
});
test("fire guard blocks imminent self-hit and records raw fire without changing movement", () => {
    const g = emptyGame();
    g.maze.walls = [{ x: 330, y: 0, w: 8, h: H }];
    g.fireGuardEnabled = true;
    g.setControls(0, { drive: "forward", turn: "right", fire: true });
    g.step(1 / 60);
    const t = g.tanks[0];
    assert.equal(g.shots[0], 0);
    assert.equal(t.firing, true);
    assert.equal(t.fireBlocked, true);
    assert.ok(t.x > 300 && t.angle > 0);
    assert.equal(t.executedDrive, "forward");
    assert.equal(t.executedTurn, "right");
    const event = g.events.find((e) => e.type === "fire_guard");
    assert.equal(event.requested, true);
    assert.equal(event.executed, false);
    assert.ok(event.self_hit_ms < 150);
});
test("fire guard permits both safe hits and safe missed shots without auto-aim", () => {
    for (const aligned of [true, false]) {
        const g = emptyGame();
        g.fireGuardEnabled = true;
        if (aligned) g.tanks[1].y = 300;
        g.setControls(0, { drive: "stop", turn: "none", fire: true });
        g.step(1 / 60);
        assert.equal(g.shots[0], 1);
        assert.equal(g.tanks[0].angle, 0);
        assert.equal(g.bullets[0].vy, 0);
    }
});
test("fire guard can be disabled and does not affect human shooting or bullet lethality", () => {
    for (const keyboard of [true, false]) {
        const g = emptyGame();
        g.maze.walls = [{ x: 330, y: 0, w: 8, h: H }];
        g.fireGuardEnabled = keyboard;
        if (!keyboard)
            g.setControls(0, { drive: "stop", turn: "none", fire: true });
        for (let i = 0; i < 30 && g.phase === "running"; i++)
            g.step(
                1 / 60,
                keyboard ? { 0: { drive: 0, turn: 0, fire: i === 0 } } : {},
            );
        assert.equal(g.tanks[0].alive, false);
        assert.ok(g.events.some((e) => e.type === "kill" && e.self));
    }
});
test("context separates existing-shell movement risk from new-shot risk and exposes obstructing walls", () => {
    const g = new Game();
    // Keep this obstruction regression independent of the open demo layouts.
    g.maze.walls = [{ x: 176, y: 0, w: 8, h: H }];
    const o = buildObservation(g, 0);
    assert.equal(o.bullets.length, 0);
    assert.equal(o.enemyVisible, false);
    assert.ok(o.frontWallDistance < 45);
    assert.ok(o.localWalls.length > 0);
    assert.equal(o.shot.selfHit, true);
    assert.ok(o.firingForecasts.some((f) => f.selfHitTime !== null));
    assert.ok(o.state.includes("Existing-shell forecasts EXCLUDE"));
    assert.ok(o.state.includes("new drive+turn lasts 600 ms then STOPS"));
    assert.ok(o.candidates.drive.every((c) => !c.description.includes("safe")));
    assert.deepEqual(o.candidates.weapon, [
        { id: "fire", description: "Fire one shell." },
        { id: "hold", description: "Do not fire a shell." },
    ]);
});
test("observations use a measured 800ms delay without pretending it is 300ms", () => {
    assert.equal(responseDelay(800), 0.8);
    const g = emptyGame();
    g.setControls(0, { drive: "forward", turn: "right", fire: false }, 0.6);
    g.step(1 / 60);
    const o = buildObservation(g, 0, { reactionDelay: responseDelay(800) });
    assert.ok(o.state.includes("predicted response delay 800 ms"));
    assert.equal(o.reactionDelay, 0.8);
    assert.ok(
        o.firingForecasts.every(
            (f) => f.launchTime === null || f.launchTime >= 0.8,
        ),
    );
});
test("movement prediction stops after the same command duration as the executor", () => {
    const g = emptyGame(),
        c = { drive: "forward", turn: "right" },
        f = forecastMovement(g, 0, c, {
            reactionDelay: 0,
            controlDuration: 0.6,
            horizon: 1.5,
        });
    g.setControls(0, { ...c, fire: false }, 0.6);
    for (let k = 0; k < 90; k++) g.step(1 / 60);
    assert.ok(distance(f, g.tanks[0]) < 3);
    assert.ok(Math.abs(f.angle - g.tanks[0].angle) < 0.06);
});
test("new-shot prediction includes a return after command expiry and after the old 1.2s horizon", () => {
    const g = emptyGame();
    g.maze.walls = [{ x: 570, y: 0, w: 8, h: H }];
    const c = { drive: "forward", turn: "none" };
    // Drive briefly while firing, then park; the later return still kills.
    const f = forecastShot(g, 0, c, {
        reactionDelay: 0,
        controlDuration: 0.1,
        horizon: 3,
    });
    g.setControls(0, { ...c, fire: true }, 0.1);
    for (let k = 0; k < 180 && g.phase === "running"; k++) g.step(1 / 60);
    const kill = g.events.find((e) => e.type === "kill" && e.self);
    assert.ok(kill);
    assert.ok(f.selfHitTime > 1.2);
    assert.ok(Math.abs(f.selfHitTime - kill.time) < 1 / 60 + 1e-8);
});
