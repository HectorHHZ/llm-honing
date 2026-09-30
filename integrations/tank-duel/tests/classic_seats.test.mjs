import test from "node:test";
import assert from "node:assert/strict";
import {
    Game,
    avoidanceContext,
    avoidanceProfile,
    assistEnabledFor,
    forecastCurrentControls,
} from "../web/classic/engine.js";
import { adapter } from "../web/classic/adapter.js";
import {
    defaultSeats,
    normalizeSeat,
    matchupPreset,
    seatProtocol,
} from "../web/shared/seat-config.js";
import { validateRequest } from "../model-bridge.mjs";

function incoming() {
    const g = new Game();
    g.maze.walls = [];
    Object.assign(g.tanks[0], { x: 250, y: 220, angle: Math.PI / 2 });
    Object.assign(g.tanks[1], { x: 650, y: 450, angle: Math.PI / 2 });
    g.bullets = [0, 1].map((i) => ({
        id: "shell-" + i,
        owner: 1 - i,
        x: g.tanks[i].x - 160,
        y: g.tanks[i].y,
        vx: 290,
        vy: 0,
        age: 1,
        bounces: 1,
        trail: [],
    }));
    g.start();
    return g;
}
test("one tank dodges while the other is hit: both directions and shell ownership", () => {
    for (const protectedId of [0, 1]) {
        const g = incoming();
        for (const id of [0, 1])
            g.setPlayerSettings(id, {
                assistEnabled: id === protectedId,
                assistProfile: "strong",
            });
        for (let k = 0; k < 50 && g.phase === "running"; k++) g.step(1 / 60);
        assert.equal(g.tanks[protectedId].alive, true);
        assert.equal(g.tanks[1 - protectedId].alive, false);
        assert.ok(
            g.events.some(
                (e) => e.type === "assist" && e.player === protectedId,
            ),
        );
        assert.ok(
            !g.events.some(
                (e) => e.type === "assist" && e.player !== protectedId,
            ),
        );
    }
});
test("changing one dodge leaves the other active burst and cooldown untouched", () => {
    const g = incoming();
    for (const id of [0, 1])
        g.setPlayerSettings(id, {
            assistEnabled: true,
            assistProfile: "strong",
        });
    g.step(1 / 60);
    assert.ok(g.tanks.every((t) => t.assistActive));
    const other = structuredClone(g.tanks[1]);
    g.setPlayerSettings(0, { assistEnabled: false });
    assert.equal(g.tanks[0].assistActive, false);
    assert.deepEqual(g.tanks[1], other);
    assert.deepEqual(forecastCurrentControls(g, g.tanks[0], 0), {
        drive: "stop",
        turn: "none",
    });
    assert.deepEqual(
        forecastCurrentControls(g, g.tanks[1], 0),
        other.avoidance.controls,
    );
});
test("different levels and shot guards persist over resets and round changes", () => {
    const g = incoming();
    g.setPlayerSettings(0, {
        assistEnabled: true,
        assistProfile: "light",
        fireGuardEnabled: true,
    });
    g.setPlayerSettings(1, {
        assistEnabled: false,
        assistProfile: "strong",
        fireGuardEnabled: false,
    });
    for (const advance of [() => {}, () => g.reset(), () => g.nextRound()]) {
        advance();
        assert.equal(assistEnabledFor(g, 0), true);
        assert.equal(assistEnabledFor(g, 1), false);
        assert.equal(avoidanceProfile(g, 0).trigger, 0.3);
        assert.equal(avoidanceProfile(g, 1).trigger, 0.9);
        assert.match(
            avoidanceContext(g, 0),
            /Shot guard=ON.*avoidance=ON.*Level=light/,
        );
        assert.match(avoidanceContext(g, 1), /Shot guard=OFF.*avoidance=OFF/);
    }
});
test("each context contains its own assist state, all shells, and the correct protocol", () => {
    const g = incoming();
    g.setPlayerSettings(0, { assistEnabled: true, assistProfile: "light" });
    g.setPlayerSettings(1, { assistEnabled: false });
    for (const [controllerMode, contextVersion] of [
        ["plans", "full"],
        ["raw", "full"],
        ["raw", "arena-v1"],
    ]) {
        for (const id of [0, 1]) {
            const obs = adapter.observe(g, id, {
                controllerMode,
                contextVersion,
                interval: 50,
                rtt: 100,
            });
            assert.equal(obs.bulletCount, 2);
            assert.equal(obs.request.avoidance_enabled, id === 0);
            assert.match(
                obs.state,
                new RegExp("avoidance=" + (id === 0 ? "ON" : "OFF")),
            );
            for (const b of g.bullets) assert.ok(obs.state.includes(b.id));
            assert.equal(
                obs.request.protocol,
                controllerMode === "plans" ? "plans-v1" : "controls-v3",
            );
            assert.ok(obs.request.context_revision);
            validateRequest(
                { ...obs.request, id: "test", player: id, round_id: 1 },
                "classic",
            );
            if (contextVersion === "arena-v1") {
                assert.ok(!obs.aim);
                assert.ok(!obs.actions);
                assert.ok(!obs.state.includes("AIM CURRENT"));
                assert.ok(
                    !JSON.stringify(obs.candidates).includes(
                        "existing_shell_hit",
                    ),
                );
            }
        }
    }
    assert.throws(
        () =>
            adapter.observe(g, 0, {
                controllerMode: "plans",
                contextVersion: "arena-v1",
                interval: 100,
            }),
        /requires full/,
    );
});
test("restored settings reject unavailable endpoints and incompatible profiles", () => {
    const models = [
        { id: "gemma", label: "DiffusionGemma" },
        { id: "nano", label: "NanoJev" },
    ];
    const seats = matchupPreset(models);
    assert.equal(seats[0].backend, "gemma");
    assert.equal(seats[1].backend, "nano");
    assert.equal(seatProtocol("classic", seats[0]), "plans-v1");
    assert.equal(seatProtocol("classic", seats[1]), "controls-v3");
    seats[0].assistEnabled = false;
    assert.equal(seats[1].assistEnabled, true);
    assert.equal(matchupPreset([models[0]]), null);
    const clean = normalizeSeat(
        {
            backend: "removed",
            controllerMode: "plans",
            contextVersion: "arena-v1",
            interval: 0,
            assistEnabled: "false",
            fireGuardEnabled: true,
        },
        models,
        defaultSeats(models)[0],
    );
    assert.equal(clean.backend, "gemma");
    assert.equal(clean.contextVersion, "full");
    assert.equal(clean.interval, 100);
    assert.equal(clean.assistEnabled, true);
    assert.equal(clean.fireGuardEnabled, false);
});
