import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { adapter as tank } from "../web/game/adapter.js";
import { circleTouchesWall, TANK_RADIUS } from "../web/shared/collision.js";
import { seatProtocol } from "../web/shared/seat-config.js";
import { validHealth, validDecision } from "../web/shared/model-client.js";
export async function smokeModel({
    url,
    edition,
    seconds = 20,
    backend = "",
    controllerMode = "plans",
    players = null,
}) {
    assert.ok(true);
    assert.ok(seconds > 0 && seconds <= 120);
    const adapter = tank,
        base = url + `/api/${edition}`;
    assert.ok(
        players === null || (Array.isArray(players) && players.length === 2),
    );
    const seats = [0, 1].map((i) => ({
        backend,
        controllerMode,
        interval: 100,
        contextVersion: "full",
        assistEnabled: true,
        assistProfile: "light",
        fireGuardEnabled: false,
        ...players?.[i],
    }));
    const query = (id) =>
        seats[id].backend
            ? `?backend=${encodeURIComponent(seats[id].backend)}`
            : "";
    const health = await Promise.all(
        seats.map(async (s, i) => {
            const r = await fetch(base + "/health" + query(i), {
                signal: AbortSignal.timeout(5000),
            });
            assert.equal(r.status, 200);
            const h = await r.json();
            assert.ok(validHealth(h, [seatProtocol(edition, s)]));
            return h;
        }),
    );
    const game = new adapter.Game();
    adapter.initialize(game);
    seats.forEach((s, i) => game.setPlayerSettings(i, s));
    game.start();
    const seatRtts = [[], []],
        identities = [new Set(), new Set()];
    const counts = [0, 0],
        rtts = [],
        pending = [null, null],
        next = [0, 0],
        errors = [];
    let serial = 0,
        discarded = 0,
        frames = 0,
        maxShells = 0;
    async function decide(id) {
        const observed = adapter.observe(game, id, {
                ...seats[id],
                rtt: seatRtts[id].at(-1),
            }),
            round = game.roundSerial;
        const body = {
                ...observed.request,
                id: `smoke-${++serial}`,
                player: id,
                round_id: round,
            },
            started = performance.now();
        next[id] = started + seats[id].interval;
        try {
            const r = await fetch(base + "/decision" + query(id), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(body),
                signal: AbortSignal.timeout(5000),
            });
            assert.equal(r.status, 200, await r.clone().text());
            const data = await r.json(),
                rtt = performance.now() - started;
            assert.ok(validDecision(data, body, health[id]));
            assert.ok(adapter.validControls(data, observed));
            counts[id]++;
            rtts.push(rtt);
            seatRtts[id].push(rtt);
            identities[id].add(data.model);
            if (
                round !== game.roundSerial ||
                game.phase !== "running" ||
                rtt > 1600
            ) {
                discarded++;
                return;
            }
            adapter.apply(game, id, data, seats[id]);
        } catch (error) {
            errors.push(error.message);
        } finally {
            pending[id] = null;
        }
    }
    const started = performance.now();
    let previous = started,
        accumulator = 0;
    while (performance.now() - started < seconds * 1000 && !errors.length) {
        const now = performance.now();
        accumulator += Math.min(0.12, (now - previous) / 1000);
        previous = now;
        while (accumulator >= 1 / 60) {
            game.step(1 / 60);
            accumulator -= 1 / 60;
            frames++;
            for (const t of game.tanks)
                assert.ok(
                    game.maze.walls.every(
                        (w) => !circleTouchesWall(t.x, t.y, TANK_RADIUS, w),
                    ),
                    "Tank crossed a wall",
                );
            if (game.tanks.every((t) => t.alive))
                assert.ok(
                    Math.hypot(
                        game.tanks[0].x - game.tanks[1].x,
                        game.tanks[0].y - game.tanks[1].y,
                    ) >=
                        2 * TANK_RADIUS - 1e-8,
                    "Tanks overlap",
                );
            maxShells = Math.max(maxShells, game.bullets.length);
        }
        if (game.phase === "running")
            for (const id of [0, 1])
                if (game.tanks[id].alive && !pending[id] && now >= next[id])
                    pending[id] = decide(id);
        await delay(4);
    }
    await Promise.all(pending.filter(Boolean));
    assert.deepEqual(errors, []);
    assert.ok(
        counts.every((n) => n > 0),
        "Both seats must receive real decisions",
    );
    const sorted = rtts.toSorted((a, b) => a - b);
    return {
        edition,
        controller: controllerMode,
        model: health[0].metadata.model,
        players: seats.map((s, i) => ({
            ...s,
            model: health[i].metadata.model,
            observed_models: [...identities[i]],
            decisions: counts[i],
            median_rtt_ms: Math.round(
                seatRtts[i].toSorted((a, b) => a - b)[
                    Math.floor(seatRtts[i].length / 2)
                ],
            ),
        })),
        seconds,
        frames,
        decisions: counts,
        discarded,
        rounds: game.round,
        scores: game.scores,
        shots: game.shots,
        enemy_hits: game.events.filter((e) => e.type === "kill" && !e.self)
            .length,
        self_hits: game.events.filter((e) => e.type === "kill" && e.self)
            .length,
        max_shells: maxShells,
        median_rtt_ms: Math.round(sorted[Math.floor(sorted.length / 2)]),
        assistance_events: [0, 1].map(
            (i) =>
                game.events.filter((e) => e.type === "assist" && e.player === i)
                    .length,
        ),
        errors: 0,
    };
}
if (
    process.argv[1] &&
    path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
    const args = process.argv.slice(2);
    const read = (flag, fallback) => {
        const i = args.indexOf(flag);
        return i < 0 ? fallback : args[i + 1];
    };
    console.log(
        JSON.stringify(
            await smokeModel({
                url: read("--url", "http://127.0.0.1:4177"),
                edition: read("--edition", "tank-war"),
                seconds: Number(read("--seconds", "20")),
                backend: read("--backend", ""),
                controllerMode: read("--controller", "plans"),
                players: JSON.parse(read("--players-json", "null")),
            }),
            null,
            2,
        ),
    );
}
