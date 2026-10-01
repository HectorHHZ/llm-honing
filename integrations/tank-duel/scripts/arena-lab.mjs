import { createHash, randomUUID } from "node:crypto";
import assert from "node:assert/strict";
import { mkdir, writeFile, appendFile, readFile } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { pathToFileURL } from "node:url";
import { adapter } from "../web/classic/adapter.js";
import { STRATEGIES } from "../web/classic/strategies.js";
import { AdaptiveCadence } from "../web/shared/cadence.js";
import { validDecision, validHealth } from "../web/shared/model-client.js";
import { blocked, distance } from "../web/classic/engine.js";
const quantile = (x, p) =>
    x.length
        ? Math.round(
              x.toSorted((a, b) => a - b)[Math.floor((x.length - 1) * p)],
          )
        : null;
const summarize = (x) => ({
    n: x.length,
    min_ms: x.length ? Math.round(Math.min(...x)) : null,
    p20_ms: quantile(x, 0.2),
    p50_ms: quantile(x, 0.5),
    p95_ms: quantile(x, 0.95),
    max_ms: x.length ? Math.round(Math.max(...x)) : null,
});
export async function endpoint(url, backend) {
    const base = `${url}/api/classic`,
        h = await fetch(
            `${base}/health?backend=${encodeURIComponent(backend)}`,
        ),
        health = await h.json();
    assert.equal(h.status, 200, JSON.stringify(health));
    assert.ok(validHealth(health, ["plans-v1"]));
    return {
        health,
        async decide(game, player, seat, cadence, tag) {
            const interval =
                seat.intervalMode === "auto" ? cadence.interval : seat.interval;
            const observed = adapter.observe(game, player, {
                ...seat,
                interval,
                rtt: cadence.prediction,
            });
            const body = {
                ...observed.request,
                id: tag,
                player,
                round_id: game.roundSerial,
                map_revision: game.maze.revision,
                configuration: seat,
            };
            const sent = performance.now(),
                response = await fetch(
                    `${base}/decision?backend=${encodeURIComponent(backend)}`,
                    {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(body),
                        signal: AbortSignal.timeout(12000),
                    },
                );
            const result = await response.json();
            assert.equal(response.status, 200, JSON.stringify(result));
            assert.ok(validDecision(result, body, health));
            assert.ok(adapter.validControls(result, observed));
            const rtt = performance.now() - sent;
            cadence.observe(rtt);
            return { observed, body, result, rtt, interval };
        },
    };
}
export async function calibrate({ url, backend, output, samples = 12 }) {
    const service = await endpoint(url, backend),
        game = new adapter.Game(200),
        cadence = [new AdaptiveCadence(), new AdaptiveCadence()],
        seat = {
            controllerMode: "plans",
            decisionVersion: "baseline",
            contextVersion: "full",
            intervalMode: "auto",
            interval: 100,
        };
    const serial = [],
        duel = [],
        responses = [];
    for (let i = 0; i < samples + 2; i++) {
        const d = await service.decide(
            game,
            0,
            seat,
            cadence[0],
            `cal-${backend}-${Date.now()}-${i}`,
        );
        responses.push({
            phase: i < 2 ? "warmup" : "serial",
            rtt_ms: d.rtt,
            request: d.body,
            response: d.result,
        });
        if (i >= 2) serial.push(d.rtt);
    }
    const began = performance.now();
    for (let i = 0; i < 4; i++) {
        const pair = await Promise.all(
            [0, 1].map((p) =>
                service.decide(
                    game,
                    p,
                    seat,
                    cadence[p],
                    `cal-duel-${backend}-${Date.now()}-${i}-${p}`,
                ),
            ),
        );
        pair.forEach((d) => {
            duel.push(d.rtt);
            responses.push({
                phase: "two-seats",
                rtt_ms: d.rtt,
                request: d.body,
                response: d.result,
            });
        });
    }
    const result = {
        backend,
        metadata: service.health.metadata,
        serial: summarize(serial),
        two_seats: summarize(duel),
        measured_duel_decisions_per_second: 8000 / (performance.now() - began),
        recommended_floor_ms: Math.max(50, quantile(serial, 0.2)),
        notes: "Observed throughput on a shared GPU; two planned decision heads; no context trimmed.",
    };
    await mkdir(output, { recursive: true });
    await writeFile(
        `${output}/calibration-${backend}.json`,
        JSON.stringify({ result, responses }, null, 2),
    );
    return result;
}
export async function round({
    service,
    backend,
    seed,
    versions,
    seconds = 35,
    output,
    tag,
    assistProfile = "light",
}) {
    const game = new adapter.Game(seed);
    adapter.initialize(game);
    game.timeLeft = seconds;
    const seats = versions.map((decisionVersion) => ({
        backend,
        decisionVersion,
        controllerMode: STRATEGIES[decisionVersion].mode,
        contextVersion: "full",
        intervalMode: "auto",
        interval: 100,
        assistEnabled: true,
        assistProfile,
        fireGuardEnabled: false,
    }));
    seats.forEach((s, i) => game.setPlayerSettings(i, s));
    game.start();
    const cadence = [new AdaptiveCadence(), new AdaptiveCadence()],
        pending = [null, null],
        next = [0, 0],
        counts = [0, 0],
        rtts = [[], []],
        records = [],
        ranges = [],
        assistFrames = [0, 0],
        errors = [];
    let previous = performance.now(),
        accumulator = 0,
        serial = 0,
        frames = 0,
        stale = 0,
        closeFrames = 0;
    const began = previous;
    async function decide(player) {
        try {
            const data = await service.decide(
                game,
                player,
                seats[player],
                cadence[player],
                `${tag}-${++serial}`,
            );
            counts[player]++;
            rtts[player].push(data.rtt);
            const record = {
                player,
                elapsed: game.elapsed,
                rtt_ms: data.rtt,
                cadence: cadence[player].stats,
                request: data.body,
                response: data.result,
                applied: false,
            };
            records.push(record);
            if (game.phase !== "running") return;
            if (data.rtt > 1600) {
                stale++;
                return;
            }
            record.execution = adapter.apply(game, player, data.result, {
                ...seats[player],
                interval: data.interval,
            });
            record.applied = true;
        } catch (e) {
            errors.push(String(e));
        } finally {
            pending[player] = null;
        }
    }
    while (
        game.phase === "running" &&
        !errors.length &&
        performance.now() - began < (seconds + 20) * 1000
    ) {
        const now = performance.now();
        accumulator += Math.min(0.12, (now - previous) / 1000);
        previous = now;
        while (accumulator >= 1 / 60 && game.phase === "running") {
            game.step(1 / 60);
            accumulator -= 1 / 60;
            frames++;
            const range = distance(...game.tanks);
            ranges.push(range);
            if (range < 150) closeFrames++;
            for (const t of game.tanks) {
                assert.ok(!blocked(game.maze, t.x, t.y), "Wall penetration");
                if (t.assistActive) assistFrames[t.id]++;
            }
            if (game.tanks.every((t) => t.alive))
                assert.ok(range >= 44 - 1e-8, "Tanks overlap");
        }
        for (let p = 0; p < 2; p++)
            if (game.phase === "running" && !pending[p] && now >= next[p]) {
                next[p] = now + cadence[p].interval;
                pending[p] = decide(p);
            }
        await delay(4);
    }
    await Promise.all(pending.filter(Boolean));
    const kills = game.events.filter((e) => e.type === "kill");
    const result = {
        tag,
        backend,
        seed,
        map: game.maze.name,
        map_revision: game.maze.revision,
        versions,
        assistProfile,
        winner: game.result,
        duration_s: game.elapsed,
        wall_time_s: (performance.now() - began) / 1000,
        complete: game.phase === "roundover",
        shots: game.shots,
        decisions: counts,
        latency: rtts.map(summarize),
        stale,
        mean_range_px:
            ranges.reduce((a, b) => a + b, 0) / Math.max(1, ranges.length),
        close_fraction: closeFrames / Math.max(1, frames),
        assist_fraction: assistFrames.map((x) => x / Math.max(1, frames)),
        self_hits: kills.filter((k) => k.self).map((k) => k.victim),
        enemy_hits: kills.filter((k) => !k.self).map((k) => k.player),
        errors,
    };
    await mkdir(output, { recursive: true });
    await writeFile(
        `${output}/${tag}.json`,
        JSON.stringify({
            result,
            metadata: service.health.metadata,
            seats,
            events: game.events,
            decisions: records,
        }),
    );
    return result;
}
export async function evaluate({
    url,
    backend,
    candidate,
    seeds,
    seconds,
    output,
}) {
    assert.ok(STRATEGIES[candidate]);
    const runId =
        new Date().toISOString().replaceAll(":", "-") +
        "-" +
        randomUUID().slice(0, 8);
    output = `${output}/${runId}`;
    await mkdir(output, { recursive: true });
    const files = [
        "web/classic/engine.js",
        "web/classic/planner.js",
        "web/classic/strategies.js",
        "web/shared/cadence.js",
        "scripts/arena-lab.mjs",
    ];
    const hashes = Object.fromEntries(
        await Promise.all(
            files.map(async (file) => [
                file,
                createHash("sha256")
                    .update(
                        await readFile(new URL("../" + file, import.meta.url)),
                    )
                    .digest("hex"),
            ]),
        ),
    );
    await writeFile(
        `${output}/manifest.json`,
        JSON.stringify(
            {
                runId,
                backend,
                candidate,
                seeds,
                seconds,
                created_at: new Date().toISOString(),
                hashes,
                strategies: STRATEGIES,
                hardware:
                    "Shared ds-serv8 GPU; real wall-clock model RTT; no mock or heuristic fallback",
            },
            null,
            2,
        ),
    );
    const service = await endpoint(url, backend),
        results = [];
    for (const seed of seeds)
        for (const swap of [false, true]) {
            const versions = swap
                    ? ["baseline", candidate]
                    : [candidate, "baseline"],
                tag = `${backend}-${candidate}-${seed}-${swap ? "red" : "blue"}`;
            const r = await round({
                service,
                backend,
                seed,
                versions,
                seconds,
                output,
                tag,
            });
            results.push(r);
            console.log(JSON.stringify({ event: "round", ...r }));
            await appendFile(`${output}/index.jsonl`, JSON.stringify(r) + "\n");
        }
    const valid = results.filter((r) => r.complete && !r.errors.length),
        wins = valid.filter(
            (r) => r.winner !== null && r.versions[r.winner] === candidate,
        ).length,
        losses = valid.filter(
            (r) => r.winner !== null && r.versions[r.winner] !== candidate,
        ).length,
        draws = valid.length - wins - losses;
    const result = {
        runId,
        output,
        backend,
        candidate,
        rounds: results.length,
        valid: valid.length,
        wins,
        losses,
        draws,
        score_rate: (wins + 0.5 * draws) / Math.max(1, valid.length),
        mean_range_px:
            valid.reduce((s, r) => s + r.mean_range_px, 0) /
            Math.max(1, valid.length),
        candidate_self_hits: valid.reduce(
            (s, r) =>
                s +
                r.self_hits.filter((p) => r.versions[p] === candidate).length,
            0,
        ),
        errors: results.flatMap((r) => r.errors),
        seeds,
        seconds,
        promotion: "pending review and held-out evaluation",
    };
    await writeFile(
        `${output}/summary-${backend}-${candidate}.json`,
        JSON.stringify(result, null, 2),
    );
    return result;
}
if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
    const args = process.argv.slice(2),
        arg = (key, fallback) =>
            args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
    const options = {
        url: arg("--url", "http://127.0.0.1:4181"),
        backend: arg("--backend", "nanojev"),
        output: arg("--output", "arena-results"),
        candidate: arg("--candidate", "multi-angle-v1"),
        seconds: Number(arg("--seconds", "35")),
        samples: Number(arg("--samples", "12")),
        seeds: arg("--seeds", "701,702,703").split(",").map(Number),
    };
    console.log(
        JSON.stringify(
            arg("--mode", "evaluate") === "calibrate"
                ? await calibrate(options)
                : await evaluate(options),
        ),
    );
}
