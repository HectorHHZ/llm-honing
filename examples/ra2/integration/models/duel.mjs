import {
    writeFileSync,
    appendFileSync,
    mkdirSync,
    readFileSync,
    existsSync,
    readdirSync,
    symlinkSync,
} from "node:fs";
import { resolve, dirname, extname } from "node:path";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";
import { loadMetadata, ROOT } from "../metadata.mjs";
import { Ra2Engine, GAME_DIR } from "../dependencies.mjs";
import { IntentPolicy } from "../laya/intent-policy.mjs";
import { PARAMETER_LIMIT } from "../laya/catalog-policy.mjs";
import { CompressedDecisionLog } from "../laya/decision-log.mjs";
import { STAGED_PROTOCOL, MAX_STAGES } from "../laya/staged.mjs";
import { createDecisionClient, decisionClientOptions } from "./factory.mjs";
import { DUEL_SEATS, normalSeats, resolveDuelStep } from "./duel-core.mjs";

const args = Object.fromEntries(
    process.argv.slice(2).map((arg) => {
        const match = /^--([^=]+)=(.*)$/.exec(arg);
        if (!match) throw new Error("Use --key=value arguments");
        return [match[1], match[2]];
    }),
);
const map = args.map ?? "mp03t4.map",
    seed = Number(args.seed ?? 104),
    ticks = Number(args.ticks ?? 18000),
    cadence = Number(args.cadence ?? 3);
assert(Number.isInteger(seed) && Number.isInteger(ticks) && ticks > 0);
assert(Number.isInteger(cadence) && cadence >= 3 && cadence % 3 === 0);
assert(args.out, "An explicit unused --out directory is required");
const out = resolve(ROOT, args.out);
assert(!existsSync(out), "Duel output already exists");
const options = Object.fromEntries(
    DUEL_SEATS.map((seat) => {
        const seatArgs = Object.fromEntries(
            Object.entries(args)
                .filter(([key]) => key.startsWith(seat + "-"))
                .map(([key, value]) => [key.slice(seat.length + 1), value]),
        );
        assert(
            seatArgs.family && seatArgs.gpu,
            `Explicit family and GPU required for ${seat}`,
        );
        assert(
            /^(0|[1-9]\d*)$/.test(seatArgs.gpu),
            "Use an explicitly reserved GPU index",
        );
        const config = decisionClientOptions(seatArgs, "intent");
        if (config.family === "laya") {
            assert(
                seatArgs["checkpoint-manifest"],
                "Use an explicit Laya checkpoint manifest or upstream",
            );
            config.checkpointManifest =
                seatArgs["checkpoint-manifest"] === "upstream"
                    ? null
                    : resolve(seatArgs["checkpoint-manifest"]);
            if (config.checkpointManifest)
                assert(existsSync(config.checkpointManifest));
        } else
            assert(
                !seatArgs["checkpoint-manifest"],
                "Native family requires its prepared manifest",
            );
        return [seat, config];
    }),
);
assert.notEqual(
    options.red.gpu,
    options.blue.gpu,
    "Duel workers require separate reserved GPUs",
);
mkdirSync(out, { recursive: true });
const save = (file, data) =>
    writeFileSync(out + "/" + file, JSON.stringify(data, null, 2) + "\n");
const append = (file, data) =>
    appendFileSync(out + "/" + file, JSON.stringify(data) + "\n");
let engine, meta, step;
const clients = {},
    logs = {},
    policies = {},
    runtimes = {},
    frames = [],
    manifests = {};
const calls = { red: 0, blue: 0 },
    stats = Object.fromEntries(
        DUEL_SEATS.map((seat) => [
            seat,
            { submitted: 0, accepted: 0, rejected: 0, filtered: 0 },
        ]),
    );
const start = performance.now();
try {
    meta = await loadMetadata(map);
    engine = await Ra2Engine.start({ gameDir: GAME_DIR });
    const seats = [
        { name: "red", country: "Russians", start: 0 },
        { name: "blue", country: "Russians", start: 1 },
    ];
    step = await engine.create({
        map,
        seed,
        max_ticks: ticks,
        record_replay: true,
        players: seats,
        options: {
            credits: 10000,
            unit_count: 0,
            short_game: true,
            super_weapons: false,
            crates: false,
            mcv_repacks: true,
        },
        observation: {
            resolution: Math.min(
                256,
                Math.max(meta.bounds.width, meta.bounds.height),
            ),
            mask_offscreen_enemies: false,
        },
        view: "normal",
        privileged: true,
    });
    const created = step;
    const snapshot = (relative) => {
        for (const item of readdirSync(ROOT + relative, {
            withFileTypes: true,
        })) {
            const file = relative + "/" + item.name;
            if (item.isDirectory() && item.name !== "__pycache__")
                snapshot(file);
            else if (item.isFile() && [".mjs", ".py"].includes(extname(file)))
                sourceFiles.push(file);
        }
    };
    const sourceFiles = [
        "README.md",
        "pins.json",
        "league/model-sources.json",
        "league/verify_native_encoding.py",
        "league/verify_duel.py",
        "league/verify_replay.py",
        "league/storage.py",
        "league/gpu_group.py",
        "league/campaign.py",
    ];
    snapshot("integration");
    const implementationSha256 = {};
    for (const file of sourceFiles.sort()) {
        const content = readFileSync(ROOT + file),
            target = out + "/source/" + file;
        mkdirSync(dirname(target), { recursive: true });
        writeFileSync(target, content);
        implementationSha256[file] = createHash("sha256")
            .update(content)
            .digest("hex");
    }
    for (const seat of DUEL_SEATS) {
        mkdirSync(out + "/" + seat);
        symlinkSync("../source", out + "/" + seat + "/source");
        symlinkSync("../actions.jsonl", out + "/" + seat + "/actions.jsonl");
        policies[seat] = new IntentPolicy({
            name: seat,
            country: "Russians",
            metadata: meta,
            catalog: created.catalog,
            seats: normalSeats(created.players, seat),
            seed: seed + Number(seat === "blue"),
            strictNormal: true,
            decisionInterval: cadence,
        });
        logs[seat] = new CompressedDecisionLog(
            out + "/" + seat + "/decisions.jsonl.gz",
        );
        clients[seat] = createDecisionClient(options[seat]);
    }
    const runtimeResults = await Promise.all(
        DUEL_SEATS.map((seat) => clients[seat].ready),
    );
    DUEL_SEATS.forEach((seat, i) => (runtimes[seat] = runtimeResults[i]));
    const manifest = {
        kind: "model-duel-v1",
        implementationSha256,
        protocol: STAGED_PROTOCOL,
        decisionMode: "staged",
        candidateMode: "intent",
        decisionInterval: cadence,
        decisionLogFormat: "gzip",
        parameterCandidateLimit: PARAMETER_LIMIT,
        maxDecisionStages: MAX_STAGES,
        map,
        seed,
        cadenceTicks: cadence,
        maxTicks: ticks,
        maxActionsPerSeat: 1,
        seats,
        modelFamilies: Object.fromEntries(
            DUEL_SEATS.map((seat) => [seat, options[seat].family]),
        ),
        runtimes,
        engine: engine.hello.build,
        normalView: true,
        maskOffscreenEnemies: false,
        strictCurrentEnemyTargets: true,
        observerDataExcluded: true,
        samePreStepObservation: true,
        opponentChoicesExcluded: true,
        noFallbackPolicy: true,
        startedAt: new Date().toISOString(),
    };
    save("manifest.json", manifest);
    writeFileSync(out + "/actions.jsonl", "");
    for (const seat of DUEL_SEATS) {
        const {
            runtimes: unusedRuntimes,
            modelFamilies: unusedFamilies,
            ...common
        } = manifest;
        manifests[seat] = {
            ...common,
            kind: "model-duel-seat-v1",
            policySeat: seat,
            opponentSeat: seat === "red" ? "blue" : "red",
            opponentKind: "decision-model",
            modelFamily: options[seat].family,
            runtime: runtimes[seat],
            inferenceCallUnit:
                options[seat].family === "laya"
                    ? "native_model_forward"
                    : "native_choice_request",
        };
        save(seat + "/manifest.json", manifests[seat]);
    }
    console.log(
        JSON.stringify({
            event: "started",
            out,
            modelFamilies: manifest.modelFamilies,
            runtimes,
        }),
    );
    const capture = () =>
        frames.push({
            tick: step.tick,
            players: step.privileged.players,
            units: Array.from(step.privileged.units.data.data),
        });
    capture();
    while (!step.done) {
        const at = step.tick;
        const resolved = await resolveDuelStep({
            tick: at,
            observations: step.observations,
            view: step.view ?? "normal",
            policies,
            clients,
        });
        const actions = {};
        for (const seat of DUEL_SEATS) {
            await logs[seat].append(resolved[seat].record);
            calls[seat] += resolved[seat].record.inferenceCalls;
            stats[seat].filtered += resolved[seat].record.dropped.length;
            actions[seat] = resolved[seat].actions;
        }
        step = await engine.step(actions, {
            ticks: Math.min(cadence, ticks - at),
            view: "normal",
            privileged: true,
        });
        for (const seat of DUEL_SEATS) {
            policies[seat].receipts(at, step.results[seat]);
            actions[seat].forEach((action, i) => {
                const result = step.results[seat]?.[i];
                stats[seat].submitted++;
                stats[seat][result?.accepted ? "accepted" : "rejected"]++;
                append("actions.jsonl", { tick: at, seat, action, result });
            });
        }
        if (step.tick % 150 === 0 || step.done) capture();
        if (step.tick % 1500 === 0 || step.done) {
            const progress = {
                tick: step.tick,
                done: step.done,
                stats,
                inferenceCalls: calls,
                elapsedSeconds: (performance.now() - start) / 1000,
            };
            save("progress.json", progress);
            console.log(JSON.stringify(progress));
        }
    }
    for (const seat of DUEL_SEATS)
        save(seat + "/decision-log.json", await logs[seat].close());
    const ending = {
        tick: step.tick,
        ended: step.ended,
        truncated: step.truncated,
        outcome: step.outcome,
        hash: await engine.hash(),
        stats,
        elapsedSeconds: (performance.now() - start) / 1000,
        finalPlayers: step.privileged.players,
    };
    writeFileSync(out + "/match.rplx", await engine.saveReplay());
    save("viewer.json", {
        map,
        bounds: meta.bounds,
        terrain: meta.tiles.map((t) => [t.rx, t.ry, t.landType, t.z ?? 0]),
        columns: created.privileged.units.columns,
        unitTypes: created.catalog.unit_types,
        frames,
    });
    for (const seat of DUEL_SEATS) {
        save(seat + "/events.json", policies[seat].memory.events);
        save(seat + "/result.json", {
            ...manifests[seat],
            ...ending,
            policyMetrics: policies[seat].metrics,
            inferenceCalls: calls[seat],
        });
    }
    save("result.json", {
        ...manifest,
        ...ending,
        inferenceCalls: calls,
        policyMetrics: Object.fromEntries(
            DUEL_SEATS.map((seat) => [seat, policies[seat].metrics]),
        ),
    });
    console.log(JSON.stringify({ event: "saved", out, ...ending }));
} catch (error) {
    save("error.json", { error: error.message, tick: step?.tick });
    throw error;
} finally {
    for (const client of Object.values(clients)) client.close();
    for (const log of Object.values(logs)) {
        try {
            await log.close();
        } catch (error) {
            console.error("Decision log failed:", error.message);
        }
    }
    if (engine) await engine.shutdown();
    meta?.dispose();
}
