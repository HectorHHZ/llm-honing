import {
    writeFileSync,
    appendFileSync,
    mkdirSync,
    readFileSync,
    existsSync,
} from "node:fs";
import { resolve, dirname } from "node:path";
import { createHash } from "node:crypto";
import { loadMetadata, ROOT } from "../metadata.mjs";
import { SupalosaAdapter } from "../adapter.mjs";
import { IntentPolicy } from "./intent-policy.mjs";
import { CatalogPolicy, PARAMETER_LIMIT } from "./catalog-policy.mjs";
import { LayaPolicy, validateAction } from "./policy.mjs";
import { rowsOf } from "./context.mjs";
import { LayaClient } from "./client.mjs";
import { predictStaged, STAGED_PROTOCOL, MAX_STAGES } from "./staged.mjs";
import { Ra2Engine, GAME_DIR } from "../dependencies.mjs";
const args = Object.fromEntries(
    process.argv.slice(2).map((a) => a.replace(/^--/, "").split("=")),
);
const map = args.map ?? "mp03t4.map",
    seed = Number(args.seed ?? 1),
    ticks = Number(args.ticks ?? 18000),
    cadence = Number(args.cadence ?? 3),
    cap = 16,
    layaSeat = args.seat ?? "red";
if (
    !["red", "blue"].includes(layaSeat) ||
    !Number.isInteger(cadence) ||
    cadence < 3 ||
    cadence % 3
)
    throw new Error("seat red/blue; cadence a positive multiple of 3");
const decisionMode = args["decision-mode"] ?? "staged";
if (!["staged", "flat"].includes(decisionMode))
    throw new Error("decision-mode must be staged or flat");
const candidateMode =
    args["candidate-mode"] ?? (decisionMode === "staged" ? "intent" : "legacy");
if (
    !["catalog", "legacy", "intent"].includes(candidateMode) ||
    (decisionMode === "flat" && candidateMode !== "legacy")
)
    throw new Error(
        "catalog candidates require staged decisions; use --candidate-mode=legacy for the previous generator",
    );
const interval = Number(args["decision-interval"] ?? 3);
if (
    !Number.isInteger(ticks) ||
    ticks <= 0 ||
    !Number.isInteger(seed) ||
    !Number.isInteger(interval) ||
    interval < cadence ||
    interval % cadence
)
    throw new Error(
        "ticks must be positive; seed integer; decision interval must be a positive multiple of cadence",
    );
const out = resolve(ROOT, args.out ?? `runs/match-${layaSeat}-seed${seed}`);
if (existsSync(out))
    throw new Error(
        `Output already exists; choose a new run directory: ${out}`,
    );
mkdirSync(out, { recursive: true });
let engine, meta, client, step, laya, baseline;
const start = performance.now();
const frames = [];
let inferenceCalls = 0;
const commandStats = {
    red: { submitted: 0, accepted: 0, rejected: 0, filtered: 0 },
    blue: { submitted: 0, accepted: 0, rejected: 0, filtered: 0 },
};
const append = (file, obj) =>
    appendFileSync(out + "/" + file, JSON.stringify(obj) + "\n");
const policySeats = (players, name) =>
    players.map((p) =>
        p.name === name ? { ...p } : { name: p.name, player_id: p.player_id },
    );
const capture = () =>
    frames.push({
        tick: step.tick,
        players: step.privileged.players,
        units: Array.from(step.privileged.units.data.data),
    });
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
    const created = step,
        other = layaSeat === "red" ? "blue" : "red";
    const conf = (name) => ({
        name,
        country: "Russians",
        metadata: meta,
        catalog: created.catalog,
        seats: policySeats(created.players, name),
        seed: seed + (name === "red" ? 0 : 1),
        strictNormal: true,
    });
    laya = new (
        candidateMode === "intent"
            ? IntentPolicy
            : candidateMode === "catalog"
            ? CatalogPolicy
            : LayaPolicy
    )({
        ...conf(layaSeat),
        decisionInterval: Number(args["decision-interval"] ?? 3),
    });
    baseline = new SupalosaAdapter(conf(other));
    baseline.refresh(step.observations[other], 0);
    baseline.start();
    client = new LayaClient({
        gpu: args.gpu ?? "0",
        model:
            args.model ??
            (candidateMode === "intent" ? "typed-decisions" : "english"),
    });
    const runtime = await client.ready;
    const sourceFiles = [
        "README.md",
        "pins.json",
        "integration/adapter.mjs",
        "integration/metadata.mjs",
        "integration/dependencies.mjs",
        "integration/paths.mjs",
        ...[
            "context.mjs",
            "policy.mjs",
            "catalog-policy.mjs",
            "intent-policy.mjs",
            "staged.mjs",
            "worker.py",
            "token_budget.py",
            "client.mjs",
            "run.mjs",
        ].map((f) => "integration/laya/" + f),
    ];
    for (const file of sourceFiles) {
        const target = out + "/source/" + file;
        mkdirSync(dirname(target), { recursive: true });
        writeFileSync(target, readFileSync(ROOT + file));
    }
    const implementationSha256 = Object.fromEntries(
        sourceFiles.map((file) => [
            file,
            createHash("sha256")
                .update(readFileSync(ROOT + file))
                .digest("hex"),
        ]),
    );
    const manifest = {
        implementationSha256,
        protocol:
            decisionMode === "staged"
                ? STAGED_PROTOCOL
                : "laya-normal-choice-v1.2",
        decisionMode,
        candidateMode,
        decisionInterval:
            candidateMode === "intent" ? laya.decisionInterval : cadence,
        parameterCandidateLimit:
            candidateMode !== "legacy" ? PARAMETER_LIMIT : null,
        maxDecisionStages: decisionMode === "staged" ? MAX_STAGES : 1,
        map,
        seed,
        layaSeat,
        baselineSeat: other,
        cadenceTicks: cadence,
        maxTicks: ticks,
        maxActionsPerSeat: cap,
        seats,
        runtime,
        engine: engine.hello.build,
        normalView: true,
        maskOffscreenEnemies: false,
        strictCurrentEnemyTargets: true,
        observerDataExcluded: true,
        designSha256: createHash("sha256")
            .update(readFileSync(ROOT + "README.md"))
            .digest("hex"),
        startedAt: new Date().toISOString(),
    };
    writeFileSync(out + "/manifest.json", JSON.stringify(manifest, null, 2));
    console.log(JSON.stringify({ event: "started", out, ...manifest }));
    capture();
    while (!step.done) {
        if (step.view && step.view !== "normal")
            throw new Error("Policy received wrong view");
        laya.refresh(step.observations[layaSeat], step.tick);
        baseline.refresh(step.observations[other], step.tick);
        const packet = laya.packet();
        // Both policy decisions use the same pre-step state. Baseline choices never enter model input.
        const rawBaseline = baseline.decide();
        const ob = step.observations[other],
            rows = rowsOf(ob, created.catalog),
            bf = {
                rows,
                own: rows.filter((u) => u.alliance === 1),
                visible: rows.filter(
                    (u) =>
                        u.alliance === 4 && u.display_type === 1 && !u.is_blip,
                ),
            };
        const baselineActions = [],
            baselineDropped = [];
        for (const action of rawBaseline) {
            const reason =
                validateAction(action, ob, bf, created.catalog) ||
                (baselineActions.length >= cap ? "action_cap" : null);
            if (reason) baselineDropped.push({ action, reason });
            else baselineActions.push(action);
        }
        // Follow-up calls only refine this frozen packet. No engine.step or new
        // opponent observation occurs until every selected action has parameters.
        const prediction =
            decisionMode === "staged"
                ? await predictStaged(packet, client)
                : packet.groups.length
                ? await client.predict(packet)
                : {
                      answers: {},
                      stateText: packet.parts.essential.join("\n"),
                      latencyMs: 0,
                      tokenAudit: {},
                      trimmed: [],
                  };
        inferenceCalls +=
            decisionMode === "staged"
                ? prediction.inferenceCalls
                : Number(packet.groups.length > 0);
        const selection = laya.select(
                decisionMode === "staged"
                    ? prediction.resolvedAnswers
                    : prediction.answers,
                cap,
            ),
            at = step.tick;
        const queryDefinition =
            decisionMode === "staged"
                ? { leafQuestions: packet.questions }
                : { questions: packet.questions };
        append("decisions.jsonl", {
            tick: at,
            seat: layaSeat,
            decisionMode,
            candidateMode,
            availability: packet.availability,
            actorExclusions: packet.actorExclusions,
            normalUnits: laya.frame.rows,
            ownVectors: {
                minerals: laya.obs.vectors.minerals,
                power_cap: laya.obs.vectors.power_cap,
                power_used: laya.obs.vectors.power_used,
                production: laya.obs.vectors.production,
                units_truncated: laya.obs.vectors.units_truncated,
            },
            contextParts: packet.parts,
            ...queryDefinition,
            candidates: packet.groups,
            omitted: packet.omitted,
            ...prediction,
            selected: selection.selected,
            dropped: selection.dropped,
        });
        const actions = {
            [layaSeat]: selection.actions,
            [other]: baselineActions,
        };
        commandStats[layaSeat].filtered += selection.dropped.length;
        commandStats[other].filtered += baselineDropped.length;
        step = await engine.step(actions, {
            ticks: Math.min(cadence, ticks - at),
            view: "normal",
            privileged: true,
        });
        laya.receipts(at, step.results[layaSeat]);
        for (const [seat, list] of Object.entries(actions))
            for (let i = 0; i < list.length; i++) {
                const result = step.results[seat]?.[i];
                commandStats[seat].submitted++;
                commandStats[seat][
                    result?.accepted ? "accepted" : "rejected"
                ]++;
                append("actions.jsonl", {
                    tick: at,
                    seat,
                    action: list[i],
                    result,
                });
            }
        if (baselineDropped.length)
            append("filtered.jsonl", {
                tick: at,
                seat: other,
                dropped: baselineDropped,
            });
        if (step.tick % 150 === 0 || step.done) capture();
        if (step.tick % 1500 === 0 || step.done) {
            writeFileSync(
                out + "/progress.json",
                JSON.stringify({
                    tick: step.tick,
                    done: step.done,
                    stats: commandStats,
                }),
            );
            console.log(
                JSON.stringify({
                    tick: step.tick,
                    done: step.done,
                    wallSeconds: Math.round((performance.now() - start) / 1000),
                    stats: commandStats,
                }),
            );
        }
    }
    const result = {
        ...manifest,
        tick: step.tick,
        ended: step.ended,
        truncated: step.truncated,
        outcome: step.outcome,
        hash: await engine.hash(),
        stats: commandStats,
        layaMetrics: laya.metrics,
        inferenceCalls,
        elapsedSeconds: (performance.now() - start) / 1000,
        finalPlayers: step.privileged.players,
    };
    writeFileSync(out + "/match.rplx", await engine.saveReplay());
    writeFileSync(
        out + "/viewer.json",
        JSON.stringify({
            map,
            bounds: meta.bounds,
            terrain: meta.tiles.map((t) => [t.rx, t.ry, t.landType, t.z ?? 0]),
            columns: created.privileged.units.columns,
            unitTypes: created.catalog.unit_types,
            frames,
        }),
    );
    writeFileSync(
        out + "/events.json",
        JSON.stringify(laya.memory.events, null, 2),
    );
    writeFileSync(out + "/result.json", JSON.stringify(result, null, 2));
    console.log(
        JSON.stringify({
            event: "saved",
            out,
            tick: step.tick,
            outcome: step.outcome,
            stats: commandStats,
        }),
    );
} catch (e) {
    writeFileSync(
        out + "/error.json",
        JSON.stringify({ error: e.message, tick: step?.tick }, null, 2),
    );
    throw e;
} finally {
    client?.close();
    if (engine) await engine.shutdown();
    meta?.dispose();
}
