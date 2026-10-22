#!/usr/bin/env node
// No game or model starts on import, --help, or without explicit --run.
import {
    appendFileSync,
    existsSync,
    readFileSync,
    writeFileSync,
} from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
    ACTION_CAP,
    CADENCE,
    HF_MODEL,
    HF_REVISION,
    RUNS_DIR,
    checkFiles,
    loadDependencies,
    ownSeatDetails,
    reserveOutput,
    snapshotSources,
    unpackReplay,
} from "./runtime.mjs";
import { enhancePacket, CONTEXT_VERSION } from "./harness/context.mjs";
import { WaitScheduler, SCHEDULER_VERSION } from "./harness/scheduler.mjs";

export const HARNESS_VERSION = "laya-v33-harness-v1";
const SEATS = ["red", "blue"];
const json = (file, value) =>
    writeFileSync(file, JSON.stringify(value, null, 2) + "\n");
const sha256 = (data) => createHash("sha256").update(data).digest("hex");
export const HELP = `Laya v33 harness v1 vs frozen Laya v33 (same CPU weights; no training)

  bash run.sh --help
  bash run.sh --check                      Check installed files; no model/game.
  bash run.sh --run --ticks 1500            Explicitly start a short test match.
  bash run.sh --run --ticks 27000 --seat blue --seed 1
  bash run.sh --report                     Summarize saved results; no game.

  --seat red|blue           Candidate harness seat (default red).
  --map mp03t4.map          Map filename.
  --seed 1                 uint32 game seed.
  --ticks 90               Maximum ticks, 15 = one game second.
  --decision-interval 30    Common base cadence, a positive multiple of 3.
  --scheduler adaptive     adaptive or fixed (H1 ablation).
  --advice on              on or off (H2 ablation).
  --max-wait-ticks 150      H1 heartbeat, multiple of 3, >= base cadence.
  --stall-ticks 150         H1 production plateau threshold, multiple of 3.
  --decision-log gzip      gzip or plain (full staged predictions).
  --out NAME               New folder under this experiment's runs/.

Both --key=value and --key value work. Both countries are Russians, normal view,
engine step = 3 ticks. Only the candidate receives H1/H2. Baseline uses frozen
IntentPolicy at the common base cadence. Default d30 is a CPU comparison profile;
use --decision-interval 3 to compare against the published v33 cadence.
One shared CPU worker; independent seat memories; both decide before time advances.
A tick-limit cutoff is unresolved, never a win. Nothing starts without --run.
`;

export function parseArgs(argv) {
    const options = {
        mode: "help",
        map: "mp03t4.map",
        seed: 1,
        ticks: 90,
        seat: "red",
        decisionInterval: 30,
        scheduler: "adaptive",
        advice: "on",
        maxWaitTicks: 150,
        stallTicks: 150,
        decisionLog: "gzip",
    };
    const names = {
        "--map": "map",
        "--seed": "seed",
        "--ticks": "ticks",
        "--seat": "seat",
        "--decision-interval": "decisionInterval",
        "--scheduler": "scheduler",
        "--advice": "advice",
        "--max-wait-ticks": "maxWaitTicks",
        "--stall-ticks": "stallTicks",
        "--decision-log": "decisionLog",
        "--out": "out",
    };
    const numeric = [
        "seed",
        "ticks",
        "decisionInterval",
        "maxWaitTicks",
        "stallTicks",
    ];
    let modeSeen = false,
        matchOption = false;
    for (let i = 0; i < argv.length; i++) {
        const token = argv[i],
            split = token.indexOf("="),
            key = split < 0 ? token : token.slice(0, split);
        if (["--run", "--help", "-h"].includes(key)) {
            if (split >= 0 || modeSeen)
                throw new Error("Choose exactly one of --help or --run");
            modeSeen = true;
            options.mode = key === "--run" ? "run" : "help";
            continue;
        }
        if (!(key in names)) throw new Error(`Unknown argument: ${token}`);
        const value = split < 0 ? argv[++i] : token.slice(split + 1),
            name = names[key];
        if (!value || value.startsWith("--"))
            throw new Error(`Missing value for ${key}`);
        if (numeric.includes(name)) {
            if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)))
                throw new Error(`${key} requires an integer`);
            options[name] = Number(value);
        } else options[name] = value;
        matchOption = true;
    }
    if (matchOption && options.mode !== "run")
        throw new Error("Match options require explicit --run");
    if (options.seed > 0xffffffff || options.ticks < 1)
        throw new Error("Invalid seed or tick limit");
    for (const name of ["decisionInterval", "maxWaitTicks", "stallTicks"])
        if (options[name] < 3 || options[name] % 3)
            throw new Error(`${name} must be a positive multiple of 3`);
    if (options.maxWaitTicks < options.decisionInterval)
        throw new Error("maxWaitTicks must be >= decisionInterval");
    for (const [name, allowed] of Object.entries({
        seat: SEATS,
        scheduler: ["adaptive", "fixed"],
        advice: ["on", "off"],
        decisionLog: ["plain", "gzip"],
    }))
        if (!allowed.includes(options[name]))
            throw new Error(`Invalid ${name}`);
    if (!/^[A-Za-z0-9_.-]+\.(?:map|mpr)$/i.test(options.map))
        throw new Error("map must be a filename");
    if (options.out && !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(options.out))
        throw new Error("out must be a single directory name");
    return options;
}

// The frozen candidate uses a d3 grid for event wakes. Its generic Wait label
// must not promise a 0.2-second reevaluation when H1 can wait up to a heartbeat.
// Other (production-specific) wait labels and every action binding stay intact.
export function relabelScheduledWait(packet) {
    const result = structuredClone(packet);
    for (const group of result.groups ?? []) {
        if (!/^Wait [\d.]+ seconds;/.test(group.waitLabel ?? "")) continue;
        const label =
            "Wait until the scheduler rechecks; existing orders continue";
        group.waitLabel = label;
        for (const choice of group.choices ?? [])
            if (choice.action === null) {
                choice.label = label;
                if (result.questions?.[group.id]?.criteria)
                    result.questions[group.id].criteria[choice.key] = label;
            }
    }
    return result;
}

export function verdict({ valid, ended, truncated, outcome }, candidateSeat) {
    if (!valid) return "invalid";
    if (truncated || !ended) return "unresolved";
    const other = candidateSeat === "red" ? "blue" : "red";
    if (outcome?.[candidateSeat] === 1 && outcome?.[other] === -1) return "win";
    if (outcome?.[candidateSeat] === -1 && outcome?.[other] === 1)
        return "loss";
    if (outcome?.[candidateSeat] === 0 && outcome?.[other] === 0) return "draw";
    return "unknown_terminal";
}

export function comparisonConfig(options) {
    return {
        harness: HARNESS_VERSION,
        model: HF_MODEL,
        revision: HF_REVISION,
        baseline: "frozen-v33-IntentPolicy",
        decisionInterval: options.decisionInterval,
        scheduler: options.scheduler,
        advice: options.advice,
        maxWaitTicks: options.maxWaitTicks,
        stallTicks: options.stallTicks,
        maxTicks: options.ticks,
        map: options.map,
        countries: { red: "Russians", blue: "Russians" },
        view: "normal",
        cadence: CADENCE,
        actionCap: ACTION_CAP,
        credits: 10000,
        unitCount: 0,
        shortGame: true,
        superWeapons: false,
        crates: false,
        mcvRepacks: true,
        buildOffAlly: false,
    };
}

export async function runMatch(options, deps, out, manifest) {
    const append = (file, row) =>
        appendFileSync(join(out, file), JSON.stringify(row) + "\n");
    const stats = Object.fromEntries(
        SEATS.map((seat) => [
            seat,
            {
                submitted: 0,
                accepted: 0,
                rejected: 0,
                filtered: 0,
                decisions: 0,
                inferenceCalls: 0,
                inferenceMs: 0,
                rootWaits: 0,
                adviceNotes: 0,
            },
        ]),
    );
    const scheduling = {
        skippedBaseSlots: 0,
        eventWakes: 0,
        heartbeats: 0,
        reasons: {},
    };
    let rawClient, client, meta, engine, step, decisionLog;
    const policies = {},
        scheduler =
            options.scheduler === "adaptive"
                ? new WaitScheduler(options)
                : null;
    const started = performance.now();
    for (const file of ["actions.jsonl", "filtered.jsonl", "scheduler.jsonl"])
        writeFileSync(join(out, file), "", { flag: "wx" });
    try {
        // The pinned model is stateless between requests. This avoids loading
        // two ~2.7 GiB CPU models in the 5.8 GiB VM. Policy memories stay separate.
        rawClient = new deps.CpuLayaClient();
        client = new deps.PermutationClient(rawClient);
        manifest.runtime = await client.ready;
        json(join(out, "manifest.json"), manifest);
        console.log(
            JSON.stringify({
                event: "model_ready",
                device: manifest.runtime.device,
                out,
            }),
        );
        meta = await deps.loadMetadata(options.map);
        const extent = Math.max(meta.bounds.width, meta.bounds.height);
        if (extent > 256)
            throw new Error(
                "Map exceeds the lossless 256-cell observation limit",
            );
        engine = await deps.Ra2Engine.start({ gameDir: deps.GAME_DIR });
        const players = SEATS.map((name, start) => ({
            name,
            country: "Russians",
            start,
        }));
        step = await engine.create({
            map: options.map,
            seed: options.seed,
            title: manifest.title,
            max_ticks: options.ticks,
            record_replay: true,
            players,
            options: {
                credits: 10000,
                unit_count: 0,
                short_game: true,
                super_weapons: false,
                crates: false,
                mcv_repacks: true,
                build_off_ally: false,
            },
            observation: { resolution: extent, mask_offscreen_enemies: false },
            view: "normal",
            privileged: false,
        });
        if (step.view !== "normal")
            throw new Error("Engine did not return normal-view observations");
        const created = step;
        for (const name of SEATS)
            policies[name] = new deps.IntentPolicy({
                name,
                country: "Russians",
                metadata: meta,
                catalog: created.catalog,
                seats: ownSeatDetails(created.players, name),
                seed: (options.seed + Number(name === "blue")) >>> 0,
                strictNormal: true,
                decisionInterval:
                    name === options.seat && scheduler
                        ? CADENCE
                        : options.decisionInterval,
            });
        manifest.engine = engine.hello;
        manifest.game = { players, catalogDigest: created.catalog.digest };
        json(join(out, "manifest.json"), manifest);
        if (options.decisionLog === "gzip")
            decisionLog = new deps.CompressedDecisionLog(
                join(out, "decisions.jsonl.gz"),
            );
        else writeFileSync(join(out, "decisions.jsonl"), "", { flag: "wx" });
        const appendDecision = (row) =>
            decisionLog
                ? decisionLog.append(row)
                : append("decisions.jsonl", row);
        console.log(
            JSON.stringify({
                event: "started",
                out,
                candidateSeat: options.seat,
                maxTicks: options.ticks,
                decisionInterval: options.decisionInterval,
                scheduler: options.scheduler,
                advice: options.advice,
            }),
        );
        while (!step.done) {
            const at = step.tick;
            if (step.view !== "normal")
                throw new Error("Policy received wrong view");
            if (!Number.isSafeInteger(at) || at < 0 || at >= options.ticks)
                throw new Error("Invalid or unbounded engine tick");
            // Refresh both from their normal observation at the SAME tick, then
            // serialize inference. Engine.step runs only after both have decided.
            for (const seat of SEATS)
                policies[seat].refresh(step.observations[seat], at);
            const actions = { red: [], blue: [] },
                selectedSeats = new Set();
            for (const seat of SEATS) {
                const policy = policies[seat],
                    candidate = seat === options.seat;
                let packet = policy.packet(),
                    gate = {
                        infer: at % options.decisionInterval === 0,
                        reason: "fixed_cadence",
                    };
                if (candidate && scheduler) {
                    gate = scheduler.evaluate({
                        tick: at,
                        observation: policy.obs,
                        frame: policy.frame,
                        packet,
                    });
                    scheduling.reasons[gate.reason] =
                        (scheduling.reasons[gate.reason] ?? 0) + 1;
                    if (
                        !gate.infer &&
                        at % options.decisionInterval === 0 &&
                        gate.reason === "unchanged_busy_wait"
                    )
                        scheduling.skippedBaseSlots++;
                    if (gate.reason === "event_wake") scheduling.eventWakes++;
                    if (gate.reason === "wait_heartbeat")
                        scheduling.heartbeats++;
                    append("scheduler.jsonl", { tick: at, seat, ...gate });
                }
                if (!gate.infer || !packet.groups?.length) continue;
                let notes = [];
                if (candidate && scheduler)
                    packet = relabelScheduledWait(packet);
                if (candidate && options.advice === "on") {
                    const enhanced = enhancePacket(packet, {
                        observation: policy.obs,
                        frame: policy.frame,
                        tick: at,
                    });
                    packet = enhanced.packet;
                    notes = enhanced.notes;
                }
                // The release select() consults current. Keep its exact bindings
                // synchronized with the final packet; neither H1 nor H2 adds actions.
                policy.current = packet;
                const before = performance.now();
                const prediction = await deps.predictStaged(packet, client);
                stats[seat].inferenceMs += performance.now() - before;
                if (
                    !Number.isInteger(prediction.inferenceCalls) ||
                    prediction.inferenceCalls < 1
                )
                    throw new Error(
                        "A decision must contain at least one real forward call",
                    );
                stats[seat].decisions++;
                stats[seat].inferenceCalls += prediction.inferenceCalls;
                stats[seat].adviceNotes += notes.length;
                const selected = policy.select(
                    prediction.resolvedAnswers,
                    ACTION_CAP,
                );
                if (
                    !Array.isArray(selected.actions) ||
                    selected.actions.length > ACTION_CAP ||
                    !Array.isArray(selected.dropped)
                )
                    throw new Error("Invalid selected actions or cap exceeded");
                if (
                    prediction.selectionPath?.length === 1 &&
                    prediction.selectionPath[0].stageIndex === 1 &&
                    prediction.selectionPath[0].dimension === "intent" &&
                    prediction.selectionPath[0].cancelled === true
                )
                    stats[seat].rootWaits++;
                if (candidate && scheduler)
                    scheduler.afterDecision({ tick: at, prediction, selected });
                selectedSeats.add(seat);
                actions[seat] = selected.actions;
                stats[seat].filtered += selected.dropped.length;
                if (selected.dropped.length)
                    append("filtered.jsonl", {
                        tick: at,
                        seat,
                        dropped: selected.dropped,
                    });
                await appendDecision({
                    tick: at,
                    seat,
                    role: candidate ? "candidate" : "baseline",
                    decisionMode: "staged",
                    candidateMode: "intent",
                    schedule: gate,
                    harnessNotes: notes,
                    availability: packet.availability,
                    actorExclusions: packet.actorExclusions,
                    normalUnits: policy.frame.rows,
                    ownVectors: policy.obs.vectors,
                    policyMemory: {
                        kind: "normal-public-search-v1",
                        searchVisits: [...(policy.searchVisits ?? new Map())],
                    },
                    contextParts: packet.parts,
                    leafQuestions: packet.questions,
                    candidates: packet.groups,
                    omitted: packet.omitted,
                    ...prediction,
                    selected: selected.selected,
                    dropped: selected.dropped,
                });
            }
            step = await engine.step(actions, {
                ticks: Math.min(CADENCE, options.ticks - at),
                view: "normal",
                privileged: false,
            });
            let missing = false;
            for (const seat of SEATS) {
                for (const [index, action] of actions[seat].entries()) {
                    const receipt = step.results?.[seat]?.[index];
                    if (
                        receipt == null ||
                        typeof receipt.accepted !== "boolean"
                    )
                        missing = true;
                    const result = receipt ?? {
                        accepted: false,
                        reason: "missing_receipt",
                    };
                    stats[seat].submitted++;
                    stats[seat][
                        result.accepted === true ? "accepted" : "rejected"
                    ]++;
                    append("actions.jsonl", { tick: at, seat, action, result });
                }
                // Pending actions persist in the frozen policy after receipts.
                // Never feed empty receipts for a skipped turn (stale pending).
                if (selectedSeats.has(seat))
                    policies[seat].receipts(at, step.results?.[seat] ?? []);
                if (seat === options.seat && scheduler && actions[seat].length)
                    scheduler.receipts({
                        tick: at,
                        results: actions[seat].map(
                            (_, i) => step.results?.[seat]?.[i],
                        ),
                    });
            }
            if (missing)
                throw new Error("Engine omitted a valid action receipt");
            if (!Number.isSafeInteger(step.tick) || step.tick <= at)
                throw new Error(
                    `Engine stopped advancing: ${at} -> ${step.tick}`,
                );
            if (step.tick > options.ticks)
                throw new Error("Engine advanced beyond requested tick limit");
            if (step.tick % 150 === 0 || step.done) {
                const progress = {
                    event: "progress",
                    tick: step.tick,
                    done: step.done,
                    elapsedSeconds: (performance.now() - started) / 1000,
                    stats,
                    scheduling,
                };
                json(join(out, "progress.json"), progress);
                console.log(JSON.stringify(progress));
            }
        }
        json(
            join(out, "decision-log.json"),
            decisionLog ? await decisionLog.close() : { format: "plain-jsonl" },
        );
        const replay = unpackReplay(await engine.saveReplay());
        writeFileSync(join(out, "match.rplx"), replay.rplx, { flag: "wx" });
        json(join(out, "replay.json"), replay.metadata);
        const result = {
            status: replay.valid ? "completed" : "invalid",
            valid: replay.valid,
            harness: HARNESS_VERSION,
            model: HF_MODEL,
            hfRevision: HF_REVISION,
            candidateSeat: options.seat,
            comparisonId: manifest.comparisonId,
            configuration: comparisonConfig(options),
            map: options.map,
            seed: options.seed,
            tick: step.tick,
            ended: step.ended,
            truncated: step.truncated,
            outcome: step.outcome,
            hash: await engine.hash(),
            stats,
            scheduling,
            policyMetrics: Object.fromEntries(
                SEATS.map((seat) => [seat, policies[seat].metrics]),
            ),
            finalOwnVectors: Object.fromEntries(
                SEATS.map((seat) => [seat, step.observations?.[seat]?.vectors]),
            ),
            replay: replay.metadata,
            replaySha256: sha256(replay.rplx),
            elapsedSeconds: (performance.now() - started) / 1000,
        };
        result.candidateResult = verdict(result, options.seat);
        json(
            join(out, "events.json"),
            Object.fromEntries(
                SEATS.map((seat) => [
                    seat,
                    policies[seat].memory?.events ?? [],
                ]),
            ),
        );
        json(join(out, "result.json"), result);
        if (!replay.valid)
            throw new Error(
                "Replay is partial or has lost_data; recording retained but run marked invalid",
            );
        console.log(
            JSON.stringify({
                event: "saved",
                out,
                replay: join(out, "match.rplx"),
                candidateResult: result.candidateResult,
                tick: step.tick,
            }),
        );
        return result;
    } catch (error) {
        const failure = {
            status: "invalid",
            valid: false,
            candidateResult: "invalid",
            candidateSeat: options.seat,
            comparisonId: manifest.comparisonId,
            configuration: comparisonConfig(options),
            seed: options.seed,
            error: error.message,
            tick: step?.tick,
            stats,
            scheduling,
        };
        json(join(out, "error.json"), failure);
        if (!existsSync(join(out, "result.json")))
            json(join(out, "result.json"), failure);
        throw error;
    } finally {
        for (const [name, cleanup] of [
            ["decision log", () => decisionLog?.close()],
            ["CPU worker", () => rawClient?.close()],
            ["engine", () => engine?.shutdown()],
            ["metadata", () => meta?.dispose()],
        ]) {
            try {
                await cleanup();
            } catch (error) {
                console.error(`${name} cleanup failed: ${error.message}`);
            }
        }
    }
}

export async function main(
    argv = process.argv.slice(2),
    loader = loadDependencies,
) {
    const options = parseArgs(argv);
    if (options.mode === "help") {
        console.log(HELP);
        return;
    }
    const deps = await loader(),
        environment = checkFiles(deps);
    const stamp = new Date().toISOString().replace(/[^0-9TZ]/g, "");
    const out = reserveOutput(
        options.out ??
            `h1-${options.seat}-vs-v33-${options.map.replace(
                /\.(map|mpr)$/i,
                "",
            )}-s${options.seed}-d${
                options.decisionInterval
            }-${stamp}-${randomUUID().slice(0, 8)}`,
        RUNS_DIR,
    );
    const sources = snapshotSources(out, deps);
    const fingerprint = sources
        .filter((f) => /\.(mjs|js|py|json)$/.test(f.snapshot))
        .map(({ snapshot, sha256 }) => ({ snapshot, sha256 }))
        .sort((a, b) => a.snapshot.localeCompare(b.snapshot));
    const manifest = {
        experiment: HARNESS_VERSION,
        createdAt: new Date().toISOString(),
        title: `H1-${basename(out)}`
            .replace(/[^A-Za-z0-9_-]/g, "_")
            .slice(0, 120),
        model: { huggingFace: HF_MODEL, hfRevision: HF_REVISION },
        configuration: comparisonConfig(options),
        candidateSeat: options.seat,
        seed: options.seed,
        comparisonId: sha256(
            JSON.stringify({
                config: comparisonConfig(options),
                sources: fingerprint,
            }),
        ),
        baseline: `Frozen v33 IntentPolicy, CPU d${options.decisionInterval}`,
        optimizations: {
            H1: {
                enabled: options.scheduler === "adaptive",
                version: SCHEDULER_VERSION,
            },
            H2: { enabled: options.advice === "on", version: CONTEXT_VERSION },
        },
        protocol: deps.STAGED_PROTOCOL,
        maxDecisionStages: deps.MAX_STAGES,
        device: "cpu",
        sharedModelWorkers: 1,
        separateSeatMemory: true,
        sequentialInferenceAtSameTick: true,
        observerDataExcluded: true,
        training: false,
        weightsModified: false,
        environment,
        sources,
    };
    json(join(out, "manifest.json"), manifest);
    return runMatch(options, deps, out, manifest);
}

if (
    process.argv[1] &&
    pathToFileURL(resolve(process.argv[1])).href === import.meta.url
)
    main().catch((error) => {
        console.error(error.stack ?? error.message);
        process.exitCode = 1;
    });
