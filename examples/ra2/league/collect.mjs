import assert from "node:assert/strict";
import {
    writeFileSync,
    appendFileSync,
    mkdirSync,
    existsSync,
    readFileSync,
} from "node:fs";
import { resolve } from "node:path";
import { createHash } from "node:crypto";
import { Ra2Engine, GAME_DIR } from "../integration/dependencies.mjs";
import { loadMetadata, ROOT } from "../integration/metadata.mjs";
import { IntentPolicy } from "../integration/laya/intent-policy.mjs";
import { SupalosaAdapter } from "../integration/adapter.mjs";
import {
    predictStaged,
    verifyStagedTrace,
} from "../integration/laya/staged.mjs";
import { validateAction } from "../integration/laya/policy.mjs";
import { rowsOf } from "../integration/laya/context.mjs";
import { MacroTeacher } from "./teacher.mjs";
const args = Object.fromEntries(
    process.argv.slice(2).map((x) => x.replace(/^--/, "").split("=")),
);
const out = resolve(ROOT, args.out ?? "runs/teacher"),
    map = args.map ?? "mp03t4.map",
    seed = Number(args.seed ?? 100),
    seat = args.seat ?? "red",
    other = seat === "red" ? "blue" : "red",
    ticks = Number(args.ticks ?? 18000);
if (
    !["red", "blue"].includes(seat) ||
    !Number.isInteger(seed) ||
    !Number.isInteger(ticks) ||
    ticks <= 0
)
    throw new Error("Invalid match settings");
if (existsSync(out)) throw new Error("Output already exists");
mkdirSync(out, { recursive: true });
const write = (name, obj) =>
    writeFileSync(out + "/" + name, JSON.stringify(obj, null, 2) + "\n");
const append = (name, obj) =>
    appendFileSync(out + "/" + name, JSON.stringify(obj) + "\n");
const sourceSha256 = Object.fromEntries(
    [
        "league/collect.mjs",
        "league/teacher.mjs",
        "integration/laya/intent-policy.mjs",
        "integration/laya/staged.mjs",
        "integration/laya/catalog-policy.mjs",
        "integration/laya/policy.mjs",
        "integration/laya/context.mjs",
        "integration/adapter.mjs",
    ].map((f) => [
        f,
        createHash("sha256")
            .update(readFileSync(ROOT + f))
            .digest("hex"),
    ]),
);
let engine,
    meta,
    step,
    samples = 0,
    decisions = 0;
const teacher = new MacroTeacher(),
    functions = {};
try {
    meta = await loadMetadata(map);
    engine = await Ra2Engine.start({ gameDir: GAME_DIR });
    step = await engine.create({
        map,
        seed,
        max_ticks: ticks,
        record_replay: true,
        players: [
            { name: "red", country: "Russians", start: 0 },
            { name: "blue", country: "Russians", start: 1 },
        ],
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
    const initial = step,
        conf = (name) => ({
            name,
            country: "Russians",
            metadata: meta,
            catalog: initial.catalog,
            seats: initial.players.map((p) =>
                p.name === name
                    ? { ...p }
                    : { name: p.name, player_id: p.player_id },
            ),
            seed: seed + (name === "red" ? 0 : 1),
            strictNormal: true,
            decisionInterval: 3,
        });
    const policy = new IntentPolicy(conf(seat)),
        baseline = new SupalosaAdapter(conf(other));
    baseline.refresh(step.observations[other], 0);
    baseline.start();
    const manifest = {
        kind: "teacher-demonstration",
        notModelEvaluation: true,
        split: "train",
        teacher: "normal-macro-v2",
        map,
        seed,
        seat,
        maxTicks: ticks,
        cadence: 3,
        studentMaxCommands: 1,
        baselineMaxCommands: 16,
        sourceSha256,
        engine: engine.hello.build,
    };
    write("manifest.json", manifest);
    for (const file of Object.keys(sourceSha256)) {
        const target = out + "/source/" + file;
        mkdirSync(target.slice(0, target.lastIndexOf("/")), {
            recursive: true,
        });
        writeFileSync(target, readFileSync(ROOT + file));
    }
    while (!step.done) {
        policy.refresh(step.observations[seat], step.tick);
        baseline.refresh(step.observations[other], step.tick);
        const packet = policy.packet(),
            leaf = teacher.choose(policy, packet);
        assert(leaf, "No wait/leaf choice");
        const records = [];
        const prediction = await predictStaged(packet, {
            predict: async (req) => {
                const answers = Object.fromEntries(
                    req.groups.map((g) => {
                        const c = g.choices.find((c) =>
                            leaf.action
                                ? c.leafKeys?.includes(leaf.key)
                                : c.kind === "cancel",
                        );
                        assert(c, "Teacher leaf missing from staged partition");
                        return [g.id, { choice: c.key }];
                    }),
                );
                records.push({
                    kind: "teacher-label",
                    split: "train",
                    map,
                    seed,
                    seat,
                    tick: step.tick,
                    stage: req.stageIndex,
                    parts: req.parts,
                    questions: req.questions,
                    answers,
                    chosenLeaf: leaf.key,
                    teacher: "normal-macro-v2",
                });
                return { answers, fixture: true, latencyMs: 0 };
            },
        });
        verifyStagedTrace(packet, prediction, assert);
        const sel = policy.select(prediction.resolvedAnswers, 16);
        assert(sel.actions.length <= 1);
        const obs = step.observations[other],
            rows = rowsOf(obs, initial.catalog),
            frame = {
                rows,
                own: rows.filter((u) => u.alliance === 1),
                visible: rows.filter(
                    (u) =>
                        u.alliance === 4 && u.display_type === 1 && !u.is_blip,
                ),
            };
        const raw = baseline.decide(),
            opponent = raw
                .filter((a) => !validateAction(a, obs, frame, initial.catalog))
                .slice(0, 16),
            at = step.tick;
        if (sel.actions.length || at < 600 || at % 30 === 0)
            for (const record of records) {
                append("samples.jsonl", record);
                samples++;
            }
        const actions = { [seat]: sel.actions, [other]: opponent };
        step = await engine.step(actions, {
            ticks: Math.min(3, ticks - at),
            view: "normal",
            privileged: true,
        });
        policy.receipts(at, step.results[seat]);
        teacher.observe(
            sel.actions.filter((_, i) => step.results[seat]?.[i]?.accepted),
            at,
        );
        decisions++;
        for (const a of sel.actions)
            functions[a.function] = (functions[a.function] ?? 0) + 1;
        append("trajectory.jsonl", {
            tick: at,
            teacherAction: sel.actions,
            baselineAction: opponent,
            receipts: step.results,
        });
        if (step.tick % 1500 === 0 || step.done) {
            write("progress.json", {
                tick: step.tick,
                done: step.done,
                samples,
                functions,
            });
            console.log(
                JSON.stringify({ tick: step.tick, samples, functions }),
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
        samples,
        decisions,
        functions,
        finalPlayers: step.privileged.players,
    };
    writeFileSync(out + "/match.rplx", await engine.saveReplay());
    write("result.json", result);
} catch (e) {
    write("error.json", { error: e.message, tick: step?.tick });
    throw e;
} finally {
    await engine?.shutdown();
    meta?.dispose();
}
