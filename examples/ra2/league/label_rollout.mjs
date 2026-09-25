// DAgger annotation uses only the learner's recorded normal observation.
// The labeler cannot advance the engine or influence the recorded match.
import assert from "node:assert/strict";
import {
    readFileSync,
    writeFileSync,
    appendFileSync,
    mkdirSync,
    existsSync,
    createReadStream,
} from "node:fs";
import { createInterface } from "node:readline";
import { resolve, dirname } from "node:path";
import { createHash } from "node:crypto";
import { loadMetadata, ROOT } from "../integration/metadata.mjs";
import {
    predictStaged,
    verifyStagedTrace,
} from "../integration/laya/staged.mjs";
import { MacroTeacher } from "./teacher.mjs";
const args = Object.fromEntries(
    process.argv.slice(2).map((x) => x.replace(/^--/, "").split("=")),
);
assert(args.match && args.out, "Use --match=... --out=...");
const input = resolve(args.match),
    out = resolve(args.out);
const result = JSON.parse(readFileSync(input + "/result.json"));
const protocol = JSON.parse(readFileSync(ROOT + "league/protocol.json"));
assert(
    protocol.train.maps.includes(result.map) &&
        protocol.train.seeds.includes(result.seed),
    "Refusing non-training rollout",
);
assert(
    result.normalView &&
        result.observerDataExcluded &&
        result.runtime.weightsSha256,
    "Expected learned normal-view rollout",
);
assert(!existsSync(out), "Output exists");
mkdirSync(out, { recursive: true });
const hash = (data) => createHash("sha256").update(data).digest("hex");
const digestFile = async (path) => {
    const h = createHash("sha256");
    for await (const block of createReadStream(path)) h.update(block);
    return h.digest("hex");
};
const sourceSha256 = {};
for (const rel of [
    "league/label_rollout.mjs",
    "league/teacher.mjs",
    "integration/laya/staged.mjs",
]) {
    const bytes = readFileSync(ROOT + rel);
    sourceSha256[rel] = hash(bytes);
    mkdirSync(dirname(out + "/source/" + rel), { recursive: true });
    writeFileSync(out + "/source/" + rel, bytes);
}
const manifest = {
    kind: "dagger-labels",
    notModelEvaluation: true,
    split: "train",
    teacher: "normal-macro-v2",
    map: result.map,
    seed: result.seed,
    seat: result.layaSeat,
    sourceSha256,
    learnerCheckpointSha256: result.runtime.weightsSha256,
    rollout: {
        path: input,
        resultSha256: hash(readFileSync(input + "/result.json")),
        decisionsSha256: await digestFile(input + "/decisions.jsonl"),
    },
};
writeFileSync(out + "/manifest.json", JSON.stringify(manifest, null, 2) + "\n");
const actions = new Map();
for (const line of readFileSync(input + "/actions.jsonl", "utf8")
    .trim()
    .split("\n")) {
    if (!line) continue;
    const row = JSON.parse(line);
    if (row.seat !== result.layaSeat || !row.result?.accepted) continue;
    if (!actions.has(row.tick)) actions.set(row.tick, []);
    actions.get(row.tick).push(row.action);
}
const meta = await loadMetadata(result.map),
    teacher = new MacroTeacher();
let decisions = 0,
    samples = 0,
    disagreements = 0;
try {
    for await (const line of createInterface({
        input: createReadStream(input + "/decisions.jsonl"),
        crlfDelay: Infinity,
    })) {
        const d = JSON.parse(line),
            own = d.normalUnits.filter((u) => u.alliance === 1);
        const policy = {
            tick: d.tick,
            meta,
            catalog: { functions: d.availability },
            frame: {
                own,
                visible: d.normalUnits.filter(
                    (u) =>
                        u.alliance === 4 && u.display_type === 1 && !u.is_blip,
                ),
            },
            obs: {
                vectors: {
                    ...d.ownVectors,
                    worker_count: own.filter(
                        (u) => meta.techno.get(u.name)?.harvester,
                    ).length,
                    is_low_power:
                        d.ownVectors.power_used > d.ownVectors.power_cap,
                },
            },
            facade: {
                res: Math.min(
                    256,
                    Math.max(meta.bounds.width, meta.bounds.height),
                ),
            },
        };
        const packet = {
            tick: d.tick,
            groups: d.candidates,
            parts: d.contextParts,
            questions: d.leafQuestions,
            availability: d.availability,
        };
        const leaf = teacher.choose(policy, packet);
        assert(leaf);
        const actual = actions.get(d.tick) ?? [],
            desired = leaf.action ? [leaf.action] : [];
        const differs = JSON.stringify(actual) !== JSON.stringify(desired);
        disagreements += Number(differs);
        decisions++;
        // Include transitions and periodic corrections, with the learner's actual history.
        if (d.tick < 600 || actual.length || d.tick % 30 === 0) {
            const prediction = await predictStaged(packet, {
                predict: async (req) => {
                    const answers = Object.fromEntries(
                        req.groups.map((g) => {
                            const c = g.choices.find((c) =>
                                leaf.action
                                    ? c.leafKeys?.includes(leaf.key)
                                    : c.kind === "cancel",
                            );
                            assert(c, "Oracle leaf lost from staged partition");
                            return [g.id, { choice: c.key }];
                        }),
                    );
                    appendFileSync(
                        out + "/samples.jsonl",
                        JSON.stringify({
                            kind: "dagger-teacher-label",
                            split: "train",
                            map: result.map,
                            seed: result.seed,
                            seat: result.layaSeat,
                            tick: d.tick,
                            stage: req.stageIndex,
                            parts: req.parts,
                            questions: req.questions,
                            answers,
                            chosenLeaf: leaf.key,
                            actualLearnerActions: actual,
                            disagrees: differs,
                            teacher: manifest.teacher,
                        }) + "\n",
                    );
                    samples++;
                    return { answers, fixture: true, latencyMs: 0 };
                },
            });
            verifyStagedTrace(packet, prediction, assert);
        }
        // Only executed learner actions affect cooldowns. Advice is never executed.
        teacher.observe(actual, d.tick);
    }
    writeFileSync(
        out + "/result.json",
        JSON.stringify(
            { ...manifest, decisions, samples, disagreements },
            null,
            2,
        ) + "\n",
    );
    console.log(JSON.stringify({ out, decisions, samples, disagreements }));
} finally {
    meta.dispose();
}
