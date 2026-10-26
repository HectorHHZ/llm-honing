import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
    playMatch,
    loadHarness,
    parseController,
    CONTROLLER_PATH,
    QUESTIONS_PATH,
} from "../scripts/match.mjs";
import { DEFAULT_QUESTIONS } from "../model-bridge.mjs";

const SECONDS = 6;
const lines = async (file) =>
    (await readFile(file, "utf8"))
        .split("\n")
        .filter(Boolean)
        .map((l) => JSON.parse(l));

// A decision server that gives option i of n the probability (i + 1) / sum, so the last option is the most
// probable and every option can be sampled. It echoes the state as Honing's server does.
async function stubServer(t) {
    const requests = [];
    const server = createServer(async (req, res) => {
        res.setHeader("Content-Type", "application/json");
        if (req.url === "/v1/health")
            return res.end(
                JSON.stringify({
                    ready: true,
                    metadata: { model: "stub", family: "laya" },
                }),
            );
        let body = "";
        for await (const chunk of req) body += chunk;
        const r = JSON.parse(body);
        requests.push(r);
        const answers = {};
        for (const [head, q] of Object.entries(r.questions)) {
            const keys = Object.keys(q.criteria),
                total = (keys.length * (keys.length + 1)) / 2;
            answers[head] = {
                type: "choice",
                choice: keys.at(-1),
                probabilities: Object.fromEntries(
                    keys.map((k, i) => [k, (i + 1) / total]),
                ),
            };
        }
        res.end(
            JSON.stringify({
                id: r.id,
                answers,
                stateText: r.parts.essential.join("\n"),
                trimmed: [],
                tokenAudit: {},
            }),
        );
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    t.after(() => server.close());
    return { url: `http://127.0.0.1:${server.address().port}`, requests };
}

async function harnessDir(files) {
    const dir = await mkdtemp(join(tmpdir(), "tank-harness-"));
    const index = [];
    for (const [path, kind, content] of files) {
        await mkdir(join(dir, path, ".."), { recursive: true });
        await writeFile(join(dir, path), content);
        index.push({ path, kind });
    }
    await writeFile(join(dir, "index.json"), JSON.stringify(index));
    return dir;
}

const outDir = async () =>
    join(await mkdtemp(join(tmpdir(), "tank-match-")), "out");

test("a teacher round is recorded and replays exactly", async () => {
    const runs = [];
    for (let i = 0; i < 2; i++) {
        const out = await outDir();
        const result = await playMatch({
            seed: 2001,
            seat: "blue",
            out,
            teacher: true,
            seconds: SECONDS,
        });
        runs.push({
            result,
            decisions: await lines(join(out, "decisions.jsonl")),
            out,
        });
    }
    assert.deepEqual(runs[0].result, runs[1].result);
    assert.deepEqual(runs[0].decisions, runs[1].decisions);
    const manifest = JSON.parse(
        await readFile(join(runs[0].out, "manifest.json"), "utf8"),
    );
    assert.equal(manifest.kind, "teacher-demonstration");
    assert.deepEqual(
        manifest.harness.map((h) => h.role),
        ["controller", "questions"],
    );
    const { result, decisions } = runs[0];
    assert.ok(["hit", "draw", "timeout"].includes(result.reason));
    assert.equal(result.ended, result.reason !== "timeout");
    // Planned steering asks for the motion, then the weapon for that motion.
    assert.deepEqual(Object.keys(decisions[0].questions), ["motion"]);
    assert.deepEqual(Object.keys(decisions[1].questions), ["weapon"]);
    for (const row of decisions) {
        for (const [head, q] of Object.entries(row.questions)) {
            assert.ok(Object.hasOwn(q.criteria, row.chosen[head]));
            assert.equal(row.answers[head].probabilities[row.chosen[head]], 1);
        }
        assert.equal(
            row.questions.motion?.instructions ?? DEFAULT_QUESTIONS.motion,
            DEFAULT_QUESTIONS.motion,
        );
    }
});

test("every harness file reaches the model: question text, controller, and operating notes", async (t) => {
    const { url, requests } = await stubServer(t);
    const questions = {
        ...DEFAULT_QUESTIONS,
        weapon: "Hold unless the shot is certain.",
    };
    const dir = await harnessDir([
        [
            CONTROLLER_PATH,
            "settings",
            JSON.stringify({ assistProfile: "strong", strategy: "baseline" }),
        ],
        [QUESTIONS_PATH, "instructions", JSON.stringify(questions)],
        ["notes/aim.md", "instructions", "Prefer ricochet shots from cover.\n"],
    ]);
    const out = await outDir();
    const result = await playMatch({
        seed: 2002,
        seat: "red",
        out,
        decisionServer: url,
        harnessDir: dir,
        seconds: SECONDS,
    });
    assert.ok(result.decisions.red > 0);
    const manifest = JSON.parse(
        await readFile(join(out, "manifest.json"), "utf8"),
    );
    assert.equal(manifest.kind, "evaluation");
    assert.equal(manifest.agent_seat.assistProfile, "strong");
    assert.deepEqual(
        manifest.harness.map((h) => [h.path, h.role]),
        [
            [CONTROLLER_PATH, "controller"],
            [QUESTIONS_PATH, "questions"],
            ["notes/aim.md", "note"],
        ],
    );
    assert.ok(
        requests.every((r) =>
            r.parts.essential[0].startsWith(
                "OPERATING NOTES (notes/aim.md):\nPrefer ricochet",
            ),
        ),
    );
    const weapon = requests.find((r) => r.questions.weapon);
    assert.equal(
        weapon.questions.weapon.instructions,
        "Hold unless the shot is certain.",
    );
    const decisions = await lines(join(out, "decisions.jsonl"));
    assert.equal(decisions.length, requests.length);
    // Unsampled, the executed motion is the most probable admissible one (the stub favors later options).
    for (const row of decisions.filter((r) => r.questions.motion)) {
        const admissible = row.constraints.admissible_motion_ids,
            order = Object.keys(row.questions.motion.criteria);
        const best = admissible.reduce((a, b) =>
            order.indexOf(b) > order.indexOf(a) ? b : a,
        );
        assert.equal(row.chosen.motion, best);
    }
});

test("a sampled round draws choices from the probabilities, reproducibly for one seed", async (t) => {
    const { url } = await stubServer(t);
    const play = async (sampleSeed) => {
        const out = await outDir();
        await playMatch({
            seed: 2003,
            seat: "blue",
            out,
            decisionServer: url,
            seconds: SECONDS,
            sampleSeed,
        });
        return { rows: await lines(join(out, "decisions.jsonl")), out };
    };
    const a = await play(7),
        b = await play(7);
    assert.deepEqual(
        a.rows.map((r) => r.chosen),
        b.rows.map((r) => r.chosen),
    );
    const manifest = JSON.parse(
        await readFile(join(a.out, "manifest.json"), "utf8"),
    );
    assert.equal(manifest.kind, "rollout");
    assert.equal(manifest.sample_seed, 7);
    const motions = a.rows.filter((r) => r.questions.motion);
    assert.ok(motions.some((r) => r.constraints.policy_motion_changed));
    for (const row of motions) {
        assert.ok(row.answers.motion.probabilities[row.chosen.motion] > 0);
        assert.equal(
            row.constraints.policy_motion_changed,
            !row.constraints.admissible_motion_ids.includes(row.chosen.motion),
        );
    }

    // The stub's argmax is always "hold" for the weapon; sampling also draws "fire".
    const weapons = new Set(
        a.rows.filter((r) => r.questions.weapon).map((r) => r.chosen.weapon),
    );
    assert.deepEqual(weapons, new Set(["fire", "hold"]));
});

test("a harness the match cannot read is refused before the round, and marked as the harness's fault", async () => {
    const cases = [
        [
            [
                CONTROLLER_PATH,
                "settings",
                JSON.stringify({ decisionInterval: 50 }),
            ],
            /unknown setting decisionInterval/,
        ],
        [[CONTROLLER_PATH, "settings", "{"], /not JSON/],
        [
            [
                CONTROLLER_PATH,
                "settings",
                JSON.stringify({
                    strategy: { id: "x", description: "y", aimBot: 1 },
                }),
            ],
            /Unknown strategy parameter/,
        ],
        [
            [
                CONTROLLER_PATH,
                "settings",
                JSON.stringify({
                    controllerMode: "plans",
                    contextVersion: "arena-v1",
                }),
            ],
            /requires contextVersion full/,
        ],
        [
            [QUESTIONS_PATH, "instructions", JSON.stringify({ motion: "x" })],
            /missing/,
        ],
        [
            [
                QUESTIONS_PATH,
                "instructions",
                JSON.stringify({ ...DEFAULT_QUESTIONS, extra: "x" }),
            ],
            /unknown keys extra/,
        ],
    ];
    for (const [file, message] of cases) {
        const dir = await harnessDir([file]);
        await assert.rejects(
            playMatch({
                seed: 2001,
                seat: "blue",
                out: await outDir(),
                teacher: true,
                harnessDir: dir,
                seconds: SECONDS,
            }),
            (error) => message.test(error.message) && error.harness === true,
        );
    }
    await assert.rejects(
        playMatch({
            seed: 2001,
            seat: "blue",
            out: await outDir(),
            teacher: true,
            harnessDir: await harnessDir([
                [
                    CONTROLLER_PATH,
                    "settings",
                    JSON.stringify({ controllerMode: "raw" }),
                ],
            ]),
        }),
        (error) =>
            /planned steering only/.test(error.message) &&
            error.harness === true,
    );
});

test("the controller defaults reproduce the browser's planned steering, and the output directory must be new", async () => {
    const seat = parseController("{}");
    assert.equal(seat.controllerMode, "plans");
    assert.equal(seat.strategy.id, "baseline");
    const defaults = await loadHarness();
    assert.deepEqual(defaults.questions, DEFAULT_QUESTIONS);
    assert.equal(defaults.notes, "");
    const out = await outDir();
    await mkdir(out);
    await assert.rejects(
        playMatch({ seed: 1, seat: "blue", out, teacher: true }),
        /already exists/,
    );
    await assert.rejects(
        playMatch({
            seed: 1,
            seat: "green",
            out: await outDir(),
            teacher: true,
        }),
        /--seat/,
    );
});

test("sampled motion uses native probabilities before the safety projection", async () => {
    const { decide } = await import("../model-bridge.mjs");
    const { adapter } = await import("../web/game/adapter.js");
    const game = new adapter.Game(2001);
    adapter.initialize(game);
    const body = {
        ...adapter.observe(game, 0, { controllerMode: "plans", interval: 200 })
            .request,
        id: "sample-projection",
        player: 0,
        round_id: 1,
    };
    const ids = body.motions.map((m) => m.id);
    body.admissible_motion_ids = [ids[1], ids[2]];
    const native = Object.fromEntries(
        ids.map((id, i) => [id, i === 0 ? 0.8 : i < 3 ? 0.1 : 0]),
    );
    const reply = await decide(
        {
            model: "fixture",
            agentIds: ["blue", "red"],
            contextLimit: null,
            ask: async (r) => ({
                id: r.id,
                stateText: r.parts.essential.join("\n"),
                trimmed: [],
                answers: Object.fromEntries(
                    Object.keys(r.questions).map((head) => [
                        head,
                        head === "motion"
                            ? { choice: ids[0], probabilities: native }
                            : {
                                  choice: "hold",
                                  probabilities: { fire: 0, hold: 1 },
                              },
                    ]),
                ),
            }),
        },
        body,
        {
            choose: (head, probabilities, candidates) => {
                if (head === "motion") {
                    assert.deepEqual(
                        candidates,
                        ids,
                        "sample all native options, not a conditioned subset",
                    );
                    assert.deepEqual(probabilities, native);
                    return ids[0];
                }
                return "hold";
            },
        },
    );
    assert.equal(reply.policy_motion_id, ids[0]);
    assert.equal(reply.motion_id, ids[1]);
    assert.ok(body.admissible_motion_ids.includes(reply.motion_id));
    assert.equal(reply.constraints.policy_motion_changed, true);
    assert.deepEqual(reply.probabilities.motion, native);
    assert.match(
        reply.weapon_context,
        new RegExp(`SELECTED MOTION ${ids[1]}:`),
    );
});
