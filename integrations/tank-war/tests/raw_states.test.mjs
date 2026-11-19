import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { once } from "node:events";
import {
    cp,
    mkdir,
    mkdtemp,
    readFile,
    readdir,
    writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";
import {
    playMatch,
    stateHash,
    OUTPUTS,
    PLATFORM,
    CONTROLLER_PATH,
    QUESTIONS_PATH,
} from "../scripts/match.mjs";
import {
    render,
    roundTrip,
    replay,
    label,
    readRun,
    LABELS_FILE,
} from "../scripts/raw-states.mjs";
import { restoreGame, snapshotGame } from "../web/game/snapshot.js";
import { adapter } from "../web/game/adapter.js";

// A sampled round against a stub model whose option i of n has probability (i + 1) / sum: seed 2002, blue, 8 s,
// shot after 22 decisions. The fixture holds every file of its run directory in one gzipped JSON object.
// Regenerate it after a change to what a round records: TANK_RECORD_FIXTURE=1 node --test tests/raw_states.test.mjs
// Rendering and labelling read the logged state, so the fixture serves them on any platform. Replay plays the
// round again, which is bit exact only on the platform that recorded it, so the replay tests record their own.
const FIXTURE = new URL(
    "./fixtures/rollout-s2002-blue.json.gz",
    import.meta.url,
);
const lines = async (file) =>
    (await readFile(file, "utf8"))
        .split("\n")
        .filter(Boolean)
        .map((l) => JSON.parse(l));
const tmp = (name) => mkdtemp(join(tmpdir(), `tank-${name}-`));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// A stub decision server. With `branches`, its health reports maxBranches and it answers a request's branches
// in the same reply, as Honing's decision server does.
async function stubServer({ branches = 0 } = {}) {
    const reply = (r) => {
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
        return {
            id: r.id,
            answers,
            stateText: r.parts.essential.join("\n"),
            trimmed: [],
            tokenAudit: {},
        };
    };
    const server = createServer(async (req, res) => {
        res.setHeader("Content-Type", "application/json");
        if (req.url === "/v1/health")
            return res.end(
                JSON.stringify({
                    ready: true,
                    metadata: { model: "stub", family: "laya" },
                    maxBranches: branches,
                }),
            );
        let body = "";
        for await (const chunk of req) body += chunk;
        const r = JSON.parse(body);
        res.end(
            JSON.stringify({
                ...reply(r),
                ...(r.branches && {
                    branches: Object.fromEntries(
                        Object.entries(r.branches).map(([k, b]) => [
                            k,
                            reply(b),
                        ]),
                    ),
                }),
            }),
        );
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    return server;
}

if (process.env.TANK_RECORD_FIXTURE) {
    const server = await stubServer(),
        out = join(await tmp("fixture"), "run");
    await playMatch({
        seed: 2002,
        seat: "blue",
        out,
        decisionServer: `http://127.0.0.1:${server.address().port}`,
        seconds: 8,
        sampleSeed: 7,
    });
    server.close();
    const files = {};
    for (const name of (await readdir(out)).sort())
        files[name] = await readFile(join(out, name), "utf8");
    await writeFile(FIXTURE, gzipSync(JSON.stringify(files), { level: 9 }));
}

// The fixture's run directory, unpacked byte for byte into a new temporary directory.
async function fixture() {
    const files = JSON.parse(gunzipSync(await readFile(FIXTURE))),
        run = await tmp("fixture");
    for (const [name, content] of Object.entries(files))
        await writeFile(join(run, name), content);
    return run;
}

// The fixture's round recorded again on this machine, once per test run, and copied for each test.
let recording;
async function recorded() {
    recording ??= (async () => {
        const server = await stubServer(),
            out = join(await tmp("recorded"), "run");
        try {
            await playMatch({
                seed: 2002,
                seat: "blue",
                out,
                decisionServer: `http://127.0.0.1:${server.address().port}`,
                seconds: 8,
                sampleSeed: 7,
            });
        } finally {
            server.close();
        }
        return out;
    })();
    const run = join(await tmp("replay"), "run");
    await cp(await recording, run, { recursive: true });
    return run;
}
const writeLines = (file, rows) =>
    writeFile(file, rows.map((l) => JSON.stringify(l) + "\n").join(""));
// Record the run's outputs as they are now, as match.mjs does when a round ends. A test that edits a run to stand
// for another recording reseals it, so the check under test is not the manifest's.
async function reseal(run) {
    const manifest = JSON.parse(await readFile(join(run, "manifest.json")));
    for (const name of OUTPUTS)
        manifest.outputs[name] = createHash("sha256")
            .update(await readFile(join(run, name)))
            .digest("hex");
    await writeFile(join(run, "manifest.json"), JSON.stringify(manifest));
}

test("every recorded decision re-renders byte for byte under its logged harness", async () => {
    const run = await fixture(),
        { states, rows, manifest } = await readRun(run);
    assert.equal(manifest.kind, "rollout");
    assert.ok(states.length >= 20);
    // The sampled policy left the admissible set at times, so the weapon was asked about a projected motion.
    assert.ok(rows.some((r) => r.constraints?.policy_motion_changed));
    assert.ok(states.some((s) => s.game.bullets.length > 0));
    assert.deepEqual(await roundTrip(run), {
        run,
        decisions: states.length,
        rows: rows.length,
    });
});

test("a snapshot restores to the same state, and observing a state leaves the game unchanged", async () => {
    const { states } = await readRun(await fixture());
    for (const s of [states[0], states.at(-1)]) {
        const game = restoreGame(s.game),
            again = snapshotGame(game);
        assert.equal(stateHash(again), s.state_sha256);
        adapter.observe(game, s.player, {
            controllerMode: "plans",
            interval: s.interval_ms,
            rtt: s.rtt_ms,
        });
        assert.deepEqual(snapshotGame(game), again);
    }
    // Numbers JSON cannot hold survive: a command without a turn deadline has turnUntil Infinity.
    const game = restoreGame(states[0].game);
    game.tanks[0].turnUntil = Infinity;
    game.bullets.push({ ...game.tanks[0], vx: -0, trail: [] });
    const back = restoreGame(snapshotGame(game));
    assert.equal(back.tanks[0].turnUntil, Infinity);
    assert.ok(Object.is(back.bullets.at(-1).vx, -0));
});

test("replaying to decision k reproduces the logged state hash at every decision up to k", async () => {
    const run = await recorded(),
        { header, states } = await readRun(run),
        last = states.length - 1;
    assert.deepEqual(header.platform, PLATFORM);
    for (const to of [0, 1, Math.floor(last / 2), last])
        assert.deepEqual(await replay(run, { to }), {
            run,
            checked: to + 1,
            to,
        });
    assert.deepEqual(await replay(run), {
        run,
        checked: states.length,
        to: null,
        events: true,
        result: true,
    });
    await assert.rejects(replay(run, { to: states.length }), /--to must be/);
});

test("replay stops at the first decision that does not reproduce", async () => {
    // A changed state: the replay names the decision and the field, and no platform, since this one recorded it.
    let run = await recorded();
    const states = await lines(join(run, "states.jsonl")),
        k = Math.floor((states.length - 1) / 2);
    states[k + 1].game.tanks[0].x += 1;
    states[k + 1].state_sha256 = stateHash(states[k + 1].game);
    await writeLines(join(run, "states.jsonl"), states);
    await reseal(run);
    await assert.rejects(
        replay(run),
        (e) =>
            e.decision === k &&
            /state\.tanks\.0\.x is/.test(e.message) &&
            !/Recorded on|floating point/.test(e.message),
    );
    // A changed choice: another admissible motion at decision j changes what the weapon question is about.
    run = await recorded();
    const rows = await lines(join(run, "decisions.jsonl")),
        motion = rows.find(
            (r) =>
                r.decision >= 1 &&
                r.questions.motion &&
                r.constraints.admissible_motion_ids.length > 1 &&
                !r.constraints.policy_motion_changed,
        ),
        other = motion.constraints.admissible_motion_ids.find(
            (id) => id !== motion.chosen.motion,
        );
    motion.chosen.motion = other;
    await writeLines(join(run, "decisions.jsonl"), rows);
    await reseal(run);
    await assert.rejects(
        replay(run),
        (e) =>
            e.decision === motion.decision &&
            /the context of blue-2002-\d+:weapon differs/.test(e.message),
    );
});

test("a run changed after its round ended is refused before replay, re-rendering or labelling reads it", async () => {
    // The #204 review's two edits: a moved tank whose state_sha256 is rewritten to match, and the weapon motion of
    // three decisions, which no state hash covers. Neither touches the manifest.
    const moved = await recorded(),
        [header, ...states] = await lines(join(moved, "states.jsonl"));
    states[0].game.tanks[0].x += 10;
    states[0].state_sha256 = stateHash(states[0].game);
    await writeLines(join(moved, "states.jsonl"), [header, ...states]);
    const steered = await recorded(),
        [steeredHeader, ...steeredStates] = await lines(
            join(steered, "states.jsonl"),
        ),
        weapons = steeredStates.filter((s) => s.motion).slice(0, 3);
    assert.equal(weapons.length, 3);
    for (const s of weapons)
        s.motion = s.motion === "stop_none" ? "forward_none" : "stop_none";
    await writeLines(join(steered, "states.jsonl"), [
        steeredHeader,
        ...steeredStates,
    ]);
    for (const run of [moved, steered]) {
        const changed = `${run}/states.jsonl does not match the SHA-256 its manifest records. The run was changed`;
        for (const use of [readRun, replay, roundTrip, label])
            await assert.rejects(use(run), { message: changed });
        await assert.rejects(readFile(join(run, LABELS_FILE)), {
            code: "ENOENT",
        });
    }
    // A manifest without the hash is from a round that did not finish, or from before raw states were hashed.
    const manifest = JSON.parse(await readFile(join(moved, "manifest.json")));
    delete manifest.outputs["states.jsonl"];
    await writeFile(join(moved, "manifest.json"), JSON.stringify(manifest));
    await assert.rejects(readRun(moved), {
        message: `${moved}/manifest.json records no SHA-256 of states.jsonl. The round did not finish, or an older match wrote it`,
    });
});

test("a saved state changed after recording is refused before replay, re-rendering or labelling uses it", async () => {
    // Only the payload changes, and the manifest is resealed. Its logged hash and the decision rows stay as recorded.
    const run = await recorded(),
        [header, ...states] = await lines(join(run, "states.jsonl")),
        rows = await readFile(join(run, "decisions.jsonl"), "utf8"),
        logged = states[0].state_sha256;
    states[0].game.tanks[0].x += 10;
    await writeLines(join(run, "states.jsonl"), [header, ...states]);
    await reseal(run);
    const message = `decision 0: the saved state hashes to ${stateHash(states[0].game)}, not to its logged state_sha256 ${logged}`,
        refused = (e) => e.decision === 0 && e.message === message;
    for (const use of [readRun, replay, roundTrip, label])
        await assert.rejects(use(run), refused);
    assert.throws(() => render(states[0], header.harness), refused);
    assert.equal(await readFile(join(run, "decisions.jsonl"), "utf8"), rows);
    await assert.rejects(readFile(join(run, LABELS_FILE)), { code: "ENOENT" });
});

test("a mismatch on a round recorded on another platform names both platforms", async () => {
    // As on CI: the round was recorded elsewhere, and one angle differs in its last bit.
    const run = await recorded(),
        [header, ...states] = await lines(join(run, "states.jsonl")),
        k = 3,
        tank = states[k].game.tanks[1];
    header.platform = { os: "linux", arch: "x64", node: "22.23.5" };
    tank.angle =
        tank.angle === 0 ? Number.MIN_VALUE : tank.angle * (1 + Number.EPSILON);
    states[k].state_sha256 = stateHash(states[k].game);
    await writeLines(join(run, "states.jsonl"), [header, ...states]);
    await reseal(run);
    const here = `${PLATFORM.os}-${PLATFORM.arch} node ${PLATFORM.node}`;
    await assert.rejects(replay(run), (e) => {
        assert.equal(e.decision, k);
        assert.match(e.message, /state\.tanks\.1\.angle is -?[\d.]+, logged/);
        assert.ok(
            e.message.endsWith(
                `, logged ${tank.angle}. Recorded on linux-x64 node 22.23.5, replayed on ${here}: floating point may differ in the last bits.`,
            ),
            e.message,
        );
        return true;
    });
    // A recording from before the header named its platform says so.
    delete header.platform;
    await writeLines(join(run, "states.jsonl"), [header, ...states]);
    await reseal(run);
    await assert.rejects(replay(run), /The recording names no platform/);
});

test("the teacher labels every question of a model's round, and its own choices on a teacher round", async () => {
    const run = await fixture(),
        { rows } = await readRun(run),
        summary = await label(run);
    assert.deepEqual(summary, {
        run,
        rows: rows.length,
        questions: rows.length,
    });
    const [header, ...labelled] = await lines(join(run, LABELS_FILE));
    assert.equal(header.source, "planner-teacher");
    assert.equal(labelled.length, rows.length);
    labelled.forEach((l, i) => {
        assert.equal(l.request_id, rows[i].request_id);
        assert.deepEqual(Object.keys(l.labels), Object.keys(rows[i].questions));
        for (const [head, choice] of Object.entries(l.labels))
            assert.ok(Object.hasOwn(rows[i].questions[head].criteria, choice));
    });
    // The weapon label is for the motion the model's weapon question asked about.
    const weapons = labelled.filter((l) => l.labels.weapon);
    assert.ok(
        weapons.every((l) =>
            rows[l.row].contextParts.essential[0].includes(
                `SELECTED MOTION ${l.motion}:`,
            ),
        ),
    );
    assert.ok(labelled.some((l, i) => !same(l.labels, rows[i].chosen)));

    const teacher = join(await tmp("teacher"), "run");
    await playMatch({
        seed: 2002,
        seat: "blue",
        out: teacher,
        teacher: true,
        seconds: 6,
    });
    const recorded = await label(teacher);
    assert.equal(recorded.recorded.matched, recorded.rows);
    assert.equal(recorded.recorded.compared, recorded.rows);
});
test("a raw-controls round and a round asked with branches round-trip and replay too", async (t) => {
    const raw = await harnessDir({ controllerMode: "raw" }),
        runs = [];
    for (const [branches, harnessDir] of [
        [0, raw],
        [16, undefined],
    ]) {
        const server = await stubServer({ branches });
        t.after(() => server.close());
        const run = join(await tmp("round"), "run");
        await playMatch({
            seed: 2003,
            seat: "red",
            out: run,
            decisionServer: `http://127.0.0.1:${server.address().port}`,
            harnessDir,
            seconds: 4,
        });
        const { states, rows } = await readRun(run);
        assert.equal(rows[0].protocol, branches ? "plans-v1" : "controls-v3");
        assert.equal((await roundTrip(run)).rows, rows.length);
        assert.equal((await replay(run)).checked, states.length);
        runs.push(run);
    }
    // The planner teacher steers by plans, so it does not label raw controls.
    await assert.rejects(label(runs[0]), /labels planned steering only/);
});

// A harness directory with the default questions and these controller settings.
async function harnessDir(controller) {
    const dir = await tmp("harness");
    await mkdir(join(dir, "tank"));
    await writeFile(join(dir, CONTROLLER_PATH), JSON.stringify(controller));
    await writeFile(
        join(dir, "index.json"),
        JSON.stringify([{ path: CONTROLLER_PATH, kind: "settings" }]),
    );
    return dir;
}

// Each harness setting that the rendering reads changes the rendered request where it should.
const DEFAULT_CONTROLLER = {
    controllerMode: "plans",
    contextVersion: "full",
    assistEnabled: true,
    assistProfile: "light",
    fireGuardEnabled: false,
    strategy: "baseline",
};
const RAW = { controllerMode: "raw", strategy: undefined };
const BASELINE = {
    id: "baseline",
    description: "Original joint steering with one preferred firing direction.",
};
const harnessWith = (
    header,
    { controller = {}, questions = {}, notes } = {},
) => {
    const files = header.harness.map((f) => {
        if (f.path === CONTROLLER_PATH)
            return {
                ...f,
                content: JSON.stringify({
                    ...DEFAULT_CONTROLLER,
                    ...controller,
                }),
            };
        if (f.path === QUESTIONS_PATH)
            return {
                ...f,
                content: JSON.stringify({
                    ...JSON.parse(f.content),
                    ...questions,
                }),
            };
        return f;
    });
    return notes
        ? [
              ...files,
              { path: "notes/probe.md", kind: "instructions", content: notes },
          ]
        : files;
};
const text = (requests, i = 0) => requests[i].parts.essential[0];
const SETTINGS = [
    [
        "controllerMode",
        { controller: RAW },
        (before, after) => {
            assert.deepEqual(
                after.map((r) => r.id.split(":")[1]),
                ["controls"],
            );
            assert.deepEqual(Object.keys(after[0].questions), [
                "drive",
                "turn",
                "weapon",
            ]);
            assert.match(text(before), /Choose a JOINT drive\/turn plan/);
        },
    ],
    [
        "contextVersion",
        { controller: { ...RAW, contextVersion: "arena-v1" } },
        (before, after) => assert.match(text(after), /CONTEXT arena-v1\./),
    ],
    [
        "assistEnabled",
        { controller: { assistEnabled: false } },
        (before, after) => {
            assert.match(text(before), /Local bullet avoidance=ON/);
            assert.match(text(after), /Local bullet avoidance=OFF/);
        },
    ],
    [
        "assistProfile",
        { controller: { assistProfile: "strong" } },
        (before, after) => {
            assert.match(text(before), /Level=light;/);
            assert.match(text(after), /Level=strong; checks every 33ms/);
        },
    ],
    [
        "fireGuardEnabled",
        { controller: { ...RAW, fireGuardEnabled: true } },
        (before, after) => assert.match(text(after), /Shot guard=ON\./),
    ],
    [
        "strategy",
        {
            controller: {
                strategy: { id: "probe", description: "Probe the corners." },
            },
        },
        (before, after) => {
            assert.match(text(before), /^STRATEGY baseline: /);
            assert.match(text(after), /^STRATEGY probe: Probe the corners\. /);
        },
    ],
    [
        "strategy.multiAim",
        {
            controller: {
                strategy: { id: "baseline", description: "x", multiAim: true },
            },
        },
        (before, after) =>
            assert.match(text(after), /Chosen barrel heading -?\d/),
    ],
    [
        "strategy.allAnglePrompt",
        {
            controller: {
                strategy: {
                    id: "baseline",
                    description: "x",
                    allAnglePrompt: true,
                },
            },
        },
        (before, after) => {
            assert.match(
                text(before),
                /Rotate the hull toward the first verified/,
            );
            assert.match(text(after), /Compare the verified directions/);
        },
    ],
    [
        "strategy.clearanceWeight",
        {
            controller: {
                strategy: {
                    id: "baseline",
                    description: "x",
                    clearanceWeight: 2,
                    clearanceMargin: 30,
                },
            },
        },
        (before, after) =>
            assert.match(text(after), /prefer at least 30px\. Enemy range/),
    ],
    [
        "strategy.spacingWeight",
        {
            controller: {
                strategy: {
                    id: "baseline",
                    description: "x",
                    multiAim: true,
                    spacingWeight: 1,
                    minRange: 120,
                    maxRange: 260,
                },
            },
        },
        (before, after) =>
            assert.match(text(after), /preferred range 120-260px/),
    ],
    [
        "questions.motion",
        { questions: { motion: "Probe motion instructions." } },
        (before, after) => {
            assert.equal(
                after[0].questions.motion.instructions,
                "Probe motion instructions.",
            );
            assert.equal(text(after), text(before));
        },
    ],
    [
        "questions.weapon",
        { questions: { weapon: "Probe weapon instructions." } },
        (before, after) =>
            assert.equal(
                after[1].questions.weapon.instructions,
                "Probe weapon instructions.",
            ),
    ],
    [
        "questions.weapon.fire",
        { questions: { "weapon.fire": "Probe fire forecast." } },
        (before, after) => {
            assert.match(
                text(after, 1),
                /weapon\/fire: Probe fire forecast\.$/m,
            );
            assert.equal(text(after, 0), text(before, 0));
        },
    ],
    [
        "questions.weapon.hold",
        { questions: { "weapon.hold": "Probe hold forecast." } },
        (before, after) =>
            assert.match(
                text(after, 1),
                /weapon\/hold: Probe hold forecast\.$/m,
            ),
    ],
    [
        "questions.drive",
        { controller: RAW, questions: { drive: "Probe drive instructions." } },
        (before, after) =>
            assert.equal(
                after[0].questions.drive.instructions,
                "Probe drive instructions.",
            ),
    ],
    [
        "questions.turn",
        { controller: RAW, questions: { turn: "Probe turn instructions." } },
        (before, after) =>
            assert.equal(
                after[0].questions.turn.instructions,
                "Probe turn instructions.",
            ),
    ],
    [
        "operating notes",
        { notes: "Probe note.\n" },
        (before, after) => {
            for (const i of [0, 1]) {
                assert.ok(
                    text(after, i).startsWith(
                        "OPERATING NOTES (notes/probe.md):\nProbe note.\n\n",
                    ),
                );
                assert.ok(text(after, i).endsWith(text(before, i)));
            }
        },
    ],
];
for (const [setting, change, check] of SETTINGS)
    test(`the harness setting ${setting} changes the re-rendered context where expected`, async () => {
        const { header, states } = await readRun(await fixture()),
            // A state with shells in flight and a verified shot, so every forecast is present.
            observed = states.find((s) => s.game.bullets.length > 0),
            before = render(observed, header.harness),
            after = render(observed, harnessWith(header, change));
        assert.notDeepEqual(after, before);
        check(before, after);
    });

// revalidateSlack acts only when a plan is executed. adaptiveHold lengthens a plan by the response delay, which
// is zero in a lockstep match, so a match never renders it differently.
test("strategy parameters that do not reach the rendering leave it as it was", async () => {
    const { header, states } = await readRun(await fixture());
    for (const parameter of [{ revalidateSlack: 40 }, { adaptiveHold: true }]) {
        const harness = harnessWith(header, {
            controller: { strategy: { ...BASELINE, ...parameter } },
        });
        for (const s of states)
            assert.deepEqual(render(s, harness), render(s, header.harness));
    }
});
