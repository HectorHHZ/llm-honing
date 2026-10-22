// Offline orchestration tests. Every model, metadata object and engine is fake.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
    comparisonConfig,
    main,
    parseArgs,
    relabelScheduledWait,
    runMatch,
    verdict,
} from "../run.mjs";
import { ownSeatDetails, reserveOutput, unpackReplay } from "../runtime.mjs";

async function withDirectory(fn) {
    const dir = mkdtempSync(join(tmpdir(), "laya-harness-runner-offline-"));
    try {
        return await fn(dir);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}
const replay = (overrides = {}) => ({
    rplx: "offline-fake-replay",
    status: "truncated",
    end_reason: "max_ticks",
    lost_data: [],
    retargeted: 0,
    ...overrides,
});
const json = (dir, name) => JSON.parse(readFileSync(join(dir, name), "utf8"));
const lines = (dir, name) =>
    readFileSync(join(dir, name), "utf8")
        .trim()
        .split("\n")
        .filter(Boolean)
        .map(JSON.parse);
const options = (...flags) =>
    parseArgs([
        "--run",
        "--ticks=6",
        "--decision-interval=3",
        "--scheduler=fixed",
        "--advice=off",
        "--decision-log=plain",
        ...flags,
    ]);
const manifest = () => ({
    title: "Offline-Mock-Only",
    comparisonId: "mock-comparison",
});

test("default/help do not load a model, metadata or engine", async () => {
    const forbidden = () => {
        throw new Error("loader must not run");
    };
    await main([], forbidden);
    await main(["--help"], forbidden);
});

test("parse supports explicit d3/d30, seat, ablation and bounded wait flags", () => {
    const defaults = parseArgs(["--run"]);
    assert.equal(defaults.decisionInterval, 30);
    assert.equal(defaults.scheduler, "adaptive");
    assert.equal(defaults.advice, "on");
    const parsed = parseArgs([
        "--run",
        "--ticks=27000",
        "--seat",
        "blue",
        "--seed=2",
        "--decision-interval",
        "3",
        "--scheduler=fixed",
        "--advice",
        "off",
        "--max-wait-ticks=300",
        "--stall-ticks",
        "90",
        "--decision-log=plain",
    ]);
    assert.equal(parsed.seat, "blue");
    assert.equal(parsed.ticks, 27000);
    assert.equal(parsed.decisionInterval, 3);
    assert.equal(parsed.scheduler, "fixed");
    assert.equal(parsed.advice, "off");
    assert.equal(parsed.maxWaitTicks, 300);
    assert.equal(parsed.stallTicks, 90);
    for (const flags of [
        ["--ticks=90"],
        ["--run", "--run"],
        ["--run", "--unknown"],
        ["--run", "--seed=-1"],
        ["--run", "--seed=4294967296"],
        ["--run", "--ticks=0"],
        ["--run", "--decision-interval=4"],
        ["--run", "--max-wait-ticks=15"],
        ["--run", "--stall-ticks=0"],
        ["--run", "--scheduler=random"],
        ["--run", "--advice=maybe"],
        ["--run", "--seat=all"],
        ["--run", "--out=../old"],
        ["--run", "--map=/tmp/mp03t4.map"],
        ["--run", "--decision-log=none"],
    ])
        assert.throws(() => parseArgs(flags), undefined, flags.join(" "));
});

test("comparison configuration keeps the same model and conditions across seats", () => {
    assert.deepEqual(
        comparisonConfig(options("--seat=red")),
        comparisonConfig(options("--seat=blue")),
    );
    assert.notDeepEqual(
        comparisonConfig(options("--advice=on")),
        comparisonConfig(options("--advice=off")),
    );
    assert.notDeepEqual(
        comparisonConfig(options("--decision-interval=3")),
        comparisonConfig(options("--decision-interval=30")),
    );
});

test("output reservation refuses overwrites and path traversal", async () =>
    withDirectory((dir) => {
        assert.equal(reserveOutput("first", dir), join(dir, "first"));
        assert.throws(() => reserveOutput("first", dir), /overwrite/);
        for (const name of ["../escape", "/absolute", ".", "..", "a/b", ""])
            assert.throws(() => reserveOutput(name, dir));
    }));

test("normal metadata exposes no other-seat start or spawn", () => {
    const players = [
        { name: "red", player_id: 0, start: 0, spawn: [0, 0] },
        { name: "blue", player_id: 1, start: 1, spawn: [90, 90] },
    ];
    assert.deepEqual(ownSeatDetails(players, "red"), [
        players[0],
        { name: "blue", player_id: 1 },
    ]);
    assert.deepEqual(ownSeatDetails(players, "blue"), [
        { name: "red", player_id: 0 },
        players[1],
    ]);
});

test("generic wait relabelling leaves bindings and production-specific waits untouched", () => {
    const packet = {
        groups: [
            {
                id: "next_action",
                waitLabel: "Wait 0.2 seconds; build no army",
                choices: [
                    {
                        key: "A",
                        action: null,
                        label: "Wait 0.2 seconds; build no army",
                    },
                    {
                        key: "L1",
                        action: { function: 60, extra: { item: 10 } },
                        label: "Place",
                    },
                ],
            },
        ],
        questions: {
            next_action: {
                criteria: { A: "Wait 0.2 seconds; build no army", L1: "Place" },
            },
        },
    };
    const before = structuredClone(packet),
        updated = relabelScheduledWait(packet);
    assert.deepEqual(packet, before);
    assert.match(updated.groups[0].waitLabel, /scheduler/);
    assert.equal(
        updated.questions.next_action.criteria.A,
        updated.groups[0].choices[0].label,
    );
    assert.deepEqual(updated.groups[0].choices[1], packet.groups[0].choices[1]);
    packet.groups[0].waitLabel = "Continue power plant production";
    assert.deepEqual(relabelScheduledWait(packet), packet);
});

function fakeDependencies({
    failReady = false,
    failPrediction = false,
    missingReceipt = false,
    stall = false,
    savedReplay = replay(),
    chooseWait = () => false,
    readyBuilding = false,
    ended = false,
    outcome = { red: 0, blue: 0 },
    noForward = false,
} = {}) {
    const trace = [],
        confs = [],
        policies = [],
        packets = [],
        originalPackets = [],
        selected = [],
        submitted = [];
    let tick = 0,
        limit;
    const obs = (seat) => ({
        seat,
        vectors: {
            minerals: 9000,
            power_cap: 150,
            power_used: 50,
            is_low_power: 0,
            army_count: 0,
            worker_count: 0,
            units_truncated: 0,
            production: [
                {
                    queue: "structures",
                    status: readyBuilding ? "Ready" : "Active",
                    size: 1,
                    item: 10,
                    progress: readyBuilding ? 100 : 1 + tick / 3,
                    available: [10],
                    prices: [600],
                },
            ],
        },
    });
    const own = (seat) => ({
        tag: seat === "red" ? 100 : 200,
        unit_type: 24,
        name: "NACNST",
        alliance: 1,
        health: 1000,
        health_ratio: 255,
        display_type: 1,
        is_blip: 0,
        x: seat === "red" ? 20 : 40,
        y: 20,
        can_move: 0,
        is_idle: 1,
        is_powered: 1,
        build_progress: 100,
        order_id_0: 0,
        order_id_1: 0,
    });
    const state = (done = false) => ({
        tick,
        done,
        view: "normal",
        ended: done && ended,
        truncated: done && !ended,
        outcome,
        observations: { red: obs("red"), blue: obs("blue") },
        catalog: { digest: "mock-catalog" },
        players: [
            { name: "red", player_id: 0, start: 0, spawn: [20, 20] },
            { name: "blue", player_id: 1, start: 1, spawn: [40, 20] },
        ],
    });
    class CpuLayaClient {
        constructor() {
            trace.push("client_constructor");
            this.ready = Promise.resolve().then(() => {
                trace.push("client_ready");
                if (failReady) throw new Error("model startup failed");
                return { device: "cpu", weightsSha256: "mock-only" };
            });
        }
        async predict(packet) {
            assert.equal(
                packet.tick,
                tick,
                "engine must remain at pre-decision tick during both predictions",
            );
            trace.push(`predict:${packet.seat}:${packet.tick}`);
            packets.push(structuredClone(packet));
            if (failPrediction) throw new Error("inference failed");
            const wait = chooseWait({ tick, seat: packet.seat });
            return {
                answers: { next_action: { choice: wait ? "A" : "L1" } },
                wait,
            };
        }
        async close() {
            trace.push("client_close");
        }
    }
    class PermutationClient {
        constructor(client) {
            trace.push("permutation_wrapper");
            this.client = client;
            this.ready = client.ready;
        }
        predict(packet) {
            return this.client.predict(packet);
        }
    }
    class IntentPolicy {
        constructor(config) {
            this.config = config;
            this.name = config.name;
            this.metrics = {};
            this.memory = { events: [] };
            this.pending = [];
            confs.push(config);
            policies.push(this);
        }
        refresh(observation, at) {
            assert.equal(observation.seat, this.name);
            this.obs = observation;
            this.at = at;
            const unit = own(this.name);
            this.frame = { rows: [unit], own: [unit], visible: [], memory: [] };
            trace.push(`refresh:${this.name}:${at}`);
        }
        packet() {
            const action = readyBuilding
                ? { function: 60, extra: { item: 10 } }
                : { function: 62, extra: { queue: "structures" } };
            const groups =
                this.at % this.config.decisionInterval === 0
                    ? [
                          {
                              id: "next_action",
                              hierarchy: "intent-function",
                              waitLabel: "Continue production",
                              choices: [
                                  {
                                      key: "A",
                                      action: null,
                                      label: "Continue production",
                                      cost: 0,
                                  },
                                  {
                                      key: "L1",
                                      action,
                                      label: readyBuilding
                                          ? "Place power plant"
                                          : "Pause production",
                                      cost: 0,
                                  },
                              ],
                          },
                      ]
                    : [];
            const packet = {
                tick: this.at,
                seat: this.name,
                groups,
                availability: [],
                actorExclusions: [],
                questions: groups.length
                    ? {
                          next_action: {
                              type: "choice",
                              criteria: {
                                  A: "Continue production",
                                  L1: "action",
                              },
                          },
                      }
                    : {},
                parts: {
                    essential: [`Frozen state for ${this.name}`],
                    details: [],
                    logs: [],
                },
            };
            this.current = packet;
            originalPackets.push(structuredClone(packet));
            return packet;
        }
        select(answers, cap) {
            assert.equal(cap, 16);
            const leaf = this.current.groups[0].choices.find(
                (c) => c.key === answers.next_action.choice,
            );
            assert.ok(leaf, "prediction must resolve to an offered leaf");
            this.pending = leaf.action ? [leaf] : [];
            trace.push(`select:${this.name}:${this.at}`);
            selected.push({
                seat: this.name,
                at: this.at,
                answers: structuredClone(answers),
            });
            return {
                actions: this.pending.map((c) => c.action),
                selected: this.pending,
                dropped: [],
            };
        }
        receipts(at, results) {
            trace.push(`receipts:${this.name}:${at}`);
            if (!missingReceipt)
                assert.equal(results.length, this.pending.length);
            // Deliberately retain pending, matching the frozen policy. The
            // runner must not accidentally send stale pending empty receipts.
        }
    }
    const engine = {
        hello: { build: "offline-fake" },
        async create(config) {
            trace.push("engine_create");
            limit = config.max_ticks;
            assert.equal(config.view, "normal");
            assert.equal(config.privileged, false);
            assert.equal(config.record_replay, true);
            assert.ok(config.players.every((p) => p.country === "Russians"));
            return state();
        },
        async step(actions, config) {
            trace.push(`engine_step:${tick}`);
            assert.equal(config.ticks, Math.min(3, limit - tick));
            assert.equal(config.privileged, false);
            assert.deepEqual(Object.keys(actions).sort(), ["blue", "red"]);
            submitted.push({ tick, actions: structuredClone(actions) });
            if (!stall) tick += config.ticks;
            return {
                ...state(tick === limit),
                results: missingReceipt
                    ? {}
                    : Object.fromEntries(
                          Object.entries(actions).map(([seat, list]) => [
                              seat,
                              list.map(() => ({ accepted: true })),
                          ]),
                      ),
            };
        },
        async saveReplay() {
            trace.push("save_replay");
            return savedReplay;
        },
        async hash() {
            return { tick, hash: "offline-fake-hash" };
        },
        async shutdown() {
            trace.push("engine_shutdown");
        },
    };
    return {
        trace,
        confs,
        policies,
        packets,
        originalPackets,
        selected,
        submitted,
        deps: {
            CpuLayaClient,
            PermutationClient,
            IntentPolicy,
            GAME_DIR: "/offline-fake-assets",
            async loadMetadata() {
                trace.push("metadata_load");
                return {
                    bounds: { width: 64, height: 64 },
                    async dispose() {
                        trace.push("metadata_dispose");
                    },
                };
            },
            Ra2Engine: {
                async start() {
                    trace.push("engine_start");
                    return engine;
                },
            },
            async predictStaged(packet, client) {
                const prediction = await client.predict(packet);
                return {
                    resolvedAnswers: prediction.answers,
                    inferenceCalls: noForward ? 0 : 1,
                    stages: [],
                    latencyMs: 1,
                    selectionPath: [
                        {
                            stageIndex: 1,
                            group: "next_action",
                            dimension: "intent",
                            cancelled: prediction.wait,
                            complete: true,
                            binding: prediction.wait
                                ? null
                                : packet.groups[0].choices[1].action,
                        },
                    ],
                };
            },
        },
    };
}

for (const seat of ["red", "blue"])
    test(`two independent policies share one worker and decide at the same pre-step tick (${seat} candidate)`, async () =>
        withDirectory(async (dir) => {
            const fake = fakeDependencies(),
                result = await runMatch(
                    options(`--seat=${seat}`),
                    fake.deps,
                    dir,
                    manifest(),
                );
            assert.equal(result.valid, true);
            assert.equal(result.tick, 6);
            assert.equal(result.candidateResult, "unresolved");
            assert.equal(
                fake.trace.filter((x) => x === "client_constructor").length,
                1,
            );
            assert.equal(
                fake.trace.filter((x) => x === "permutation_wrapper").length,
                1,
            );
            assert.equal(fake.policies.length, 2);
            assert.notEqual(fake.policies[0], fake.policies[1]);
            assert.notEqual(fake.policies[0].memory, fake.policies[1].memory);
            assert.ok(
                fake.trace.indexOf("client_ready") <
                    fake.trace.indexOf("metadata_load"),
            );
            for (const at of [0, 3])
                for (const name of ["red", "blue"]) {
                    assert.ok(
                        fake.trace.indexOf(`refresh:red:${at}`) <
                            fake.trace.indexOf(`predict:${name}:${at}`),
                    );
                    assert.ok(
                        fake.trace.indexOf(`refresh:blue:${at}`) <
                            fake.trace.indexOf(`predict:${name}:${at}`),
                    );
                    assert.ok(
                        fake.trace.indexOf(`select:${name}:${at}`) <
                            fake.trace.indexOf(`engine_step:${at}`),
                    );
                }
            for (const conf of fake.confs) {
                assert.equal(conf.strictNormal, true);
                assert.equal(
                    conf.seats.find((p) => p.name !== conf.name).start,
                    undefined,
                );
            }
            assert.equal(result.stats.red.accepted, 2);
            assert.equal(result.stats.blue.accepted, 2);
            assert.equal(
                readFileSync(join(dir, "match.rplx"), "utf8"),
                "offline-fake-replay",
            );
            for (const cleanup of [
                "client_close",
                "engine_shutdown",
                "metadata_dispose",
            ])
                assert.ok(fake.trace.includes(cleanup));
        }));

for (const seat of ["red", "blue"])
    test(`H2 applies only to ${seat}; baseline packet and every action binding stay frozen`, async () =>
        withDirectory(async (dir) => {
            const fake = fakeDependencies({ readyBuilding: true });
            const result = await runMatch(
                options(`--seat=${seat}`, "--advice=on"),
                fake.deps,
                dir,
                manifest(),
            );
            for (const packet of fake.packets) {
                const original = fake.originalPackets.find(
                    (p) => p.seat === packet.seat && p.tick === packet.tick,
                );
                assert.deepEqual(packet.groups, original.groups);
                assert.deepEqual(packet.questions, original.questions);
                if (packet.seat === seat) {
                    assert.equal(
                        packet.parts.essential.length,
                        original.parts.essential.length + 1,
                    );
                    assert.match(
                        packet.parts.essential.at(-1),
                        /Harness v1.*placement|Harness v1.*Ready/,
                    );
                } else assert.deepEqual(packet, original);
            }
            assert.equal(result.stats[seat].adviceNotes, 2);
            assert.equal(
                result.stats[seat === "red" ? "blue" : "red"].adviceNotes,
                0,
            );
        }));

test("fixed d30 explicitly queries both seats only on d30 slots and never sends stale receipts", async () =>
    withDirectory(async (dir) => {
        const fake = fakeDependencies();
        await runMatch(
            options("--ticks=36", "--decision-interval=30"),
            fake.deps,
            dir,
            manifest(),
        );
        assert.deepEqual(
            fake.packets.map((p) => [p.seat, p.tick]),
            [
                ["red", 0],
                ["blue", 0],
                ["red", 30],
                ["blue", 30],
            ],
        );
        assert.deepEqual(
            fake.trace.filter((x) => x.startsWith("receipts:")),
            [
                "receipts:red:0",
                "receipts:blue:0",
                "receipts:red:30",
                "receipts:blue:30",
            ],
        );
        for (const step of fake.submitted.filter(
            (s) => ![0, 30].includes(s.tick),
        ))
            assert.deepEqual(step.actions, { red: [], blue: [] });
    }));

test("adaptive H1 skips only candidate busy waits; baseline remains fixed d30", async () =>
    withDirectory(async (dir) => {
        const fake = fakeDependencies({ chooseWait: () => true });
        const result = await runMatch(
            options(
                "--ticks=63",
                "--decision-interval=30",
                "--scheduler=adaptive",
                "--seat=blue",
            ),
            fake.deps,
            dir,
            manifest(),
        );
        assert.equal(
            fake.confs.find((c) => c.name === "blue").decisionInterval,
            3,
        );
        assert.equal(
            fake.confs.find((c) => c.name === "red").decisionInterval,
            30,
        );
        assert.deepEqual(
            fake.packets.filter((p) => p.seat === "blue").map((p) => p.tick),
            [0],
        );
        assert.deepEqual(
            fake.packets.filter((p) => p.seat === "red").map((p) => p.tick),
            [0, 30, 60],
        );
        assert.equal(result.scheduling.skippedBaseSlots, 2);
        assert.equal(result.stats.blue.decisions, 1);
        assert.equal(result.stats.red.decisions, 3);
        assert.deepEqual(
            fake.trace.filter((x) => x.startsWith("receipts:blue:")),
            ["receipts:blue:0"],
        );
        assert.equal(lines(dir, "scheduler.jsonl").length, 21);
    }));

test("failed worker startup never creates metadata or game and still closes worker", async () =>
    withDirectory(async (dir) => {
        const fake = fakeDependencies({ failReady: true });
        await assert.rejects(
            runMatch(options(), fake.deps, dir, manifest()),
            /startup failed/,
        );
        assert.ok(!fake.trace.includes("metadata_load"));
        assert.ok(!fake.trace.includes("engine_start"));
        assert.ok(fake.trace.includes("client_close"));
        assert.equal(json(dir, "result.json").valid, false);
    }));

for (const [failure, pattern] of [
    ["failPrediction", /inference failed/],
    ["missingReceipt", /receipt/],
    ["stall", /advancing/],
    ["noForward", /real forward/],
])
    test(`${failure} aborts with invalid result and cleanup rather than silently continuing`, async () =>
        withDirectory(async (dir) => {
            const fake = fakeDependencies({ [failure]: true });
            await assert.rejects(
                runMatch(options(), fake.deps, dir, manifest()),
                pattern,
            );
            if (["failPrediction", "noForward"].includes(failure))
                assert.ok(
                    !fake.trace.some((x) => x.startsWith("engine_step:")),
                );
            assert.equal(json(dir, "result.json").valid, false);
            assert.equal(json(dir, "result.json").candidateResult, "invalid");
            for (const cleanup of [
                "client_close",
                "engine_shutdown",
                "metadata_dispose",
            ])
                assert.ok(fake.trace.includes(cleanup));
        }));

test("partial or lost replay retained for diagnosis but never a valid result", async () => {
    for (const savedReplay of [
        replay({ status: "partial" }),
        replay({ lost_data: ["commands"] }),
    ])
        await withDirectory(async (dir) => {
            const fake = fakeDependencies({ savedReplay });
            await assert.rejects(
                runMatch(options(), fake.deps, dir, manifest()),
                /marked invalid/,
            );
            assert.equal(json(dir, "result.json").valid, false);
            assert.equal(json(dir, "result.json").candidateResult, "invalid");
            assert.equal(
                readFileSync(join(dir, "match.rplx"), "utf8"),
                "offline-fake-replay",
            );
        });
});

test("time-limit cutoff is unresolved regardless of score; only terminal outcomes win", () => {
    const score = { red: 1, blue: -1 };
    assert.equal(
        verdict(
            { valid: true, ended: false, truncated: true, outcome: score },
            "red",
        ),
        "unresolved",
    );
    assert.equal(
        verdict(
            { valid: true, ended: true, truncated: true, outcome: score },
            "red",
        ),
        "unresolved",
    );
    assert.equal(
        verdict(
            { valid: true, ended: true, truncated: false, outcome: score },
            "red",
        ),
        "win",
    );
    assert.equal(
        verdict(
            { valid: true, ended: true, truncated: false, outcome: score },
            "blue",
        ),
        "loss",
    );
    assert.equal(
        verdict(
            {
                valid: true,
                ended: true,
                truncated: false,
                outcome: { red: 0, blue: 0 },
            },
            "red",
        ),
        "draw",
    );
    assert.equal(
        verdict({ valid: false, ended: true, outcome: score }, "red"),
        "invalid",
    );
    assert.equal(
        verdict(
            {
                valid: true,
                ended: true,
                truncated: false,
                outcome: { red: 1, blue: 1 },
            },
            "red",
        ),
        "unknown_terminal",
    );
});

test("SDK envelope accepts text replay, flags lost data, and refuses invalid envelope", () => {
    assert.equal(unpackReplay(replay()).rplx, "offline-fake-replay");
    assert.equal(unpackReplay(replay({ status: "partial" })).valid, false);
    assert.throws(() => unpackReplay("old-format"), /SavedReplay/);
    assert.throws(() => unpackReplay(replay({ rplx: "" })), /nonempty/);
    assert.throws(() => unpackReplay(replay({ retargeted: -1 })), /metadata/);
});
