import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { LayaClient } from "../integration/laya/client.mjs";
import {
    normalSeats,
    resolveDuelStep,
} from "../integration/models/duel-core.mjs";

function fixture() {
    const log = [],
        observations = {},
        policies = {},
        clients = {};
    for (const seat of ["red", "blue"]) {
        const own = {
            vectors: { minerals: seat === "red" ? 123 : 456 },
            privateFor: seat,
        };
        observations[seat] = own;
        clients[seat] = { seat };
        policies[seat] = {
            frame: { rows: [{ tag: seat }] },
            refresh(ob, tick) {
                assert.equal(ob, own);
                this.obs = ob;
                this.tick = tick;
                log.push(`refresh:${seat}:${tick}`);
            },
            packet() {
                log.push(`packet:${seat}`);
                return {
                    tick: this.tick,
                    parts: { essential: [this.obs.privateFor] },
                    groups: [],
                    questions: {},
                    omitted: [],
                };
            },
            select(answers, cap) {
                assert.equal(cap, 1);
                log.push(`select:${seat}`);
                const action = { function: 1, unit_tags: [seat] };
                return {
                    actions: [action],
                    selected: [{ action }],
                    dropped: [],
                };
            },
        };
    }
    return { tick: 6, observations, view: "normal", policies, clients, log };
}

test("opponent player configuration exposes identity only", () => {
    const players = [
        { name: "red", player_id: 1, credits: 100, private: "red" },
        { name: "blue", player_id: 2, credits: 500, private: "blue" },
    ];
    assert.deepEqual(normalSeats(players, "red"), [
        players[0],
        { name: "blue", player_id: 2 },
    ]);
    assert.deepEqual(normalSeats(players, "blue"), [
        { name: "red", player_id: 1 },
        players[1],
    ]);
});

test("both normal packets precede either model and wait for both decisions", async () => {
    const f = fixture(),
        completed = {};
    const pending = resolveDuelStep({
        ...f,
        predict: (packet, client) => {
            assert.deepEqual(f.log.slice(0, 4), [
                "refresh:red:6",
                "packet:red",
                "refresh:blue:6",
                "packet:blue",
            ]);
            assert.deepEqual(packet.parts.essential, [client.seat]);
            assert.equal(packet.tick, 6);
            return new Promise(
                (resolve) =>
                    (completed[client.seat] = () =>
                        resolve({
                            resolvedAnswers: {},
                            inferenceCalls: 2,
                            stages: [],
                        })),
            );
        },
    });
    completed.blue();
    await Promise.resolve();
    assert(
        !f.log.some((entry) => entry.startsWith("select:")),
        "Partial opponent result must not be acted upon",
    );
    completed.red();
    const result = await pending;
    for (const seat of ["red", "blue"]) {
        assert.equal(result[seat].record.seat, seat);
        assert.equal(result[seat].record.tick, 6);
        assert.deepEqual(result[seat].record.contextParts.essential, [seat]);
        assert.equal(
            result[seat].record.ownVectors.minerals,
            seat === "red" ? 123 : 456,
        );
        assert(!("privileged" in result[seat].record));
    }
});

test("one failed model aborts the joint decision without fallback actions", async () => {
    const f = fixture();
    await assert.rejects(
        resolveDuelStep({
            ...f,
            predict: async (_, client) => {
                if (client.seat === "blue") throw new Error("native failure");
                return { resolvedAnswers: {}, inferenceCalls: 1 };
            },
        }),
        /native failure/,
    );
    assert(!f.log.some((entry) => entry.startsWith("select:")));
});

test("privileged or missing observation fails before inference", async () => {
    const f = fixture();
    let calls = 0;
    const predict = async () => {
        calls++;
    };
    await assert.rejects(
        resolveDuelStep({ ...f, view: "privileged", predict }),
    );
    await assert.rejects(resolveDuelStep({ ...f, observations: {}, predict }));
    assert.equal(calls, 0);
});

test("a policy exceeding the equal action budget fails closed", async () => {
    const f = fixture();
    f.policies.blue.select = () => ({ actions: [{}, {}], selected: [{}, {}] });
    await assert.rejects(
        resolveDuelStep({
            ...f,
            predict: async () => ({ resolvedAnswers: {}, inferenceCalls: 1 }),
        }),
        /one-command budget/,
    );
});

function spawnFixture(collected) {
    return (_python, _args, options) => {
        collected.push(options.env);
        const child = new EventEmitter();
        child.stdin = new PassThrough();
        child.stdout = new PassThrough();
        child.kill = () => child.stdout.end();
        queueMicrotask(() =>
            child.stdout.write(
                JSON.stringify({ ready: true, metadata: { fixture: true } }) +
                    "\n",
            ),
        );
        return child;
    };
}

test("different Laya seats receive independent checkpoint environments", async () => {
    const original = process.env.LAYA_CHECKPOINT_MANIFEST;
    process.env.LAYA_CHECKPOINT_MANIFEST = "/fixture/inherited.json";
    const environments = [],
        clients = [];
    try {
        for (const checkpointManifest of [
            "/fixture/red.json",
            "/fixture/blue.json",
            null,
            undefined,
        ])
            clients.push(
                new LayaClient({
                    checkpointManifest,
                    spawnProcess: spawnFixture(environments),
                }),
            );
        await Promise.all(clients.map((client) => client.ready));
        assert.deepEqual(
            environments.map((env) => env.LAYA_CHECKPOINT_MANIFEST),
            [
                "/fixture/red.json",
                "/fixture/blue.json",
                undefined,
                "/fixture/inherited.json",
            ],
        );
        assert.equal(
            process.env.LAYA_CHECKPOINT_MANIFEST,
            "/fixture/inherited.json",
        );
        assert.notEqual(environments[0], environments[1]);
    } finally {
        clients.forEach((client) => client.close());
        if (original === undefined) delete process.env.LAYA_CHECKPOINT_MANIFEST;
        else process.env.LAYA_CHECKPOINT_MANIFEST = original;
    }
});
