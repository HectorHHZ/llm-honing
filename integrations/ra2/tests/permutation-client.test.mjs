import test from "node:test";
import assert from "node:assert/strict";
import {
    PermutationClient,
    verifyEnsemble,
    METHOD,
} from "../integration/laya/permutation-client.mjs";
const packet = {
    parts: { essential: ["normal public state"] },
    dimensions: { next_action: "target" },
    questions: {
        next_action: {
            type: "choice",
            instructions: "Choose a target.",
            criteria: {
                A: "Cancel this action",
                B: "Own base",
                C: "Unexplored start",
            },
        },
    },
};
function fixture(values) {
    const calls = [];
    return {
        calls,
        ready: Promise.resolve({ weightsSha256: "fixture" }),
        close() {},
        async predict(request) {
            calls.push(structuredClone(request));
            const ps = values[calls.length - 1],
                keys = Object.keys(request.questions.next_action.criteria);
            const p = Object.fromEntries(keys.map((k, i) => [k, ps[i]])),
                choice = keys.reduce((a, b) => (p[b] > p[a] ? b : a), keys[0]);
            const answers = { next_action: { choice, probabilities: p } };
            return {
                id: calls.length,
                answers,
                modelResult: { answers: structuredClone(answers) },
                stateText: "normal public state",
                tokenAudit: {
                    next_action: {
                        options: 3,
                        stateTruncated: false,
                        stateTokens: 4,
                        stateRoom: 1900,
                        inputTokens: 120,
                    },
                },
                trimmed: [],
                latencyMs: 1,
            };
        },
    };
}
test("mean maps permuted scores back to exact canonical target bindings", async () => {
    const raw = fixture([
            [0, 0.6, 0.4],
            [0, 0.9, 0.1],
        ]),
        client = new PermutationClient(raw),
        before = structuredClone(packet);
    assert.equal((await client.ready).selectionMethod, METHOD);
    const r = await client.predict(packet);
    assert.equal(r.answers.next_action.choice, "C");
    assert(Math.abs(r.answers.next_action.probabilities.C - 0.65) < 1e-10);
    assert.equal(r.forwardCalls, 2);
    assert.equal(r.latencyMs, 2);
    assert.deepEqual(packet, before);
    assert.deepEqual(raw.calls[0].parts, raw.calls[1].parts);
    assert.deepEqual(raw.calls[1].questions.next_action.criteria, {
        A: "Cancel this action",
        B: "Unexplored start",
        C: "Own base",
    });
    verifyEnsemble({ ...packet, ...r });
    const bad = structuredClone(r);
    bad.answers.next_action.choice = "B";
    assert.throws(() => verifyEnsemble({ ...packet, ...bad }));
});
test("root requests retain the single original model call", async () => {
    const raw = fixture([[0, 0.6, 0.4]]),
        client = new PermutationClient(raw);
    const p = { ...packet, dimensions: { next_action: "intent" } };
    const r = await client.predict(p);
    assert.equal(raw.calls.length, 1);
    assert.equal(r.answers.next_action.choice, "B");
    verifyEnsemble({ ...p, ...r });
});
test("failed or differently packed second call does not substitute a local action", async () => {
    const raw = fixture([
            [0, 0.6, 0.4],
            [0, 0.9, 0.1],
        ]),
        predict = raw.predict.bind(raw);
    raw.predict = async (p) => {
        const r = await predict(p);
        if (raw.calls.length === 2) r.stateText += " changed";
        return r;
    };
    await assert.rejects(
        new PermutationClient(raw).predict(packet),
        /retained context/,
    );
    const invalid = fixture([
            [0, 0.6, 0.4],
            [0, 0.9, 0.1],
        ]),
        next = invalid.predict.bind(invalid);
    invalid.predict = async (p) => {
        const r = await next(p);
        if (invalid.calls.length === 2)
            delete r.answers.next_action.probabilities.C;
        return r;
    };
    await assert.rejects(new PermutationClient(invalid).predict(packet));
});

test("remote token accounting survives the actual target permutation path", async () => {
    const raw = fixture([
            [0, 0.6, 0.4],
            [0, 0.9, 0.1],
        ]),
        predict = raw.predict.bind(raw);
    raw.predict = async (p) => {
        const r = await predict(p);
        r.tokenAudit.next_action = {
            ...r.tokenAudit.next_action,
            encoding: "djev-remote-prompt",
            accounting: "upstream-request-total",
            stateTokens: null,
            stateRoom: null,
        };
        return r;
    };
    const r = await new PermutationClient(raw).predict(packet);
    verifyEnsemble({ ...packet, ...r });
    assert.equal(r.answers.next_action.choice, "C");
    assert.equal(r.tokenAudit.next_action.stateTokens, null);
    const bad = structuredClone(r);
    bad.ensemble.original.tokenAudit.next_action.stateRoom = 8192;
    assert.throws(() => verifyEnsemble({ ...packet, ...bad }));
});
