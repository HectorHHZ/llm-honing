import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import {
    CpuLayaClient,
    WEIGHTS_SHA256,
    SOURCE_REVISION,
} from "../cpu-client.mjs";

const packet = () => ({
    parts: { essential: ["test context"], details: [], memory: [], logs: [] },
    questions: {
        next: { type: "choice", criteria: { A: "Wait", B: "Deploy" } },
    },
    dimensions: { next: "intent" },
});
const metadata = () => ({
    repo: "jayzou3773/laya-ra2-v33",
    weightsSha256: WEIGHTS_SHA256,
    sourceCommit: SOURCE_REVISION,
    device: "cpu",
    releaseTransport: true,
});
const response = (id) => ({
    id,
    answers: {
        next: {
            type: "choice",
            choice: "B",
            probabilities: { A: 0.2, B: 0.8 },
        },
    },
    modelResult: { answers: {} },
    stateText: "test context",
    tokenAudit: { next: {} },
    trimmed: [],
    latencyMs: 1,
});
function fixture(options = {}) {
    const child = new EventEmitter();
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.pid = 123;
    child.signals = [];
    child.kill = (signal) => {
        child.signals.push(signal);
        setImmediate(() => child.emit("exit", null, signal));
        return true;
    };
    let invocation;
    const client = new CpuLayaClient({
        startupTimeoutMs: 1000,
        requestTimeoutMs: 1000,
        ...options,
        spawnProcess: (...args) => {
            invocation = args;
            return child;
        },
    });
    const send = (value) => child.stdout.write(JSON.stringify(value) + "\n");
    const ready = () => send({ ready: true, metadata: metadata() });
    const onRequest = (f) =>
        child.stdin.on("data", (data) => f(JSON.parse(String(data))));
    return { client, child, invocation, send, ready, onRequest };
}

test("CPU worker is pinned, preserves packet and returns one unpermuted answer", async () => {
    const f = fixture();
    try {
        assert.equal(f.invocation[2].env.LAYA_DEVICE, "cpu");
        assert.match(f.invocation[1][1], /ra2\/integration\/laya\/worker.py$/);
        f.ready();
        assert.equal((await f.client.ready).weightsSha256, WEIGHTS_SHA256);
        let calls = 0;
        f.onRequest((request) => {
            calls++;
            assert.deepEqual(request.questions, packet().questions);
            assert.deepEqual(request.parts, packet().parts);
            assert.equal(Object.hasOwn(request, "dimensions"), false);
            f.send(response(request.id));
        });
        assert.equal(
            (await f.client.predict(packet())).answers.next.choice,
            "B",
        );
        assert.equal(calls, 1);
    } finally {
        await f.client.close();
    }
    assert.deepEqual(f.child.signals, ["SIGTERM"]);
    assert.equal(f.client.terminated, true);
});

test("startup timeout rejects readiness and terminates its worker", async () => {
    const f = fixture({ startupTimeoutMs: 15 });
    await assert.rejects(f.client.ready, /startup timeout/);
    await f.client.close();
    assert.equal(f.client.pending.size, 0);
});

test("inference timeout rejects all requests and terminates worker", async () => {
    const f = fixture({ requestTimeoutMs: 15 });
    f.ready();
    await assert.rejects(f.client.predict(packet()), /inference timeout/);
    await f.client.close();
    assert.equal(f.client.pending.size, 0);
});

test("unknown response ids fail instead of silently hanging", async () => {
    const f = fixture();
    f.ready();
    f.onRequest((request) => f.send(response(request.id + 1)));
    await assert.rejects(
        f.client.predict(packet()),
        /Unexpected worker response id/,
    );
    await f.client.close();
});

test("multiple answers for one request fail the protocol", async () => {
    const f = fixture();
    f.ready();
    f.onRequest((request) => {
        f.send(response(request.id));
        f.send(response(request.id));
    });
    await assert.rejects(
        f.client.predict(packet()),
        /Unexpected worker response id/,
    );
    await f.client.close();
});

test("stdout EOF before process exit terminates the invalid worker", async () => {
    const f = fixture();
    f.ready();
    f.onRequest(() => f.child.stdout.end());
    await assert.rejects(f.client.predict(packet()), /stdout closed/);
    await f.client.close();
});

test("worker errors reject pending inference and future calls", async () => {
    const f = fixture();
    f.ready();
    f.onRequest(() => f.send({ error: "out of memory" }));
    await assert.rejects(f.client.predict(packet()), /out of memory/);
    await assert.rejects(f.client.predict(packet()), /out of memory/);
    await f.client.close();
});

test("duplicate ready messages are fatal", async () => {
    const f = fixture();
    f.ready();
    await f.client.ready;
    f.ready();
    await assert.rejects(
        f.client.predict(packet()),
        /Duplicate\/invalid worker ready/,
    );
    await f.client.close();
});

test("wrong device/hash cannot masquerade as the pinned CPU model", async () => {
    for (const patch of [{ device: "cuda:0" }, { weightsSha256: "other" }]) {
        const f = fixture();
        f.send({ ready: true, metadata: { ...metadata(), ...patch } });
        await assert.rejects(f.client.ready, /metadata/);
        await f.client.close();
    }
});

test("unexpected child exit rejects a pending request", async () => {
    const f = fixture();
    f.ready();
    f.onRequest(() => f.child.emit("exit", 137, null));
    await assert.rejects(f.client.predict(packet()), /exited \(137\)/);
    await f.client.close();
    assert.equal(f.client.terminated, true);
});

test("spawn error rejects readiness and closes the child", async () => {
    const f = fixture();
    f.child.emit("error", new Error("spawn ENOENT"));
    await assert.rejects(f.client.ready, /ENOENT/);
    await f.client.close();
});

test("close rejects outstanding work, is idempotent and awaits termination", async () => {
    const f = fixture();
    f.ready();
    f.onRequest(() => f.client.close());
    await assert.rejects(f.client.predict(packet()), /worker closed/);
    const a = f.client.close(),
        b = f.client.close();
    assert.equal(a, b);
    await a;
    assert.equal(f.client.terminated, true);
});

test("illegal model answer or probability distribution is fatal", async () => {
    for (const answer of [
        { type: "choice", choice: "Z", probabilities: { A: 0.2, B: 0.8 } },
        { type: "choice", choice: "B", probabilities: { A: 0.8, B: 0.8 } },
        { type: "choice", choice: "A", probabilities: { A: 0.2, B: 0.8 } },
    ]) {
        const f = fixture();
        f.ready();
        f.onRequest((request) =>
            f.send({ ...response(request.id), answers: { next: answer } }),
        );
        await assert.rejects(f.client.predict(packet()), /Invalid/);
        await f.client.close();
    }
});

test("non-JSON stdout is fatal, never treated as a model answer", async () => {
    const f = fixture();
    f.child.stdout.write("unexpected banner\n");
    await assert.rejects(f.client.ready);
    await f.client.close();
});

test("invalid local request is rejected without poisoning ready worker", async () => {
    const f = fixture();
    try {
        f.ready();
        await assert.rejects(
            f.client.predict({ parts: {}, questions: {} }),
            /nonempty/,
        );
        assert.equal(f.client.closed, false);
    } finally {
        await f.client.close();
    }
});
