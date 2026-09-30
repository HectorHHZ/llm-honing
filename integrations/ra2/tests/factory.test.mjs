import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import {
    checkServedFamily,
    decisionClientOptions,
} from "../integration/models/factory.mjs";
import { LayaClient } from "../integration/laya/client.mjs";

test("the laya, kev, and nanojev families are accepted, anything else is refused", () => {
    assert.deepEqual(decisionClientOptions({}, "intent"), {
        family: "laya",
        gpu: "0",
        model: "typed-decisions",
    });
    assert.deepEqual(
        decisionClientOptions({ family: "kev", gpu: "3" }, "intent"),
        {
            family: "kev",
            gpu: "3",
            model: "kev-0.8b",
        },
    );
    assert.deepEqual(
        decisionClientOptions(
            { family: "kev", "decision-server": "http://127.0.0.1:8801" },
            "intent",
        ),
        { family: "kev", decisionServer: "http://127.0.0.1:8801" },
    );
    assert.equal(
        decisionClientOptions(
            { family: "kev", gpu: "0", model: "kev-4b" },
            "intent",
        ).model,
        "kev-4b",
    );
    assert.equal(
        decisionClientOptions({ family: "nanojev", gpu: "1" }, "intent").model,
        "nanojev",
    );
    assert.throws(
        () => decisionClientOptions({ family: "jev-omni" }, "intent"),
        /jev-omni is not supported; use one of laya, kev, nanojev, djev/,
    );
});

test("a decision server must have loaded the family the run asked for", () => {
    assert.deepEqual(
        checkServedFamily(
            { family: "kev" },
            { family: "kev", model: "kev-0.8b" },
        ),
        { family: "kev", model: "kev-0.8b" },
    );
    // A Laya server reports no family field.
    assert.deepEqual(checkServedFamily({ family: "laya" }, { revision: "r" }), {
        revision: "r",
    });
    assert.throws(
        () => checkServedFamily({ family: "kev" }, { revision: "r" }),
        /Asked for a kev model, but the decision server loaded laya/,
    );
});

function fakeSpawn(calls) {
    return (command, args, options) => {
        calls.push({ command, args, env: options.env });
        const child = new EventEmitter();
        child.stdout = new PassThrough();
        child.stdin = new PassThrough();
        child.kill = () => {};
        return child;
    };
}

test("a kev worker runs with KEV_PYTHON and the kev settings", () => {
    const calls = [];
    const saved = process.env.KEV_PYTHON;
    process.env.KEV_PYTHON = "/envs/kev/bin/python";
    try {
        const client = new LayaClient({
            family: "kev",
            gpu: "2",
            model: "kev-0.8b",
            checkpointManifest: "/runs/train/checkpoint.json",
            spawnProcess: fakeSpawn(calls),
        });
        client.ready.catch(() => {});
        client.close();
    } finally {
        if (saved === undefined) delete process.env.KEV_PYTHON;
        else process.env.KEV_PYTHON = saved;
    }
    const [{ command, args, env }] = calls;
    assert.equal(command, "/envs/kev/bin/python");
    assert.match(args[1], /integration\/laya\/worker\.py$/);
    assert.equal(env.DECISION_FAMILY, "kev");
    assert.equal(env.LAYA_GPU, "2");
    assert.equal(env.KEV_MODEL, "kev-0.8b");
    assert.equal(env.KEV_CHECKPOINT_MANIFEST, "/runs/train/checkpoint.json");
});

test("a nanojev worker runs with NANOJEV_PYTHON and the nanojev settings", () => {
    const calls = [];
    const saved = process.env.NANOJEV_PYTHON;
    process.env.NANOJEV_PYTHON = "/envs/nanojev/bin/python";
    try {
        const client = new LayaClient({
            family: "nanojev",
            gpu: "4",
            model: "nanojev",
            checkpointManifest: null,
            spawnProcess: fakeSpawn(calls),
        });
        client.ready.catch(() => {});
        client.close();
    } finally {
        if (saved === undefined) delete process.env.NANOJEV_PYTHON;
        else process.env.NANOJEV_PYTHON = saved;
    }
    const [{ command, env }] = calls;
    assert.equal(command, "/envs/nanojev/bin/python");
    assert.equal(env.DECISION_FAMILY, "nanojev");
    assert.equal(env.NANOJEV_MODEL, "nanojev");
    assert.equal(env.LAYA_GPU, "4");
    assert.equal(env.NANOJEV_CHECKPOINT_MANIFEST, undefined);
});

test("a djev worker runs with DJEV_PYTHON, and the family names DiffusionGemma by default", () => {
    assert.deepEqual(
        decisionClientOptions({ family: "djev", gpu: "2" }, "intent"),
        { family: "djev", gpu: "2", model: "diffusiongemma-26b-a4b-it" },
    );
    const calls = [];
    const saved = process.env.DJEV_PYTHON;
    process.env.DJEV_PYTHON = "/envs/djev/bin/python";
    try {
        const client = new LayaClient({
            family: "djev",
            gpu: "2",
            model: "diffusiongemma-26b-a4b-it",
            spawnProcess: fakeSpawn(calls),
        });
        client.ready.catch(() => {});
        client.close();
    } finally {
        if (saved === undefined) delete process.env.DJEV_PYTHON;
        else process.env.DJEV_PYTHON = saved;
    }
    const [{ command, env }] = calls;
    assert.equal(command, "/envs/djev/bin/python");
    assert.equal(env.DECISION_FAMILY, "djev");
    assert.equal(env.DJEV_MODEL, "diffusiongemma-26b-a4b-it");
});
