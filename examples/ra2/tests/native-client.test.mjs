import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import {
    NativeClient,
    bridgeNativePrediction,
    verifyNativeRecord,
} from "../integration/models/client.mjs";
import {
    PermutationClient,
    verifyEnsemble,
} from "../integration/laya/permutation-client.mjs";
import {
    STAGED_PROTOCOL,
    predictStaged,
    verifyStagedTrace,
} from "../integration/laya/staged.mjs";

const parts = {
    essential: ["Normal-view CPU fixture, not a model evaluation."],
    details: [],
    memory: [],
    logs: [],
};
const request = {
    parts,
    questions: {
        next_action: {
            type: "choice",
            instructions: "Choose target",
            criteria: { A: "Cancel action", B: "North", C: "South" },
        },
    },
    dimensions: { next_action: "target" },
};
function metadata(family) {
    return {
        family,
        protocol: "native-choice-jsonl-v1",
        maxLen: 2048,
        modelWeightsLoaded: true,
        weightSetSha256: "1".repeat(64),
        preparedManifestSha256: "2".repeat(64),
        verifiedFilesSha256: { "fixture-only": "3".repeat(64) },
        parameterStorage: "float32",
        forwardAutocast: "bfloat16",
    };
}
function response(packet, family, id = 1) {
    const answers = Object.fromEntries(
        Object.entries(packet.questions).map(([key, q]) => {
            const keys = Object.keys(q.criteria),
                choice = keys[1];
            return [
                key,
                {
                    choice,
                    probabilities: Object.fromEntries(
                        keys.map((k) => [k, Number(k === choice)]),
                    ),
                },
            ];
        }),
    );
    const upstream =
        family === "nanojev"
            ? {
                  states: [
                      {
                          id: "decision",
                          answers: Object.fromEntries(
                              Object.entries(answers).map(([k, a]) => [
                                  k,
                                  { ...structuredClone(a), type: "choice" },
                              ]),
                          ),
                      },
                  ],
              }
            : { family, answers: structuredClone(answers) };
    const tokenAudit = Object.fromEntries(
        Object.entries(packet.questions).map(([k, q]) => [
            k,
            {
                options: Object.keys(q.criteria).length,
                stateTruncated: false,
                maxInputTokens: 2048,
                inputTokens: 100,
                stateTokens: 50,
                stateRoom: 1998,
                ...(family === "nanojev"
                    ? {
                          encoding: "nanojev-native-candidate-paths",
                          candidatePathSha256: Object.keys(q.criteria).map(() =>
                              "4".repeat(64),
                          ),
                      }
                    : {
                          encoding: "jev-omni-native-numbered-options",
                          promptSha256: "5".repeat(64),
                      }),
            },
        ]),
    );
    return {
        id,
        answers,
        stateText: ["essential", "details", "memory", "logs"]
            .flatMap((s) => packet.parts[s] ?? [])
            .join("\n"),
        tokenAudit,
        trimmed: [],
        modelResult: upstream,
        latencyMs: 1,
    };
}
function fakeSpawn(family, mutate = (r) => r) {
    const process = new EventEmitter();
    process.stdin = new PassThrough();
    process.stdout = new PassThrough();
    process.kill = () => {
        process.killed = true;
        queueMicrotask(() => process.emit("exit", 0));
    };
    process.stdin.on("data", (chunk) => {
        for (const line of chunk.toString().trim().split("\n")) {
            const packet = JSON.parse(line),
                result = mutate(response(packet, family, packet.id));
            if (result)
                queueMicrotask(() =>
                    process.stdout.write(JSON.stringify(result) + "\n"),
                );
        }
    });
    return (...args) => {
        process.command = args;
        queueMicrotask(() =>
            process.stdout.write(
                JSON.stringify({ ready: true, metadata: metadata(family) }) +
                    "\n",
            ),
        );
        return process;
    };
}
const options = (family) => ({
    family,
    preparedManifest: "/fixture/manifest.json",
    sourceRoot: family === "nanojev" ? "/fixture/source" : undefined,
    gpu: "5",
});

for (const family of ["nanojev", "jev-omni"]) {
    test(`${family} field bridge preserves raw native evidence and two-order target inference`, async () => {
        const client = new NativeClient(options(family), fakeSpawn(family));
        const mean = new PermutationClient(client);
        try {
            const runtime = await mean.ready;
            const result = await mean.predict(request);
            verifyEnsemble({ ...request, ...result });
            assert.equal(result.forwardCalls, 2);
            assert.equal(result.answers.next_action.choice, "B");
            verifyNativeRecord(request, result.ensemble.original, runtime);
            verifyNativeRecord(
                { ...request, questions: result.ensemble.permutedQuestions },
                result.ensemble.swapped,
                runtime,
            );
            assert.equal(result.ensemble.original.id, 1);
            assert.equal(result.ensemble.swapped.id, 2);
            assert.equal(
                client.process.command[1].includes("integration.models.worker"),
                true,
            );
        } finally {
            mean.close();
        }
    });
}

test("native evidence rejects altered upstream choices, context, partial probabilities and encoding bounds", () => {
    const raw = response(request, "nanojev"),
        runtime = metadata("nanojev");
    const changed = [
        (r) => {
            r.modelResult.states[0].answers.next_action.choice = "C";
        },
        (r) => {
            r.stateText += "\nextra information";
        },
        (r) => {
            delete r.answers.next_action.probabilities.C;
        },
        (r) => {
            r.tokenAudit.next_action.inputTokens = 2049;
        },
        (r) => {
            r.tokenAudit.next_action.candidatePathSha256.pop();
        },
        (r) => {
            r.trimmed = [{ section: "essential", text: parts.essential[0] }];
        },
    ];
    for (const change of changed) {
        const copy = structuredClone(raw);
        change(copy);
        assert.throws(() => bridgeNativePrediction(request, copy, runtime));
    }
    assert.deepEqual(raw, response(request, "nanojev"));
});

test("worker errors and unknown IDs reject pending choices without fallback", async () => {
    const missing = new NativeClient({
        ...options("nanojev"),
        python: "/nonexistent/native-python",
    });
    await assert.rejects(missing.ready, /ENOENT/);
    for (const mutate of [
        (r) => ({ ...r, id: r.id + 1 }),
        () => ({ error: "native fixture failure" }),
    ]) {
        const client = new NativeClient(
            options("nanojev"),
            fakeSpawn("nanojev", mutate),
        );
        await assert.rejects(client.predict(request));
        assert(client.closed && client.process.killed);
        assert.equal(client.pending.size, 0);
    }
});

test("native inference timeout stops the worker rather than manufacturing an action", async () => {
    const client = new NativeClient(
        { ...options("jev-omni"), inferenceTimeoutMs: 20 },
        fakeSpawn("jev-omni", () => null),
    );
    await assert.rejects(client.predict(request), /timeout/);
    assert.equal(client.pending.size, 0);
    assert(client.closed && client.process.killed);
});

test("native bridge carries staged choices to a grounded engine leaf with preserved evidence", async () => {
    const choices = [
        { key: "A", label: "Wait", action: null, cost: 0 },
        ...[100, 101]
            .flatMap((actor) =>
                [200, 201].map((target) => ({
                    cost: 0,
                    label: `Attack visible target #${target} with unit #${actor}`,
                    action: {
                        function: 23,
                        unit_tags: [actor],
                        target_unit_tag: target,
                    },
                })),
            )
            .map((c, i) => ({ key: String.fromCharCode(66 + i), ...c })),
    ];
    const group = {
        id: "combat",
        instructions: "Choose a combat action",
        choices,
    };
    const packet = {
        tick: 0,
        parts,
        groups: [group],
        omitted: [],
        questions: {
            combat: {
                type: "choice",
                instructions: group.instructions,
                criteria: Object.fromEntries(
                    choices.map((c) => [c.key, c.label]),
                ),
            },
        },
    };
    const raw = new NativeClient(options("nanojev"), fakeSpawn("nanojev")),
        client = new PermutationClient(raw);
    try {
        const result = await predictStaged(packet, client);
        verifyStagedTrace(packet, result, assert);
        assert.equal(result.inferenceCalls, 4);
        assert.equal(result.resolvedAnswers.combat.choice, "B");
        assert(result.stages[1].previousChoices.length > 0);
        assert.equal(result.stages.at(-1).tick, 0);
        // Synthetic fixture, not a game: exercise the production audit CLI.
        const out = fs.mkdtempSync(path.join(os.tmpdir(), "ra2-native-audit-"));
        const manifest = {
            protocol: STAGED_PROTOCOL,
            decisionMode: "staged",
            candidateMode: "intent",
            decisionInterval: 3,
            cadenceTicks: 3,
            tick: 3,
            layaSeat: "red",
            modelFamily: "nanojev",
            runtime: await client.ready,
            inferenceCalls: result.inferenceCalls,
            inferenceCallUnit: "native_choice_request",
            maxActionsPerSeat: 16,
            stats: {
                red: { submitted: 1, accepted: 1 },
                blue: { submitted: 0, accepted: 0 },
            },
        };
        const selected = { ...choices[1], group: "combat" };
        const row = {
            ...result,
            tick: 0,
            seat: "red",
            decisionMode: "staged",
            contextParts: parts,
            candidates: packet.groups,
            leafQuestions: packet.questions,
            selected: [selected],
            dropped: [],
            omitted: [],
            normalUnits: [
                { tag: 100, alliance: 1 },
                { tag: 200, alliance: 4, display_type: 1, is_blip: false },
            ],
        };
        const write = (name, value) =>
            fs.writeFileSync(
                path.join(out, name),
                JSON.stringify(value) + "\n",
            );
        const audit = () =>
            spawnSync(
                process.execPath,
                [
                    fileURLToPath(
                        new URL(
                            "../integration/laya/audit.mjs",
                            import.meta.url,
                        ),
                    ),
                    out,
                    "--check-only",
                ],
                { encoding: "utf8" },
            );
        try {
            write("result.json", manifest);
            write("decisions.jsonl", row);
            write("actions.jsonl", {
                tick: 0,
                seat: "red",
                action: selected.action,
                result: { accepted: true },
            });
            const passed = audit();
            assert.equal(passed.status, 0, passed.stderr);
            assert.equal(
                JSON.parse(passed.stdout).latencyUnit,
                "native_choice_request",
            );
            write("result.json", { ...manifest, tick: 6 });
            assert.notEqual(audit().status, 0, "Missing cadence must fail");
            write("result.json", manifest);
            const upstream =
                row.stages[0].modelResult.upstream.states[0].answers;
            upstream[Object.keys(upstream)[0]].choice = "A";
            write("decisions.jsonl", row);
            assert.notEqual(
                audit().status,
                0,
                "Altered native result must fail",
            );
        } finally {
            fs.rmSync(out, { recursive: true, force: true });
        }
    } finally {
        client.close();
    }
});

test("native dispatch is explicit and leaves default Laya selection unchanged", async () => {
    const { decisionClientOptions } = await import(
        "../integration/models/factory.mjs"
    );
    assert.deepEqual(decisionClientOptions({}, "intent"), {
        family: "laya",
        gpu: "0",
        model: "typed-decisions",
    });
    const native = decisionClientOptions(
        { family: "nanojev", "source-root": "/pinned/source", gpu: "5" },
        "intent",
    );
    assert.equal(native.family, "nanojev");
    assert.equal(native.maxLength, 2048);
    assert.equal(native.sourceRoot, "/pinned/source");
    assert.throws(() => decisionClientOptions({ family: "unknown" }, "intent"));
    assert.throws(() =>
        decisionClientOptions(
            { family: "jev-omni", model: "english" },
            "intent",
        ),
    );
    assert.throws(() =>
        decisionClientOptions({ "native-max-length": "2048" }, "intent"),
    );
});
