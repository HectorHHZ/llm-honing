import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { realpathSync } from "node:fs";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { ROOT } from "../paths.mjs";

export const NATIVE_PROTOCOL = "native-choice-jsonl-v1";
const families = new Set(["nanojev", "jev-omni"]);
const sections = ["essential", "details", "memory", "logs"];
const hash = (value) =>
    typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const keysEqual = (left, right) =>
    assert.deepEqual(Object.keys(left).sort(), Object.keys(right).sort());

export function verifyNativeMetadata(metadata, family, maxLength) {
    assert(families.has(family));
    assert.equal(metadata.family, family);
    assert.equal(metadata.protocol, NATIVE_PROTOCOL);
    assert.equal(metadata.maxLen, maxLength);
    assert.equal(metadata.modelWeightsLoaded, true);
    assert(hash(metadata.weightSetSha256));
    assert(hash(metadata.preparedManifestSha256));
    assert.equal(metadata.parameterStorage, "float32");
    assert.equal(metadata.forwardAutocast, "bfloat16");
    assert(Object.keys(metadata.verifiedFilesSha256).length > 0);
    assert(Object.values(metadata.verifiedFilesSha256).every(hash));
    if (metadata.learnedHead !== undefined) {
        assert.equal(metadata.checkpointKind, "trained-native-decision-head");
        const head = metadata.learnedHead,
            base = metadata.baseModel;
        assert(hash(base?.weightSetSha256));
        assert(
            hash(head.headSha256) &&
                hash(head.manifestSha256) &&
                hash(head.independentVerificationSha256),
        );
        assert.equal(head.trainingScope, "native-head-only");
        assert(
            Number.isInteger(head.actualOptimizerSteps) &&
                head.actualOptimizerSteps > 0,
        );
        assert(
            typeof head.manifestPath === "string" &&
                head.manifestPath.length > 0,
        );
        const expected = createHash("sha256")
            .update(
                JSON.stringify({
                    baseWeightSetSha256: base.weightSetSha256,
                    nativeHeadSha256: head.headSha256,
                }),
            )
            .digest("hex");
        assert.equal(metadata.weightSetSha256, expected);
        assert.equal(metadata.revision, expected);
        assert.equal(metadata.repo, "local:ra2/trained-native-decision-head");
    } else {
        assert(
            [undefined, "prepared-upstream-model"].includes(
                metadata.checkpointKind,
            ),
        );
        assert.equal(metadata.baseModel, undefined);
    }
}

export function verifyNativePrediction(packet, response, metadata) {
    const { questions, parts } = packet;
    assert(families.has(metadata.family));
    assert(Number.isFinite(response.latencyMs) && response.latencyMs >= 0);
    keysEqual(response.answers, questions);
    keysEqual(response.tokenAudit, questions);
    const retained = Object.fromEntries(
        sections.map((s) => [s, [...(parts[s] ?? [])]]),
    );
    assert(retained.essential.length > 0);
    assert(Array.isArray(response.trimmed));
    for (const removed of response.trimmed) {
        const section = ["logs", "memory", "details"].find(
            (s) => retained[s].length,
        );
        assert.equal(
            removed.section,
            section,
            "Native trimming changed essential context or pruning order",
        );
        const text =
            section === "logs"
                ? retained[section].shift()
                : retained[section].pop();
        assert.equal(removed.text, text);
    }
    assert.equal(
        response.stateText,
        sections.flatMap((s) => retained[s]).join("\n"),
    );
    const raw = response.modelResult;
    let upstream;
    if (metadata.family === "nanojev") {
        assert.equal(raw.states?.length, 1);
        assert.equal(raw.states[0].id, "decision");
        upstream = raw.states[0].answers;
    } else {
        assert.equal(raw.family, "jev-omni");
        upstream = raw.answers;
    }
    keysEqual(upstream, questions);
    for (const [id, question] of Object.entries(questions)) {
        assert.equal(question.type, "choice");
        const answer = response.answers[id],
            keys = Object.keys(question.criteria);
        assert(keys.length >= 2 && keys.length <= 255);
        keysEqual(answer.probabilities, question.criteria);
        const probabilities = keys.map((key) => answer.probabilities[key]);
        assert(
            probabilities.every(
                (p) =>
                    typeof p === "number" &&
                    Number.isFinite(p) &&
                    p >= 0 &&
                    p <= 1,
            ),
        );
        assert(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) <= 1e-5);
        const best = keys.reduce(
            (best, key) =>
                answer.probabilities[key] > answer.probabilities[best]
                    ? key
                    : best,
            keys[0],
        );
        assert.equal(answer.choice, best);
        assert.equal(upstream[id].choice, answer.choice);
        assert.deepEqual(upstream[id].probabilities, answer.probabilities);
        if (metadata.family === "nanojev")
            assert.equal(upstream[id].type, "choice");
        const audit = response.tokenAudit[id];
        assert.equal(audit.options, keys.length);
        assert.equal(audit.stateTruncated, false);
        assert.equal(audit.maxInputTokens, metadata.maxLen);
        assert(
            Number.isInteger(audit.inputTokens) &&
                audit.inputTokens > 0 &&
                audit.inputTokens <= metadata.maxLen,
        );
        assert(
            Number.isInteger(audit.stateTokens) &&
                audit.stateTokens >= 0 &&
                audit.stateTokens <= audit.inputTokens,
        );
        assert.equal(
            audit.stateRoom,
            metadata.maxLen - (audit.inputTokens - audit.stateTokens),
        );
        if (metadata.family === "nanojev") {
            assert.equal(audit.encoding, "nanojev-native-candidate-paths");
            assert.equal(audit.candidatePathSha256?.length, keys.length);
            assert(audit.candidatePathSha256.every(hash));
        } else {
            assert.equal(audit.encoding, "jev-omni-native-numbered-options");
            assert(hash(audit.promptSha256));
        }
    }
}

// The common record preserves upstream evidence while exposing the staged interface.
// This maps fields only; native probabilities and the chosen option never change.
export function bridgeNativePrediction(packet, response, metadata) {
    verifyNativePrediction(packet, response, metadata);
    return {
        ...response,
        modelResult: {
            protocol: NATIVE_PROTOCOL,
            family: metadata.family,
            answers: structuredClone(response.answers),
            upstream: response.modelResult,
        },
    };
}

export function verifyNativeRecord(packet, record, metadata) {
    assert.equal(record.modelResult.protocol, NATIVE_PROTOCOL);
    assert.equal(record.modelResult.family, metadata.family);
    assert.deepEqual(record.modelResult.answers, record.answers);
    verifyNativePrediction(
        packet,
        { ...record, modelResult: record.modelResult.upstream },
        metadata,
    );
}

export class NativeClient {
    constructor(
        {
            family,
            preparedManifest,
            headManifest,
            sourceRoot,
            gpu = "0",
            maxLength = 2048,
            python = process.env.RA2_NATIVE_PYTHON ??
                process.env.LAYA_PYTHON ??
                "python3",
            startupTimeoutMs = 600000,
            inferenceTimeoutMs = 60000,
        },
        spawnProcess = spawn,
    ) {
        if (
            !families.has(family) ||
            !preparedManifest ||
            (family === "nanojev" && !sourceRoot)
        )
            throw new Error(
                "Native family, prepared manifest and NanoJev source root are required",
            );
        if (!Number.isInteger(maxLength) || maxLength <= 0)
            throw new Error("Invalid native input limit");
        this.headManifest = headManifest
            ? realpathSync(headManifest)
            : undefined;
        this.seq = 0;
        this.lastResponse = 0;
        this.pending = new Map();
        this.closed = false;
        this.inferenceTimeoutMs = inferenceTimeoutMs;
        this.ready = new Promise((resolve, reject) => {
            this.resolveReady = resolve;
            this.rejectReady = reject;
        });
        const command = [
            "-u",
            "-m",
            "integration.models.worker",
            "--family",
            family,
            "--prepared-manifest",
            preparedManifest,
            "--gpu",
            String(gpu),
            "--max-length",
            String(maxLength),
        ];
        if (sourceRoot) command.push("--source-root", sourceRoot);
        if (this.headManifest)
            command.push("--head-manifest", this.headManifest);
        this.process = spawnProcess(python, command, {
            cwd: ROOT,
            env: process.env,
            stdio: ["pipe", "pipe", "inherit"],
        });
        this.startTimer = setTimeout(
            () => this.fail(new Error("Native worker startup timeout")),
            startupTimeoutMs,
        );
        this.ready.then(
            () => clearTimeout(this.startTimer),
            () => clearTimeout(this.startTimer),
        );
        this.process.on("error", (error) => this.fail(error));
        this.process.stdin.on("error", (error) => this.fail(error));
        this.process.on("exit", (code) => {
            if (!this.closed)
                this.fail(new Error(`Native worker exited ${code}`));
        });
        createInterface({ input: this.process.stdout }).on("line", (line) => {
            if (this.closed) return;
            try {
                const response = JSON.parse(line);
                if (response.error) throw new Error(response.error);
                if (response.ready) {
                    assert(
                        !this.metadata,
                        "Native worker sent a second ready record",
                    );
                    verifyNativeMetadata(response.metadata, family, maxLength);
                    assert.equal(
                        Boolean(response.metadata.learnedHead),
                        Boolean(this.headManifest),
                        "Native worker loaded a different checkpoint kind",
                    );
                    if (this.headManifest)
                        assert.equal(
                            response.metadata.learnedHead.manifestPath,
                            this.headManifest,
                            "Native worker loaded another learned head",
                        );
                    this.metadata = response.metadata;
                    this.resolveReady(this.metadata);
                    return;
                }
                assert(
                    this.metadata,
                    "Native worker answered before loading weights",
                );
                assert(
                    Number.isInteger(response.id) &&
                        response.id > this.lastResponse,
                );
                const request = this.pending.get(response.id);
                assert(request, "Native worker returned an unknown request ID");
                const bridged = bridgeNativePrediction(
                    request.packet,
                    response,
                    this.metadata,
                );
                this.lastResponse = response.id;
                clearTimeout(request.timer);
                this.pending.delete(response.id);
                request.resolve(bridged);
            } catch (error) {
                this.fail(error);
            }
        });
    }
    async predict(packet) {
        await this.ready;
        if (this.closed) throw new Error("Native worker is closed");
        const id = ++this.seq;
        const request = structuredClone({
            parts: packet.parts,
            questions: packet.questions,
        });
        return new Promise((resolve, reject) => {
            const timer = setTimeout(
                () => this.fail(new Error("Native inference timeout")),
                this.inferenceTimeoutMs,
            );
            this.pending.set(id, { packet: request, resolve, reject, timer });
            this.process.stdin.write(JSON.stringify({ id, ...request }) + "\n");
        });
    }
    fail(error) {
        if (this.closed) return;
        this.closed = true;
        clearTimeout(this.startTimer);
        this.rejectReady(error);
        for (const pending of this.pending.values()) {
            clearTimeout(pending.timer);
            pending.reject(error);
        }
        this.pending.clear();
        this.process.kill("SIGTERM");
    }
    close() {
        this.fail(new Error("Native worker closed"));
    }
}
