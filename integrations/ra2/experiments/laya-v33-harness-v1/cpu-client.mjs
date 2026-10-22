import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import { homedir } from "node:os";

export const RELEASE_REVISION = "51d2f5a0b36c0a9921a450f14a5afb7cb6d30685";
export const WEIGHTS_SHA256 =
    "2ed46ac94c366db2dc68751d0eabfe945ceaeaf430e9fea1ec9e6c60c0207925";
export const SOURCE_REVISION = "970dc8c5f63d7b886a68409493f37d569424f933";

const object = (value) =>
    value !== null && typeof value === "object" && !Array.isArray(value);
const labPath = (...parts) =>
    resolve(process.env.RA2_LAB_DIR ?? resolve(homedir(), "ra2-lab"), ...parts);
function duration(value, name) {
    const n = Number(value);
    if (!Number.isSafeInteger(n) || n <= 0 || n > 2147483647)
        throw new Error(
            `${name} must be a positive integer up to 2147483647 ms`,
        );
    return n;
}
export function validateMetadata(metadata) {
    if (
        !object(metadata) ||
        metadata.device !== "cpu" ||
        metadata.releaseTransport !== true
    )
        throw new Error("Expected the release CPU worker metadata");
    if (
        metadata.repo !== "jayzou3773/laya-ra2-v33" ||
        metadata.weightsSha256 !== WEIGHTS_SHA256 ||
        metadata.sourceCommit !== SOURCE_REVISION
    )
        throw new Error(
            "Worker metadata does not identify the pinned Laya v33 weights/source",
        );
    return metadata;
}
export function validatePacket(packet) {
    if (
        !object(packet) ||
        !object(packet.parts) ||
        !object(packet.questions) ||
        !Object.keys(packet.questions).length
    )
        throw new Error("Expected nonempty parts/questions request");
    for (const section of ["essential", "details", "memory", "logs"])
        if (
            !Array.isArray(packet.parts[section]) ||
            packet.parts[section].some((value) => typeof value !== "string")
        )
            throw new Error(`Invalid context section ${section}`);
    for (const [id, question] of Object.entries(packet.questions)) {
        if (
            !object(question) ||
            question.type !== "choice" ||
            !object(question.criteria) ||
            !Object.keys(question.criteria).length ||
            Object.values(question.criteria).some((v) => typeof v !== "string")
        )
            throw new Error(`Invalid choice question ${id}`);
    }
}
export function validateResponse(response, packet) {
    if (
        !object(response.answers) ||
        Object.keys(response.answers).sort().join("\0") !==
            Object.keys(packet.questions).sort().join("\0")
    )
        throw new Error("Worker returned missing or unexpected answers");
    if (
        typeof response.stateText !== "string" ||
        !object(response.modelResult) ||
        !object(response.tokenAudit) ||
        !Array.isArray(response.trimmed) ||
        !Number.isFinite(response.latencyMs) ||
        response.latencyMs < 0
    )
        throw new Error("Invalid inference response schema");
    for (const [id, question] of Object.entries(packet.questions)) {
        const answer = response.answers[id],
            keys = Object.keys(question.criteria);
        if (
            !object(answer) ||
            answer.type !== "choice" ||
            !keys.includes(answer.choice) ||
            !object(answer.probabilities) ||
            Object.keys(answer.probabilities).sort().join("\0") !==
                [...keys].sort().join("\0")
        )
            throw new Error(
                `Invalid selected choice/probability keys for ${id}`,
            );
        const values = keys.map((key) => answer.probabilities[key]);
        const sum = values.reduce((total, value) => total + value, 0);
        if (
            values.some(
                (v) =>
                    typeof v !== "number" ||
                    !Number.isFinite(v) ||
                    v < 0 ||
                    v > 1,
            ) ||
            Math.abs(sum - 1) >= 0.005 ||
            sum <= 0 ||
            answer.probabilities[answer.choice] + 0.0001 < Math.max(...values)
        )
            throw new Error(
                `Invalid choice probability distribution for ${id}`,
            );
    }
    return response;
}

// Transport only: the frozen PermutationClient wraps this once, outside this class.
export class CpuLayaClient {
    constructor({
        releaseDir = process.env.LAYA_RELEASE_DIR ??
            labPath("laya-ra2-v33-51d2f5a"),
        python = process.env.LAYA_PYTHON ??
            labPath("laya-v33-cpu-venv/bin/python"),
        modelPath = process.env.LAYA_MODEL_PATH ?? releaseDir,
        startupTimeoutMs = process.env.LAYA_STARTUP_TIMEOUT_MS ?? 600000,
        requestTimeoutMs = process.env.LAYA_REQUEST_TIMEOUT_MS ?? 600000,
        shutdownTimeoutMs = 2000,
        spawnProcess = spawn,
    } = {}) {
        const root = resolve(releaseDir);
        if (resolve(modelPath) !== root)
            throw new Error(
                "LAYA_MODEL_PATH must equal LAYA_RELEASE_DIR for the frozen v33 comparison",
            );
        this.startupTimeoutMs = duration(startupTimeoutMs, "startup timeout");
        this.requestTimeoutMs = duration(requestTimeoutMs, "request timeout");
        this.shutdownTimeoutMs = duration(
            shutdownTimeoutMs,
            "shutdown timeout",
        );
        this.seq = 0;
        this.pending = new Map();
        this.closed = false;
        this.readySeen = false;
        this.terminated = false;
        this.ready = new Promise((resolveReady, rejectReady) => {
            this.readyResolve = resolveReady;
            this.readyReject = rejectReady;
        });
        // close() may be called before its consumer has attached a ready handler.
        this.ready.catch(() => {});
        this.exited = new Promise((resolveExit) => {
            this.resolveExit = resolveExit;
        });
        this.process = spawnProcess(
            python,
            ["-u", resolve(root, "ra2/integration/laya/worker.py")],
            {
                cwd: root,
                env: {
                    ...process.env,
                    LAYA_RELEASE_DIR: root,
                    LAYA_MODEL_PATH: root,
                    LAYA_SOURCE_DIR:
                        process.env.LAYA_SOURCE_DIR ?? labPath("laya-970dc8c"),
                    LAYA_DEVICE: "cpu",
                    USE_TF: "0",
                    TOKENIZERS_PARALLELISM: "false",
                    HF_HUB_OFFLINE: "1",
                    PYTHONDONTWRITEBYTECODE: "1",
                },
                stdio: ["pipe", "pipe", "inherit"],
            },
        );
        this.reader = createInterface({ input: this.process.stdout });
        this.reader.on("line", (line) => this.onLine(line));
        this.reader.on("error", (error) => this.fail(error));
        this.reader.on("close", () => {
            if (!this.closed)
                this.fail(
                    new Error("Laya CPU worker stdout closed unexpectedly"),
                );
        });
        this.process.stdout.on("error", (error) => this.fail(error));
        this.process.stdin.on("error", (error) => this.fail(error));
        this.process.on("error", (error) => this.fail(error));
        this.process.on("exit", (code, signal) => this.onExit(code, signal));
        this.process.on("close", (code, signal) => this.onExit(code, signal));
        this.startTimer = setTimeout(
            () => this.fail(new Error("Laya CPU startup timeout")),
            this.startupTimeoutMs,
        );
    }
    onLine(line) {
        if (this.closed) return;
        try {
            const response = JSON.parse(line);
            if (!object(response))
                throw new Error("Worker protocol requires a JSON object");
            if (Object.hasOwn(response, "error"))
                throw new Error(`Laya worker: ${String(response.error)}`);
            if (Object.hasOwn(response, "ready")) {
                if (
                    response.ready !== true ||
                    this.readySeen ||
                    Object.hasOwn(response, "id")
                )
                    throw new Error("Duplicate/invalid worker ready message");
                this.metadata = {
                    ...validateMetadata(response.metadata),
                    transport: {
                        kind: "laya-v33-cpu-stdio-v1",
                        releaseRevision: RELEASE_REVISION,
                        startupTimeoutMs: this.startupTimeoutMs,
                        requestTimeoutMs: this.requestTimeoutMs,
                    },
                };
                this.readySeen = true;
                clearTimeout(this.startTimer);
                this.readyResolve(this.metadata);
                return;
            }
            if (
                !this.readySeen ||
                !Number.isSafeInteger(response.id) ||
                !this.pending.has(response.id)
            )
                throw new Error(
                    `Unexpected worker response id ${String(response.id)}`,
                );
            const pending = this.pending.get(response.id);
            validateResponse(response, pending.packet);
            clearTimeout(pending.timer);
            this.pending.delete(response.id);
            pending.resolve(response);
        } catch (error) {
            this.fail(error);
        }
    }
    onExit(code, signal) {
        if (this.terminated) return;
        this.terminated = true;
        clearTimeout(this.killTimer);
        this.resolveExit({ code, signal });
        if (!this.closed)
            this.fail(new Error(`Laya CPU worker exited (${code ?? signal})`));
    }
    rejectOutstanding(error) {
        clearTimeout(this.startTimer);
        this.readyReject(error);
        for (const pending of this.pending.values()) {
            clearTimeout(pending.timer);
            pending.reject(error);
        }
        this.pending.clear();
    }
    fail(error) {
        if (this.closed) return;
        this.failure =
            error instanceof Error ? error : new Error(String(error));
        this.rejectOutstanding(this.failure);
        void this.close();
    }
    async predict(packet) {
        validatePacket(packet);
        await this.ready;
        if (this.closed)
            throw this.failure ?? new Error("Laya CPU worker closed");
        // Store a detached request: callers must not mutate validation data while pending.
        const request = structuredClone({
            parts: packet.parts,
            questions: packet.questions,
        });
        const id = ++this.seq;
        const response = await new Promise((resolveResponse, reject) => {
            const timer = setTimeout(
                () =>
                    this.fail(
                        new Error(`Laya CPU inference timeout (id ${id})`),
                    ),
                this.requestTimeoutMs,
            );
            this.pending.set(id, {
                resolve: resolveResponse,
                reject,
                timer,
                packet: request,
            });
            try {
                this.process.stdin.write(
                    JSON.stringify({ id, ...request }) + "\n",
                    (error) => {
                        if (error) this.fail(error);
                    },
                );
            } catch (error) {
                this.fail(error);
            }
        });
        if (this.failure) throw this.failure;
        return response;
    }
    close() {
        if (this.closePromise) return this.closePromise;
        this.closed = true;
        this.rejectOutstanding(
            this.failure ?? new Error("Laya CPU worker closed"),
        );
        this.closePromise = this.exited.then(() => {
            this.reader.close();
        });
        if (!this.terminated) {
            this.process.stdin.end();
            this.process.kill("SIGTERM");
            this.killTimer = setTimeout(() => {
                if (!this.terminated) this.process.kill("SIGKILL");
            }, this.shutdownTimeoutMs);
            this.killTimer.unref?.();
        }
        return this.closePromise;
    }
}
