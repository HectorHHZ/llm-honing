import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { ROOT } from "../paths.mjs";
// Each family runs in its own model environment, and its worker reads its own settings.
const PYTHON_VARIABLE = {
    laya: "LAYA_PYTHON",
    kev: "KEV_PYTHON",
    nanojev: "NANOJEV_PYTHON",
    startlux: "STARTLUX_PYTHON",
};
const SETTING_PREFIX = {
    laya: "LAYA",
    kev: "KEV",
    nanojev: "NANOJEV",
    startlux: "STARTLUX",
};
export class LayaClient {
    constructor({
        family = "laya",
        gpu = "0",
        model = "english",
        python = process.env[PYTHON_VARIABLE[family]] ?? "python3",
        checkpointManifest = undefined,
        spawnProcess = spawn,
    } = {}) {
        this.seq = 0;
        this.pending = new Map();
        this.closed = false;
        this.ready = new Promise((resolve, reject) => {
            this.readyResolve = resolve;
            this.readyReject = reject;
        });
        if (!Object.hasOwn(SETTING_PREFIX, family))
            throw new Error(`Unknown decision model family ${family}`);
        const prefix = SETTING_PREFIX[family];
        const manifestVar = `${prefix}_CHECKPOINT_MANIFEST`;
        const workerEnv = {
            ...process.env,
            DECISION_FAMILY: family,
            LAYA_GPU: String(gpu),
            [family === "laya" ? "LAYA_MODEL_TYPE" : `${prefix}_MODEL`]: model,
        };
        if (checkpointManifest === null) delete workerEnv[manifestVar];
        else if (checkpointManifest !== undefined) {
            if (typeof checkpointManifest !== "string" || !checkpointManifest)
                throw new Error("Invalid per-player checkpoint manifest");
            workerEnv[manifestVar] = checkpointManifest;
        }
        this.process = spawnProcess(
            python,
            ["-u", ROOT + "integration/laya/worker.py"],
            {
                cwd: ROOT,
                env: workerEnv,
                stdio: ["pipe", "pipe", "inherit"],
            },
        );
        createInterface({ input: this.process.stdout }).on("line", (line) => {
            try {
                const r = JSON.parse(line);
                if (r.error) throw new Error(r.error);
                if (r.ready) {
                    this.metadata = r.metadata;
                    this.readyResolve(r.metadata);
                    return;
                }
                const p = this.pending.get(r.id);
                if (p) {
                    clearTimeout(p.timer);
                    this.pending.delete(r.id);
                    p.resolve(r);
                }
            } catch (e) {
                this.fail(e);
            }
        });
        this.process.on("error", (e) => this.fail(e));
        this.process.on("exit", (code) => {
            if (!this.closed)
                this.fail(new Error(`Laya worker exited ${code}`));
        });
        this.startTimer = setTimeout(
            () => this.fail(new Error("Laya startup timeout")),
            180000,
        );
        this.ready.then(
            () => clearTimeout(this.startTimer),
            () => clearTimeout(this.startTimer),
        );
    }
    fail(e) {
        this.readyReject(e);
        for (const p of this.pending.values()) {
            clearTimeout(p.timer);
            p.reject(e);
        }
        this.pending.clear();
        this.close();
    }
    async predict(packet) {
        await this.ready;
        if (this.closed) throw new Error("Laya worker closed");
        const id = ++this.seq;
        return new Promise((resolve, reject) => {
            const timer = setTimeout(
                () => this.fail(new Error("Laya inference timeout")),
                60000,
            );
            this.pending.set(id, { resolve, reject, timer });
            this.process.stdin.write(
                JSON.stringify({
                    id,
                    parts: packet.parts,
                    questions: packet.questions,
                }) + "\n",
            );
        });
    }
    close() {
        this.closed = true;
        clearTimeout(this.startTimer);
        this.process.kill("SIGTERM");
    }
}
