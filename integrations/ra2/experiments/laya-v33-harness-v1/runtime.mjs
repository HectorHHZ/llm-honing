#!/usr/bin/env node
// Explicit opt-in runner. Imports and --help never initialize a game or model.
import {
    accessSync,
    constants,
    copyFileSync,
    existsSync,
    mkdirSync,
    readFileSync,
    readdirSync,
    statSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { verifyRelease } from "./release-integrity.mjs";

export const EXPERIMENT_DIR = dirname(fileURLToPath(import.meta.url));
export const RUNS_DIR = join(EXPERIMENT_DIR, "runs");
export const HF_MODEL = "jayzou3773/laya-ra2-v33";
export const HF_REVISION = "51d2f5a0b36c0a9921a450f14a5afb7cb6d30685";
export const CADENCE = 3;
export const ACTION_CAP = 16;
const sha256 = (value) => createHash("sha256").update(value).digest("hex");

export function ownSeatDetails(players, seat) {
    return players.map((p) =>
        p.name === seat ? { ...p } : { name: p.name, player_id: p.player_id },
    );
}

export function unpackReplay(saved) {
    if (
        !saved ||
        typeof saved !== "object" ||
        typeof saved.rplx !== "string" ||
        !saved.rplx.trim()
    )
        throw new Error(
            "Expected SDK 0.1.1 SavedReplay object with nonempty rplx text",
        );
    if (
        !["final", "truncated", "partial"].includes(saved.status) ||
        typeof saved.end_reason !== "string" ||
        !Array.isArray(saved.lost_data) ||
        !saved.lost_data.every((x) => typeof x === "string") ||
        !Number.isInteger(saved.retargeted) ||
        saved.retargeted < 0
    )
        throw new Error("Invalid SavedReplay metadata");
    const { rplx, status, end_reason, lost_data, retargeted } = saved;
    return {
        rplx,
        metadata: { status, end_reason, lost_data, retargeted },
        valid: status !== "partial" && lost_data.length === 0,
    };
}

export function reserveOutput(name, runsDir = RUNS_DIR) {
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(name))
        throw new Error("Invalid output directory name");
    const out = join(resolve(runsDir), name);
    if (existsSync(out))
        throw new Error(`Refusing to overwrite output: ${out}`);
    mkdirSync(runsDir, { recursive: true });
    mkdirSync(out); // Atomic refusal if another process already created it.
    return out;
}

export async function loadDependencies() {
    if (!process.env.LAYA_RELEASE_DIR)
        throw new Error("LAYA_RELEASE_DIR is required; use run.sh");
    const release = resolve(process.env.LAYA_RELEASE_DIR),
        integration = join(release, "ra2");
    const releaseIntegrity = verifyRelease(release);
    const load = (name) => import(pathToFileURL(join(integration, name)).href);
    const [
        metadata,
        intent,
        staged,
        dependencies,
        paths,
        ensemble,
        log,
        transport,
    ] = await Promise.all([
        load("integration/metadata.mjs"),
        load("integration/laya/intent-policy.mjs"),
        load("integration/laya/staged.mjs"),
        load("integration/dependencies.mjs"),
        load("integration/paths.mjs"),
        load("integration/laya/permutation-client.mjs"),
        load("integration/laya/decision-log.mjs"),
        import("./cpu-client.mjs"),
    ]);
    return {
        release,
        integration,
        releaseIntegrity,
        loadMetadata: metadata.loadMetadata,
        IntentPolicy: intent.IntentPolicy,
        predictStaged: staged.predictStaged,
        STAGED_PROTOCOL: staged.STAGED_PROTOCOL,
        MAX_STAGES: staged.MAX_STAGES,
        Ra2Engine: dependencies.Ra2Engine,
        GAME_DIR: paths.GAME_DIR,
        SDK_DIR: paths.SDK_DIR,
        SUPALOSA_DIR: paths.SUPALOSA_DIR,
        PINS: paths.PINS,
        PermutationClient: ensemble.PermutationClient,
        CompressedDecisionLog: log.CompressedDecisionLog,
        CpuLayaClient: transport.CpuLayaClient,
    };
}

export function checkFiles(deps) {
    if (Number(process.versions.node.split(".")[0]) < 22)
        throw new Error("Node.js 22 or newer is required");
    for (const name of ["ra2.mix", "language.mix"]) {
        const file = join(deps.GAME_DIR, name);
        accessSync(file, constants.R_OK);
        if (!statSync(file).isFile() || !statSync(file).size)
            throw new Error(`Empty or missing asset: ${file}`);
    }
    const binary = resolve(
        process.env.RA2_ENGINE_BIN ??
            join(
                deps.SDK_DIR,
                "bin",
                `${process.platform}-${process.arch}`,
                process.platform === "win32" ? "ra2-engine.exe" : "ra2-engine",
            ),
    );
    accessSync(binary, constants.X_OK);
    if (typeof deps.Ra2Engine.start !== "function")
        throw new Error("Invalid engine SDK");
    return {
        node: process.versions.node,
        platform: `${process.platform}-${process.arch}`,
        binary,
        gameDir: deps.GAME_DIR,
        sdkDir: deps.SDK_DIR,
        releaseDir: deps.release,
        supalosaDir: deps.SUPALOSA_DIR,
        releaseIntegrity: deps.releaseIntegrity,
    };
}

export function snapshotSources(out, deps) {
    const files = [];
    const collect = (dir, prefix, accept = () => true) => {
        for (const item of readdirSync(dir, { withFileTypes: true })) {
            if (item.name === "__pycache__") continue;
            const source = join(dir, item.name),
                target = `${prefix}/${item.name}`;
            if (item.isDirectory()) collect(source, target, accept);
            else if (item.isFile() && accept(item.name))
                files.push([source, target]);
        }
    };
    // Only code/config, never model weights or prior runs.
    for (const name of [
        "run.mjs",
        "runtime.mjs",
        "release-integrity.mjs",
        "cpu-client.mjs",
        "check.mjs",
        "report.mjs",
        "run.sh",
        "env.sh",
        ".env.example",
        "setup-linux.sh",
        "download-release.py",
        "README.md",
    ])
        if (existsSync(join(EXPERIMENT_DIR, name)))
            files.push([join(EXPERIMENT_DIR, name), `experiment/${name}`]);
    collect(join(EXPERIMENT_DIR, "harness"), "experiment/harness");
    collect(
        join(deps.integration, "integration"),
        "release/ra2/integration",
        (n) => /\.(mjs|js|json|py|md)$/.test(n),
    );
    for (const name of ["pins.json", "README.md"])
        files.push([join(deps.integration, name), `release/ra2/${name}`]);
    for (const name of [
        "README.md",
        "config.json",
        "rl_agent_config.json",
        "release.json",
        "inference.py",
        "requirements.txt",
        "model_info.json",
        "SHA256SUMS",
        ".snapshot.json",
    ])
        if (existsSync(join(deps.release, name)))
            files.push([join(deps.release, name), `release/${name}`]);
    for (const name of ["encoder", "tokenizer"])
        collect(join(deps.release, name), `release/${name}`, (name) =>
            /\.json$/.test(name),
        );
    collect(
        join(deps.SUPALOSA_DIR, "packages/chronodivide-bot/src"),
        "supalosa/src",
        (name) => /\.(ts|js|mjs|json)$/.test(name),
    );
    files.push(
        [
            join(deps.SUPALOSA_DIR, "package-lock.json"),
            "supalosa/package-lock.json",
        ],
        [join(deps.SDK_DIR, "package.json"), "sdk/package.json"],
        [join(deps.SDK_DIR, "dist/index.js"), "sdk/index.js"],
    );
    return files.map(([source, target]) => {
        const file = join(out, "source", target);
        mkdirSync(dirname(file), { recursive: true });
        copyFileSync(source, file);
        return {
            original: source,
            snapshot: `source/${target}`,
            sha256: sha256(readFileSync(file)),
        };
    });
}
