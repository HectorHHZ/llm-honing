import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
    mkdtempSync,
    mkdirSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { CpuLayaClient, WEIGHTS_SHA256 } from "../cpu-client.mjs";
import {
    readReleaseEntries,
    verifyRelease,
    verifyReleaseFiles,
} from "../release-integrity.mjs";
import { snapshotSources } from "../runtime.mjs";

const digest = (text) => createHash("sha256").update(text).digest("hex");
function withDirectory(fn) {
    const root = mkdtempSync(join(tmpdir(), "v33-migration-"));
    try {
        return fn(root);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}
function put(root, name, text = "fixture") {
    const path = join(root, name);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, text);
}

test("release manifest paths reject escape, duplicates and a changed weight pin", () => {
    const weights = `${WEIGHTS_SHA256}  model.safetensors\n`;
    const valid =
        weights + `${digest("code")}  ra2/integration/laya/policy.mjs\n`;
    assert.equal(readReleaseEntries(valid).size, 2);
    for (const path of [
        "../escape",
        "/absolute",
        "a/../b",
        "a\\b",
        "a//b",
        "./file",
    ])
        assert.throws(
            () => readReleaseEntries(weights + `${digest("a")}  ${path}\n`),
            /Unsafe/,
        );
    assert.throws(() => readReleaseEntries(valid + weights), /duplicate/);
    assert.throws(
        () => readReleaseEntries(`${digest("changed")}  model.safetensors\n`),
        /expected v33 weights/,
    );
    assert.throws(() => readReleaseEntries("malformed"), /checksum/);
});

test("frozen config and tokenizer drift fail without loading weights or a model", () =>
    withDirectory((root) => {
        const manifest = `${WEIGHTS_SHA256}  model.safetensors\n${digest(
            "config",
        )}  rl_agent_config.json\n${digest(
            "tokenizer",
        )}  tokenizer/tokenizer.json\n`;
        put(root, "rl_agent_config.json", "config");
        put(root, "tokenizer/tokenizer.json", "tokenizer");
        const entries = readReleaseEntries(manifest);
        // Intentionally no weights: this validation must not touch them.
        assert.equal(verifyReleaseFiles(root, entries), 2);
        put(root, "tokenizer/tokenizer.json", "changed tokenizer");
        assert.throws(
            () => verifyReleaseFiles(root, entries),
            /tokenizer\/tokenizer.json/,
        );
        put(root, "tokenizer/tokenizer.json", "tokenizer");
        put(root, "rl_agent_config.json", "changed config");
        assert.throws(
            () => verifyReleaseFiles(root, entries),
            /rl_agent_config.json/,
        );
    }));

test("unrecognized manifest is rejected before importing release code", () =>
    withDirectory((root) => {
        put(root, "SHA256SUMS", `${WEIGHTS_SHA256}  model.safetensors\n`);
        assert.throws(() => verifyRelease(root), /SHA256SUMS does not match/);
    }));

test("another model directory is rejected before spawning any worker", () => {
    let spawned = false;
    assert.throws(
        () =>
            new CpuLayaClient({
                releaseDir: "/pinned-release",
                modelPath: "/other-tokenizer",
                spawnProcess: () => {
                    spawned = true;
                },
            }),
        /LAYA_MODEL_PATH must equal/,
    );
    assert.equal(spawned, false);
});

test("snapshots include setup and model configuration but exclude weights, secrets and prior runs", () =>
    withDirectory((root) => {
        const release = join(root, "release"),
            sdk = join(root, "sdk"),
            supalosa = join(root, "supalosa"),
            out = join(root, "out");
        for (const name of [
            "ra2/integration/laya/intent-policy.mjs",
            "ra2/pins.json",
            "ra2/README.md",
            "rl_agent_config.json",
            "tokenizer/tokenizer.json",
            "encoder/config.json",
            "model.safetensors",
            "runs/previous/match.rplx",
            "checks/old.json",
            ".env",
        ])
            put(release, name);
        for (const name of ["package.json", "dist/index.js"]) put(sdk, name);
        for (const name of [
            "package-lock.json",
            "packages/chronodivide-bot/src/bot.ts",
            "packages/chronodivide-bot/src/.env",
        ])
            put(supalosa, name);
        const sources = snapshotSources(out, {
            release,
            integration: join(release, "ra2"),
            SDK_DIR: sdk,
            SUPALOSA_DIR: supalosa,
        });
        const paths = sources.map((item) => item.snapshot);
        for (const path of [
            "experiment/env.sh",
            "experiment/.env.example",
            "experiment/setup-linux.sh",
            "experiment/download-release.py",
            "experiment/release-integrity.mjs",
            "release/rl_agent_config.json",
            "release/tokenizer/tokenizer.json",
            "release/encoder/config.json",
        ])
            assert.ok(paths.includes(`source/${path}`), path);
        assert.ok(
            paths.every(
                (path) =>
                    !/model\.safetensors|\/runs\/|\/checks\/|\/\.env$/.test(
                        path,
                    ),
            ),
        );
        for (const item of sources)
            assert.equal(
                digest(readFileSync(join(out, item.snapshot))),
                item.sha256,
            );
    }));
