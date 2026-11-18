import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { check } from "../scripts/check-v0-faults.mjs";
import { loadHarness } from "../scripts/match.mjs";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
const sha256 = (value) => createHash("sha256").update(value).digest("hex");

async function fixture(root) {
    const harnessDir = join(root, "harness");
    await mkdir(join(harnessDir, "tank"), { recursive: true });
    const settings = JSON.parse(
        await readFile(join(ROOT, "config/controller.json"), "utf8"),
    );
    settings.strategy = {
        id: "extreme-standoff-v1",
        description:
            "Keep far beyond the opponent's normal engagement distance.",
        spacingWeight: 1000,
        minRange: 1800,
        maxRange: 2200,
    };
    await writeFile(
        join(harnessDir, "tank/controller.json"),
        `${JSON.stringify(settings, null, 2)}\n`,
    );
    await writeFile(
        join(harnessDir, "tank/questions.json"),
        await readFile(join(ROOT, "config/questions.json")),
    );
    await writeFile(
        join(harnessDir, "index.json"),
        JSON.stringify([
            { path: "tank/controller.json", kind: "settings" },
            { path: "tank/questions.json", kind: "instructions" },
        ]),
    );
    const faultText = `${JSON.stringify({ layer: "controller", kind: "drop_fire_after_selection", family: "drop_fire_after_selection", instance: "drop_fire_after_selection-v1" }, null, 2)}\n`;
    const faultFile = join(root, "controller-fault.json");
    await writeFile(faultFile, faultText);
    const clean = await loadHarness();
    const changed = await loadHarness(harnessDir);
    const sources = {};
    for (const name of ["engine.js", "planner.js", "adapter.js"])
        sources[`web/game/${name}`] = sha256(
            await readFile(join(ROOT, "web/game", name)),
        );
    const controllerHash = (settingsHash, faultHash) =>
        sha256(
            JSON.stringify({
                engine: sources["web/game/engine.js"],
                planner: sources["web/game/planner.js"],
                adapter: sources["web/game/adapter.js"],
                settings: settingsHash,
                fault: faultHash,
            }),
        );
    const cleanSettings = clean.files.find(
        (file) => file.role === "controller",
    ).sha256;
    const changedSettings = changed.files.find(
        (file) => file.role === "controller",
    ).sha256;
    return {
        harnessDir,
        faultFile,
        faultText,
        freeze: {
            game: { sources_sha256: sources },
            clean_harness: {
                recorded_harness_sha256: sha256(JSON.stringify(clean.files)),
                recorded_controller_sha256: controllerHash(cleanSettings, null),
            },
            selected_faults: {
                harness: {
                    recorded_harness_sha256: sha256(
                        JSON.stringify(changed.files),
                    ),
                    recorded_controller_sha256: controllerHash(
                        changedSettings,
                        null,
                    ),
                },
                controller: {
                    builder_kind: "drop_fire_after_selection",
                    instance: "drop_fire_after_selection-v1",
                    recorded_controller_sha256: controllerHash(
                        cleanSettings,
                        sha256(faultText),
                    ),
                },
            },
        },
    };
}

test("altered harness controller pin is checked", async () => {
    const root = mkdtempSync(join(tmpdir(), "tank-v0-controller-"));
    try {
        const { freeze, harnessDir, faultText } = await fixture(root);
        await check(freeze, harnessDir, faultText);
        freeze.selected_faults.harness.recorded_controller_sha256 = "0".repeat(
            64,
        );
        await assert.rejects(
            check(freeze, harnessDir, faultText),
            /Harness controller hash differs/,
        );
    } finally {
        await rm(root, { recursive: true, force: true });
    }
});

test("edited checkpoint checker rejects changed non-weight files", async () => {
    const root = mkdtempSync(join(tmpdir(), "tank-v0-checkpoint-"));
    try {
        const { freeze, harnessDir, faultFile } = await fixture(root);
        const weightsRoot = join(root, "weights");
        const checkpoint = join(
            weightsRoot,
            "reverse_strength_3p0",
            "checkpoint",
        );
        await mkdir(checkpoint, { recursive: true });
        const weights = Buffer.from("synthetic checkpoint weights");
        await writeFile(join(checkpoint, "best.safetensors"), weights);
        const auxiliary = {};
        for (const name of [
            "config.json",
            "backbone_config/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/chat_template.jinja",
        ]) {
            await mkdir(resolve(checkpoint, name, ".."), { recursive: true });
            const content = Buffer.from(`frozen ${name}`);
            await writeFile(join(checkpoint, name), content);
            auxiliary[name] = sha256(content);
        }
        const digest = sha256(weights);
        await writeFile(
            join(weightsRoot, "reverse_strength_3p0", "checkpoint.json"),
            JSON.stringify({ weightsSha256: digest }),
        );
        freeze.selected_faults.weights = [
            { instance: "reverse_strength_3p0-v1", weights_sha256: digest },
        ];
        freeze.selected_faults.weights_builder_inputs = {
            checkpoint_aux_sha256: auxiliary,
        };
        const freezePath = join(root, "freeze.json");
        await writeFile(freezePath, JSON.stringify(freeze));
        const args = [
            join(ROOT, "scripts/check-v0-faults.mjs"),
            `--freeze=${freezePath}`,
            `--harness-dir=${harnessDir}`,
            `--controller-fault-file=${faultFile}`,
            `--weights-root=${weightsRoot}`,
        ];
        assert.equal(spawnSync(process.execPath, args).status, 0);
        await writeFile(
            join(checkpoint, "tokenizer/tokenizer.json"),
            "changed",
        );
        const result = spawnSync(process.execPath, args);
        assert.notEqual(result.status, 0);
        assert.match(
            result.stderr.toString(),
            /Checkpoint non-weight file differs/,
        );
        await writeFile(
            join(checkpoint, "tokenizer/tokenizer.json"),
            "frozen tokenizer/tokenizer.json",
        );
        await writeFile(join(checkpoint, "config.json"), "changed");
        const changedConfig = spawnSync(process.execPath, args);
        assert.notEqual(changedConfig.status, 0);
        assert.match(
            changedConfig.stderr.toString(),
            /Checkpoint non-weight file differs/,
        );
    } finally {
        await rm(root, { recursive: true, force: true });
    }
});
