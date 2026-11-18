// Check that the v0 builders reproduce the frozen harness, controller and checkpoint identities.
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { loadHarness } from "./match.mjs";

const sha256 = (value) => createHash("sha256").update(value).digest("hex");

async function sha256File(path) {
    const digest = createHash("sha256");
    for await (const chunk of createReadStream(path)) digest.update(chunk);
    return digest.digest("hex");
}

async function checkpointFiles(root, prefix = "") {
    const files = [];
    for (const entry of await readdir(resolve(root, prefix), {
        withFileTypes: true,
    })) {
        const name = prefix ? `${prefix}/${entry.name}` : entry.name;
        if (entry.isSymbolicLink())
            throw Error(`Copied checkpoint contains a symlink: ${name}`);
        if (entry.isDirectory())
            files.push(...(await checkpointFiles(root, name)));
        else if (entry.isFile()) files.push(name);
        else throw Error(`Unsupported checkpoint entry: ${name}`);
    }
    return files.sort();
}

function options(argv) {
    const result = {};
    for (const item of argv) {
        const match = /^--([a-z-]+)=(.+)$/.exec(item);
        if (!match) throw Error(`Invalid option ${item}`);
        result[match[1]] = resolve(match[2]);
    }
    for (const name of ["freeze", "harness-dir", "controller-fault-file"])
        if (!result[name]) throw Error(`Missing --${name}`);
    return result;
}

export async function check(freeze, harnessDir, controllerFaultText) {
    const clean = await loadHarness();
    const changed = await loadHarness(harnessDir);
    const harnessHash = (harness) => sha256(JSON.stringify(harness.files));
    if (harnessHash(clean) !== freeze.clean_harness.recorded_harness_sha256)
        throw Error("Clean harness hash differs from frozen sample");
    if (
        harnessHash(changed) !==
        freeze.selected_faults.harness.recorded_harness_sha256
    )
        throw Error("Harness builder output differs from frozen sample");
    const fault = JSON.parse(controllerFaultText);
    if (
        fault.layer !== "controller" ||
        fault.kind !== freeze.selected_faults.controller.builder_kind ||
        fault.instance !== freeze.selected_faults.controller.instance
    )
        throw Error("Controller builder output has the wrong fault identity");
    const sources = freeze.game.sources_sha256;
    const cleanSettings = clean.files.find(
        (file) => file.role === "controller",
    )?.sha256;
    const changedSettings = changed.files.find(
        (file) => file.role === "controller",
    )?.sha256;
    const controllerHash = (settings, faultHash) =>
        sha256(
            JSON.stringify({
                engine: sources["web/game/engine.js"],
                planner: sources["web/game/planner.js"],
                adapter: sources["web/game/adapter.js"],
                settings,
                fault: faultHash,
            }),
        );
    if (
        controllerHash(cleanSettings, null) !==
        freeze.clean_harness.recorded_controller_sha256
    )
        throw Error("Clean controller hash differs from frozen sample");
    if (
        controllerHash(changedSettings, null) !==
        freeze.selected_faults.harness.recorded_controller_sha256
    )
        throw Error("Harness controller hash differs from frozen sample");
    if (
        controllerHash(cleanSettings, sha256(controllerFaultText)) !==
        freeze.selected_faults.controller.recorded_controller_sha256
    )
        throw Error("Controller builder output differs from frozen sample");
    return {
        clean_harness_sha256: harnessHash(clean),
        fault_harness_sha256: harnessHash(changed),
        clean_controller_sha256: controllerHash(cleanSettings, null),
        harness_controller_sha256: controllerHash(changedSettings, null),
        fault_controller_sha256: controllerHash(
            cleanSettings,
            sha256(controllerFaultText),
        ),
    };
}

if (
    process.argv[1] &&
    import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
    const args = options(process.argv.slice(2));
    const freeze = JSON.parse(await readFile(args.freeze, "utf8"));
    const controllerFault = await readFile(
        args["controller-fault-file"],
        "utf8",
    );
    const result = await check(freeze, args["harness-dir"], controllerFault);
    if (args["weights-root"]) {
        const auxiliary =
            freeze.selected_faults.weights_builder_inputs.checkpoint_aux_sha256;
        const expectedAuxiliary = [
            "config.json",
            "backbone_config/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/chat_template.jinja",
        ];
        if (
            JSON.stringify(Object.keys(auxiliary).sort()) !==
            JSON.stringify(expectedAuxiliary.sort())
        )
            throw Error("Freeze omits non-weight inference files");
        result.edited_checkpoints_verified = 0;
        for (const item of freeze.selected_faults.weights) {
            const name = item.instance.replace(/-v1$/, "");
            const checkpoint = resolve(
                args["weights-root"],
                name,
                "checkpoint",
            );
            const weights = resolve(checkpoint, "best.safetensors");
            const manifest = JSON.parse(
                await readFile(
                    resolve(args["weights-root"], name, "checkpoint.json"),
                    "utf8",
                ),
            );
            if (
                (await sha256File(weights)) !== item.weights_sha256 ||
                manifest.weightsSha256 !== item.weights_sha256
            )
                throw Error(
                    `Edited checkpoint differs from frozen sample: ${name}`,
                );
            if (
                JSON.stringify(await checkpointFiles(checkpoint)) !==
                JSON.stringify(
                    ["best.safetensors", ...expectedAuxiliary].sort(),
                )
            )
                throw Error(
                    `Checkpoint file list differs from freeze: ${name}`,
                );
            for (const [file, digest] of Object.entries(auxiliary))
                if ((await sha256File(resolve(checkpoint, file))) !== digest)
                    throw Error(
                        `Checkpoint non-weight file differs from freeze: ${name}/${file}`,
                    );
            result.edited_checkpoints_verified++;
        }
    }
    console.log(JSON.stringify(result));
}
