// Check that the v0 builders reproduce the frozen harness/controller identities.
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { loadHarness } from "./match.mjs";

const sha256 = (value) => createHash("sha256").update(value).digest("hex");

async function sha256File(path) {
    const digest = createHash("sha256");
    for await (const chunk of createReadStream(path)) digest.update(chunk);
    return digest.digest("hex");
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
    const settings = clean.files.find(
        (file) => file.role === "controller",
    )?.sha256;
    const controllerHash = (faultHash) =>
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
        controllerHash(null) !== freeze.clean_harness.recorded_controller_sha256
    )
        throw Error("Clean controller hash differs from frozen sample");
    if (
        controllerHash(sha256(controllerFaultText)) !==
        freeze.selected_faults.controller.recorded_controller_sha256
    )
        throw Error("Controller builder output differs from frozen sample");
    return {
        clean_harness_sha256: harnessHash(clean),
        fault_harness_sha256: harnessHash(changed),
        clean_controller_sha256: controllerHash(null),
        fault_controller_sha256: controllerHash(sha256(controllerFaultText)),
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
        result.edited_checkpoints_verified = 0;
        for (const item of freeze.selected_faults.weights) {
            const name = item.instance.replace(/-v1$/, "");
            const weights = resolve(
                args["weights-root"],
                name,
                "checkpoint/best.safetensors",
            );
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
            result.edited_checkpoints_verified++;
        }
    }
    console.log(JSON.stringify(result));
}
