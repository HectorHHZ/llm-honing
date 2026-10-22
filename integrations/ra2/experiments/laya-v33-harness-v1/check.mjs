// Read-only preflight: no CpuLayaClient, metadata load, game or weight loading.
import { accessSync, constants, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
    checkFiles,
    HF_MODEL,
    HF_REVISION,
    loadDependencies,
} from "./runtime.mjs";
import { SOURCE_REVISION, WEIGHTS_SHA256 } from "./cpu-client.mjs";

export function assertSupportedPlatform(
    platform = process.platform,
    architecture = process.arch,
) {
    if (platform !== "linux" || architecture !== "x64")
        throw new Error(
            "Model inference and engine checks require x86_64 Linux",
        );
}

export function validateReleaseIdentity(release, snapshot) {
    for (const item of [release, snapshot]) {
        if (
            item?.repo !== HF_MODEL ||
            item.weightsSha256 !== WEIGHTS_SHA256 ||
            item.sourceRevision !== SOURCE_REVISION
        )
            throw new Error(
                "Release identity differs from the pinned v33 weights/source",
            );
    }
    if (snapshot.revision !== HF_REVISION)
        throw new Error(
            "Release snapshot differs from the pinned Hugging Face revision",
        );
}

export async function check() {
    assertSupportedPlatform();
    const deps = await loadDependencies();
    const environment = checkFiles(deps);
    const release = JSON.parse(
        readFileSync(resolve(deps.release, "release.json"), "utf8"),
    );
    const snapshot = JSON.parse(
        readFileSync(resolve(deps.release, ".snapshot.json"), "utf8"),
    );
    validateReleaseIdentity(release, snapshot);
    for (const file of [
        "model.safetensors",
        "tokenizer/tokenizer.json",
        "ra2/integration/laya/worker.py",
    ])
        accessSync(resolve(deps.release, file), constants.R_OK);
    accessSync(process.env.LAYA_PYTHON, constants.X_OK);
    console.log(
        JSON.stringify(
            {
                event: "preflight_passed",
                environment,
                modelCreated: false,
                gameCreated: false,
                weightsHashVerifiedNow: false,
                note: "The unchanged release loader verifies the weight hash at model startup.",
            },
            null,
            2,
        ),
    );
}

if (
    process.argv[1] &&
    pathToFileURL(resolve(process.argv[1])).href === import.meta.url
) {
    if (process.argv.length > 2) {
        console.error("--check accepts no additional arguments");
        process.exitCode = 1;
    } else
        check().catch((error) => {
            console.error(error.stack ?? error.message);
            process.exitCode = 1;
        });
}
