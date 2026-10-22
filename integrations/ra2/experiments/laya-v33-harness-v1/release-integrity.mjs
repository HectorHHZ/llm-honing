// Verify the frozen baseline before importing any code from its release.
// Weight bytes are verified by the unchanged release worker at model startup.
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { WEIGHTS_SHA256 } from "./cpu-client.mjs";

export const RELEASE_MANIFEST_SHA256 =
    "e9a1699e87f675582de111ac146faf21425085204d5baa8c6e5a8fe6cd309939";
const digest = (value) => createHash("sha256").update(value).digest("hex");

export function readReleaseEntries(text) {
    const entries = new Map();
    for (const line of text.split(/\r?\n/).filter((line) => line.trim())) {
        const match = /^([0-9a-f]{64})\s+\*?(.+)$/.exec(line);
        if (!match) throw new Error("Invalid release checksum entry");
        const [, checksum, name] = match;
        if (
            !/^[A-Za-z0-9_.\/-]+$/.test(name) ||
            name.startsWith("/") ||
            name
                .split("/")
                .some((part) => !part || part === "." || part === "..") ||
            entries.has(name)
        )
            throw new Error(`Unsafe or duplicate release path: ${name}`);
        entries.set(name, checksum);
    }
    if (entries.get("model.safetensors") !== WEIGHTS_SHA256)
        throw new Error(
            "Release manifest does not pin the expected v33 weights",
        );
    return entries;
}

export function verifyRelease(releaseDir) {
    const root = resolve(releaseDir);
    if (
        process.env.LAYA_MODEL_PATH &&
        resolve(process.env.LAYA_MODEL_PATH) !== root
    )
        throw new Error(
            "LAYA_MODEL_PATH must equal LAYA_RELEASE_DIR for the frozen v33 comparison",
        );
    const manifest = readFileSync(resolve(root, "SHA256SUMS"));
    if (digest(manifest) !== RELEASE_MANIFEST_SHA256)
        throw new Error(
            "SHA256SUMS does not match the pinned v33 release; use setup-linux.sh --install",
        );
    const entries = readReleaseEntries(manifest.toString("utf8"));
    const verifiedFiles = verifyReleaseFiles(root, entries);
    return {
        manifestSha256: RELEASE_MANIFEST_SHA256,
        verifiedNonWeightFiles: verifiedFiles,
        weightsVerification:
            "The release worker hashes model.safetensors at startup.",
    };
}

export function verifyReleaseFiles(root, entries) {
    let verifiedFiles = 0;
    for (const [name, checksum] of entries) {
        if (name === "model.safetensors") continue;
        if (digest(readFileSync(resolve(root, name))) !== checksum)
            throw new Error(`Frozen v33 release file has changed: ${name}`);
        verifiedFiles++;
    }
    return verifiedFiles;
}
