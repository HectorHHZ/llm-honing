import { LayaClient } from "../laya/client.mjs";
import { PermutationClient } from "../laya/permutation-client.mjs";
import { NativeClient } from "./client.mjs";
import { ROOT } from "../paths.mjs";

export function decisionClientOptions(args, candidateMode) {
    const family = args.family ?? "laya";
    if (!["laya", "nanojev", "jev-omni"].includes(family))
        throw new Error("Unknown decision model family");
    if (family === "laya") {
        if (
            args["head-manifest"] ||
            args["prepared-manifest"] ||
            args["source-root"] ||
            args["native-max-length"]
        )
            throw new Error("Native settings require a native model family");
        return {
            family,
            gpu: args.gpu ?? "0",
            model:
                args.model ??
                (candidateMode === "intent" ? "typed-decisions" : "english"),
        };
    }
    if (args.model)
        throw new Error(
            "Native family uses a prepared manifest, not a Laya model name",
        );
    const maxLength = Number(args["native-max-length"] ?? 2048);
    if (!Number.isInteger(maxLength) || maxLength <= 0)
        throw new Error("Invalid native input limit");
    const sourceRoot = args["source-root"] ?? process.env.NANOJEV_SOURCE_DIR;
    if (family === "nanojev" && !sourceRoot)
        throw new Error("Set --source-root or NANOJEV_SOURCE_DIR for NanoJev");
    return {
        family,
        gpu: args.gpu ?? "0",
        preparedManifest:
            args["prepared-manifest"] ?? ROOT + `.cache/models/${family}.json`,
        sourceRoot: family === "nanojev" ? sourceRoot : undefined,
        headManifest: args["head-manifest"],
        maxLength,
    };
}

export function createDecisionClient(options) {
    const raw =
        options.family === "laya"
            ? new LayaClient(options)
            : new NativeClient(options);
    return new PermutationClient(raw);
}
