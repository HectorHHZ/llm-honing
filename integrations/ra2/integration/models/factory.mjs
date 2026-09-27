import { LayaClient } from "../laya/client.mjs";
import { PermutationClient } from "../laya/permutation-client.mjs";

// Only the Laya family runs here. The native families (NanoJev, Jev-Omni) come
// with the decision server and the trainer; see #52 and #56.
export function decisionClientOptions(args, candidateMode) {
    const family = args.family ?? "laya";
    if (family !== "laya")
        throw new Error(
            `Model family ${family} is not supported yet; only laya runs in this integration`,
        );
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

export function createDecisionClient(options) {
    return new PermutationClient(new LayaClient(options));
}
