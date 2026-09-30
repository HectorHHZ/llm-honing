import { LayaClient } from "../laya/client.mjs";
import { PermutationClient } from "../laya/permutation-client.mjs";
import { HttpDecisionClient } from "../laya/http-client.mjs";

// Three decision-model families run here: Laya, Kev, and NanoJev. All answer the same
// typed questions with the same response fields (honing/backends/jev/laya.py, kev.py,
// nanojev.py).
export const FAMILIES = ["laya", "kev", "nanojev"];
const DEFAULT_MODEL = {
    laya: (candidateMode) =>
        candidateMode === "intent" ? "typed-decisions" : "english",
    kev: () => "kev-0.8b",
    nanojev: () => "nanojev",
};
export function decisionClientOptions(args, candidateMode) {
    const family = args.family ?? "laya";
    if (!FAMILIES.includes(family))
        throw new Error(
            `Model family ${family} is not supported; use one of ${FAMILIES.join(
                ", ",
            )}`,
        );
    if (
        args["head-manifest"] ||
        args["prepared-manifest"] ||
        args["source-root"] ||
        args["native-max-length"]
    )
        throw new Error("Native settings require a native model family");
    if (args["decision-server"])
        return { family, decisionServer: args["decision-server"] };
    return {
        family,
        gpu: args.gpu ?? "0",
        model: args.model ?? DEFAULT_MODEL[family](candidateMode),
    };
}

// A decision server reports the family it loaded; a Laya server predates the field.
export function checkServedFamily(options, metadata) {
    const served = metadata?.family ?? "laya";
    if (served !== options.family)
        throw new Error(
            `Asked for a ${options.family} model, but the decision server loaded ${served}`,
        );
    return metadata;
}

export function createDecisionClient(options) {
    const raw = options.decisionServer
        ? new HttpDecisionClient({ url: options.decisionServer })
        : new LayaClient(options);
    return new PermutationClient(raw);
}
