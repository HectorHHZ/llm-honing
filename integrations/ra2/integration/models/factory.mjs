import { LayaClient } from "../laya/client.mjs";
import { PermutationClient } from "../laya/permutation-client.mjs";
import { HttpDecisionClient } from "../laya/http-client.mjs";

// Local Laya, Kev, NanoJev and remote djev answer the same
// typed questions with the same response fields (honing/backends/jev/laya.py, kev.py,
// nanojev.py).
export const FAMILIES = ["laya", "kev", "nanojev", "djev"];
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
    if (family === "djev")
        throw new Error(
            "djev requires --decision-server pointing to the Honing djev bridge",
        );
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
    if (options.family === "djev" && !options.decisionServer)
        throw new Error("djev requires a decision server");
    const raw = options.decisionServer
        ? new HttpDecisionClient({ url: options.decisionServer })
        : new LayaClient(options);
    return new PermutationClient(raw);
}

export function inferenceCallUnit(family) {
    return family === "djev"
        ? "structured_decision_request"
        : "native_model_forward";
}
