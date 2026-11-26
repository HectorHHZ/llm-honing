import { LayaClient } from "../laya/client.mjs";
import { PermutationClient } from "../laya/permutation-client.mjs";
import { HttpDecisionClient } from "../laya/http-client.mjs";

// Local Laya, Kev, NanoJev, StartLux, Nimble and remote djev answer the same
// typed questions with the same response fields (honing/backends/jev/laya.py, kev.py,
// nanojev.py, startlux.py, nimble.py).
export const FAMILIES = [
    "laya",
    "kev",
    "nanojev",
    "startlux",
    "nimble",
    "djev",
];
const DEFAULT_MODEL = {
    laya: (candidateMode) =>
        candidateMode === "intent" ? "typed-decisions" : "english",
    kev: () => "kev-0.8b",
    nanojev: () => "nanojev",
    startlux: () => "startlux-4b",
    nimble: () => "nimble-v4-max",
};
// The tags Honing's Reef records carry: TAG_NAMES in honing/backends/reef/records.py.
export const TAG_NAMES = [
    "episode",
    "task",
    "release",
    "harness",
    "policy",
    "game",
];

// --reef-tags, checked: a JSON object of TAG_NAMES to non-empty strings without surrounding spaces. The run sends
// them as the x-reef-tag-* headers of every decision request, so a decision server that records in Reef tags its
// records with them (#223), and its manifest keeps them under "reef_tags". The "episode" tag must be new for every
// attempt of every episode. Follows Reef's x-reef-tag-* headers (with_request_tags, reef/api/wire.py, Reef bc8c150).
export function parseReefTags(text) {
    let tags;
    try {
        tags = JSON.parse(text);
    } catch (e) {
        throw new Error(`--reef-tags is not JSON: ${e.message}`);
    }
    if (!tags || typeof tags !== "object" || Array.isArray(tags))
        throw new Error("--reef-tags must be a JSON object");
    for (const [name, value] of Object.entries(tags)) {
        if (!TAG_NAMES.includes(name))
            throw new Error(
                `--reef-tags has unknown tag ${name}; the tags are ${TAG_NAMES}`,
            );
        if (typeof value !== "string" || !value || value !== value.trim())
            throw new Error(
                `--reef-tags: ${name} must be a non-empty string without surrounding spaces`,
            );
    }
    return tags;
}

// --reef-tags as the headers of a decision request.
export function reefTagHeaders(text) {
    return Object.fromEntries(
        Object.entries(parseReefTags(text)).map(([name, value]) => [
            "x-reef-tag-" + name,
            value,
        ]),
    );
}

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
        return {
            family,
            decisionServer: args["decision-server"],
            ...(args["reef-tags"] === undefined
                ? {}
                : { headers: reefTagHeaders(args["reef-tags"]) }),
        };
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
        ? new HttpDecisionClient({
              url: options.decisionServer,
              headers: options.headers,
          })
        : new LayaClient(options);
    return new PermutationClient(raw);
}

export function inferenceCallUnit(family) {
    return family === "djev"
        ? "structured_decision_request"
        : "native_model_forward";
}
