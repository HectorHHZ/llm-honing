import { STRATEGIES, strategyFor } from "../classic/strategies.js";
// Public versions bundle observation content and decision controls; internal revisions stay stable.
export const CLASSIC_VERSIONS = Object.freeze([
    Object.freeze({
        id: "v1",
        label: "V1 · Basic",
        decisionVersion: "raw",
        contextVersion: "arena-v1",
        assistEnabled: false,
    }),
    Object.freeze({
        id: "v2",
        label: "V2 · Tactical",
        decisionVersion: "raw",
        contextVersion: "full",
        assistEnabled: false,
    }),
    Object.freeze({
        id: "v3",
        label: "V3 · Joint Planning",
        decisionVersion: "baseline",
        contextVersion: "full",
        assistEnabled: false,
    }),
    Object.freeze({
        id: "v4",
        label: "V4 · Joint Planning + Dodge",
        decisionVersion: "baseline",
        contextVersion: "full",
        assistEnabled: true,
    }),
]);
export function classicVersionFor(seat) {
    return CLASSIC_VERSIONS.find(
        (v) =>
            v.decisionVersion ===
                (seat.decisionVersion ||
                    (seat.controllerMode === "raw" ? "raw" : "baseline")) &&
            v.contextVersion === seat.contextVersion &&
            (v.decisionVersion !== "baseline" ||
                v.assistEnabled === !!seat.assistEnabled),
    );
}
export function selectClassicVersion(seat, id, models) {
    const version = CLASSIC_VERSIONS.find((v) => v.id === id);
    if (!version) return seat;
    return normalizeSeat(
        {
            ...seat,
            decisionVersion: version.decisionVersion,
            contextVersion: version.contextVersion,
            assistEnabled: version.assistEnabled,
            assistProfile: "strong",
        },
        models,
        seat,
    );
}
export const INTERVALS = [50, 100, 200, 350, 600, 1000];
export function defaultSeats(models, classic = true) {
    return [0, 1].map(() => ({
        backend: models[0]?.id || "",
        controllerMode: classic ? "plans" : "raw",
        decisionVersion: classic ? "baseline" : "raw",
        intervalMode: classic ? "auto" : "manual",
        contextVersion: "full",
        interval: 100,
        assistEnabled: classic,
        assistProfile: "light",
        fireGuardEnabled: false,
    }));
}
// The public game never restores lab selections or substitutes another model.
export function humanChallengeSeats(models) {
    return defaultSeats(models).map((seat, i) => ({
        ...selectClassicVersion(seat, "v3", models),
        backend:
            i === 1 && models.some((m) => m.id === "nanojev") ? "nanojev" : "",
        assistEnabled: false,
        assistProfile: "standard",
        intervalMode: "auto",
        fireGuardEnabled: false,
    }));
}
export function normalizeSeat(input, models, fallback) {
    const s = { ...fallback };
    if (!input || typeof input !== "object") return s;
    if (models.some((m) => m.id === input.backend)) s.backend = input.backend;
    if (["plans", "raw"].includes(input.controllerMode))
        s.controllerMode = input.controllerMode;
    if (Object.hasOwn(STRATEGIES, input.decisionVersion))
        s.decisionVersion = input.decisionVersion;
    else if (input.controllerMode)
        s.decisionVersion = input.controllerMode === "raw" ? "raw" : "baseline";
    s.controllerMode = strategyFor(s).mode;
    if (["auto", "manual"].includes(input.intervalMode))
        s.intervalMode = input.intervalMode;
    if (["full", "arena-v1"].includes(input.contextVersion))
        s.contextVersion = input.contextVersion;
    if (s.controllerMode === "plans") s.contextVersion = "full";
    if (INTERVALS.includes(input.interval)) s.interval = input.interval;
    for (const k of ["assistEnabled", "fireGuardEnabled"])
        if (typeof input[k] === "boolean") s[k] = input[k];
    if (["light", "standard", "strong"].includes(input.assistProfile))
        s.assistProfile = input.assistProfile;
    if (s.controllerMode === "plans") s.fireGuardEnabled = false;
    return s;
}
export function seatProtocol(edition, seat) {
    return edition === "auto-aim"
        ? "controls-v2"
        : seat.controllerMode === "plans"
          ? "plans-v1"
          : "controls-v3";
}
export function contextRevision(seat) {
    return seat.controllerMode === "plans"
        ? strategyFor(seat).id === "baseline"
            ? "planned-aim-v1"
            : strategyFor(seat).id
        : seat.contextVersion === "arena-v1"
          ? "arena-v1"
          : "state-v7-avoidance-levels";
}
export function matchupPreset(models) {
    const gemma = models.find((m) => /diffusiongemma/i.test(m.label));
    const nano = models.find((m) => /nanojev/i.test(m.label));
    if (!gemma || !nano) return null;
    const seats = defaultSeats(models);
    seats[0].backend = gemma.id;
    seats[1].backend = nano.id;
    seats[1].controllerMode = "raw";
    seats[1].decisionVersion = "raw";
    return seats;
}
