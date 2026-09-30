export const INTERVALS = [50, 100, 200, 350, 600, 1000];
export function defaultSeats(models, classic = true) {
    return [0, 1].map(() => ({
        backend: models[0]?.id || "",
        controllerMode: classic ? "plans" : "raw",
        contextVersion: "full",
        interval: 100,
        assistEnabled: classic,
        assistProfile: "light",
        fireGuardEnabled: false,
    }));
}
export function normalizeSeat(input, models, fallback) {
    const s = { ...fallback };
    if (!input || typeof input !== "object") return s;
    if (models.some((m) => m.id === input.backend)) s.backend = input.backend;
    if (["plans", "raw"].includes(input.controllerMode))
        s.controllerMode = input.controllerMode;
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
        ? "planned-aim-v1"
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
    return seats;
}
