// Published controller strategies. A Honing harness can also supply its own
// planner parameters (see strategyFrom); the browser offers only these two.
export const STRATEGIES = Object.freeze({
    baseline: Object.freeze({
        id: "baseline",
        label: "Planned Steering",
        mode: "plans",
        status: "baseline",
        multiAim: false,
        spacingWeight: 0,
        description:
            "Original joint steering with one preferred firing direction.",
    }),
    raw: Object.freeze({
        id: "raw",
        label: "Free Controls",
        mode: "raw",
        status: "baseline",
        multiAim: false,
        spacingWeight: 0,
        description: "The model independently chooses drive, turn and fire.",
    }),
});
// Planner parameters a harness may set, with the type each must have.
export const STRATEGY_PARAMETERS = Object.freeze({
    id: "string",
    description: "string",
    multiAim: "boolean",
    allAnglePrompt: "boolean",
    adaptiveHold: "boolean",
    spacingWeight: "number",
    minRange: "number",
    maxRange: "number",
    shotSpeedWeight: "number",
    safetyHorizon: "number",
    clearanceWeight: "number",
    clearanceMargin: "number",
    revalidateSlack: "number",
});
// A planned-steering strategy from a published id or an object of planner
// parameters. Unknown keys and wrong types throw, so a harness that names a
// parameter the planner does not read is refused before a match starts.
export function strategyFrom(value) {
    if (typeof value === "string") {
        const s = STRATEGIES[value];
        if (!s || s.mode !== "plans")
            throw Error(`Unknown planned strategy ${JSON.stringify(value)}`);
        return s;
    }
    if (!value || typeof value !== "object" || Array.isArray(value))
        throw Error("A strategy is a published id or an object");
    for (const [key, v] of Object.entries(value)) {
        const type = STRATEGY_PARAMETERS[key];
        if (!type) throw Error(`Unknown strategy parameter ${key}`);
        if (typeof v !== type || (type === "number" && !Number.isFinite(v)))
            throw Error(`Strategy parameter ${key} must be a ${type}`);
    }
    if (!value.id || !value.description)
        throw Error("A strategy needs an id and a description");
    if (
        value.spacingWeight &&
        !(value.minRange >= 0 && value.maxRange >= value.minRange)
    )
        throw Error("spacingWeight needs 0 <= minRange <= maxRange");
    if (value.clearanceWeight && !(value.clearanceMargin >= 0))
        throw Error("clearanceWeight needs a clearanceMargin >= 0");
    return Object.freeze({
        multiAim: false,
        spacingWeight: 0,
        ...value,
        label: value.id,
        mode: "plans",
        status: "harness",
    });
}
export const strategyFor = (seat = {}) =>
    seat.strategy && typeof seat.strategy === "object"
        ? seat.strategy
        : STRATEGIES[seat.decisionVersion] ||
          STRATEGIES[seat.controllerMode === "raw" ? "raw" : "baseline"];
export const publishedStrategies = () =>
    Object.values(STRATEGIES).filter((s) => s.status === "baseline");
export function rangeCost(distance, strategy) {
    if (!strategy.spacingWeight) return 0;
    return distance < strategy.minRange
        ? strategy.minRange - distance
        : Math.max(0, distance - strategy.maxRange) * 0.35;
}
