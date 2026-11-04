// IDs are immutable experiment versions. Promotion requires paired arena evidence.
export const STRATEGIES = Object.freeze({
    baseline: Object.freeze({
        id: "baseline",
        label: "Classic Aim",
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
    "multi-angle-v1": Object.freeze({
        id: "multi-angle-v1",
        label: "Angle Hunter v1",
        mode: "plans",
        status: "candidate",
        multiAim: true,
        spacingWeight: 0,
        description: "Compare direct and ricochet angles for each moving shot.",
    }),
    "steady-aim-v1": Object.freeze({
        id: "steady-aim-v1",
        label: "Steady Aim v1",
        mode: "plans",
        status: "candidate",
        multiAim: false,
        spacingWeight: 0,
        revalidateSlack: 60,
        description:
            "Retain the model's selected motion when it remains safe and useful, reducing late shot cancellations.",
    }),
    "quick-shot-v1": Object.freeze({
        id: "quick-shot-v1",
        label: "Quick Shot v1",
        mode: "plans",
        status: "candidate",
        multiAim: true,
        allAnglePrompt: true,
        spacingWeight: 0.25,
        minRange: 180,
        maxRange: 360,
        shotSpeedWeight: 100,
        description:
            "Compare several firing angles, favor a short time-to-hit, and retain room to retreat.",
    }),
    "safe-lanes-v1": Object.freeze({
        id: "safe-lanes-v1",
        label: "Safe Lanes v1",
        mode: "plans",
        status: "candidate",
        multiAim: false,
        spacingWeight: 0.5,
        minRange: 180,
        maxRange: 360,
        safetyHorizon: 1.15,
        clearanceWeight: 4,
        clearanceMargin: 45,
        description:
            "Keep a shell clearance margin and room to retreat; favor usable firing lanes over constant backing away.",
    }),
    "standoff-v1": Object.freeze({
        id: "standoff-v1",
        allAnglePrompt: true,
        label: "Keep Distance v1",
        mode: "plans",
        status: "candidate",
        multiAim: true,
        spacingWeight: 1.6,
        minRange: 220,
        maxRange: 360,
        description:
            "Compare firing angles while maintaining a 220–360 px engagement range.",
    }),
});
export const strategyFor = (seat = {}) =>
    STRATEGIES[seat.decisionVersion] ||
    STRATEGIES[seat.controllerMode === "raw" ? "raw" : "baseline"];
export const publishedStrategies = () =>
    Object.values(STRATEGIES).filter((s) => s.status !== "candidate");
export function rangeCost(distance, strategy) {
    if (!strategy.spacingWeight) return 0;
    return distance < strategy.minRange
        ? strategy.minRange - distance
        : Math.max(0, distance - strategy.maxRange) * 0.35;
}
