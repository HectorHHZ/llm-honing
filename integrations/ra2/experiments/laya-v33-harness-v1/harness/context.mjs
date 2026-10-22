// A bounded, normal-observation-only supplement to the frozen v33 context.
// It never changes candidates, leaf keys, questions, model weights, or commands.
export const CONTEXT_VERSION = "laya-v33-context-advice-v1";
export const MAX_NOTE_CHARS = 320;
export const NEAR_BASE_CELLS = 12; // Advisory distance, not weapon range.
const MILITARY_QUEUES = ["infantry", "vehicles", "aircraft", "ships"];
const BUILDING_QUEUES = ["structures", "armory"];
const FACTORIES = ["NAWEAP", "GAWEAP"];
const ECONOMY = [
    "NACNST",
    "GACNST",
    "NAREFN",
    "GAREFN",
    "NAWEAP",
    "GAWEAP",
    "HARV",
    "CMIN",
];
const NON_MILITARY = ["HARV", "CMIN", "SMCV", "AMCV"];
const live = (u) => Number.isFinite(u.health) && u.health > 0 && !u.is_in_cargo;
const point = (u) => Number.isFinite(u.x) && Number.isFinite(u.y);
const distance = (a, b) =>
    point(a) && point(b) ? Math.hypot(a.x - b.x, a.y - b.y) : Infinity;

/**
 * Append at most one grounded advisory. Observation must be the seat's normal
 * observation; frame is its own IntentPolicy.frame. We recheck alliances and
 * visibility and deliberately never consult frame.memory or other seats.
 *
 * Notes are for audit, not an action plan. The existing staged model still
 * chooses every action, actor, and target, including the unchanged wait option.
 * The release tokenizer retains its strict lossless budget checks; this helper
 * bounds added characters rather than guessing a tokenizer-specific token count.
 */
export function enhancePacket(
    packet,
    { observation, frame, tick = packet?.tick } = {},
) {
    if (
        !packet ||
        !Array.isArray(packet.parts?.essential) ||
        !Array.isArray(packet.groups)
    )
        throw new TypeError(
            "enhancePacket requires a frozen-v33 packet with parts and groups",
        );
    if (
        !observation?.vectors ||
        !Array.isArray(frame?.own) ||
        !Array.isArray(frame?.visible)
    )
        throw new TypeError(
            "enhancePacket requires the seat's observation and frame",
        );
    if (packet.tick !== undefined && tick !== packet.tick)
        throw new Error(
            "Context evidence and packet must refer to the same tick",
        );
    const result = {
        packet: structuredClone(packet),
        notes: [],
        version: CONTEXT_VERSION,
    };
    const leaves = packet.groups
        .flatMap((g) => g.choices ?? [])
        .filter((c) => c.action);
    if (!leaves.length) return result;
    const v = observation.vectors,
        queues = v.production ?? [];
    const own = frame.own.filter((u) => u.alliance === 1 && live(u));
    const visible = frame.visible.filter(
        (u) =>
            u.alliance === 4 && u.display_type === 1 && !u.is_blip && live(u),
    );
    const units = new Map(own.map((u) => [u.tag, u]));
    const coverage = new Map(
        (packet.availability ?? []).map((f) => [f.id, f.name]),
    );
    const affordable = (c) =>
        Number.isFinite(c.cost) &&
        c.cost >= 0 &&
        Number.isFinite(v.minerals) &&
        c.cost <= v.minerals;
    const militaryActors = (c, idleOnly = false) => {
        const actors = (c.action.unit_tags ?? []).map((tag) => units.get(tag));
        return (
            c.intent?.id === "combat" &&
            actors.length > 0 &&
            actors.every(
                (u) =>
                    u &&
                    u.can_move &&
                    !NON_MILITARY.includes(u.name) &&
                    (!idleOnly || u.is_idle),
            )
        );
    };
    const note = (code, observed, advice, choices, evidence) => {
        const text = `Harness v1. Observed: ${observed} Advice (not a command): ${advice}`;
        if (text.length > MAX_NOTE_CHARS)
            throw new Error(
                "Harness context note exceeds its character budget",
            );
        result.packet.parts.essential.push(text);
        result.notes.push({
            code,
            tick,
            text,
            evidence,
            candidateKeys: [...new Set(choices.map((c) => c.key))],
        });
        return result;
    };

    // An available attack is not proof that the enemy is attacking us. State
    // only observed proximity, and keep the military response a suggestion.
    const assets = own.filter((u) => ECONOMY.includes(u.name));
    const nearby = visible.filter((enemy) =>
        assets.some((asset) => distance(enemy, asset) <= NEAR_BASE_CELLS),
    );
    const defend = leaves.filter(
        (c) =>
            c.action.function === 23 &&
            militaryActors(c) &&
            nearby.some((enemy) => enemy.tag === c.action.target_unit_tag),
    );
    if (defend.length)
        return note(
            "visible-enemy-near-economy",
            "a visible enemy is within 12 cells of an economic or production asset; a military attack is offered.",
            "Consider defending with combat troops before routine waiting; keep miners working.",
            defend,
            {
                visibleTargetTags: [
                    ...new Set(defend.map((c) => c.action.target_unit_tag)),
                ],
                radiusCells: NEAR_BASE_CELLS,
            },
        );

    const ready = queues.filter(
        (q) =>
            BUILDING_QUEUES.includes(q.queue) &&
            q.status === "Ready" &&
            q.size > 0,
    );
    const place = leaves.filter(
        (c) =>
            c.action.function === 60 &&
            ready.some((q) => q.item === c.action.extra?.item),
    );
    if (place.length)
        return note(
            "ready-building-placement",
            "a building queue is Ready and its placement options are offered.",
            "Consider placing the paid-for building now so it can work and free the queue.",
            place,
            { queues: ready.map((q) => q.queue) },
        );

    const resume = leaves.filter(
        (c) =>
            c.action.function === 63 &&
            queues.some(
                (q) =>
                    q.queue === c.action.extra?.queue &&
                    q.status === "OnHold" &&
                    q.size > 0,
            ),
    );
    if (resume.length && v.minerals > 0)
        return note(
            "paused-production",
            "production is OnHold, credits are available, and Resume is offered.",
            "Consider resuming useful production; waiting alone does not resume a paused queue.",
            resume,
            { queues: [...new Set(resume.map((c) => c.action.extra.queue))] },
        );

    // Start/stop repair use the same function. Only the release's explicit
    // Start label supports advice to start; never accidentally suggest stopping.
    const repair = leaves.filter(
        (c) =>
            c.action.function === 50 &&
            /^Start repairing /.test(c.label ?? "") &&
            (c.action.unit_tags ?? []).some((tag) => {
                const u = units.get(tag);
                return (
                    u &&
                    ECONOMY.includes(u.name) &&
                    !u.can_move &&
                    u.health_ratio > 0 &&
                    u.health_ratio < 128
                );
            }),
    );
    if (repair.length && v.minerals > 0)
        return note(
            "damaged-key-building",
            "an economic or production building has under half health and Start repair is offered.",
            "Consider repairing this building to preserve its capability.",
            repair,
            {
                unitTags: [
                    ...new Set(repair.flatMap((c) => c.action.unit_tags ?? [])),
                ],
            },
        );

    // This reports a present capability gap; it never infers destruction from
    // missing/truncated unit rows. Complete observed prerequisites avoid pushing
    // a second building while the first one is still unfolding.
    const factoryBuild = leaves.filter(
        (c) =>
            ["Build_NAWEAP", "Build_GAWEAP"].includes(
                coverage.get(c.action.function),
            ) && affordable(c),
    );
    const economyReady =
        own.some(
            (u) =>
                ["NAREFN", "GAREFN"].includes(u.name) &&
                u.build_progress === 100,
        ) &&
        own.some((u) => ["HARV", "CMIN"].includes(u.name)) &&
        !v.is_low_power &&
        v.power_cap >= v.power_used;
    if (
        !v.units_truncated &&
        economyReady &&
        !own.some((u) => FACTORIES.includes(u.name)) &&
        factoryBuild.length
    )
        return note(
            "missing-vehicle-production",
            "no own war factory is present in complete unit rows; an affordable Build war factory option is offered.",
            "Consider establishing vehicle production while preserving mining income.",
            factoryBuild,
            { completeOwnRows: true, missingCapability: "war-factory" },
        );

    const construction = queues.filter(
        (q) =>
            BUILDING_QUEUES.includes(q.queue) &&
            q.status === "Active" &&
            q.size > 0,
    );
    const militaryQueues = queues.filter((q) =>
        MILITARY_QUEUES.includes(q.queue),
    );
    const emptyMilitaryQueues = MILITARY_QUEUES.every((name) =>
        militaryQueues.some(
            (q) => q.queue === name && q.status === "Idle" && q.size === 0,
        ),
    );
    const recruit = leaves.filter(
        (c) =>
            c.intent?.id === "army" &&
            c.action.function >= 2000 &&
            c.action.function < 3000 &&
            affordable(c),
    );
    if (
        economyReady &&
        construction.length &&
        emptyMilitaryQueues &&
        recruit.length
    )
        return note(
            "parallel-military-production",
            "construction is active, military queues are empty, and affordable recruitment is offered.",
            "Consider recruiting combat troops in parallel; keep the building queue running and reserve funds for it.",
            recruit,
            {
                activeBuildingQueues: construction.map((q) => q.queue),
                emptyMilitaryQueues: [...MILITARY_QUEUES],
            },
        );

    const idleAttack = leaves.filter(
        (c) =>
            c.action.function === 23 &&
            militaryActors(c, true) &&
            visible.some((enemy) => enemy.tag === c.action.target_unit_tag),
    );
    if (idleAttack.length)
        return note(
            "idle-military-visible-target",
            "idle combat troops have offered attacks against currently visible enemies.",
            "Consider a suitable attack instead of leaving all these troops idle; assess the visible targets.",
            idleAttack,
            {
                visibleTargetTags: [
                    ...new Set(idleAttack.map((c) => c.action.target_unit_tag)),
                ],
            },
        );

    const scout = leaves.filter(
        (c) => c.action.function === 22 && militaryActors(c, true),
    );
    if (!visible.length && scout.length)
        return note(
            "idle-military-scout",
            "combat troops are idle and attack-move options are offered; no enemy is currently visible.",
            "Consider scouting an offered unvisited or stale public location. Unseen enemies remain unknown.",
            scout,
            {
                idleUnitTags: [
                    ...new Set(scout.flatMap((c) => c.action.unit_tags ?? [])),
                ],
            },
        );
    return result;
}
