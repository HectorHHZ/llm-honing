// Conditional model calls over a frozen set of grounded leaf actions.
// This module has no engine, observer, GPU, or game-clock handle.
export const STAGED_PROTOCOL = "laya-normal-staged-v1.5";
export const MAX_STAGES = 8;
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const actorKey = (c) => JSON.stringify(c.action.unit_tags ?? []);
const targetKey = (c) =>
    JSON.stringify({
        world: c.action.world,
        target_unit_tag: c.action.target_unit_tag,
    });
const functionLabels = {
    20: "Move units to a location",
    21: "Force-move units",
    22: "Attack-move to scout or search a location",
    23: "Attack a visible target",
    24: "Attack a ground location",
    25: "Guard an area",
    26: "Guard in place",
    27: "Stop units",
    28: "Deploy or undeploy units",
    29: "Move the observation camera",
    80: "Change an alliance",
    81: "Resign from this match",
    40: "Scatter units",
    41: "Enter a transport",
    42: "Unload passengers",
    43: "Repair a target",
    44: "Capture a structure",
    45: "Garrison a structure",
    46: "Gather resources",
    47: "Dock at a facility",
    48: "Place a bomb",
    49: "Sell an owned object",
    50: "Toggle repair on an owned building",
    60: "Place a completed building",
    61: "Cancel a production item",
    62: "Pause a production queue",
    63: "Resume a production queue",
};
function partition(leaves, key) {
    const groups = new Map();
    for (const leaf of leaves) {
        const k = key(leaf);
        if (!groups.has(k)) groups.set(k, []);
        groups.get(k).push(leaf);
    }
    return [...groups.values()];
}
function commonBindings(leaves) {
    const out = {};
    for (const key of [
        "function",
        "unit_tags",
        "target_unit_tag",
        "world",
        "extra",
        "queued",
        "repeat",
    ]) {
        const value = leaves[0].action[key];
        if (
            value !== undefined &&
            leaves.every((c) => equal(c.action[key], value))
        )
            out[key] = value;
    }
    return out;
}
function actorsLabel(leaves) {
    if (leaves[0].actorLabel) return `Use ${leaves[0].actorLabel}`;
    const tags = leaves[0].action.unit_tags ?? [];
    return tags.length
        ? `Use ${tags.length} owned unit(s): ${tags
              .slice(0, 3)
              .map((t) => `#${t}`)
              .join(", ")}${tags.length > 3 ? " ..." : ""}`
        : "No acting unit is required";
}
function bindingText(binding) {
    if (!binding) return "no action";
    const fields =
        binding.function === undefined ? [] : [`function=${binding.function}`];
    if (binding.unit_tags) fields.push(actorsLabel([{ action: binding }]));
    if (binding.target_unit_tag !== undefined)
        fields.push(`target=#${binding.target_unit_tag}`);
    if (binding.world !== undefined) fields.push(`world cell=${binding.world}`);
    if (binding.extra) fields.push(`extra=${JSON.stringify(binding.extra)}`);
    return fields.join("; ");
}
function intentLabel(leaves) {
    if (leaves.length === 1) return leaves[0].label;
    if (leaves[0].intent?.label) return leaves[0].intent.label;
    if (leaves[0].action.function === 60) {
        const item = leaves[0].label.match(/^Place (.+) at /)?.[1];
        if (item) return `Place completed ${item}; choose its location next`;
    }
    return (
        functionLabels[leaves[0].action.function] ??
        `Choose parameters for function ${leaves[0].action.function}`
    );
}
function dimensionLabel(dimension, leaves) {
    if (dimension === "intent") return intentLabel(leaves);
    if (dimension === "function")
        return (
            leaves[0].functionLabel ??
            functionLabels[leaves[0].action.function] ??
            leaves[0].label
        );
    if (dimension === "actors") return actorsLabel(leaves);
    return leaves[0].label;
}
function branch(state) {
    const leaves = state.leaves;
    if (
        state.started &&
        leaves.length === 1 &&
        leaves[0].requiresConfirmation &&
        !state.confirmed
    )
        return { dimension: "confirmation", buckets: [leaves] };
    if (!state.started)
        return {
            dimension: "intent",
            buckets: partition(
                leaves,
                (c) => c.intent?.id ?? String(c.action.function),
            ),
        };
    if (state.group.hierarchy === "intent-function") {
        const functions = partition(leaves, (c) => String(c.action.function));
        if (functions.length > 1)
            return { dimension: "function", buckets: functions };
    }
    const actors = partition(leaves, actorKey);
    if (actors.length > 1) return { dimension: "actors", buckets: actors };
    const targets = partition(leaves, targetKey);
    if (targets.length > 1) return { dimension: "target", buckets: targets };
    return { dimension: "variant", buckets: leaves.map((c) => [c]) };
}

export class StagedDecision {
    constructor(packet) {
        this.base = structuredClone(packet);
        this.states = new Map();
        this.answers = {};
        this.history = [];
        this.round = 0;
        this.current = null;
        for (const group of this.base.groups) {
            const waits = group.choices.filter((c) => !c.action),
                leaves = group.choices.filter((c) => c.action);
            if (waits.length !== 1 || !leaves.length)
                throw new Error(`Invalid leaf choices for ${group.id}`);
            if (
                new Set(group.choices.map((c) => c.key)).size !==
                group.choices.length
            )
                throw new Error(`Duplicate keys for ${group.id}`);
            this.states.set(group.id, {
                group,
                leaves,
                wait: waits[0],
                started: false,
            });
        }
    }
    get done() {
        return Object.keys(this.answers).length === this.states.size;
    }
    nextPacket() {
        if (this.current) return this.current;
        if (this.done) return null;
        if (this.round >= MAX_STAGES)
            throw new Error(
                "Staged decision exceeded its bounded parameter depth",
            );
        const groups = [],
            dimensions = {};
        for (const [id, state] of this.states) {
            if (this.answers[id]) continue;
            const { dimension, buckets: rawBuckets } = branch(state);
            dimensions[id] = dimension;
            const paged = rawBuckets.length > 12,
                width = paged ? Math.ceil(rawBuckets.length / 12) : 1;
            const buckets = [];
            for (let i = 0; i < rawBuckets.length; i += width)
                buckets.push(rawBuckets.slice(i, i + width).flat());
            const choices = [
                {
                    key: "A",
                    label: state.started
                        ? (dimension === "confirmation" &&
                              state.leaves[0].cancelLabel) ||
                          "Cancel this pending action; submit no command for it."
                        : state.group.waitLabel ??
                          "Take no action for this group this turn.",
                    kind: "cancel",
                    leafKeys: [state.wait.key],
                    binding: null,
                },
            ];
            for (const [pageIndex, leaves] of buckets.entries()) {
                const smallLabel = (members) =>
                    dimensionLabel(dimension, members)
                        .replace(/; cost .*/, "")
                        .slice(0, 64);
                // Summarize the parameter actually being selected. Leaf labels
                // start with a target and can be identical across actor pages.
                const first = rawBuckets[pageIndex * width];
                const last =
                    rawBuckets[
                        Math.min((pageIndex + 1) * width, rawBuckets.length) - 1
                    ];
                const label = paged
                    ? `More ${dimension} options: ${smallLabel(
                          first,
                      )} ... ${smallLabel(last)}`
                    : dimension === "confirmation"
                    ? leaves[0].confirmationLabel ??
                      `Confirm: ${leaves[0].label}`
                    : dimensionLabel(dimension, leaves);
                choices.push({
                    key: String.fromCharCode(65 + choices.length),
                    label,
                    kind: "choose",
                    refinesIntent: dimension === "intent" && paged,
                    leafKeys: leaves.map((c) => c.key),
                    binding: commonBindings(leaves),
                });
            }
            groups.push({
                id,
                dimension,
                instructions:
                    dimension === "intent"
                        ? `${state.group.instructions} Select the action first; unresolved parameters will be asked next.`
                        : `For the previously selected action, choose ${
                              dimension === "actors"
                                  ? "the acting units"
                                  : dimension === "target"
                                  ? "the target or location"
                                  : dimension === "confirmation"
                                  ? "whether to confirm this action"
                                  : "the remaining parameters"
                          }, or cancel that pending action.`,
                choices,
            });
        }
        const stageIndex = this.round + 1;
        const history = [...this.states.keys()].flatMap((id) => {
            const path = this.history.filter((h) => h.group === id);
            if (!path.length) return [];
            const last = path.at(-1);
            return [
                `Prior selections (${id}): ${path
                    .map((h) => `s${h.stageIndex} ${h.key}=${h.label}`)
                    .join(" -> ")}. Bound now: ${bindingText(last.binding)}. ${
                    last.cancelled
                        ? "Cancelled."
                        : last.complete
                        ? "Complete; not submitted."
                        : "Parameters pending."
                }`,
            ];
        });
        const essential = this.base.groups.some((g) => g.contextVersion === 2)
            ? [
                  `Decision stage ${stageIndex}, tick ${this.base.tick}. Simulation paused; this decision's actions have not executed yet.`,
                  ...history,
                  ...this.base.parts.essential,
              ]
            : [
                  `Decision stage ${stageIndex} at tick ${this.base.tick}. The simulation is paused at this same tick. No choices in this decision have been submitted or executed yet.`,
                  ...history,
                  "Previous choices above are commitments for this decision, not observations of completed actions. Keep their bound parameters. Choose the remaining parameters only; cancellation abandons that group. Shared spending and actor conflicts are checked before submission.",
                  ...this.base.parts.essential,
              ];
        const questions = Object.fromEntries(
            groups.map((g) => [
                g.id,
                {
                    type: "choice",
                    instructions: g.instructions,
                    criteria: Object.fromEntries(
                        g.choices.map((c) => [c.key, c.label]),
                    ),
                },
            ]),
        );
        this.current = {
            tick: this.base.tick,
            stageIndex,
            dimensions,
            groups,
            questions,
            parts: { ...this.base.parts, essential },
            previousChoices: structuredClone(this.history),
            omitted: this.base.omitted ?? [],
        };
        return this.current;
    }
    accept(answers) {
        const packet = this.current;
        if (!packet) throw new Error("No pending stage to answer");
        if (
            !equal(
                Object.keys(answers).sort(),
                Object.keys(packet.questions).sort(),
            )
        )
            throw new Error("Stage answer groups do not match its questions");
        // Validate every answer before changing any plan.
        const choices = packet.groups.map((g) => {
            const c = g.choices.find((c) => c.key === answers[g.id]?.choice);
            if (!c)
                throw new Error(
                    `Invalid ${g.id} stage choice: ${answers[g.id]?.choice}`,
                );
            return [g, c];
        });
        for (const [group, choice] of choices) {
            const state = this.states.get(group.id),
                cancelled = choice.kind === "cancel";
            if (cancelled) this.answers[group.id] = { choice: state.wait.key };
            else {
                state.leaves = state.leaves.filter((c) =>
                    choice.leafKeys.includes(c.key),
                );
                if (!choice.refinesIntent) state.started = true;
                if (group.dimension === "confirmation") state.confirmed = true;
                if (!state.leaves.length)
                    throw new Error(
                        "Selected parameter branch has no grounded action",
                    );
                if (
                    state.leaves.length === 1 &&
                    (!state.leaves[0].requiresConfirmation || state.confirmed)
                )
                    this.answers[group.id] = { choice: state.leaves[0].key };
            }
            this.history.push({
                stageIndex: packet.stageIndex,
                group: group.id,
                dimension: group.dimension,
                key: choice.key,
                label: choice.label,
                binding: structuredClone(choice.binding),
                cancelled,
                complete: !!this.answers[group.id],
            });
        }
        this.round++;
        this.current = null;
    }
    resolvedAnswers() {
        if (!this.done)
            throw new Error("Cannot submit a partially specified decision");
        return structuredClone(this.answers);
    }
}

export async function predictStaged(packet, client) {
    const session = new StagedDecision(packet),
        stages = [];
    let request;
    while ((request = session.nextPacket())) {
        const prediction = await client.predict(request);
        session.accept(prediction.answers);
        stages.push({
            stageIndex: request.stageIndex,
            tick: request.tick,
            dimensions: request.dimensions,
            contextParts: request.parts,
            questions: request.questions,
            candidates: request.groups,
            previousChoices: request.previousChoices,
            ...prediction,
        });
    }
    return {
        decisionMode: "staged",
        stages,
        resolvedAnswers: session.resolvedAnswers(),
        selectionPath: session.history,
        inferenceCalls: stages.length,
        latencyMs: stages.reduce((n, s) => n + s.latencyMs, 0),
    };
}

// Reconstruct the conditional branches from logged grounded candidates. Used by
// offline audit; exact parameter choices must lead to the submitted leaf key.
export function verifyStagedTrace(packet, record, assert) {
    const session = new StagedDecision(packet);
    for (const stage of record.stages) {
        const expected = session.nextPacket();
        assert(
            expected,
            "Unexpected model call after all actions were resolved",
        );
        assert.equal(stage.tick, expected.tick);
        assert.equal(stage.stageIndex, expected.stageIndex);
        assert.deepEqual(stage.questions, expected.questions);
        assert.deepEqual(stage.candidates, expected.groups);
        assert.deepEqual(stage.contextParts, expected.parts);
        assert.deepEqual(stage.previousChoices, expected.previousChoices);
        assert.deepEqual(stage.dimensions, expected.dimensions);
        session.accept(stage.answers);
    }
    assert(session.done, "Unresolved parameters at engine submission");
    assert.deepEqual(record.resolvedAnswers, session.resolvedAnswers());
    assert.deepEqual(record.selectionPath, session.history);
    return session.resolvedAnswers();
}
