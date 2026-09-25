import assert from "node:assert/strict";
import {
    StagedDecision,
    predictStaged,
    verifyStagedTrace,
} from "../integration/laya/staged.mjs";
const checks = [],
    traces = [];
const ok = (name) => {
    checks.push(name);
    console.log("PASS", name);
};
const parts = {
    essential: [
        "Normal-view test fixture; command only owned units and currently visible targets.",
    ],
    details: [],
    memory: [],
    logs: [],
};
function fixtureGroup(id, leaves) {
    return {
        id,
        instructions: "Choose an action for this group.",
        choices: [
            { key: "A", label: "Wait", action: null, cost: 0 },
            ...leaves.map((c, i) => ({
                key: String.fromCharCode(66 + i),
                cost: 0,
                ...c,
            })),
        ],
    };
}
function fixturePacket(groups) {
    return {
        tick: 120,
        groups,
        parts,
        omitted: [],
        questions: Object.fromEntries(
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
        ),
    };
}
function prediction(request, choose) {
    const answers = Object.fromEntries(
        request.groups.map((g) => {
            const key = choose(g, request);
            return [
                g.id,
                {
                    choice: key,
                    probabilities: Object.fromEntries(
                        g.choices.map((c) => [c.key, Number(c.key === key)]),
                    ),
                },
            ];
        }),
    );
    return {
        answers,
        stateText: ["essential", "details", "memory", "logs"]
            .flatMap((k) => request.parts[k])
            .join("\n"),
        latencyMs: 0,
        trimmed: [],
        tokenAudit: {},
        fixture: true,
    };
}
const attack = fixtureGroup(
    "combat",
    [100, 101].flatMap((actor) =>
        [200, 201].map((target) => ({
            label: `Attack visible target #${target} with unit #${actor}`,
            action: {
                function: 23,
                unit_tags: [actor],
                target_unit_tag: target,
            },
        })),
    ),
);
const packet = fixturePacket([attack]);
let calls = 0;
const three = await predictStaged(packet, {
    predict: async (request) => {
        calls++;
        assert.equal(request.tick, 120);
        if (calls === 2) {
            assert.equal(request.previousChoices[0].binding.function, 23);
            assert(
                request.parts.essential.some((s) =>
                    s.includes("Prior selection"),
                ),
            );
        }
        if (calls === 3) {
            assert.deepEqual(request.previousChoices[1].binding.unit_tags, [
                101,
            ]);
            assert(request.parts.essential.some((s) => s.includes("#101")));
        }
        return prediction(request, () => ["B", "C", "C"][calls - 1]);
    },
});
assert.equal(calls, 3);
assert.equal(three.resolvedAnswers.combat.choice, "E");
assert.deepEqual(attack.choices.find((c) => c.key === "E").action, {
    function: 23,
    unit_tags: [101],
    target_unit_tag: 201,
});
verifyStagedTrace(packet, three, assert);
traces.push({ name: "three-stage-binding", packet, record: three });
ok(
    "three calls bind action, actor and target; each later context carries the actual prior choices",
);

const rootWait = await predictStaged(packet, {
    predict: async (r) => prediction(r, () => "A"),
});
assert.equal(rootWait.inferenceCalls, 1);
assert.equal(rootWait.resolvedAnswers.combat.choice, "A");
const cancel = await predictStaged(packet, {
    predict: async (r) => prediction(r, () => (r.stageIndex === 1 ? "B" : "A")),
});
assert.equal(cancel.inferenceCalls, 2);
assert.equal(cancel.resolvedAnswers.combat.choice, "A");
verifyStagedTrace(packet, cancel, assert);
ok(
    "root wait avoids extra calls; parameter cancellation produces no leaf action",
);

const session = new StagedDecision(packet);
session.nextPacket();
session.accept({ combat: { choice: "B" } });
assert.throws(() => session.resolvedAnswers(), /partially specified/);
session.nextPacket();
assert.throws(() => session.accept({ combat: { choice: "Z" } }), /Invalid/);
assert.throws(() => session.resolvedAnswers(), /partially specified/);
let failedCalls = 0;
await assert.rejects(
    predictStaged(packet, {
        predict: async (r) => {
            failedCalls++;
            if (failedCalls === 2) throw new Error("simulated worker failure");
            return prediction(r, () => "B");
        },
    }),
    /simulated worker failure/,
);
assert.equal(failedCalls, 2);
ok(
    "partial plans, invalid model keys and inference failures never resolve to executable actions",
);

const production = fixtureGroup("produce", [
    {
        label: "Produce infantry; cost 100",
        action: { function: 2001 },
        cost: 100,
    },
]);
const multi = fixturePacket([production, attack]);
const mixed = await predictStaged(multi, {
    predict: async (r) => {
        if (r.stageIndex > 1) {
            assert(!Object.hasOwn(r.questions, "produce"));
            assert(
                r.previousChoices.some(
                    (h) => h.group === "produce" && h.complete,
                ),
            );
        }
        return prediction(r, () => "B");
    },
});
assert.equal(mixed.resolvedAnswers.produce.choice, "B");
assert.equal(mixed.inferenceCalls, 3);
verifyStagedTrace(multi, mixed, assert);
ok(
    "simple actions resolve in the first call; only unresolved groups receive follow-up questions",
);

const noQuestions = await predictStaged(fixturePacket([]), {
    predict: () => {
        throw new Error("unexpected inference");
    },
});
assert.equal(noQuestions.inferenceCalls, 0);
assert.deepEqual(noQuestions.resolvedAnswers, {});
ok("empty candidate sets do not call the model");

const badHistory = structuredClone(three);
badHistory.stages[1].contextParts.essential =
    badHistory.stages[1].contextParts.essential.filter(
        (s) => !s.startsWith("Prior selection"),
    );
assert.throws(() => verifyStagedTrace(packet, badHistory, assert));
const badBinding = structuredClone(three);
badBinding.resolvedAnswers.combat.choice = "B";
assert.throws(() => verifyStagedTrace(packet, badBinding, assert));
const badTick = structuredClone(three);
badTick.stages[1].tick++;
assert.throws(() => verifyStagedTrace(packet, badTick, assert));
ok(
    "audit rejects altered history, a substituted leaf action and a changed simulation tick",
);

const many = fixturePacket([
    fixtureGroup(
        "many",
        Array.from({ length: 51 }, (_, i) => ({
            key: `L${i}`,
            label: `Produce unit ${i}`,
            action: { function: 2000 + i },
        })),
    ),
]);
for (const leaf of many.groups[0].choices.filter((x) => x.action)) {
    const result = await predictStaged(many, {
        predict: async (r) =>
            prediction(
                r,
                (g) => g.choices.find((c) => c.leafKeys.includes(leaf.key)).key,
            ),
    });
    assert.equal(result.resolvedAnswers.many.choice, leaf.key);
    assert(
        result.stages.every((s) =>
            s.candidates.every((g) => g.choices.length <= 13),
        ),
    );
    verifyStagedTrace(many, result, assert);
}
const crowded = fixturePacket([
    fixtureGroup(
        "crowded",
        Array.from({ length: 25 }, (_, i) => ({
            key: `L${i}`,
            label: `Attack with unit ${i}`,
            action: {
                function: 23,
                unit_tags: [100 + i],
                target_unit_tag: 200,
            },
        })),
    ),
]);
const lateActor = await predictStaged(crowded, {
    predict: async (r) =>
        prediction(
            r,
            (g) => g.choices.find((c) => c.leafKeys.includes("L24")).key,
        ),
});
assert.equal(lateActor.resolvedAnswers.crowded.choice, "L24");
verifyStagedTrace(crowded, lateActor, assert);
ok(
    "pagination preserves all 51 production functions and later actor choices instead of truncating action families",
);
const confirmPacket = fixturePacket([
    fixtureGroup("diplomacy", [
        {
            label: "Resign and concede this match immediately",
            action: { function: 81 },
            requiresConfirmation: true,
        },
    ]),
]);
const declined = await predictStaged(confirmPacket, {
    predict: async (r) => prediction(r, () => (r.stageIndex === 1 ? "B" : "A")),
});
assert.equal(declined.resolvedAnswers.diplomacy.choice, "A");
assert.equal(declined.inferenceCalls, 2);
const confirmed = await predictStaged(confirmPacket, {
    predict: async (r) => prediction(r, () => "B"),
});
assert.equal(confirmed.resolvedAnswers.diplomacy.choice, "B");
assert.equal(confirmed.inferenceCalls, 2);
verifyStagedTrace(confirmPacket, confirmed, assert);
ok(
    "resignation has a separate model confirmation stage and can be cancelled there",
);

const sameTargetActors = fixturePacket([
    fixtureGroup(
        "same_target",
        Array.from({ length: 25 }, (_, i) => ({
            key: `unit-${i}`,
            label: "Attack visible ore refinery #200 with this unit",
            actorLabel: `${i < 20 ? "conscript infantry" : "Rhino tank"} #${
                100 + i
            }`,
            action: {
                function: 23,
                unit_tags: [100 + i],
                target_unit_tag: 200,
            },
        })),
    ),
]);
let actorPagesChecked = false;
const groundedPage = await predictStaged(sameTargetActors, {
    predict: async (request) =>
        prediction(request, (group) => {
            if (
                group.dimension === "actors" &&
                group.choices.some((c) =>
                    c.label.startsWith("More actors options:"),
                )
            ) {
                const labels = group.choices
                    .filter((c) => c.kind === "choose")
                    .map((c) => c.label);
                assert.equal(
                    new Set(labels).size,
                    labels.length,
                    "Actor pages must be distinguishable despite identical target labels",
                );
                assert(labels.some((s) => s.includes("Rhino tank #124")));
                assert(labels.every((s) => !s.includes("ore refinery #200")));
                actorPagesChecked = true;
            }
            return group.choices.find((c) => c.leafKeys.includes("unit-24"))
                .key;
        }),
});
assert(actorPagesChecked);
assert.equal(groundedPage.resolvedAnswers.same_target.choice, "unit-24");
verifyStagedTrace(sameTargetActors, groundedPage, assert);
ok(
    "actor page descriptions identify owned units even when all target labels match",
);
