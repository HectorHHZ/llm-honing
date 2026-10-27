import { test } from "node:test";
import assert from "node:assert/strict";

import { decide } from "../model-bridge.mjs";
import { adapter } from "../web/game/adapter.js";

// A plans-v1 request with two admissible motions.
function plansRequest(id) {
    const game = new adapter.Game(2001);
    adapter.initialize(game);
    const body = {
        ...adapter.observe(game, 0, { controllerMode: "plans", interval: 200 })
            .request,
        id,
        player: 0,
        round_id: 1,
    };
    const ids = body.motions.map((m) => m.id);
    body.admissible_motion_ids = [ids[1], ids[2]];
    return { body, ids };
}

// A stand-in decision server: the motion question prefers ids[0], and the
// weapon question fires only after the motion its state names, so a reply from
// the wrong branch shows. It answers branches as the server does.
function server(ids, asked) {
    const answer = (r) => ({
        id: r.id,
        stateText: r.parts.essential.join("\n"),
        trimmed: [],
        answers: Object.fromEntries(
            Object.keys(r.questions).map((head) => {
                if (head === "motion")
                    return [
                        head,
                        {
                            choice: ids[0],
                            probabilities: Object.fromEntries(
                                ids.map((id, i) => [
                                    id,
                                    [0.5, 0.2, 0.3][i] ?? 0,
                                ]),
                            ),
                        },
                    ];
                const fire = r.parts.essential[0].includes(
                    `SELECTED MOTION ${ids[2]}:`,
                );
                return [
                    head,
                    {
                        choice: fire ? "fire" : "hold",
                        probabilities: fire
                            ? { fire: 0.9, hold: 0.1 }
                            : { fire: 0.1, hold: 0.9 },
                    },
                ];
            }),
        ),
    });
    return async (r) => {
        asked.push(r);
        const reply = answer(r);
        if (r.branches)
            reply.branches = Object.fromEntries(
                Object.entries(r.branches).map(([k, b]) => [k, answer(b)]),
            );
        return reply;
    };
}

const decision = (ids, asked, branches, options) =>
    decide(
        {
            model: "fixture",
            agentIds: ["blue", "red"],
            contextLimit: null,
            branches,
            ask: server(ids, asked),
        },
        options.body,
        options.choose ? { choose: options.choose } : {},
    );

test("with enough branches a plans-v1 decision is one request with the sequential decision's result", async () => {
    const { body, ids } = plansRequest("together");
    const sequentialAsked = [],
        togetherAsked = [];
    const sequential = await decision(ids, sequentialAsked, 0, { body });
    const together = await decision(ids, togetherAsked, 16, { body });
    assert.equal(sequentialAsked.length, 2);
    assert.equal(togetherAsked.length, 1);
    const [sent] = togetherAsked;
    assert.deepEqual(Object.keys(sent.branches), body.admissible_motion_ids);
    // The selected motion's branch is the weapon request the sequential decision sent.
    assert.equal(sequential.motion_id, ids[2]);
    assert.deepEqual(sent.branches[ids[2]], sequentialAsked[1]);
    const { branches: _, ...motion } = sent;
    assert.deepEqual(motion, sequentialAsked[0]);
    for (const key of [
        "controls",
        "probabilities",
        "motion_id",
        "raw_weapon",
        "weapon_choice",
        "weapon_context",
        "constraints",
        "honing_results",
    ])
        assert.deepEqual(together[key], sequential[key], key);
    assert.equal(together.controls.fire, sequential.controls.fire);
    assert.equal(together.structured_requests, 1);
    assert.equal(sequential.structured_requests, 2);
});

test("a sampled motion outside the admissible set keeps the branch of the motion it projects to", async () => {
    const { body, ids } = plansRequest("sampled");
    const asked = [];
    const reply = await decision(ids, asked, 16, {
        body,
        choose: (head, probabilities, candidates) =>
            head === "motion" ? ids[0] : "hold",
    });
    assert.equal(asked.length, 1);
    assert.equal(reply.policy_motion_id, ids[0]);
    assert.equal(reply.motion_id, ids[2]);
    assert.match(
        reply.honing_results[1].stateText,
        new RegExp(`SELECTED MOTION ${ids[2]}:`),
    );
    assert.equal(reply.raw_weapon, "fire");
});

test("too few branches for the admissible motions means two requests", async () => {
    const { body, ids } = plansRequest("few");
    const asked = [];
    await decision(ids, asked, 1, { body });
    assert.equal(asked.length, 2);
    assert.ok(asked.every((r) => r.branches === undefined));
});

test("a reply without the selected branch is refused", async () => {
    const { body, ids } = plansRequest("missing");
    await assert.rejects(
        decide(
            {
                model: "fixture",
                agentIds: ["blue", "red"],
                contextLimit: null,
                branches: 16,
                ask: async (r) => {
                    const { branches: _, ...reply } = await server(ids, [])(r);
                    return reply;
                },
            },
            body,
        ),
        /did not preserve complete decision context/,
    );
});
