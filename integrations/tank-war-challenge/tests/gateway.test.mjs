// The gateway's bridge, run in this checkout: it reads integrations/tank-war, asks with the legacy wording,
// and sends a plans-v1 decision as one request with its weapon branches.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { decide, validateQuestions } from "../compat-bridge.mjs";
import { adapter } from "../web/classic/adapter.js";

const legacy = JSON.parse(readFileSync(new URL("../legacy-questions.json", import.meta.url)));

test("the legacy question text passes the bridge's checks", () => {
    assert.equal(validateQuestions(legacy), legacy);
});

test("a plans-v1 decision is one request with the legacy wording and a weapon branch per admissible motion", async () => {
    const game = new adapter.Game(2001);
    adapter.initialize(game);
    game.start();
    const body = { ...adapter.observe(game, 1, { controllerMode: "plans", interval: 400, rtt: 500 }).request, id: "g", player: 1, round_id: 1 };
    const asked = [];
    const answer = (r) => ({
        id: r.id,
        stateText: r.parts.essential.join("\n"),
        trimmed: [],
        answers: Object.fromEntries(Object.entries(r.questions).map(([head, q]) => {
            const keys = Object.keys(q.criteria);
            return [head, { choice: keys[0], probabilities: Object.fromEntries(keys.map((k, i) => [k, i ? 0 : 1])) }];
        })),
    });
    const reply = await decide(
        {
            model: "fixture",
            agentIds: ["a", "b"],
            contextLimit: null,
            branches: 16,
            ask: async (r) => {
                asked.push(r);
                return { ...answer(r), branches: Object.fromEntries(Object.entries(r.branches ?? {}).map(([k, b]) => [k, answer(b)])) };
            },
        },
        body,
        { questions: legacy, legacyNanoJev: true },
    );
    assert.equal(asked.length, 1);
    assert.deepEqual(Object.keys(asked[0].branches), body.admissible_motion_ids);
    assert.equal(asked[0].questions.motion.instructions, legacy.motion);
    // The legacy wording keeps each option's own description instead of a pointer into the state.
    const [first] = body.motions;
    assert.equal(asked[0].questions.motion.criteria[first.id], first.description);
    assert.ok(body.admissible_motion_ids.includes(reply.motion_id));
});
