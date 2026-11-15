import test from "node:test";
import assert from "node:assert/strict";
import {
    probabilityProbe,
    textDifference,
} from "../scripts/probe-recorded-window.mjs";

test("rerender probe ignores equal objects and reports a real text change", () => {
    assert.equal(textDifference({ aim: ["right"] }, { aim: ["right"] }), null);
    const difference = textDifference(
        { state: "Shot guard=OFF; assistance=OFF" },
        { state: "Shot guard=OFF; assistance=ON" },
    );
    assert.match(difference.observed_excerpt, /assistance=OFF/);
    assert.match(difference.reference_excerpt, /assistance=ON/);
});

test("same-request probe compares every branch and option probability", () => {
    const observed = {
        answers: {
            motion: {
                choice: "left",
                probabilities: { left: 0.6, right: 0.4 },
            },
        },
        branches: {
            left: {
                answers: {
                    weapon: {
                        choice: "hold",
                        probabilities: { fire: 0.2, hold: 0.8 },
                    },
                },
            },
        },
    };
    assert.equal(
        probabilityProbe(observed, structuredClone(observed))
            .max_abs_probability_difference,
        0,
    );
    const reference = structuredClone(observed);
    reference.branches.left.answers.weapon.probabilities = {
        fire: 0.8,
        hold: 0.2,
    };
    reference.branches.left.answers.weapon.choice = "fire";
    const result = probabilityProbe(observed, reference);
    assert.ok(result.max_abs_probability_difference > 0.5);
    assert.equal(result.distributions.length, 2);
    assert.equal(result.distributions[1].reference_choice, "fire");
});
