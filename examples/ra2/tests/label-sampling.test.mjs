import test from "node:test";
import assert from "node:assert/strict";
import { sampleCorrection } from "../league/label_sampling.mjs";

test("periodic correction thins repeated waits while preserving transient labels", () => {
    const action = [{ function: 60, world: 100, extra: { item: 2 } }];
    const wait = JSON.stringify([]);
    assert.equal(
        sampleCorrection(597, [], [], wait, "transition-periodic"),
        true,
    );
    assert.equal(
        sampleCorrection(600, [], [], wait, "transition-periodic"),
        true,
    );
    assert.equal(
        sampleCorrection(
            603,
            [{ function: 21 }],
            [],
            wait,
            "transition-periodic",
        ),
        false,
    );
    assert.equal(
        sampleCorrection(606, [], action, wait, "transition-periodic"),
        true,
    );
    assert.equal(
        sampleCorrection(
            609,
            [],
            [],
            JSON.stringify(action),
            "transition-periodic",
        ),
        true,
    );
    assert.equal(
        sampleCorrection(630, [], [], wait, "transition-periodic"),
        true,
    );
});

test("legacy mode retains executed action samples and unknown modes fail", () => {
    assert.equal(
        sampleCorrection(603, [{ function: 21 }], [], "[]", "actions-periodic"),
        true,
    );
    assert.throws(
        () => sampleCorrection(603, [], [], "[]", "guess"),
        /Unknown/,
    );
});
