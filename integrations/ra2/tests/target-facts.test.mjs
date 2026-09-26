import assert from "node:assert/strict";
import test from "node:test";
import { attackTargetFacts } from "../integration/laya/context.mjs";

test("each visible target has current HP, position and distance from the chosen actors", () => {
    const frame = {
        own: [
            { tag: 1, alliance: 1, x: 0, y: 0 },
            { tag: 2, alliance: 1, x: 3, y: 4 },
            { tag: 3, alliance: 1, x: 60, y: 80 },
        ],
        visible: Array.from({ length: 7 }, (_, i) => ({
            tag: 100 + i,
            alliance: 4,
            display_type: 1,
            is_blip: 0,
            health: 150 - i,
            x: 6,
            y: 8,
        })),
    };
    const action = { function: 23, unit_tags: [1, 2], target_unit_tag: 106 };
    const before = structuredClone({ frame, action });
    assert.equal(
        attackTargetFacts(frame, action),
        "HP 144; at 6,8; direct 5.0-10.0 cells",
    );
    assert.equal(
        attackTargetFacts(frame, { ...action, unit_tags: [2] }),
        "HP 144; at 6,8; direct 5.0 cells",
    );
    assert.deepEqual({ frame, action }, before);
});

test("fogged, radar-only, friendly, absent and dead targets expose no facts", () => {
    const target = {
        tag: 7,
        alliance: 4,
        display_type: 1,
        is_blip: 0,
        health: 100,
        x: 5,
        y: 5,
    };
    const action = { function: 23, target_unit_tag: 7, unit_tags: [1] };
    for (const change of [
        { display_type: 2 },
        { is_blip: 1 },
        { alliance: 1 },
        { tag: 8 },
        { health: 0 },
    ]) {
        assert.equal(
            attackTargetFacts(
                { visible: [{ ...target, ...change }], own: [] },
                action,
            ),
            "",
        );
    }
    assert.equal(attackTargetFacts({ visible: [], own: [] }, action), "");
    assert.equal(
        attackTargetFacts(
            { visible: [{ ...target, x: undefined }], own: [] },
            action,
        ),
        "HP 100",
    );
});
