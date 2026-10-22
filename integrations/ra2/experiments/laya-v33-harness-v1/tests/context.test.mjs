import test from "node:test";
import assert from "node:assert/strict";
import {
    enhancePacket,
    CONTEXT_VERSION,
    MAX_NOTE_CHARS,
} from "../harness/context.mjs";

const unit = (tag, name, overrides = {}) => ({
    tag,
    name,
    alliance: 1,
    health: 100,
    health_ratio: 255,
    build_progress: 100,
    can_move: 0,
    is_idle: 1,
    is_in_cargo: 0,
    x: 10,
    y: 10,
    display_type: 1,
    is_blip: 0,
    ...overrides,
});
const queue = (name, overrides = {}) => ({
    queue: name,
    status: "Idle",
    size: 0,
    item: 0,
    progress: 0,
    available: [],
    prices: [],
    ...overrides,
});
const leaf = (fn, key = "L1", overrides = {}) => ({
    key,
    label: "An offered option",
    cost: 0,
    intent: { id: "base" },
    action: { function: fn },
    ...overrides,
});
function fixture(leaves = []) {
    const choices = [
        { key: "A", label: "Continue production", action: null, cost: 0 },
        ...leaves,
    ];
    return {
        packet: {
            tick: 300,
            groups: [
                {
                    id: "next_action",
                    hierarchy: "intent-function",
                    contextVersion: 6,
                    instructions: "Choose the next useful action.",
                    choices,
                },
            ],
            questions: {
                next_action: {
                    type: "choice",
                    instructions: "Choose the next useful action.",
                    criteria: Object.fromEntries(
                        choices.map((c) => [c.key, c.label]),
                    ),
                },
            },
            parts: {
                essential: [
                    "Original essential state",
                    "Original recommended objective",
                ],
                details: ["detail"],
                memory: [],
                logs: [],
            },
            availability: [
                { id: 1001, name: "Build_NAWEAP", status: "offered" },
            ],
        },
        env: {
            tick: 300,
            observation: {
                vectors: {
                    minerals: 4000,
                    power_cap: 150,
                    power_used: 50,
                    is_low_power: 0,
                    units_truncated: 0,
                    production: [
                        "structures",
                        "armory",
                        "infantry",
                        "vehicles",
                        "aircraft",
                        "ships",
                    ].map((n) => queue(n)),
                },
            },
            frame: {
                own: [
                    unit(1, "NACNST"),
                    unit(2, "NAREFN"),
                    unit(3, "HARV", { can_move: 1 }),
                ],
                visible: [],
                memory: [],
            },
        },
    };
}
function apply(f) {
    return enhancePacket(f.packet, f.env);
}
function deepFreeze(obj) {
    Object.freeze(obj);
    for (const x of Object.values(obj))
        if (x && typeof x === "object" && !Object.isFrozen(x)) deepFreeze(x);
    return obj;
}

test("wait, sell, pause and diplomacy alone produce no unsolicited advice", () => {
    const f = fixture([
        leaf(49),
        leaf(62, "L2"),
        leaf(80, "L3"),
        leaf(81, "L4"),
    ]);
    const result = apply(f);
    assert.deepEqual(result.notes, []);
    assert.deepEqual(result.packet, f.packet);
    assert.notEqual(result.packet, f.packet);
    assert.equal(result.version, CONTEXT_VERSION);
});

test("only actually offered ready placement gets a bounded note, all original content is unchanged", () => {
    const f = fixture([
        leaf(60, "L1", {
            action: { function: 60, world: 10, extra: { item: 16 } },
        }),
    ]);
    f.env.observation.vectors.production[0] = queue("structures", {
        status: "Ready",
        size: 1,
        item: 16,
        progress: 100,
    });
    deepFreeze(f);
    const result = apply(f);
    assert.equal(result.notes[0].code, "ready-building-placement");
    assert.deepEqual(
        result.packet.parts.essential.slice(0, -1),
        f.packet.parts.essential,
    );
    for (const key of ["groups", "questions", "availability"])
        assert.deepEqual(result.packet[key], f.packet[key]);
    for (const key of ["details", "memory", "logs"])
        assert.deepEqual(result.packet.parts[key], f.packet.parts[key]);
    assert.match(result.notes[0].text, /Observed:.*Advice \(not a command\):/);
    assert.ok(result.notes[0].text.length <= MAX_NOTE_CHARS);
    assert.deepEqual(result.notes[0].candidateKeys, ["L1"]);
});

test("ready queue without matching placement action does not invent an option", () => {
    const f = fixture([
        leaf(60, "L1", { action: { function: 60, extra: { item: 99 } } }),
    ]);
    f.env.observation.vectors.production[0] = queue("structures", {
        status: "Ready",
        size: 1,
        item: 16,
    });
    assert.deepEqual(apply(f).notes, []);
});

test("parallel military recruitment requires active construction, income, empty queues and affordability", () => {
    const f = fixture([
        leaf(2002, "L1", { cost: 900, intent: { id: "army" } }),
    ]);
    f.env.observation.vectors.production[0] = queue("structures", {
        status: "Active",
        size: 1,
        item: 15,
    });
    assert.equal(apply(f).notes[0].code, "parallel-military-production");
    f.env.observation.vectors.production[3] = queue("vehicles", {
        status: "Active",
        size: 1,
        item: 200,
    });
    assert.deepEqual(apply(f).notes, []);
    f.env.observation.vectors.production[3] = queue("vehicles");
    f.env.observation.vectors.minerals = 200;
    assert.deepEqual(apply(f).notes, []);
    f.env.observation.vectors.minerals = 4000;
    f.env.frame.own = f.env.frame.own.filter((u) => u.name !== "HARV");
    assert.deepEqual(apply(f).notes, []);
});

test("missing production note does not call an incomplete or absent factory destroyed", () => {
    const f = fixture([leaf(1001, "L1", { cost: 2000 })]);
    let result = apply(f);
    assert.equal(result.notes[0].code, "missing-vehicle-production");
    assert.doesNotMatch(result.notes[0].text, /destroyed|lost/i);
    f.env.observation.vectors.units_truncated = 1;
    assert.deepEqual(apply(f).notes, []);
    f.env.observation.vectors.units_truncated = 0;
    f.env.frame.own.push(unit(7, "NAWEAP", { build_progress: 40 }));
    assert.deepEqual(apply(f).notes, []);
});

test("nearby visible target response outranks routine ready placement without claiming attack", () => {
    const f = fixture([
        leaf(23, "L1", {
            action: { function: 23, unit_tags: [4], target_unit_tag: 20 },
            intent: { id: "combat" },
        }),
        leaf(60, "L2", { action: { function: 60, extra: { item: 16 } } }),
    ]);
    f.env.frame.own.push(unit(4, "HTNK", { can_move: 1 }));
    f.env.frame.visible = [unit(20, "HTNK", { alliance: 4, x: 16 })];
    f.env.observation.vectors.production[0] = queue("structures", {
        status: "Ready",
        size: 1,
        item: 16,
    });
    const result = apply(f);
    assert.equal(result.notes.length, 1);
    assert.equal(result.notes[0].code, "visible-enemy-near-economy");
    assert.doesNotMatch(
        result.notes[0].text,
        /enemy is attacking|under attack/,
    );
    assert.deepEqual(result.notes[0].evidence.visibleTargetTags, [20]);
});

test("hidden, remembered, blip and allied objects never become visible attack evidence", () => {
    const f = fixture([
        leaf(23, "L1", {
            action: { function: 23, unit_tags: [4], target_unit_tag: 20 },
            intent: { id: "combat" },
        }),
    ]);
    f.env.frame.own.push(unit(4, "HTNK", { can_move: 1 }));
    f.env.frame.memory = [unit(20, "HTNK", { alliance: 4 })];
    for (const bad of [
        [],
        [unit(20, "HTNK", { alliance: 4, display_type: 2 })],
        [unit(20, "HTNK", { alliance: 4, is_blip: 1 })],
        [unit(20, "HTNK", { alliance: 2 })],
    ]) {
        f.env.frame.visible = bad;
        assert.deepEqual(apply(f).notes, []);
    }
});

test("scouting advice needs idle military actors and an offered attack-move", () => {
    const f = fixture([
        leaf(22, "L1", {
            action: { function: 22, unit_tags: [4], world: 100 },
            intent: { id: "combat" },
        }),
    ]);
    f.env.frame.own.push(unit(4, "HTNK", { can_move: 1 }));
    assert.equal(apply(f).notes[0].code, "idle-military-scout");
    f.env.frame.own.at(-1).is_idle = 0;
    assert.deepEqual(apply(f).notes, []);
    f.env.frame.own.at(-1).is_idle = 1;
    f.env.frame.own.at(-1).name = "HARV";
    assert.deepEqual(apply(f).notes, []);
    f.env.frame.own.at(-1).name = "HTNK";
    f.env.frame.own.at(-1).alliance = 4;
    assert.deepEqual(apply(f).notes, []);
});

test("paused queue resume needs matching queue and available credits", () => {
    const f = fixture([
        leaf(63, "L1", {
            action: { function: 63, extra: { queue: "structures" } },
        }),
    ]);
    f.env.observation.vectors.production[0] = queue("structures", {
        status: "OnHold",
        size: 1,
        item: 16,
    });
    assert.equal(apply(f).notes[0].code, "paused-production");
    f.env.observation.vectors.minerals = 0;
    assert.deepEqual(apply(f).notes, []);
});

test("repair advice never reverses active repair or invents a damaged target", () => {
    const f = fixture([
        leaf(50, "L1", {
            label: "Start repairing NACNST #1",
            action: { function: 50, unit_tags: [1] },
        }),
    ]);
    f.env.frame.own[0].health_ratio = 100;
    assert.equal(apply(f).notes[0].code, "damaged-key-building");
    f.packet.groups[0].choices[1].label = "Stop repairing NACNST #1";
    assert.deepEqual(apply(f).notes, []);
});

test("no decision packet, missing evidence and mismatched ticks are explicit", () => {
    const f = fixture();
    f.packet.groups = [];
    assert.deepEqual(apply(f).notes, []);
    assert.throws(
        () => enhancePacket(f.packet, { ...f.env, tick: 303 }),
        /same tick/,
    );
    assert.throws(() => enhancePacket(f.packet), /observation and frame/);
    assert.throws(() => enhancePacket({}), /frozen-v33 packet/);
});
