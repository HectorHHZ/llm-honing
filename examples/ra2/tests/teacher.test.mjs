import assert from "node:assert/strict";
import test from "node:test";
import { MacroTeacher } from "../league/teacher.mjs";

function state() {
    const own = ["NACNST", "NAPOWR", "NAREFN", "NAHAND", "NAWEAP"].map(
        (name, i) => ({ name, tag: i + 1, x: 0, y: 0, build_progress: 100 }),
    );
    own.push(
        ...Array.from({ length: 5 }, (_, i) => ({
            name: "HTNK",
            tag: 10 + i,
            x: 0,
            y: 0,
            health: 100,
        })),
    );
    return {
        tick: 0,
        frame: { own, visible: [] },
        facade: { res: 100 },
        catalog: { functions: [] },
        obs: {
            vectors: {
                production: [{ queue: "vehicles", size: 2 }],
                worker_count: 3,
                minerals: 0,
                power_cap: 100,
                power_used: 0,
            },
        },
        meta: {
            techno: new Map([["NACNST", { constructionYard: true, type: 2 }]]),
            general: { baseUnit: ["SMCV"] },
        },
    };
}

test("training teacher searches public locations using observed arrival", () => {
    const teacher = new MacroTeacher(),
        policy = state();
    const candidate = (key, world) => ({
        key,
        label: "unexplored possible enemy base",
        action: { function: 22, unit_tags: [10, 11, 12, 13, 14], world },
    });
    const packet = {
        groups: [
            {
                choices: [
                    { key: "A", action: null },
                    candidate("near", 20),
                    candidate("far", 50),
                ],
            },
        ],
    };
    assert.equal(teacher.choose(policy, packet).key, "near");
    teacher.observe(
        [{ function: 22, unit_tags: [10, 11, 12, 13, 14], world: 20 }],
        0,
    );
    policy.tick = 3;
    assert.equal(teacher.choose(policy, packet).key, "A");
    policy.tick = 150;
    policy.frame.own.find((u) => u.tag === 10).x = 20;
    assert.equal(teacher.choose(policy, packet).key, "far");
});

test("training teacher counts an unfolding building as already present", () => {
    const teacher = new MacroTeacher(),
        policy = state();
    policy.frame.own.find((u) => u.name === "NAHAND").build_progress = 25;
    policy.frame.own = policy.frame.own.filter((u) => u.name !== "NAWEAP");
    policy.catalog.functions = [
        { id: 1008, name: "Build_NAHAND" },
        { id: 1011, name: "Build_NAWEAP" },
    ];
    const packet = {
        groups: [
            {
                choices: [
                    { key: "A", action: null },
                    { key: "duplicate", action: { function: 1008 } },
                    { key: "factory", action: { function: 1011 } },
                ],
            },
        ],
    };
    assert.equal(teacher.choose(policy, packet).key, "factory");
});

test("teacher uses an offered squad move when tank attack actors were omitted", () => {
    const teacher = new MacroTeacher(),
        policy = state();
    policy.meta.techno.set("NAPOWR", { type: 2 });
    policy.frame.visible = [{ tag: 999, name: "NAPOWR", x: 20, y: 30 }];
    const action = (world) => ({
        function: 22,
        unit_tags: [10, 11, 12, 13, 14],
        world,
    });
    const packet = {
        groups: [
            {
                choices: [
                    { key: "A", action: null },
                    { key: "home", label: "own base", action: action(0) },
                    {
                        key: "assault",
                        label: "visible NAPOWR",
                        action: action(3020),
                    },
                ],
            },
        ],
    };
    assert.equal(teacher.choose(policy, packet).key, "assault");
});

test("teacher uses offered attack-move while most reinforcements are distant", () => {
    const teacher = new MacroTeacher(),
        policy = state();
    policy.meta.techno.set("NAPOWR", { type: 2 });
    policy.frame.visible = [
        { tag: 999, name: "NAPOWR", x: 30, y: 0, health: 100 },
    ];
    const tags = [10, 11, 12, 13, 14];
    const attack = {
        key: "target",
        label: "visible power plant",
        action: { function: 23, unit_tags: tags, target_unit_tag: 999 },
    };
    const advance = {
        key: "advance",
        label: "visible target location",
        action: { function: 22, unit_tags: tags, world: 30 },
    };
    const packet = {
        groups: [{ choices: [{ key: "A", action: null }, attack, advance] }],
    };
    assert.equal(teacher.choose(policy, packet).key, "advance");
    for (const tag of tags.slice(0, 3))
        policy.frame.own.find((u) => u.tag === tag).x = 25;
    assert.equal(teacher.choose(policy, packet).key, "target");
    packet.groups[0].choices.pop();
    for (const tag of tags) policy.frame.own.find((u) => u.tag === tag).x = 0;
    assert.equal(teacher.choose(policy, packet).key, "target");
});

test("teacher revisits stale public locations and excludes the offered home search", () => {
    const teacher = new MacroTeacher(),
        policy = state();
    policy.searchVisits = new Map([
        [20, 150],
        [50, 30],
    ]);
    const make = (key, world, label) => ({
        key,
        label,
        action: { function: 22, unit_tags: [10, 11, 12, 13, 14], world },
    });
    const packet = {
        groups: [
            {
                choices: [
                    { key: "A", action: null },
                    make("home", 0, "(own base)"),
                    make("recent", 20, "map waypoint"),
                    make("stale", 50, "map waypoint"),
                ],
            },
        ],
    };
    const before = JSON.stringify([...policy.searchVisits]);
    assert.equal(teacher.choose(policy, packet).key, "stale");
    assert.equal(JSON.stringify([...policy.searchVisits]), before);
});
