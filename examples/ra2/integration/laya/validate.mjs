import assert from "node:assert/strict";
import { writeFileSync, mkdirSync } from "node:fs";
import { ObservationMemory, rowsOf } from "./context.mjs";
import { LayaPolicy, validateAction } from "./policy.mjs";
import { SupalosaAdapter } from "../adapter.mjs";
import { loadMetadata, ROOT } from "../metadata.mjs";
import { Ra2Engine, GAME_DIR } from "../dependencies.mjs";
const checks = [];
const ok = (name) => {
    checks.push(name);
    console.log("PASS", name);
};
let meta, engine;
try {
    meta = await loadMetadata("mp03t4.map");
    engine = await Ra2Engine.start({ gameDir: GAME_DIR });
    let c = await engine.create({
        map: "mp03t4.map",
        seed: 1,
        max_ticks: 3000,
        players: [
            { name: "red", country: "Russians", start: 0 },
            { name: "blue", country: "Russians", start: 1 },
        ],
        options: { unit_count: 0, credits: 10000 },
        observation: { resolution: 146, mask_offscreen_enemies: false },
        view: "normal",
        privileged: true,
    });
    const seats = c.players.map((p) =>
        p.name === "red" ? p : { name: p.name, player_id: p.player_id },
    );
    assert(!("start_cell" in seats[1]) && !("country" in seats[1]));
    const config = {
        name: "red",
        country: "Russians",
        metadata: meta,
        catalog: c.catalog,
        seats,
        strictNormal: true,
    };
    let policy = new LayaPolicy(config);
    policy.refresh(c.observations.red, 0);
    let packet = policy.packet();
    assert.equal(policy.frame.visible.length, 0);
    assert.equal(policy.frame.memory.length, 0);
    assert(
        packet.groups
            .find((g) => g.id === "utility")
            .choices.some((c) => c.action?.function === 28),
    );
    ok(
        "real initial normal view has no hidden enemy and offers concrete MCV deployment",
    );
    const before = JSON.stringify(packet),
        poisoned = structuredClone(c.observations.red);
    poisoned.privileged = {
        players: [{ credits: 999999 }],
        units: { tag: 7654321 },
    };
    poisoned.enemyActions = [{ function: 1001 }];
    poisoned.outcome = { red: 1 };
    const isolated = new LayaPolicy(config);
    isolated.refresh(poisoned, 0);
    assert.equal(JSON.stringify(isolated.packet()), before);
    ok(
        "privileged state, opponent commands and outcome injection do not affect context or candidates",
    );
    const hash = await engine.hash();
    await engine.observe({ view: "god", privileged: true });
    assert.deepEqual(await engine.hash(), hash);
    const normal = await engine.observe({ view: "normal" });
    const p2 = new LayaPolicy(config);
    p2.refresh(normal.observations.red, 0);
    assert.equal(JSON.stringify(p2.packet()), before);
    ok("god observer cannot contaminate later normal policy input");
    const obs = structuredClone(c.observations.red),
        features = obs.units.features,
        w = features.length;
    const enemyRow = Array.from(obs.units.data.data).slice(0, w);
    const set = (key, val) => (enemyRow[features.indexOf(key)] = val);
    set("tag", 876543);
    set("alliance", 4);
    set("owner", 1);
    set("display_type", 1);
    set("is_blip", 0);
    obs.units.data.data = Int32Array.from([
        ...obs.units.data.data,
        ...enemyRow,
    ]);
    obs.units.data.shape[0]++;
    const memory = new ObservationMemory();
    memory.update(obs, c.catalog, 0);
    set("display_type", 2);
    obs.units.data.data.set(enemyRow, w);
    const stale = memory.update(obs, c.catalog, 15);
    assert.equal(stale.visible.length, 0);
    assert.equal(stale.memory[0].lastSeenTick, 0);
    assert.equal(stale.memory[0].ageTicks, 15);
    assert(memory.events.some((e) => e.kind === "lost_sight"));
    assert(!memory.events.some((e) => e.text.includes("killed")));
    assert.equal(
        validateAction(
            {
                function: 23,
                unit_tags: [stale.own[0].tag],
                target_unit_tag: 876543,
            },
            obs,
            stale,
            c.catalog,
        ),
        "target_not_currently_visible",
    );
    const strict = new SupalosaAdapter(config);
    strict.refresh(obs, 15);
    assert.equal(strict.game.getUnitData(876543), undefined);
    assert.equal(strict.game.getVisibleUnits("red", "enemy").length, 0);
    assert.equal(memory.update(obs, c.catalog, 916).memory.length, 0);
    ok(
        "enemy snapshots stay stale, expire, cannot be targeted by either policy and do not assert kills",
    );
    const capped = new ObservationMemory();
    capped.update(c.observations.red, c.catalog, 0);
    const empty = structuredClone(c.observations.red);
    empty.units.data.shape[0] = 0;
    empty.units.data.data = new Int32Array();
    empty.vectors.units_truncated = 1;
    capped.update(empty, c.catalog, 15);
    assert(!capped.events.some((e) => e.kind === "own_count"));
    ok("observation row cap does not manufacture unit-loss events");
    const decisions = Object.fromEntries(
        packet.groups.map((g) => [
            g.id,
            {
                choice:
                    g.choices.find((c) => c.action?.function === 28)?.key ??
                    "A",
            },
        ]),
    );
    let selected = policy.select(decisions);
    assert(selected.actions.some((a) => a.function === 28));
    c = await engine.step(
        { red: selected.actions },
        { ticks: 60, view: "normal", privileged: true },
    );
    policy.receipts(0, c.results.red);
    policy.refresh(c.observations.red, c.tick);
    assert(policy.frame.own.some((u) => u.name === "NACNST"));
    ok(
        "grounded deployment actually creates a construction yard in authoritative engine",
    );
    packet = policy.packet();
    const build = packet.groups.find((g) => g.id === "produce_structures");
    assert(build);
    const plant = build.choices.find((c) => c.label.includes("NAPOWR"));
    assert(plant);
    let answers = Object.fromEntries(
        packet.groups.map((g) => [
            g.id,
            { choice: g.id === build.id ? plant.key : "A" },
        ]),
    );
    selected = policy.select(answers);
    c = await engine.step(
        { red: selected.actions },
        { ticks: 600, view: "normal", privileged: true },
    );
    policy.receipts(60, c.results.red);
    policy.refresh(c.observations.red, c.tick);
    assert.equal(
        c.observations.red.vectors.production.find(
            (q) => q.queue === "structures",
        ).status,
        "Ready",
    );
    let placed = false;
    for (let attempt = 0; attempt < 15; attempt++) {
        packet = policy.packet();
        const group = packet.groups.find((g) => g.id === "place_structures");
        assert(group);
        answers = Object.fromEntries(
            packet.groups.map((g) => [
                g.id,
                { choice: g.id === group.id ? g.choices[1].key : "A" },
            ]),
        );
        selected = policy.select(answers);
        const tick = c.tick;
        c = await engine.step(
            { red: selected.actions },
            { ticks: 15, view: "normal", privileged: true },
        );
        policy.receipts(tick, c.results.red);
        policy.refresh(c.observations.red, c.tick);
        if (policy.frame.own.some((u) => u.name === "NAPOWR")) {
            placed = true;
            break;
        }
    }
    assert(placed);
    ok(
        "production and selected placement create a real power plant, not just accepted receipts",
    );
    packet = policy.packet();
    const own = policy.frame.own[0];
    assert.equal(
        validateAction(
            { function: 28, unit_tags: [876543] },
            policy.obs,
            policy.frame,
            policy.catalog,
        ),
        "not_owned",
    );
    policy.current = {
        groups: [
            {
                id: "x",
                choices: [
                    {
                        key: "a",
                        action: { function: 28, unit_tags: [own.tag] },
                        cost: 20000,
                    },
                ],
            },
            {
                id: "y",
                choices: [
                    {
                        key: "a",
                        action: { function: 28, unit_tags: [own.tag] },
                        cost: 0,
                    },
                ],
            },
            {
                id: "z",
                choices: [
                    {
                        key: "a",
                        action: { function: 28, unit_tags: [own.tag] },
                        cost: 0,
                    },
                ],
            },
        ],
    };
    const conflicts = policy.select({
        x: { choice: "a" },
        y: { choice: "a" },
        z: { choice: "a" },
    });
    assert(conflicts.dropped.some((d) => d.reason === "shared_budget"));
    assert(conflicts.dropped.some((d) => d.reason === "actor_conflict"));
    assert.equal(conflicts.actions.length, 1);
    ok(
        "shared budget and duplicate actor conflicts cannot create extra actions",
    );
    assert.equal(
        validateAction(
            { function: 47, unit_tags: [own.tag], target_unit_tag: own.tag },
            policy.obs,
            policy.frame,
            policy.catalog,
        ),
        null,
    );
    const neutral = {
        ...own,
        tag: 876542,
        alliance: 3,
        display_type: 1,
        is_blip: 0,
    };
    const nf = { ...policy.frame, rows: [...policy.frame.rows, neutral] };
    assert.equal(
        validateAction(
            {
                function: 45,
                unit_tags: [own.tag],
                target_unit_tag: neutral.tag,
            },
            policy.obs,
            nf,
            policy.catalog,
        ),
        null,
    );
    const maskProbe = structuredClone(policy.obs);
    maskProbe.vectors.function_mask.data.fill(0);
    assert.equal(
        validateAction(
            { function: policy.facade.buildFns.get("NAPOWR") },
            maskProbe,
            policy.frame,
            policy.catalog,
        ),
        null,
    );
    ok(
        "owned and visible neutral targets remain legal; masks do not preempt authoritative action receipts",
    );
    mkdirSync(ROOT + "runs/validation-validation", { recursive: true });
    writeFileSync(
        ROOT + "runs/validation-validation/result.json",
        JSON.stringify({ passed: true, checks }, null, 2),
    );
} finally {
    if (engine) await engine.shutdown();
    meta?.dispose();
}
