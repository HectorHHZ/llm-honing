import assert from "node:assert/strict";
import { IntentPolicy } from "./intent-policy.mjs";
import { predictStaged, verifyStagedTrace } from "./staged.mjs";
import { loadMetadata, ROOT } from "../metadata.mjs";
import { Ra2Engine, GAME_DIR } from "../dependencies.mjs";
let meta, engine;
try {
    meta = await loadMetadata("mp03t4.map");
    engine = await Ra2Engine.start({ gameDir: GAME_DIR });
    let s = await engine.create({
        map: "mp03t4.map",
        seed: 1,
        max_ticks: 2000,
        players: [
            { name: "red", country: "Russians", start: 0 },
            { name: "blue", country: "Russians", start: 1 },
        ],
        options: { unit_count: 0, credits: 10000, super_weapons: false },
        observation: { resolution: 146, mask_offscreen_enemies: false },
        view: "normal",
        privileged: true,
    });
    const initial = structuredClone(s.observations.red);
    const p = new IntentPolicy({
        name: "red",
        country: "Russians",
        metadata: meta,
        catalog: s.catalog,
        seats: s.players.map((x) =>
            x.name === "red" ? x : { name: x.name, player_id: x.player_id },
        ),
        strictNormal: true,
        decisionInterval: 3,
    });
    async function choose(fn, ticks) {
        p.refresh(s.observations.red, s.tick);
        const packet = p.packet(),
            leaf = packet.groups[0].choices.find(
                (c) => c.action?.function === fn,
            );
        assert(leaf);
        const hash = await engine.hash();
        const decision = await predictStaged(packet, {
            predict: async (req) => {
                assert.deepEqual(await engine.hash(), hash);
                const c = req.groups[0].choices.find((c) =>
                    c.leafKeys.includes(leaf.key),
                );
                assert(c);
                return {
                    answers: { next_action: { choice: c.key } },
                    latencyMs: 0,
                    fixture: true,
                };
            },
        });
        verifyStagedTrace(packet, decision, assert);
        const sel = p.select(decision.resolvedAnswers);
        assert.equal(sel.actions.length, 1);
        const tick = s.tick;
        s = await engine.step(
            { red: sel.actions },
            { ticks, view: "normal", privileged: true },
        );
        p.receipts(tick, s.results.red);
        return decision;
    }
    await choose(28, 60);
    p.refresh(s.observations.red, s.tick);
    let packet = p.packet();
    assert(!packet.groups[0].choices.some((c) => c.action?.function === 28));
    assert(
        packet.parts.logs.some(
            (x) => x.includes("construction yard") && x.includes("0 -> 1"),
        ),
    );
    const before = JSON.stringify(packet);
    await engine.observe({ view: "god", privileged: true });
    assert.equal(JSON.stringify(p.packet()), before);
    await choose(p.facade.buildFns.get("NAPOWR"), 600);
    const decision = await choose(60, 120);
    assert(decision.stages.length >= 2);
    p.refresh(s.observations.red, s.tick);
    packet = p.packet();
    assert(p.frame.own.some((u) => u.name === "NAPOWR"));
    const sale = await predictStaged(packet, {
        predict: async (req) => ({
            answers: {
                next_action: {
                    choice:
                        req.groups[0].dimension === "confirmation"
                            ? "A"
                            : req.groups[0].choices.find((c) =>
                                  c.leafKeys.some(
                                      (k) =>
                                          packet.groups[0].choices.find(
                                              (x) => x.key === k,
                                          )?.action?.function === 49,
                                  ),
                              ).key,
                },
            },
            latencyMs: 0,
        }),
    });
    assert.equal(p.select(sale.resolvedAnswers).actions.length, 0);
    const obs = structuredClone(initial),
        features = obs.units.features,
        width = features.length,
        template = Object.fromEntries(
            features.map((k, i) => [k, obs.units.data.data[i]]),
        );
    const specs = [
        { name: "HARV", tag: 9001, order_id_0: 46, is_idle: 0 },
        { name: "HTNK", tag: 9002 },
        { name: "HTNK", tag: 9003 },
        {
            name: "E2",
            tag: 9010,
            alliance: 4,
            owner: 1,
            x: template.x + 3,
            display_type: 1,
            is_blip: 0,
        },
        {
            name: "E2",
            tag: 9020,
            alliance: 4,
            owner: 1,
            x: template.x + 30,
            display_type: 1,
            is_blip: 0,
        },
    ];
    const rows = specs.map((x) => ({
        ...template,
        ...x,
        unit_type: p.catalog.unit_types.indexOf(x.name),
        health: meta.techno.get(x.name).strength,
        health_ratio: 255,
        build_progress: 100,
        can_move: 1,
    }));
    obs.units.data.shape[0] = rows.length;
    obs.units.data.data = Int32Array.from(
        rows.flatMap((u) => features.map((k) => u[k] ?? 0)),
    );
    p.refresh(obs, 1200);
    packet = p.packet();
    const leaves = packet.groups.flatMap((g) =>
        g.choices.filter((c) => c.action),
    );
    assert(
        !leaves.some(
            (c) =>
                c.action.function === 22 && c.action.unit_tags.includes(9001),
        ),
    );
    assert(
        leaves.some(
            (c) =>
                c.action.function === 22 && c.action.unit_tags.includes(9002),
        ),
    );
    assert(
        leaves.some(
            (c) =>
                c.action.function === 23 &&
                c.action.unit_tags.includes(9001) &&
                c.action.target_unit_tag === 9010,
        ),
    );
    assert(
        !leaves.some(
            (c) =>
                c.action.function === 23 &&
                c.action.unit_tags.includes(9001) &&
                c.action.target_unit_tag === 9020,
        ),
    );
    assert(
        leaves.some(
            (c) =>
                c.action.function === 23 &&
                c.action.unit_tags.length === 2 &&
                c.action.unit_tags.includes(9002) &&
                c.action.unit_tags.includes(9003),
        ),
    );
    const attackLeaves = leaves.filter((c) => c.action.function === 23);
    assert(attackLeaves.length > 0);
    for (const c of attackLeaves) {
        const target = rows.find((u) => u.tag === c.action.target_unit_tag);
        assert(
            c.label.includes(`HP ${target.health}; at ${target.x},${target.y}`),
        );
        assert(c.label.includes("direct "));
    }
    assert.equal(
        packet.availability.find((x) => x.id === 29).status,
        "wait_equivalent",
    );
    assert(!leaves.some((c) => c.action.function === 29));
    console.log(
        "PASS synthetic normal-only role fixtures retain nearby truck defense and squad attacks, exclude distant truck attacks, and merge cosmetic camera controls with wait",
    );
    console.log(
        "PASS intent hierarchy resolves one grounded command with frozen state; confirmed effects enter context; no-op yard deploy removed; sale can be cancelled; god observer does not affect candidates",
    );
    // Normal-observation macro diagnostics must not manufacture income or losses.
    p.obs = {
        vectors: {
            minerals: 0,
            army_count: 0,
            units_truncated: false,
            production: [
                {
                    queue: "vehicles",
                    status: "Active",
                    size: 25,
                    item: 345,
                    progress: 44,
                },
            ],
        },
    };
    p.frame = { own: [{ name: "HARV", is_idle: 1, order_id_0: 0 }] };
    p.tick = 900;
    p.macroHistory = [{ tick: 0, credits: 900, army: 5, truncated: false }];
    p.queueProgress = new Map([["vehicles", { since: 0 }]]);
    const choices = [{ action: { function: 46 } }],
        unchanged = structuredClone(choices);
    let macro = p.macroContext(choices, 25);
    assert.match(macro.objective, /Restore recurring income/);
    assert.match(macro.facts, /NOT gross income/);
    assert.match(macro.facts, /44% unchanged 60s/);
    assert.deepEqual(choices, unchanged);
    p.frame.own[0].is_idle = 0;
    p.frame.own[0].order_id_0 = 46;
    p.obs.vectors.minerals = 2000;
    p.obs.vectors.production = [];
    macro = p.macroContext(choices, 0);
    assert.match(macro.facts, /1 have Gather\/Dock/);
    assert.match(macro.objective, /Recruit replacements/);
    assert.match(p.macroContext(choices, 4).objective, /already queued/);
    p.obs.vectors.units_truncated = true;
    assert.equal(p.macroContext([], 0).objective, null);
    console.log(
        "PASS macro context diagnoses diverted miners and stalled queues, preserves choices, distinguishes net credits from income, and limits replenishment inference",
    );
} finally {
    await engine?.shutdown();
    meta?.dispose();
}
