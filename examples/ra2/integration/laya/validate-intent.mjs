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
    const attackLeaves = leaves.filter((c) => c.action.function === 23);
    assert(attackLeaves.length > 0);
    for (const c of attackLeaves) {
        const target = rows.find((u) => u.tag === c.action.target_unit_tag);
        assert(
            c.label.includes(`HP ${target.health}; at ${target.x},${target.y}`),
        );
        assert(c.label.includes("direct "));
    }
    // Many idle infantry must not hide the tank actor type at the 12-actor cap.
    const crowded = [
        ...Array.from({ length: 14 }, (_, i) => ({
            ...rows[1],
            name: "E2",
            tag: 9100 + i,
            unit_type: p.catalog.unit_types.indexOf("E2"),
            is_idle: 1,
        })),
        { ...rows[1], is_idle: 0 },
        { ...rows[2], is_idle: 0 },
        rows[0],
        rows[3],
    ];
    obs.units.data.shape[0] = crowded.length;
    obs.units.data.data = Int32Array.from(
        crowded.flatMap((u) => features.map((k) => u[k] ?? 0)),
    );
    p.refresh(obs, 1500);
    const covered = p
        .packet()
        .groups.flatMap((g) => g.choices)
        .filter((c) => c.action);
    assert(
        covered.some(
            (c) =>
                c.action.function === 23 && c.action.unit_tags.includes(9002),
        ),
    );
    const combined = covered.filter((c) =>
        c.actorLabel?.startsWith("Combined"),
    );
    assert(combined.length > 0);
    assert(combined.every((c) => !c.action.unit_tags.includes(9001)));
    assert(
        combined.some(
            (c) =>
                c.action.unit_tags.includes(9002) &&
                c.action.unit_tags.includes(9100),
        ),
    );
    assert.match(p.packet().groups[0].waitLabel, /0.2 seconds/);
    const chosen = covered.find(
        (c) => c.action.function === 23 && c.action.unit_tags.includes(9002),
    );
    p.select({ next_action: { choice: chosen.key } }, 1);
    p.receipts(1500, [{ accepted: true }]);
    p.refresh(obs, 1503);
    assert(
        p
            .packet()
            .parts.essential.some(
                (s) =>
                    s.includes("0.2s ago") &&
                    s.includes("Last accepted military"),
            ),
    );
    console.log(
        "PASS crowded actors retain tanks and explicit mixed squads exclude harvesters; context records actual cadence and accepted order age",
    );
    const pointsBefore = new Set(
        p
            .packet()
            .groups.flatMap((g) => g.choices)
            .filter(
                (c) =>
                    c.action?.function === 22 &&
                    c.label.includes("map waypoint"),
            )
            .map((c) => c.action.world),
    );
    p.refresh(obs, 1800);
    const pointsAfter = new Set(
        p
            .packet()
            .groups.flatMap((g) => g.choices)
            .filter(
                (c) =>
                    c.action?.function === 22 &&
                    c.label.includes("map waypoint"),
            )
            .map((c) => c.action.world),
    );
    assert(pointsBefore.size > 0 && pointsAfter.size > 0);
    assert([...pointsAfter].some((c) => !pointsBefore.has(c)));
    const frozenSearch = JSON.stringify(p.packet());
    await engine.observe({ view: "god", privileged: true });
    assert.equal(JSON.stringify(p.packet()), frozenSearch);
    console.log(
        "PASS public search targets rotate with game time and remain independent of god observations",
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
    p.frame.own[0].order_id_0 = 20;
    p.frame.own[0].order_id_1 = 47;
    assert.match(
        p.macroContext([], 4).facts,
        /1 have Gather\/Dock orders, 0 idle or diverted/,
    );
    p.obs.vectors.units_truncated = true;
    assert.equal(p.macroContext([], 0).objective, null);
    console.log(
        "PASS macro context diagnoses diverted miners and stalled queues, preserves choices, distinguishes net credits from income, and limits replenishment inference",
    );
    const searchObs = structuredClone(initial);
    const scoutRows = [
        { ...rows[1], tag: 9301, is_idle: 0 },
        { ...rows[2], tag: 9302, is_idle: 0 },
    ];
    searchObs.units.data.shape[0] = scoutRows.length;
    searchObs.units.data.data = Int32Array.from(
        scoutRows.flatMap((u) => features.map((k) => u[k] ?? 0)),
    );
    p.refresh(searchObs, 6000);
    const firstSearchPacket = p.packet();
    const publicMove = firstSearchPacket.groups
        .flatMap((g) => g.choices)
        .find(
            (c) =>
                c.action?.function === 22 &&
                meta.starts.some(
                    (start) =>
                        p.facade.cell(start.x, start.y) === c.action.world,
                ) &&
                Math.hypot(
                    scoutRows[0].x - (c.action.world % p.facade.res),
                    scoutRows[0].y - Math.floor(c.action.world / p.facade.res),
                ) > 20,
        );
    assert(publicMove, "fixture needs public spawn target");
    const location = publicMove.action.world;
    const layerOffset =
        searchObs.world.channels.indexOf("visibility_map") * p.facade.res ** 2;
    searchObs.world.data.data[layerOffset + location] = 1;
    p.refresh(searchObs, 6015);
    let searchPacket = p.packet();
    assert(
        searchPacket.groups
            .flatMap((g) => g.choices)
            .some(
                (c) =>
                    c.action?.function === 22 &&
                    c.action.world === location &&
                    c.label.includes(
                        "previously explored public spawn; now fogged",
                    ),
            ),
    );
    const scout = scoutRows[0];
    scout.x = location % p.facade.res;
    scout.y = Math.floor(location / p.facade.res);
    searchObs.units.data.data = Int32Array.from(
        scoutRows.flatMap((u) => features.map((k) => u[k] ?? 0)),
    );
    p.refresh(searchObs, 6030);
    searchPacket = p.packet();
    assert.equal(p.searchVisits.get(location), 6030);
    assert(
        searchPacket.groups
            .flatMap((g) => g.choices)
            .some(
                (c) =>
                    c.action?.function === 22 &&
                    c.action.world === location &&
                    c.label.includes("own combat troops nearby now"),
            ),
    );
    scout.x = rows[1].x;
    scout.y = rows[1].y;
    searchObs.units.data.data = Int32Array.from(
        scoutRows.flatMap((u) => features.map((k) => u[k] ?? 0)),
    );
    p.refresh(searchObs, 6180);
    searchPacket = p.packet();
    assert.equal(p.searchVisits.get(location), 6030);
    assert(
        searchPacket.groups
            .flatMap((g) => g.choices)
            .some(
                (c) =>
                    c.action?.function === 22 &&
                    c.action.world === location &&
                    c.label.includes("troops visited 10s ago"),
            ),
    );
    const searchBeforeGod = JSON.stringify(searchPacket);
    await engine.observe({ view: "god", privileged: true });
    assert.equal(JSON.stringify(p.packet()), searchBeforeGod);
    console.log(
        "PASS public search labels distinguish fogged terrain, retain own troop visit recency, and ignore privileged observer state",
    );
} finally {
    await engine?.shutdown();
    meta?.dispose();
}
