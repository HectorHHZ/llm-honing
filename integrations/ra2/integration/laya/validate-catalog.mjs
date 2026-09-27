import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { CatalogPolicy } from "./catalog-policy.mjs";
import { validateAction } from "./policy.mjs";
import { predictStaged, verifyStagedTrace } from "./staged.mjs";
import { loadMetadata, ROOT } from "../metadata.mjs";
import { Ra2Engine, GAME_DIR } from "../dependencies.mjs";
const out = ROOT + "runs/validation-catalog-validation";
mkdirSync(out, { recursive: true });
const checks = [],
    snapshots = [],
    traces = [];
const ok = (name) => {
    checks.push(name);
    console.log("PASS", name);
};
const leaves = (p) =>
    p.groups.flatMap((g) => g.choices.filter((c) => c.action));
const offered = (p, id) =>
    p.availability.find((a) => a.id === id)?.status === "offered";
let meta, engine;
try {
    meta = await loadMetadata("mp03t4.map");
    engine = await Ra2Engine.start({ gameDir: GAME_DIR });
    let c = await engine.create({
        map: "mp03t4.map",
        seed: 1,
        max_ticks: 4000,
        players: [
            { name: "red", country: "Russians", start: 0 },
            { name: "blue", country: "Russians", start: 1 },
        ],
        options: {
            unit_count: 0,
            credits: 10000,
            super_weapons: false,
            mcv_repacks: true,
        },
        observation: { resolution: 146, mask_offscreen_enemies: false },
        view: "normal",
        privileged: true,
    });
    const config = {
        name: "red",
        country: "Russians",
        metadata: meta,
        catalog: c.catalog,
        seats: c.players.map((p) =>
            p.name === "red" ? p : { name: p.name, player_id: p.player_id },
        ),
        strictNormal: true,
    };
    const policy = new CatalogPolicy(config);
    policy.refresh(c.observations.red, c.tick);
    let p = policy.packet();
    const catalog = c.catalog,
        initialObs = structuredClone(c.observations.red),
        unitTemplate = { ...policy.frame.own[0] };
    snapshots.push({
        name: "initial-normal",
        packet: p,
        normalUnits: policy.frame.rows,
    });
    assert.equal(p.availability.length, 120);
    assert(!p.availability.some((a) => a.status === "unsupported"));
    assert(offered(p, 28));
    assert(!offered(p, 23));
    assert(!offered(p, 44));
    assert(!offered(p, 46));
    assert(!p.availability.some((a) => a.id >= 3000 && a.status === "offered"));
    assert.equal(p.availability.find((a) => a.id === 28).mask, false);
    for (const leaf of leaves(p))
        assert.equal(
            validateAction(leaf.action, policy.obs, policy.frame, catalog),
            null,
        );
    const before = JSON.stringify(p);
    await engine.observe({ view: "god", privileged: true });
    assert.equal(JSON.stringify(policy.packet()), before);
    ok(
        "120 functions accounted for; initial prerequisites, hidden targets, special-ability masks and normal/god isolation checked",
    );
    async function choose(fn, { name, target, ticks = 3 } = {}) {
        p = policy.packet();
        const group = p.groups.find((g) =>
            g.choices.some(
                (x) =>
                    x.action?.function === fn &&
                    (!name || x.label.includes(name)) &&
                    (!target || x.action.unit_tags?.includes(target)),
            ),
        );
        assert(group, `Function ${fn} (${name ?? ""}) has no offered branch`);
        const leaf = group.choices.find(
            (x) =>
                x.action?.function === fn &&
                (!name || x.label.includes(name)) &&
                (!target || x.action.unit_tags?.includes(target)),
        );
        const frozen = await engine.hash();
        const decision = await predictStaged(p, {
            predict: async (req) => {
                assert.deepEqual(await engine.hash(), frozen);
                const answers = Object.fromEntries(
                    req.groups.map((g) => {
                        const choice =
                            g.id === group.id
                                ? g.choices.find((x) =>
                                      x.leafKeys.includes(leaf.key),
                                  )
                                : g.choices[0];
                        assert(choice);
                        return [g.id, { choice: choice.key }];
                    }),
                );
                return { answers, latencyMs: 0, fixture: true };
            },
        });
        verifyStagedTrace(p, decision, assert);
        const selection = policy.select(decision.resolvedAnswers);
        assert.equal(selection.actions.length, 1);
        const tick = c.tick;
        c = await engine.step(
            { red: selection.actions },
            { ticks, view: "normal", privileged: true },
        );
        policy.receipts(tick, c.results.red);
        policy.refresh(c.observations.red, c.tick);
        assert(c.results.red[0].accepted, JSON.stringify(c.results.red[0]));
        traces.push({
            tick,
            fn,
            decision,
            action: selection.actions[0],
            receipt: c.results.red[0],
        });
    }
    await choose(28, { ticks: 60 });
    assert(policy.frame.own.some((u) => u.name === "NACNST"));
    const buildPower = policy.facade.buildFns.get("NAPOWR");
    p = policy.packet();
    assert(offered(p, buildPower));
    assert(!offered(p, policy.facade.buildFns.get("APOC")));
    await choose(buildPower, { ticks: 30 });
    p = policy.packet();
    assert(offered(p, 61));
    assert(offered(p, 62));
    assert(!offered(p, 63));
    assert(!offered(p, buildPower));
    await choose(62);
    assert.equal(
        c.observations.red.vectors.production.find(
            (q) => q.queue === "structures",
        ).status,
        "OnHold",
    );
    p = policy.packet();
    assert(offered(p, 63));
    assert(!offered(p, 62));
    const held = c.observations.red.vectors.production.find(
        (q) => q.queue === "structures",
    ).progress;
    c = await engine.step({}, { ticks: 60, view: "normal", privileged: true });
    policy.refresh(c.observations.red, c.tick);
    assert.equal(
        c.observations.red.vectors.production.find(
            (q) => q.queue === "structures",
        ).progress,
        held,
    );
    await choose(63);
    assert.equal(
        c.observations.red.vectors.production.find(
            (q) => q.queue === "structures",
        ).status,
        "Active",
    );
    const credits = c.observations.red.vectors.minerals;
    await choose(61);
    assert.equal(
        c.observations.red.vectors.production.find(
            (q) => q.queue === "structures",
        ).size,
        0,
    );
    assert(c.observations.red.vectors.minerals > credits);
    ok(
        "real engine deploys; production prerequisites update; pause holds progress, resume restarts and cancel refunds credits",
    );
    await choose(buildPower, { ticks: 600 });
    p = policy.packet();
    snapshots.push({
        name: "completed-power-plant-normal",
        packet: p,
        normalUnits: policy.frame.rows,
    });
    assert(offered(p, 60));
    await choose(60, { ticks: 30 });
    const plant = policy.frame.own.find((u) => u.name === "NAPOWR");
    assert(plant);
    p = policy.packet();
    assert(!offered(p, 50), "Undamaged buildings should not get start-repair");
    const creditBeforeSell = c.observations.red.vectors.minerals;
    assert(
        !leaves(p).some(
            (x) =>
                x.action.function === 49 &&
                x.action.unit_tags.includes(plant.tag),
        ),
    );
    c = await engine.step({}, { ticks: 90, view: "normal", privileged: true });
    policy.refresh(c.observations.red, c.tick);
    await choose(policy.facade.buildFns.get("NAHAND"), { ticks: 600 });
    await choose(60, { ticks: 120 });
    assert(
        policy.frame.own.some(
            (u) => u.name === "NAHAND" && u.build_progress === 100,
        ),
    );
    const train = policy.facade.buildFns.get("E2");
    await choose(train, { ticks: 3 });
    assert.equal(
        c.observations.red.vectors.production.find(
            (q) => q.queue === "infantry",
        ).size,
        1,
    );
    p = policy.packet();
    assert(offered(p, train));
    await choose(train, { ticks: 3 });
    assert.equal(
        c.observations.red.vectors.production.find(
            (q) => q.queue === "infantry",
        ).size,
        2,
    );
    ok(
        "busy infantry queue can accept another eligible train action, while busy building production is excluded",
    );
    const beforeSale = c.observations.red.vectors.minerals;
    await choose(49, { target: plant.tag, ticks: 120 });
    assert(!policy.frame.own.some((u) => u.tag === plant.tag));
    assert(c.observations.red.vectors.minerals > beforeSale);
    ok(
        "staged placement creates the specified building and staged sell removes it with a real credit refund",
    );

    // Fabricated normal observations exercise role/visibility branches only;
    // these are not model matches or claims of live execution for every function.
    function synthetic(specs, { ore = true, abilities = false } = {}) {
        const obs = structuredClone(initialObs),
            f = obs.units.features,
            w = f.length;
        const rows = specs.map((s, i) => {
            const rule = meta.techno.get(s.name);
            assert(rule, s.name);
            return {
                ...unitTemplate,
                tag: 9000 + i,
                unit_type: catalog.unit_types.indexOf(s.name),
                health: rule.strength,
                health_ratio: 255,
                can_move: Number(rule.type !== 2),
                cargo_space_max:
                    rule.passengers ?? rule.maxNumberOccupants ?? 0,
                ...s,
            };
        });
        obs.units.data.shape[0] = rows.length;
        obs.units.data.data = Int32Array.from(
            rows.flatMap((u) => f.map((k) => u[k] ?? 0)),
        );
        const counts = obs.vectors.unit_counts.data ?? obs.vectors.unit_counts;
        counts.fill(0);
        for (const u of rows.filter((u) => u.alliance === 1))
            counts[u.unit_type]++;
        obs.vectors.function_mask.data.fill(0);
        if (abilities)
            for (let i = 0; i < catalog.functions.length; i++)
                if (catalog.functions[i].id >= 3000)
                    obs.vectors.function_mask.data[i] = 1;
        const cell = unitTemplate.y * 146 + unitTemplate.x;
        for (const [name, value] of [
            ["visibility_map", 2],
            ["resource_map", ore ? 100 : 0],
        ]) {
            const k = obs.world.channels.indexOf(name);
            if (k >= 0) obs.world.data.data[k * 146 * 146 + cell] = value;
        }
        const pol = new CatalogPolicy(config);
        pol.refresh(obs, 100);
        return { pol, packet: pol.packet() };
    }
    const specs = [
        { name: "SENGINEER" },
        { name: "E2" },
        { name: "HARV" },
        { name: "IVAN" },
        { name: "HTK", cargo_space_taken: 1, cargo_space_max: 5 },
        { name: "NACNST", health: 500, health_ratio: 120 },
        { name: "NAREFN" },
        { name: "CAHOSP", alliance: 3 },
        { name: "CAHOSP", alliance: 4, tag: 99998, display_type: 2 },
        { name: "CAHOSP", alliance: 4, tag: 99999, is_blip: 1 },
        { name: "HTNK", alliance: 4, owner: 1 },
    ];
    const garrison = [...meta.techno].find(
        ([name, r]) =>
            r.type === 2 && r.canBeOccupied && r.maxNumberOccupants > 0,
    );
    assert(garrison);
    specs.push({
        name: garrison[0],
        alliance: 3,
        cargo_space_max: garrison[1].maxNumberOccupants,
    });
    const rich = synthetic(specs, { abilities: true });
    for (const id of [
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 80, 81,
    ])
        assert(offered(rich.packet, id), `Missing synthetic function ${id}`);
    for (const fn of catalog.functions.filter((x) => x.id >= 3000))
        assert(offered(rich.packet, fn.id));
    for (const leaf of leaves(rich.packet)) {
        assert.equal(
            validateAction(leaf.action, rich.pol.obs, rich.pol.frame, catalog),
            null,
        );
        assert(![99998, 99999].includes(leaf.action.target_unit_tag));
    }
    const removed = synthetic([{ name: "E2" }], { ore: false });
    for (const id of [41, 42, 43, 44, 45, 46, 47, 48, 49, 50])
        assert(
            !offered(removed.packet, id),
            `Missing role still offered ${id}`,
        );
    ok(
        "synthetic normal-state fixtures cover transport, repairs, capture, garrison, mining, dock, bombing and ability binders; hidden/blip targets excluded",
    );
    snapshots.push({
        name: "synthetic-role-coverage-not-a-match",
        packet: rich.packet,
        normalUnits: rich.pol.frame.rows,
    });
    writeFileSync(out + "/snapshots.json", JSON.stringify(snapshots, null, 2));
    writeFileSync(out + "/engine-traces.json", JSON.stringify(traces, null, 2));
    writeFileSync(
        out + "/result.json",
        JSON.stringify(
            {
                passed: true,
                fixtureChoices: true,
                fullModelMatches: 0,
                catalogFunctions: catalog.functions.length,
                checks,
            },
            null,
            2,
        ),
    );
} finally {
    if (engine) await engine.shutdown();
    meta?.dispose();
}
