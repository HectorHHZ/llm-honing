// Catalog-wide action families grounded only in normal observations and public rules.
import { LayaPolicy } from "./policy.mjs";
import { contextParts, layerAt } from "./context.mjs";
export const PARAMETER_LIMIT = 12;
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const shortActors = (units) =>
    units.length === 1
        ? `${units[0].name} #${units[0].tag}`
        : `${units.length} ${units[0].name} units`;

export class CatalogPolicy extends LayaPolicy {
    role(name) {
        const label = super.role(name);
        return this.meta.techno.get(name)?.type === 2 &&
            ["combat unit", "support unit"].includes(label)
            ? "building"
            : label;
    }
    packet() {
        const v = this.obs.vectors,
            f = this.facade,
            r = f.res,
            meta = this.meta;
        const own = this.frame.own.filter(
                (u) =>
                    u.health > 0 &&
                    !u.is_in_cargo &&
                    (meta.techno.get(u.name)?.type !== 2 ||
                        u.build_progress === 100),
            ),
            visible = this.frame.visible;
        const observed = this.frame.rows.filter(
            (u) =>
                u.health > 0 &&
                !u.is_in_cargo &&
                (u.alliance === 1 ||
                    u.alliance === 2 ||
                    (u.display_type === 1 && !u.is_blip)),
        );
        const rule = (u) => meta.techno.get(u.name) ?? {},
            building = (u) => rule(u).type === 2;
        const friendly = (u) => u.alliance === 1 || u.alliance === 2;
        const base = own.find((u) => rule(u).constructionYard) ??
            own.find(building) ??
            own[0] ?? { x: 0, y: 0 };
        const cell = (u) => u.y * r + u.x;
        const catalog = this.catalog.functions,
            masks = v.function_mask?.data ?? v.function_mask ?? [];
        const mask = new Map(catalog.map((fn, i) => [fn.id, !!masks[i]]));
        const groups = new Map(),
            signatures = new Map(),
            omitted = [],
            coverage = new Map(
                catalog.map((fn) => [
                    fn.id,
                    {
                        id: fn.id,
                        name: fn.name,
                        mask: mask.get(fn.id),
                        status: "unavailable",
                        reason: "No matching actors, targets or prerequisites in current normal observation.",
                        candidates: 0,
                    },
                ]),
            );
        const groupFor = (id) => {
            if (!groups.has(id))
                groups.set(id, {
                    id,
                    instructions: `Choose one currently offered ${id.replaceAll(
                        "_",
                        " ",
                    )} action, or wait. Parameters are selected in later calls.`,
                    choices: [],
                });
            return groups.get(id);
        };
        const reason = (id, text) => {
            if (coverage.has(id)) coverage.get(id).reason = text;
        };
        const add = (id, label, action, options = {}) => {
            if (!coverage.has(action.function)) return;
            const group = groupFor(id),
                item = { label, action, cost: 0, ...options };
            const signature = JSON.stringify(action);
            if (!signatures.has(id)) signatures.set(id, new Set());
            if (signatures.get(id).has(signature)) return;
            signatures.get(id).add(signature);
            group.choices.push(item);
            Object.assign(coverage.get(action.function), {
                status: "offered",
                reason: "Grounded candidate exists; engine receipt and later observations determine execution.",
                candidates: coverage.get(action.function).candidates + 1,
            });
        };
        const shortlist = (items, name, limit = PARAMETER_LIMIT) => {
            if (items.length > limit)
                omitted.push({
                    group: name,
                    available: items.length,
                    offered: limit,
                    scope: "parameter_candidates_only",
                });
            return items.slice(0, limit);
        };
        const actors = (units, name, { single = false } = {}) => {
            const ordered = units
                .slice()
                .sort(
                    (a, b) =>
                        b.is_idle - a.is_idle ||
                        a.health_ratio - b.health_ratio ||
                        a.tag - b.tag,
                );
            if (single) return shortlist(ordered, name).map((u) => [u]);
            const byType = new Map();
            for (const u of ordered) {
                if (!byType.has(u.name)) byType.set(u.name, []);
                byType.get(u.name).push(u);
            }
            const batches = [...byType.values()].flatMap((us) =>
                Array.from({ length: Math.ceil(us.length / 64) }, (_, i) =>
                    us.slice(i * 64, (i + 1) * 64),
                ),
            );
            // Bound type-based batches; any omitted parameter batches are logged.
            return shortlist(batches, name);
        };
        const targets = (units, name) =>
            shortlist(
                units
                    .slice()
                    .sort(
                        (a, b) =>
                            distance(a, base) - distance(b, base) ||
                            a.tag - b.tag,
                    ),
                name,
            );
        const weapons = (u) =>
            [rule(u).primary, rule(u).secondary]
                .filter(Boolean)
                .map((n) => meta.rules.getWeapon(n));
        const armed = (u) => weapons(u).some((w) => Number(w?.damage) > 0);
        const mobile = own.filter((u) => u.can_move && !building(u));
        const combat = own.filter(armed);
        const points = new Map();
        const point = (world, label) => {
            if (
                !Number.isInteger(world) ||
                world < 0 ||
                world >= r * r ||
                !f.tileFromGrid(world % r, Math.floor(world / r))
            )
                return;
            if (!points.has(world)) points.set(world, { world, label });
        };
        point(cell(base), "own base");
        for (const u of visible) point(cell(u), `visible ${u.name}`);
        for (const m of this.frame.memory)
            point(cell(m), `last seen ${m.name}; ${m.ageTicks} ticks old`);
        for (const s of meta.starts)
            point(f.cell(s.x, s.y), "public possible starting position");
        for (let y = 8; y < r; y += 16)
            for (let x = 8; x < r; x += 16) {
                const c = y * r + x;
                if (
                    layerAt(this.obs, "pathable", c) ||
                    layerAt(this.obs, "navigable", c)
                )
                    point(c, "map waypoint");
            }
        const worldTargets = shortlist([...points.values()], "world_waypoints");
        const ore = shortlist(
            [...new Set(meta.tiles.map((t) => f.cell(t.rx, t.ry)))]
                .filter(
                    (c) =>
                        c >= 0 &&
                        layerAt(this.obs, "visibility_map", c) > 0 &&
                        layerAt(this.obs, "resource_map", c) > 0,
                )
                .sort(
                    (a, b) =>
                        distance({ x: a % r, y: Math.floor(a / r) }, base) -
                        distance({ x: b % r, y: Math.floor(b / r) }, base),
                ),
            "observed_ore_targets",
        );
        const at = (p) => `${p.world % r},${Math.floor(p.world / r)}`;
        const worldActions = (
            fn,
            verb,
            units,
            positions = worldTargets,
            group = "unit_orders",
        ) => {
            for (const us of actors(units, `${fn}:actors`))
                for (const p of positions)
                    add(
                        group,
                        `${verb} ${shortActors(us)} to ${at(p)} (${p.label})`,
                        {
                            function: fn,
                            unit_tags: us.map((u) => u.tag),
                            world: p.world,
                        },
                        { actorLabel: shortActors(us) },
                    );
        };
        const pairActions = (
            fn,
            verb,
            units,
            objects,
            compatible = () => true,
        ) => {
            // Filter before shortlisting, so an irrelevant nearby target cannot hide all valid choices.
            for (const us of actors(
                units.filter((u) => objects.some((t) => compatible(u, t))),
                `${fn}:actors`,
                { single: true },
            ))
                for (const target of targets(
                    objects.filter((t) => compatible(us[0], t)),
                    `${fn}:${us[0].tag}:targets`,
                ))
                    add(
                        fn >= 43 ? "support_actions" : "unit_orders",
                        `${verb} ${target.name} #${
                            target.tag
                        } using ${shortActors(us)}`,
                        {
                            function: fn,
                            unit_tags: us.map((u) => u.tag),
                            target_unit_tag: target.tag,
                        },
                        { actorLabel: shortActors(us) },
                    );
        };
        const simpleActions = (
            fn,
            verb,
            units,
            group = "unit_orders",
            single = false,
        ) => {
            for (const us of actors(units, `${fn}:actors`, { single }))
                add(
                    group,
                    `${verb} ${shortActors(us)}${
                        fn === 28 && rule(us[0]).deploysInto
                            ? `; becomes ${rule(us[0]).deploysInto}`
                            : ""
                    }`,
                    { function: fn, unit_tags: us.map((u) => u.tag) },
                    { actorLabel: shortActors(us) },
                );
        };

        // Every Build/Train function is accounted for. Only unlocked queue offers with
        // an enabled engine mask and affordable full reservation are offered here.
        for (const fn of catalog.filter((fn) =>
            /^(Build|Train)_/.test(fn.name),
        ))
            reason(
                fn.id,
                "Not in an unlocked production queue, building queue occupied, mask disabled, or insufficient currently reserved credits.",
            );
        for (const q of v.production) {
            const group = `queue_${q.queue}`;
            if (q.item && q.size > 0)
                add(
                    group,
                    `Cancel ${this.catalog.unit_types[q.item]} production in ${
                        q.queue
                    }; request refund`,
                    { function: 61, extra: { item: q.item } },
                );
            if (q.status === "Active")
                add(group, `Pause ${q.queue} production to stop spending`, {
                    function: 62,
                    extra: { queue: q.queue },
                });
            if (q.status === "OnHold")
                add(group, `Resume ${q.queue} production`, {
                    function: 63,
                    extra: { queue: q.queue },
                });
            if (q.status === "Ready") {
                const name = this.catalog.unit_types[q.item];
                const sites = meta.tiles
                    .filter((t) => f.canPlace(name, t))
                    .map((t) => ({ t, world: f.cell(t.rx, t.ry) }))
                    .filter(
                        (p) =>
                            p.world >= 0 &&
                            (this.lastRejected.get(`${name}:${p.world}`) ??
                                -Infinity) <=
                                this.tick - 90,
                    )
                    .sort(
                        (a, b) =>
                            distance(
                                { x: a.world % r, y: Math.floor(a.world / r) },
                                base,
                            ) -
                            distance(
                                { x: b.world % r, y: Math.floor(b.world / r) },
                                base,
                            ),
                    );
                for (const p of shortlist(sites, `${q.queue}:placement`))
                    add(
                        group,
                        `Place ${name} ${this.role(name)} at ${at(p)}`,
                        {
                            function: 60,
                            world: p.world,
                            extra: { item: q.item },
                        },
                        { placementKey: `${name}:${p.world}` },
                    );
            }
            // Building queues accept only one pending building despite a permissive mask.
            if (q.size > 0 && ["structures", "armory"].includes(q.queue))
                continue;
            for (let i = 0; i < q.available.length; i++) {
                const item = q.available[i],
                    name = this.catalog.unit_types[item],
                    fn = f.buildFns.get(name),
                    price = q.prices?.[i] ?? meta.techno.get(name)?.cost ?? 0;
                if (fn !== undefined && mask.get(fn) && price <= v.minerals)
                    add(
                        group,
                        `${
                            q.size > 0 ? "Queue another" : "Produce"
                        } ${name} ${this.role(name)}; cost ${price}`,
                        { function: fn },
                        { cost: price },
                    );
            }
        }
        reason(
            0,
            "Mapped to the wait choice of every group; all-wait submits an empty action list without stopping existing orders.",
        );
        if (coverage.has(0)) coverage.get(0).status = "wait_equivalent";
        worldActions(20, "Move", mobile);
        worldActions(21, "Force-move", mobile);
        worldActions(22, "Attack-move", mobile.filter(armed));
        pairActions(23, "Attack visible", combat, visible);
        worldActions(24, "Force-attack ground with", combat);
        worldActions(25, "Guard area with", mobile.filter(armed));
        simpleActions(26, "Guard in place with", combat);
        simpleActions(
            27,
            "Stop current orders of",
            own.filter((u) => u.can_move || armed(u)),
        );
        simpleActions(
            28,
            "Deploy or undeploy",
            own.filter((u) => {
                const q = rule(u);
                return (
                    q.deploysInto ||
                    q.undeploysInto ||
                    q.deployer ||
                    q.deployFire
                );
            }),
        );
        for (const p of worldTargets)
            add("observation", `Move observation camera to ${at(p)}`, {
                function: 29,
                world: p.world,
            });
        simpleActions(40, "Scatter", mobile);
        pairActions(
            41,
            "Enter transport",
            mobile.filter((u) => rule(u).type !== 1),
            observed.filter(
                (t) =>
                    friendly(t) &&
                    rule(t).passengers > 0 &&
                    t.cargo_space_taken < t.cargo_space_max,
            ),
            (u, t) =>
                u.tag !== t.tag &&
                (rule(u).size ?? 1) <= rule(t).sizeLimit &&
                (rule(u).size ?? 1) <= t.cargo_space_max - t.cargo_space_taken,
        );
        simpleActions(
            42,
            "Unload passengers from",
            own.filter(
                (u) => rule(u).passengers > 0 && u.cargo_space_taken > 0,
            ),
        );
        pairActions(
            43,
            "Repair",
            own.filter(
                (u) =>
                    rule(u).engineer ||
                    weapons(u).some((w) => Number(w?.damage) < 0),
            ),
            observed.filter((t) => friendly(t) && t.health_ratio < 255),
            (u, t) =>
                u.tag !== t.tag &&
                (!rule(u).engineer || (building(t) && rule(t).repairable)),
        );
        pairActions(
            44,
            "Capture",
            own.filter((u) => rule(u).engineer),
            observed.filter(
                (t) => !friendly(t) && building(t) && rule(t).capturable,
            ),
        );
        pairActions(
            45,
            "Garrison",
            own.filter((u) => rule(u).occupier),
            observed.filter(
                (t) =>
                    building(t) &&
                    rule(t).canBeOccupied &&
                    (!t.cargo_space_taken || friendly(t)) &&
                    t.cargo_space_taken <
                        (rule(t).maxNumberOccupants || t.cargo_space_max),
            ),
        );
        worldActions(
            46,
            "Gather observed ore with",
            own.filter((u) => rule(u).harvester),
            ore.map((world) => ({ world, label: "observed ore" })),
            "support_actions",
        );
        pairActions(
            47,
            "Dock at",
            mobile,
            observed.filter(
                (t) =>
                    friendly(t) &&
                    building(t) &&
                    (rule(t).refinery ||
                        rule(t).unitRepair ||
                        rule(t).numberOfDocks > 0),
            ),
            (u, t) =>
                rule(u).harvester
                    ? !!rule(t).refinery
                    : rule(u).type === 1
                    ? rule(t).numberOfDocks > 0
                    : !!rule(t).unitRepair,
        );
        pairActions(
            48,
            "Plant bomb on",
            own.filter((u) => rule(u).c4 || rule(u).ivan),
            observed.filter((t) => !friendly(t)),
            (u, t) => (rule(u).ivan ? !!rule(t).bombable : !!rule(t).canC4),
        );
        const saleFacilities = own.filter(
            (u) => building(u) && rule(u).unitSell,
        );
        simpleActions(
            49,
            "Sell",
            own.filter(
                (u) =>
                    !rule(u).unsellable &&
                    (building(u) ||
                        (rule(u).type === 7 &&
                            saleFacilities.some((t) => distance(u, t) <= 3))),
            ),
            "support_actions",
            true,
        );
        for (const u of own.filter(
            (u) =>
                building(u) &&
                rule(u).clickRepairable &&
                (f.repairs.has(u.tag) ||
                    (u.health_ratio < 255 && v.minerals > 0)),
        ))
            add(
                "support_actions",
                `${f.repairs.has(u.tag) ? "Stop" : "Start"} repairing ${
                    u.name
                } #${u.tag}`,
                { function: 50, unit_tags: [u.tag] },
            );
        for (const seat of f.seats.filter((s) => s.name !== this.name))
            add(
                "diplomacy",
                `Toggle alliance with player ${seat.name} #${seat.player_id}; changes hostility`,
                { function: 80, extra: { player: seat.player_id } },
                { requiresConfirmation: true },
            );
        add(
            "diplomacy",
            "Resign and concede this match immediately",
            { function: 81 },
            { requiresConfirmation: true },
        );
        for (const fn of catalog.filter((fn) =>
            fn.name.startsWith("Activate_"),
        )) {
            if (!mask.get(fn.id)) {
                reason(
                    fn.id,
                    "Special ability is not currently available according to the engine mask (ownership, charge or match settings).",
                );
                continue;
            }
            for (const p of worldTargets)
                add(
                    "special_abilities",
                    `Use ${fn.name.slice(9)} at ${at(p)}`,
                    { function: fn.id, world: p.world },
                    {
                        intent: {
                            id: String(fn.id),
                            label: `Use ${fn.name.slice(
                                9,
                            )}; choose a location next`,
                        },
                    },
                );
        }
        const known = new Set([
            0,
            ...Array.from({ length: 10 }, (_, i) => 20 + i),
            ...Array.from({ length: 11 }, (_, i) => 40 + i),
            60,
            61,
            62,
            63,
            80,
            81,
        ]);
        for (const fn of catalog)
            if (!known.has(fn.id) && !/^(Build|Train|Activate)_/.test(fn.name))
                Object.assign(coverage.get(fn.id), {
                    status: "unsupported",
                    reason: "Unknown function family; no parameter binder implemented.",
                });
        const finalGroups = [...groups.values()]
            .filter((g) => g.choices.length)
            .map((g) => ({
                ...g,
                choices: [
                    {
                        key: "A",
                        label: "Take no action this turn.",
                        action: null,
                        cost: 0,
                    },
                    ...g.choices.map((c, i) => ({ ...c, key: `L${i + 1}` })),
                ],
            }));
        const questions = Object.fromEntries(
            finalGroups.map((g) => [
                g.id,
                {
                    type: "choice",
                    instructions: g.instructions,
                    criteria: Object.fromEntries(
                        g.choices.map((c) => [c.key, c.label]),
                    ),
                },
            ]),
        );
        const availability = [...coverage.values()],
            offered = availability.filter((x) => x.status === "offered");
        const parts = contextParts(
            this.obs,
            this.catalog,
            this.frame,
            this.memory,
            this.tick,
            `Action catalog coverage: ${offered.length} of ${
                catalog.length
            } functions currently have grounded candidates; waiting remains available. Available action groups: ${finalGroups
                .map((g) => g.id)
                .join(
                    ", ",
                )}. Candidate types follow current normal state and public rules; target lists are bounded. A selected action can be followed by actor and target questions. Shared spending cannot exceed current credits.`,
        );
        parts.essential.push(
            `Current selectable action families: ${
                offered.map((x) => x.name).join(", ") || "none"
            }. Sell_Unit and Cancel_Production, when offered, can recover funds; refunds are only usable after the engine executes them. Observe existing orders before issuing replacements.`,
        );
        this.current = {
            tick: this.tick,
            groups: finalGroups,
            questions,
            parts,
            omitted,
            availability,
            candidateMode: "catalog",
        };
        return this.current;
    }
    receipts(tick, results) {
        const wasRepairing = new Set(this.facade.repairs),
            pending = this.pending;
        super.receipts(tick, results);
        for (let i = 0; i < pending.length; i++)
            if (pending[i].action.function === 50 && results?.[i]?.accepted)
                for (const tag of pending[i].action.unit_tags ?? [])
                    if (wasRepairing.has(tag)) this.facade.repairs.delete(tag);
    }
}
