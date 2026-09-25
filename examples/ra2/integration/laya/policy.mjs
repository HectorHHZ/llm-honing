import { SupalosaAdapter } from "../adapter.mjs";
import { ObservationMemory, contextParts, layerAt } from "./context.mjs";

const NAMES = {
    NACNST: "construction yard",
    NAPOWR: "power plant",
    NAREFN: "ore refinery with harvester",
    NAHAND: "barracks",
    NAWEAP: "war factory",
    NARADR: "radar tech",
    NATECH: "battle lab",
    NADEPT: "repair depot",
    NAFLAK: "anti-air",
    NALASR: "sentry gun",
    TESLA: "Tesla defense",
    NABNKR: "battle bunker",
    NAWALL: "wall",
    NAYARD: "naval yard",
    NAINDP: "industrial plant",
    SMCV: "construction vehicle",
    HARV: "ore harvester",
    E2: "conscript infantry",
    E1: "GI infantry",
    HTNK: "Rhino tank",
    V3: "V3 siege launcher",
    APOC: "Apocalypse tank",
    FLAKT: "Flak trooper",
    SENGINEER: "engineer",
    DTRUCK: "demolition truck",
    DRON: "terror drone",
    TTNK: "Tesla tank",
    DTRK: "demolition truck",
};
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
export class LayaPolicy {
    constructor(config) {
        this.name = config.name;
        this.facade = new SupalosaAdapter(config);
        this.catalog = config.catalog;
        this.meta = config.metadata;
        this.memory = new ObservationMemory();
        this.pending = [];
        this.lastRejected = new Map();
        this.metrics = { decisions: 0, selected: 0, dropped: 0 };
    }
    refresh(obs, tick) {
        this.obs = obs;
        this.tick = tick;
        this.facade.refresh(obs, tick);
        this.frame = this.memory.update(obs, this.catalog, tick);
    }
    allowed(id) {
        const i = this.catalog.functions.findIndex((f) => f.id === id);
        const mask = this.obs.vectors.function_mask;
        return i >= 0 && (id < 1000 || !!(mask?.data ?? mask)?.[i]);
    }
    cell(row) {
        return row.y * this.facade.res + row.x;
    }
    role(name) {
        const r = this.meta.techno.get(name);
        return (
            NAMES[name] ??
            (r?.harvester
                ? "harvester"
                : r?.type === 0
                ? "building"
                : r?.primary
                ? "combat unit"
                : "support unit")
        );
    }
    packet() {
        const { own, visible, memory } = this.frame,
            v = this.obs.vectors,
            f = this.facade;
        const base = own.find(
            (u) => this.meta.techno.get(u.name)?.constructionYard,
        ) ||
            own.find((u) => this.meta.techno.get(u.name)?.type === 0) ||
            own[0] || { x: 0, y: 0 };
        const groups = [],
            omitted = [];
        const add = (id, instructions, choices, max = 12) => {
            const unique = [
                ...new Map(
                    choices
                        .filter((c) => this.allowed(c.action.function))
                        .map((c) => [JSON.stringify(c.action), c]),
                ).values(),
            ];
            if (!unique.length) return;
            if (unique.length > max)
                omitted.push({
                    group: id,
                    available: unique.length,
                    offered: max,
                });
            const shortlist = unique.slice(0, max);
            groups.push({
                id,
                instructions,
                choices: [
                    {
                        key: "A",
                        label: "Take no action this turn.",
                        action: null,
                        cost: 0,
                    },
                    ...shortlist.map((c, i) => ({
                        ...c,
                        key: String.fromCharCode(66 + i),
                    })),
                ],
            });
        };
        for (const q of v.production) {
            if (q.status === "Ready") {
                const name = this.catalog.unit_types[q.item];
                const tiles = this.meta.tiles
                    .filter((t) => f.canPlace(name, t))
                    .sort(
                        (a, b) =>
                            dist(
                                {
                                    x: f.cell(a.rx, a.ry) % f.res,
                                    y: Math.floor(f.cell(a.rx, a.ry) / f.res),
                                },
                                base,
                            ) -
                            dist(
                                {
                                    x: f.cell(b.rx, b.ry) % f.res,
                                    y: Math.floor(f.cell(b.rx, b.ry) / f.res),
                                },
                                base,
                            ),
                    );
                const choices = [];
                for (const t of tiles) {
                    const cell = f.cell(t.rx, t.ry),
                        key = `${name}:${cell}`;
                    if (
                        (this.lastRejected.get(key) ?? -Infinity) >
                        this.tick - 90
                    )
                        continue;
                    if (
                        choices.length &&
                        choices.some(
                            (c) =>
                                Math.abs(
                                    (c.action.world % f.res) - (cell % f.res),
                                ) +
                                    Math.abs(
                                        Math.floor(c.action.world / f.res) -
                                            Math.floor(cell / f.res),
                                    ) <
                                3,
                        )
                    )
                        continue;
                    choices.push({
                        label: `Place ${name} ${this.role(name)} at ${
                            cell % f.res
                        },${Math.floor(cell / f.res)}`,
                        action: {
                            function: 60,
                            world: cell,
                            extra: { item: q.item },
                        },
                        cost: 0,
                        placementKey: key,
                    });
                    if (choices.length === 3) break;
                }
                add(
                    `place_${q.queue}`,
                    "Choose placement for this completed building, or wait.",
                    choices,
                );
                continue;
            }
            if (q.status === "OnHold") {
                add(
                    `resume_${q.queue}`,
                    "Choose whether to resume this paused production queue.",
                    [
                        {
                            label: `Resume ${q.queue} production`,
                            action: { function: 63, extra: { queue: q.queue } },
                            cost: 0,
                        },
                    ],
                );
                continue;
            }
            if (q.size > 0) continue;
            const choices = q.available.flatMap((type, i) => {
                const name = this.catalog.unit_types[type],
                    fn = f.buildFns.get(name),
                    price =
                        q.prices?.[i] ?? this.meta.techno.get(name)?.cost ?? 0;
                return fn !== undefined && price <= v.minerals
                    ? [
                          {
                              label: `Produce ${name} ${this.role(
                                  name,
                              )}; cost ${price}`,
                              action: { function: fn },
                              cost: price,
                          },
                      ]
                    : [];
            });
            add(
                `produce_${q.queue}`,
                `Choose one ${q.queue} production action to strengthen economy or army, or save money.`,
                choices,
            );
        }
        const utility = [];
        const mcvs = own.filter(
            (u) =>
                this.meta.techno.get(u.name)?.deploysInto &&
                this.meta.general.baseUnit.includes(u.name),
        );
        for (const u of mcvs)
            utility.push({
                label: `Deploy ${u.name} #${u.tag} into construction yard; enable building production`,
                action: { function: 28, unit_tags: [u.tag] },
                cost: 0,
            });
        const repair = own
            .filter(
                (u) =>
                    this.meta.techno.get(u.name)?.type === 0 &&
                    u.health_ratio < 180 &&
                    !f.repairs.has(u.tag),
            )
            .sort((a, b) => a.health_ratio - b.health_ratio)
            .slice(0, 2);
        for (const u of repair)
            utility.push({
                label: `Start repairing ${u.name} #${u.tag} HP ${u.health}`,
                action: { function: 50, unit_tags: [u.tag] },
                cost: 0,
            });
        const workers = own.filter(
            (u) =>
                this.meta.techno.get(u.name)?.harvester &&
                u.is_idle &&
                !u.is_in_cargo,
        );
        for (const u of workers.slice(0, 2)) {
            const ore = this.meta.tiles
                .map((t) => ({
                    x: f.cell(t.rx, t.ry) % f.res,
                    y: Math.floor(f.cell(t.rx, t.ry) / f.res),
                    cell: f.cell(t.rx, t.ry),
                }))
                .filter(
                    (t) =>
                        layerAt(this.obs, "visibility_map", t.cell) > 0 &&
                        layerAt(this.obs, "resource_map", t.cell) > 0,
                )
                .sort((a, b) => dist(a, u) - dist(b, u))[0];
            if (ore)
                utility.push({
                    label: `Send idle harvester #${u.tag} to observed ore ${ore.x},${ore.y}`,
                    action: {
                        function: 46,
                        unit_tags: [u.tag],
                        world: ore.cell,
                    },
                    cost: 0,
                });
        }
        const gis = own.filter((u) => u.name === "E1");
        for (const deployed of [0, 1]) {
            const tags = gis
                .filter((u) => u.is_deployed === deployed)
                .slice(0, 64)
                .map((u) => u.tag);
            if (tags.length)
                utility.push({
                    label: `${deployed ? "Undeploy" : "Deploy"} ${
                        tags.length
                    } GI infantry`,
                    action: { function: 28, unit_tags: tags },
                    cost: 0,
                });
        }
        add(
            "utility",
            "Choose a concrete base or economy action. An undeployed MCV cannot build a base.",
            utility,
            8,
        );
        const combat = own
            .filter((u) => {
                const r = this.meta.techno.get(u.name);
                return (
                    r?.type !== 0 &&
                    (r?.primary || r?.secondary) &&
                    !r?.harvester &&
                    !mcvs.some((m) => m.tag === u.tag) &&
                    !u.is_in_cargo
                );
            })
            .sort((a, b) => a.tag - b.tag);
        // Two bounded groups; excess units retain prior orders, explicitly logged.
        if (combat.length > 128)
            omitted.push({
                group: "tactics_units",
                available: combat.length,
                offered: 128,
            });
        for (let n = 0; n < Math.min(2, Math.ceil(combat.length / 64)); n++) {
            const group = combat.slice(n * 64, (n + 1) * 64),
                tags = group.map((u) => u.tag),
                center = {
                    x: group.reduce((s, u) => s + u.x, 0) / group.length,
                    y: group.reduce((s, u) => s + u.y, 0) / group.length,
                };
            const choices = visible
                .slice()
                .sort((a, b) => dist(a, center) - dist(b, center))
                .slice(0, 3)
                .map((u) => ({
                    label: `Attack visible ${u.name} #${u.tag} HP${u.health} at ${u.x},${u.y}`,
                    action: {
                        function: 23,
                        unit_tags: tags,
                        target_unit_tag: u.tag,
                    },
                    cost: 0,
                }));
            choices.push({
                label: `Defend own base at ${base.x},${base.y}`,
                action: {
                    function: 25,
                    unit_tags: tags,
                    world: this.cell(base),
                },
                cost: 0,
            });
            if (dist(center, base) > 8)
                choices.push({
                    label: `Retreat ${group.length} units to own base`,
                    action: {
                        function: 20,
                        unit_tags: tags,
                        world: this.cell(base),
                    },
                    cost: 0,
                });
            for (const m of memory.slice(0, 1))
                choices.push({
                    label: `Search last seen ${m.name} position ${m.x},${m.y}; age ${m.ageTicks} ticks`,
                    action: {
                        function: 22,
                        unit_tags: tags,
                        world: this.cell(m),
                    },
                    cost: 0,
                });
            const targets = this.meta.starts
                .map((t) => {
                    const cell = f.cell(t.x, t.y);
                    return {
                        x: cell % f.res,
                        y: Math.floor(cell / f.res),
                        cell,
                    };
                })
                .filter(
                    (t) =>
                        t.cell >= 0 &&
                        dist(t, base) > 15 &&
                        layerAt(this.obs, "visibility_map", t.cell) !== 2,
                )
                .sort((a, b) => dist(a, center) - dist(b, center));
            if (!targets.length) {
                for (let y = 8; y < f.res; y += 16)
                    for (let x = 8; x < f.res; x += 16) {
                        const cell = y * f.res + x;
                        if (
                            layerAt(this.obs, "visibility_map", cell) === 0 &&
                            layerAt(this.obs, "pathable", cell)
                        )
                            targets.push({ x, y, cell });
                    }
                targets.sort((a, b) => dist(a, center) - dist(b, center));
            }
            for (const t of targets.slice(0, 2))
                choices.push({
                    label: `Scout by attack-moving to unexplored ${t.x},${t.y}`,
                    action: { function: 22, unit_tags: tags, world: t.cell },
                    cost: 0,
                });
            add(
                `tactics_${n}`,
                `Choose an order for combat group of ${
                    group.length
                } units at ${Math.round(center.x)},${Math.round(
                    center.y,
                )}. Seek enemy, fight favorable battles, protect base.`,
                choices,
                9,
            );
        }
        const questions = Object.fromEntries(
            groups.map((g) => [
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
        const parts = contextParts(
            this.obs,
            this.catalog,
            this.frame,
            this.memory,
            this.tick,
            `Choices available in: ${
                groups.map((g) => g.id).join(", ") || "none"
            }. Shared spending cannot exceed credits. Action conflicts are dropped.`,
        );
        this.current = { tick: this.tick, groups, questions, parts, omitted };
        return this.current;
    }
    select(answers, maxActions = 16) {
        const selected = [],
            dropped = [];
        let budget = this.obs.vectors.minerals;
        const used = new Set();
        for (const g of this.current.groups) {
            const key = answers[g.id]?.choice;
            if (!g.choices.some((c) => c.key === key))
                throw new Error(
                    `Model returned invalid choice for ${g.id}: ${key}`,
                );
            const c = g.choices.find((c) => c.key === key);
            if (!c.action) continue;
            const reason =
                validateAction(c.action, this.obs, this.frame, this.catalog) ||
                (selected.length >= maxActions
                    ? "action_cap"
                    : c.cost > budget
                    ? "shared_budget"
                    : c.action.unit_tags?.some((id) => used.has(id))
                    ? "actor_conflict"
                    : null);
            if (reason) {
                dropped.push({ group: g.id, key, reason });
                continue;
            }
            selected.push({ ...c, group: g.id });
            budget -= c.cost;
            for (const id of c.action.unit_tags ?? []) used.add(id);
        }
        this.pending = selected;
        this.metrics.decisions++;
        this.metrics.selected += selected.length;
        this.metrics.dropped += dropped.length;
        return { actions: selected.map((c) => c.action), selected, dropped };
    }
    receipts(tick, results) {
        this.memory.receipts(tick, this.pending, results);
        for (let i = 0; i < this.pending.length; i++) {
            const c = this.pending[i];
            if (c.action.function === 50 && results?.[i]?.accepted)
                for (const id of c.action.unit_tags)
                    this.facade.repairs.add(id);
            if (c.placementKey) this.lastRejected.set(c.placementKey, tick);
        }
    }
}
export function validateAction(action, obs, frame, catalog) {
    const fn = catalog.functions.findIndex((f) => f.id === action.function);
    // Masks guide model candidate generation; the engine decides submitted action legality.
    if (fn < 0) return "unknown_function";
    if (action.unit_tags?.length > 64) return "too_many_units";
    if (action.unit_tags?.some((id) => !frame.own.some((u) => u.tag === id)))
        return "not_owned";
    const observed = frame.rows ?? [...frame.own, ...frame.visible];
    if (
        action.target_unit_tag !== undefined &&
        !observed.some(
            (u) =>
                u.tag === action.target_unit_tag &&
                (u.alliance === 1 ||
                    u.alliance === 2 ||
                    (u.display_type === 1 && !u.is_blip)),
        )
    )
        return "target_not_currently_visible";
    const r = obs.world.data.shape[1];
    if (
        action.world !== undefined &&
        (!Number.isInteger(action.world) ||
            action.world < 0 ||
            action.world >= r * r)
    )
        return "off_map";
    return null;
}
