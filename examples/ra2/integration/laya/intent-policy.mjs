// Coordinated catalog decisions. All engine commands remain model-selected leaves.
import { CatalogPolicy } from "./catalog-policy.mjs";
import { countsOf, layerAt } from "./context.mjs";
const countsText = (rows) =>
    Object.entries(countsOf(rows))
        .map(([n, c]) => `${n}:${c}`)
        .join(", ") || "none";
export class IntentPolicy extends CatalogPolicy {
    role(name) {
        return (
            {
                SHK: "Tesla trooper",
                DOG: "attack dog",
                HTK: "flak transport",
                FLAKT: "anti-air infantry",
            }[name] ?? super.role(name)
        );
    }
    constructor(config) {
        super(config);
        this.decisionInterval = config.decisionInterval ?? 15;
        if (
            !Number.isInteger(this.decisionInterval) ||
            this.decisionInterval < 3 ||
            this.decisionInterval % 3
        )
            throw new Error(
                "decisionInterval must be a positive multiple of 3",
            );
        this.commandHistory = [];
        this.effects = [];
        this.previousNormal = null;
    }
    refresh(obs, tick) {
        super.refresh(obs, tick);
        const current = {
            truncated: !!obs.vectors.units_truncated,
            counts: countsOf(this.frame.own),
            queues: obs.vectors.production.map((q) => ({
                queue: q.queue,
                status: q.status,
                item: q.item,
                progress: q.progress,
            })),
            credits: obs.vectors.minerals,
        };
        if (
            this.previousNormal &&
            !current.truncated &&
            !this.previousNormal.truncated
        ) {
            const before = this.previousNormal;
            for (const n of new Set([
                ...Object.keys(before.counts),
                ...Object.keys(current.counts),
            ]))
                if ((before.counts[n] ?? 0) !== (current.counts[n] ?? 0))
                    this.effects.push({
                        tick,
                        text: `Observed own ${this.role(n)} (${n}) count ${
                            before.counts[n] ?? 0
                        } -> ${current.counts[n] ?? 0}.`,
                    });
            for (const q of current.queues) {
                const p = before.queues.find((p) => p.queue === q.queue);
                if (
                    p &&
                    p.status !== q.status &&
                    ["Ready", "OnHold"].includes(q.status)
                )
                    this.effects.push({
                        tick,
                        text: `Observed ${q.queue} ${q.status} at ${q.progress}%.`,
                    });
            }
        }
        this.previousNormal = current;
        this.effects = this.effects
            .filter((e) => tick - e.tick <= 1800)
            .slice(-24);
        this.commandHistory = this.commandHistory.filter(
            (e) => tick - e.tick <= 1800,
        );
    }
    packet() {
        const packet = super.packet(),
            v = this.obs.vectors,
            own = this.frame.own,
            rule = (u) => this.meta.techno.get(u.name) ?? {};
        const yards = own.filter((u) => rule(u).constructionYard),
            mcvs = own.filter((u) =>
                this.meta.general.baseUnit.includes(u.name),
            );
        const refineries = own.filter((u) => rule(u).refinery),
            barracks = own.filter((u) => ["NAHAND", "GAPILE"].includes(u.name)),
            factories = own.filter((u) =>
                ["NAWEAP", "GAWEAP"].includes(u.name),
            );
        const readyBuilding = v.production.find(
            (q) =>
                q.status === "Ready" &&
                ["structures", "armory"].includes(q.queue),
        );
        const unfolding = own.filter(
            (u) => rule(u).type === 2 && u.build_progress < 100,
        );
        const activeBuilding = v.production.find(
            (q) =>
                q.status === "Active" &&
                ["structures", "armory"].includes(q.queue),
        );
        const queuedMilitary = v.production
            .filter(
                (q) =>
                    ["infantry", "vehicles", "aircraft", "ships"].includes(
                        q.queue,
                    ) &&
                    q.size > 0 &&
                    !this.meta.techno.get(this.catalog.unit_types[q.item])
                        ?.harvester,
            )
            .reduce((n, q) => n + q.size, 0);
        const priority = unfolding.length
            ? `Allow the placed ${this.role(
                  unfolding[0].name,
              )} to finish construction before treating its services as missing.`
            : readyBuilding
            ? `Place the completed ${this.role(
                  this.catalog.unit_types[readyBuilding.item],
              )} to activate it and free the construction queue.`
            : activeBuilding && v.army_count < 8
            ? `Allow the ${this.role(
                  this.catalog.unit_types[activeBuilding.item],
              )} already in production to finish, then place it.`
            : !yards.length
            ? "Deploy the construction vehicle to establish the base."
            : !v.power_cap
            ? "Build and place the first power plant."
            : !refineries.length
            ? "Build and place an ore refinery with a harvester for recurring income."
            : !v.worker_count
            ? "Restore a harvester to mine ore."
            : v.is_low_power
            ? "Restore sufficient electricity for the existing economy and army."
            : !barracks.length
            ? "Build and place barracks to unlock infantry and a war factory."
            : !factories.length
            ? "Build and place a war factory to unlock battle tanks."
            : v.army_count < 8 && v.army_count + queuedMilitary >= 8
            ? `Let the ${queuedMilitary} already ordered military units finish training; the planned army is sufficient to begin scouting.`
            : v.army_count < 8
            ? "Recruit a combat army, especially Rhino tanks, while keeping income working."
            : !this.frame.visible.length
            ? "Scout public possible enemy starting locations using attack-move with the army."
            : "Attack visible enemy buildings with the army; defend the economy against immediate threats.";
        const active = v.production.filter((q) => q.status === "Active"),
            ready = v.production.filter(
                (q) =>
                    q.status === "Ready" &&
                    ["structures", "armory"].includes(q.queue),
            ),
            paused = v.production.filter((q) => q.status === "OnHold");
        const queueName = (q) => this.role(this.catalog.unit_types[q.item]);
        const waitLabel = unfolding.length
            ? `Wait for the placed ${this.role(
                  unfolding[0].name,
              )} to finish construction`
            : ready.length
            ? `Issue no new command; completed ${queueName(
                  ready[0],
              )} remains unplaced`
            : paused.length
            ? `Issue no new command; ${queueName(paused[0])} remains paused`
            : active.length
            ? v.army_count >= 8
                ? "Wait one second; leave the army on its existing orders"
                : `Continue ${active
                      .map((q) => queueName(q))
                      .join(
                          " and ",
                      )} production and existing orders without interruption`
            : "Wait one second; build no army or new buildings";
        const leaves = [];
        let excludedDeploy = 0;
        const actorExclusions = [];
        for (const g of packet.groups)
            for (const original of g.choices) {
                if (!original.action) continue;
                const c = structuredClone(original),
                    a = c.action,
                    fn = a.function,
                    actors = (a.unit_tags ?? []).map((t) =>
                        own.find((u) => u.tag === t),
                    );
                // Camera changes do not alter the visibility used by this policy; merge this cosmetic choice with wait.
                if (fn === 29) {
                    actorExclusions.push({
                        function: fn,
                        reason: "camera_does_not_change_policy_observation",
                    });
                    continue;
                }
                // Role-grounded parameters: combat squads perform distant attacks. Armed ore
                // trucks retain nearby self-defense, gather/dock, base relocation and scatter.
                const workerActors =
                    actors.length && actors.every((u) => rule(u).harvester);
                if (workerActors && [22, 24, 25].includes(fn)) {
                    actorExclusions.push({
                        function: fn,
                        reason: "distant_military_order_requires_combat_troops",
                    });
                    continue;
                }
                if (
                    workerActors &&
                    fn === 26 &&
                    !actors.some((u) =>
                        this.frame.visible.some(
                            (t) => Math.hypot(u.x - t.x, u.y - t.y) <= 8,
                        ),
                    )
                ) {
                    actorExclusions.push({
                        function: fn,
                        reason: "worker_guard_is_local_self_defense_only",
                    });
                    continue;
                }
                if (workerActors && fn === 23) {
                    const target = this.frame.visible.find(
                        (u) => u.tag === a.target_unit_tag,
                    );
                    if (
                        !target ||
                        actors.some(
                            (u) =>
                                Math.hypot(u.x - target.x, u.y - target.y) > 8,
                        )
                    ) {
                        actorExclusions.push({
                            function: fn,
                            reason: "harvester_attack_is_local_self_defense_only",
                        });
                        continue;
                    }
                }
                if (
                    workerActors &&
                    [20, 21].includes(fn) &&
                    !original.label.includes("(own base)")
                ) {
                    actorExclusions.push({
                        function: fn,
                        reason: "worker_relocation_targets_base; observed_ore_uses_gather",
                    });
                    continue;
                }
                // Actual isolated engine probes show Deploy on construction yards is a no-op.
                // Preserve functioning MCV/infantry deployment, do not claim yard packing works.
                if (
                    [20, 21, 22].includes(fn) &&
                    actors.length &&
                    actors.every(
                        (u) =>
                            u.is_idle &&
                            Math.hypot(
                                u.x - (a.world % this.facade.res),
                                u.y - Math.floor(a.world / this.facade.res),
                            ) < 2,
                    )
                )
                    continue;
                if (fn === 28 && actors.some((u) => rule(u).type === 2)) {
                    excludedDeploy++;
                    continue;
                }
                const q = v.production.find(
                    (q) =>
                        q.queue === a.extra?.queue ||
                        (a.extra?.item !== undefined &&
                            q.item === a.extra.item),
                );
                let category, label;
                if (
                    [20, 21, 22].includes(fn) &&
                    actors.length &&
                    actors.every(
                        (u) =>
                            u.is_idle &&
                            Math.hypot(
                                u.x - (a.world % this.facade.res),
                                u.y - Math.floor(a.world / this.facade.res),
                            ) < 2,
                    )
                )
                    continue;
                if (fn === 28 && actors.some((u) => rule(u).deploysInto)) {
                    category = "base";
                    label = "Establish or expand the base and its economy";
                    c.label = `Deploy ${actors
                        .map((u) => this.role(u.name))
                        .join(
                            ", ",
                        )} into a construction yard; enable building production`;
                } else if (
                    (fn >= 1000 && fn < 2000) ||
                    fn === 60 ||
                    (fn >= 2000 &&
                        fn < 3000 &&
                        this.meta.techno.get(
                            this.catalog.functions
                                .find((f) => f.id === fn)
                                ?.name.replace(/^Train_/, ""),
                        )?.harvester)
                ) {
                    category = "base";
                    label = ready.length
                        ? `Place completed ${queueName(
                              ready[0],
                          )}; activate its benefit`
                        : !refineries.length && v.power_cap
                        ? "Build ore refinery, barracks or power plant"
                        : !barracks.length
                        ? "Build barracks, power plant or ore refinery"
                        : !factories.length
                        ? "Build war factory, power plant or other base structures"
                        : "Construct more military or economic buildings";
                } else if (fn >= 2000 && fn < 3000) {
                    category = "army";
                    label = "Recruit additional combat or specialist units";
                } else if ([49, 61, 62].includes(fn)) {
                    category = "interrupt";
                    label = ready.length
                        ? `Discard completed ${queueName(
                              ready[0],
                          )} or demolish owned buildings`
                        : "Demolish owned buildings or abort current production";
                } else if ([80, 81].includes(fn)) {
                    category = "diplomacy";
                    label = "Change diplomatic relations or concede this match";
                } else if (fn === 29) {
                    category = "camera";
                    label =
                        "Move only the observation camera; does not reveal fog or command troops";
                } else if ([43, 46, 47, 50, 63].includes(fn)) {
                    category = "support";
                    label =
                        "Maintain resource collection, resume paused production or repair damage";
                } else {
                    category = "combat";
                    label = this.frame.visible.length
                        ? "Command troops to fight the visible enemy"
                        : v.army_count
                        ? "Send combat troops to scout for the enemy base"
                        : "Move or change orders of support vehicles";
                }
                if (
                    actors.length &&
                    actors.every((u) => rule(u).harvester) &&
                    ![43, 46, 47].includes(fn)
                ) {
                    category = "worker_orders";
                    label =
                        "Reposition or defend ore harvesters; these vehicles provide mining income";
                }
                if (fn >= 1000 && fn < 3000) {
                    const name = this.catalog.functions
                        .find((f) => f.id === fn)
                        .name.replace(/^(Build|Train)_/, "");
                    const count = own.filter((u) => u.name === name).length;
                    let effect =
                        {
                            NAPOWR: `supply power, current ${v.power_cap}/${v.power_used}`,
                            NAHAND: "unlock infantry and war factory",
                            NAREFN: "add ore refinery and a harvester for income",
                            NAWEAP: "unlock Rhino battle tanks",
                            HTNK: "strong battle tank for attacking enemy buildings",
                            E2: "basic rifle infantry",
                            HARV: "mine ore for recurring income",
                        }[name] ?? this.role(name);
                    if (count > 0 && name === "NAWEAP")
                        effect = "add another vehicle production building";
                    if (count > 0 && name === "NAHAND")
                        effect = "add another infantry production building";
                    const queue = v.production.find((q) =>
                        q.available.includes(
                            this.catalog.unit_types.indexOf(name),
                        ),
                    );
                    c.label = `${fn < 2000 ? "Build" : "Recruit"} ${this.role(
                        name,
                    )}; ${effect}; own ${count}; cost ${c.cost}${
                        fn >= 2000 ? `; ${queue?.size ?? 0} already queued` : ""
                    }`;
                }
                if (fn === 60) {
                    const n = this.catalog.unit_types[a.extra.item];
                    c.functionLabel = `Place completed ${this.role(
                        n,
                    )}; cost already paid; activate its benefit`;
                    c.label = `Place ${this.role(n)} at ${
                        a.world % this.facade.res
                    },${Math.floor(a.world / this.facade.res)}`;
                }
                if (fn === 61)
                    c.label = `Cancel ${q ? queueName(q) : "item"} at ${
                        q?.progress ?? "?"
                    }%; discard this build and refund paid credits`;
                if (fn === 62)
                    c.label = `Pause ${q ? queueName(q) : a.extra.queue} at ${
                        q?.progress ?? "?"
                    }%; freeze construction and spending`;
                if (fn === 63)
                    c.label = `Resume ${q ? queueName(q) : a.extra.queue} at ${
                        q?.progress ?? "?"
                    }%; restart useful construction`;
                if (fn === 49) {
                    const u = actors[0];
                    const loss = rule(u).constructionYard
                        ? "lose the ability to construct buildings"
                        : rule(u).refinery
                        ? "lose this ore unloading site"
                        : u.name === "NAPOWR"
                        ? "lose electricity supply"
                        : "lose its production or combat capability";
                    c.label = `Demolish ${this.role(u.name)} #${
                        u.tag
                    }; ${loss}; recover partial cost once`;
                    c.requiresConfirmation = true;
                    c.cancelLabel = `Keep ${this.role(
                        u.name,
                    )} functioning; retain its capability`;
                    c.confirmationLabel = `Demolish ${this.role(
                        u.name,
                    )}; ${loss}`;
                }
                if (fn === 22) {
                    c.functionLabel =
                        "Attack-move army to scout enemy territory and fight along the route";
                    c.label = c.label.replace(
                        "public possible starting position",
                        layerAt(this.obs, "visibility_map", a.world) === 2
                            ? "currently visible public spawn"
                            : "unexplored possible enemy base",
                    );
                }
                if (fn === 20)
                    c.functionLabel =
                        "Move units without engaging enemies en route";
                if (fn === 23)
                    c.functionLabel = "Attack a currently visible enemy target";
                const functionLabels = {
                    20: "Move units to a location",
                    21: "Force-move units through obstacles or traffic",
                    24: "Fire at ground; can hit friendly objects",
                    25: "Guard a location",
                    26: "Guard current positions",
                    27: "Stop existing unit orders",
                    28: "Deploy selected units",
                    29: "Move the observation camera",
                    40: "Scatter units",
                    41: "Enter transport",
                    42: "Unload passengers",
                    43: "Repair a damaged friendly target",
                    44: "Capture an enemy or neutral building",
                    45: "Garrison a building",
                    46: "Order harvesters to collect observed ore",
                    47: "Dock at a suitable friendly facility",
                    48: "Plant a bomb on a target",
                    49: "Demolish an owned asset for partial refund",
                    50: "Toggle building repairs",
                    61: "Cancel a production item and discard its progress",
                    62: "Pause production and stop its progress",
                    63: "Resume paused production",
                    80: "Change diplomatic relations",
                    81: "Concede this match",
                };
                c.functionLabel ??=
                    fn >= 1000 ? c.label : functionLabels[fn] ?? c.label;
                if (actors.length)
                    c.actorLabel = `${actors.length} ${this.role(
                        actors[0].name,
                    )}${actors.length > 1 ? " units" : ` #${actors[0].tag}`}${
                        rule(actors[0]).harvester ? " (economy)" : ""
                    }`;
                c.intent = { id: category, label };
                c.key = `L${leaves.length + 1}`;
                leaves.push(c);
            }
        // Retain individual Attack choices and add whole-type squads, up to the raw 64-actor limit.
        const squads = new Map();
        for (const c of leaves.filter((c) => c.action.function === 23)) {
            const u = own.find((u) => u.tag === c.action.unit_tags[0]),
                key = `${u.name}:${c.action.target_unit_tag}`;
            if (!squads.has(key)) squads.set(key, { sample: c, name: u.name });
        }
        for (const { sample, name } of squads.values()) {
            const units = own
                .filter(
                    (u) => u.name === name && u.health > 0 && !u.is_in_cargo,
                )
                .slice(0, 64);
            if (units.length < 2) continue;
            const target = this.frame.visible.find(
                (u) => u.tag === sample.action.target_unit_tag,
            );
            if (!target) continue;
            const c = structuredClone(sample);
            c.action.unit_tags = units.map((u) => u.tag);
            c.actorLabel = `All ${units.length} ${this.role(name)} units${
                this.meta.techno.get(name)?.harvester ? " (economy)" : ""
            }`;
            c.label = `Attack visible ${this.role(target.name)} #${
                target.tag
            } with ${units.length} ${this.role(name)} units`;
            c.key = `L${leaves.length + 1}`;
            leaves.push(c);
        }
        // Root labels describe only currently grounded choices, not unavailable strategic wishes.
        const byIntent = new Map();
        for (const c of leaves) {
            if (!byIntent.has(c.intent.id)) byIntent.set(c.intent.id, []);
            byIntent.get(c.intent.id).push(c);
        }
        for (const [id, choices] of byIntent) {
            let label;
            if (id === "base" || id === "army") {
                const placed = choices.find((c) => c.action.function === 60);
                const names = [
                    ...new Set(
                        choices
                            .filter(
                                (c) =>
                                    c.action.function >= 1000 &&
                                    c.action.function < 3000,
                            )
                            .map((c) =>
                                this.catalog.functions
                                    .find((f) => f.id === c.action.function)
                                    .name.replace(/^(Build|Train)_/, ""),
                            ),
                    ),
                ];
                label = placed
                    ? `Place completed ${this.role(
                          this.catalog.unit_types[placed.action.extra.item],
                      )}; activate its benefit`
                    : names.length
                    ? `${id === "army" ? "Recruit" : "Build"} ${names
                          .slice(0, 4)
                          .map((n) => this.role(n))
                          .join(", ")}${
                          names.length > 4 ? " or other available types" : ""
                      }`
                    : choices[0].intent.label;
                if (id === "army" && names.includes("HTNK"))
                    label =
                        queuedMilitary >= 8
                            ? `Add more Rhino tanks or infantry; ${queuedMilitary} military units already queued`
                            : "Recruit Rhino tanks, infantry or other combat troops";
            } else if (id === "support") {
                const verbs = {
                    43: "repair damaged targets",
                    46: "gather ore",
                    47: "dock units",
                    50: "repair buildings",
                    63: "resume paused production",
                };
                label = `Support: ${[
                    ...new Set(choices.map((c) => verbs[c.action.function])),
                ]
                    .filter(Boolean)
                    .join(", ")}`;
            }
            if (label) for (const c of choices) c.intent.label = label;
        }
        for (const cov of packet.availability) {
            const count = leaves.filter(
                (c) => c.action.function === cov.id,
            ).length;
            if (count) {
                cov.candidates = count;
                cov.status = "offered";
            } else if (cov.status === "offered") {
                cov.candidates = 0;
                cov.status =
                    cov.id === 29 ? "wait_equivalent" : "policy_filtered";
                cov.reason =
                    [
                        ...new Set(
                            actorExclusions
                                .filter((x) => x.function === cov.id)
                                .map((x) => x.reason),
                        ),
                    ].join("; ") ||
                    "Redundant movement at the current location omitted";
            }
        }
        if (excludedDeploy) {
            const cov = packet.availability.find((f) => f.id === 28);
            cov.candidates = leaves.filter(
                (c) => c.action.function === 28,
            ).length;
            cov.status = cov.candidates ? "offered" : "unsupported";
            cov.reason =
                "Construction-yard Deploy excluded after verified no-effect engine probes; functioning vehicle/infantry deployment retained.";
        }
        const commandCounts = {};
        for (const c of this.commandHistory.filter(
            (c) => this.tick - c.tick <= 450,
        )) {
            const name =
                this.catalog.functions.find((f) => f.id === c.function)?.name ??
                c.function;
            commandCounts[name] = (commandCounts[name] ?? 0) + 1;
        }
        const parts = {
            essential: [
                "Red Alert 2. Win by destroying the enemy base, while preserving a viable base, income and combat forces. Only normal fog-of-war information is available. Unseen enemies are unknown.",
                "Short game: losing all base buildings without a replacement construction vehicle causes defeat. Preserve the ability to construct and collect income.",
                `Tick ${this.tick} (${(this.tick / 15).toFixed(1)}s). Credits ${
                    v.minerals
                }. Power ${v.power_cap} supply/${
                    v.power_used
                } demand. Harvesters ${v.worker_count}, idle ${
                    v.idle_worker_count
                }. Army count ${
                    v.army_count
                }; military units queued ${queuedMilitary}. Own objects: ${Object.entries(
                    countsOf(own),
                )
                    .map(([n, c]) => `${this.role(n)}:${c}`)
                    .join(", ")}.`,
                `Buildings finishing placement: ${
                    unfolding
                        .map((u) => `${this.role(u.name)} ${u.build_progress}%`)
                        .join(", ") || "none"
                }. Current production: ${v.production
                    .map(
                        (q) =>
                            `${q.queue}=${q.status}${
                                q.size
                                    ? ` ${queueName(q)} ${q.progress}%, ${
                                          q.size
                                      } queued`
                                    : ""
                            }`,
                    )
                    .join("; ")}.`,
                `Recommended strategic objective: ${priority} This is advice; choose the action and its parameters.`,
                "Economy first: power plant unlocks ore refinery; refinery provides a harvester. Barracks unlocks war factory; war factory produces Rhino tanks. Extra power without demand does not provide income or an army.",
                "Preserve useful ongoing production. Active queues advance without a new command. OnHold queues need Resume. Ready buildings need Place. Cancelling loses the build; pausing prevents progress. Do not repeatedly reverse your own orders.",
                `Currently visible enemy: ${countsText(
                    this.frame.visible,
                )}. ${this.frame.visible
                    .slice(0, 5)
                    .map(
                        (u) =>
                            `${this.role(u.name)} #${u.tag} at ${u.x},${
                                u.y
                            } HP${u.health}`,
                    )
                    .join("; ")}`,
                `Commands submitted in previous ${
                    Math.min(this.tick, 450) / 15
                }s: ${JSON.stringify(
                    commandCounts,
                )}. Submission is not proof of execution.`,
                "Production can run while existing troops scout and attack. Use combat troops for battle and keep ore harvesters mining. Reinforce a useful existing army instead of leaving it idle.",
                "Choose one useful action goal, then its function, acting units and target. Every further call preserves your previous selections at this same game tick. Existing orders continue between decisions.",
            ],
            details: own
                .slice(0, 16)
                .map(
                    (u) =>
                        `Own ${this.role(u.name)} #${u.tag} at ${u.x},${
                            u.y
                        }, HP${u.health}, idle=${u.is_idle}, current order=${
                            u.order_id_0
                        }.`,
                ),
            memory: this.frame.memory
                .slice(0, 5)
                .map(
                    (m) =>
                        `Last seen ${this.role(m.name)} at ${m.x},${m.y}, ${
                            m.ageTicks / 15
                        }s ago; current survival and position unknown.`,
                ),
            logs: [
                ...this.effects.slice(-4).map((e) => `t${e.tick}: ${e.text}`),
                ...this.commandHistory
                    .slice(-2)
                    .map((e) => `t${e.tick}: chose ${e.label}; ${e.result}.`),
            ],
        };
        // Compact state: current facts and immediate bottleneck precede secondary history.
        parts.essential = [
            parts.essential[0],
            parts.essential[2],
            parts.essential[3],
            parts.essential[4],
            parts.essential[7],
            "A finished building must be placed before it can work. Queued production continues while combat troops scout and attack. Ore harvesters collect income; use military units for distant attacks.",
            "Keep useful buildings and the construction yard functioning. Build an income source, military production, and an army; use the army to find and destroy enemy buildings.",
            parts.essential[8],
        ];
        const types = [...new Set(own.map((u) => u.name))];
        parts.details = types
            .filter((n) => this.meta.techno.get(n)?.type !== 2)
            .slice(0, 10)
            .map((n) => {
                const us = own.filter((u) => u.name === n);
                return `${us.length} ${this.role(n)}: ${
                    us.filter((u) => u.is_idle).length
                } idle; orders ${[...new Set(us.map((u) => u.order_id_0))].join(
                    ",",
                )}; positions ${us
                    .slice(0, 3)
                    .map((u) => `${u.x},${u.y}`)
                    .join("; ")}.`;
            });
        parts.logs = parts.logs.slice(-3);
        // One decision per second by default. The engine and opponent still step every three ticks.
        const groups =
            this.tick % this.decisionInterval === 0 && leaves.length
                ? [
                      {
                          id: "next_action",
                          hierarchy: "intent-function",
                          contextVersion: 2,
                          waitLabel,
                          instructions:
                              "Choose the next action that advances your strategic objective. Keep useful processes running and protect the means to win.",
                          choices: [
                              {
                                  key: "A",
                                  label: waitLabel,
                                  action: null,
                                  cost: 0,
                              },
                              ...leaves,
                          ],
                      },
                  ]
                : [];
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
        this.current = {
            ...packet,
            groups,
            questions,
            parts,
            actorExclusions: [
                ...new Map(
                    actorExclusions.map((x) => [JSON.stringify(x), x]),
                ).values(),
            ],
            candidateMode: "intent",
        };
        return this.current;
    }
    select(answers, cap) {
        const result = super.select(answers, cap);
        for (const d of result.dropped)
            this.effects.push({
                tick: this.tick,
                text: `Selected action was filtered: ${d.reason}; no command was submitted.`,
            });
        return result;
    }
    receipts(tick, results) {
        for (let i = 0; i < this.pending.length; i++) {
            const c = this.pending[i];
            this.commandHistory.push({
                tick,
                function: c.action.function,
                label: c.label,
                result: results?.[i]?.accepted
                    ? "accepted for processing; effect not yet verified"
                    : `rejected: ${results?.[i]?.reason ?? "missing receipt"}`,
            });
        }
        super.receipts(tick, results);
    }
}
