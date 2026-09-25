// Policy-only encoding. This module has no engine or observer handle.
export function rowsOf(obs, catalog) {
    const { features, data } = obs.units,
        width = features.length,
        rows = [];
    for (let i = 0; i < data.shape[0]; i++) {
        const r = Object.fromEntries(
            features.map((k, j) => [k, data.data[i * width + j]]),
        );
        rows.push({ ...r, name: catalog.unit_types[r.unit_type] ?? "unknown" });
    }
    return rows;
}
export function layerAt(obs, name, cell) {
    const index = obs.world.channels.indexOf(name),
        size = obs.world.data.shape[1] ** 2;
    return index < 0 || cell < 0 || cell >= size
        ? 0
        : obs.world.data.data[index * size + cell];
}
export const countsOf = (rows) =>
    Object.fromEntries(
        [...new Set(rows.map((r) => r.name))]
            .sort()
            .map((n) => [n, rows.filter((r) => r.name === n).length]),
    );
export class ObservationMemory {
    constructor() {
        this.last = null;
        this.enemies = new Map();
        this.events = [];
        this.seenVisible = new Set();
        this.noCommandTurns = 0;
        this.lastCommandTick = null;
        this.strategicProgress = null;
        this.creditHistory = [];
    }
    event(tick, kind, text) {
        this.events.push({ tick, kind, text });
        this.events = this.events
            .filter((e) => tick - e.tick <= 900)
            .slice(-64);
    }
    update(obs, catalog, tick) {
        const rows = rowsOf(obs, catalog),
            own = rows.filter((r) => r.alliance === 1),
            visible = rows.filter(
                (r) => r.alliance === 4 && r.display_type === 1 && !r.is_blip,
            );
        const ids = new Set(visible.map((r) => r.tag));
        for (const r of visible) {
            if (!this.seenVisible.has(r.tag))
                this.event(
                    tick,
                    "sighted",
                    `Sighted ${r.name} at ${r.x},${r.y}.`,
                );
            this.enemies.set(r.tag, {
                tag: r.tag,
                name: r.name,
                x: r.x,
                y: r.y,
                lastSeenTick: tick,
                health: r.health,
            });
        }
        if (!obs.vectors.units_truncated)
            for (const tag of this.seenVisible)
                if (!ids.has(tag))
                    this.event(
                        tick,
                        "lost_sight",
                        `Enemy ${
                            this.enemies.get(tag)?.name ?? tag
                        } no longer visible; fate unknown.`,
                    );
        if (this.last) {
            for (const r of own) {
                const prev = this.last.own.get(r.tag);
                if (prev && r.health < prev.health)
                    this.event(
                        tick,
                        "damage",
                        `${r.name} ${r.tag} lost ${
                            prev.health - r.health
                        } HP; now ${r.health}.`,
                    );
            }
            if (!obs.vectors.units_truncated && !this.last.truncated) {
                const before = countsOf([...this.last.own.values()]),
                    after = countsOf(own);
                for (const n of new Set([
                    ...Object.keys(before),
                    ...Object.keys(after),
                ]))
                    if ((before[n] ?? 0) !== (after[n] ?? 0))
                        this.event(
                            tick,
                            "own_count",
                            `Own ${n} count ${before[n] ?? 0}->${
                                after[n] ?? 0
                            }; production, transformation or loss.`,
                        );
            }
        }
        for (const [tag, m] of this.enemies)
            if (tick - m.lastSeenTick > 900) this.enemies.delete(tag);
        this.events = this.events
            .filter((e) => tick - e.tick <= 900)
            .slice(-64);
        this.seenVisible = ids;
        this.last = {
            own: new Map(own.map((r) => [r.tag, r])),
            truncated: obs.vectors.units_truncated,
        };
        return {
            rows,
            own,
            visible,
            memory: [...this.enemies.values()]
                .filter((m) => !ids.has(m.tag))
                .sort((a, b) => b.lastSeenTick - a.lastSeenTick)
                .map((m) => ({
                    ...m,
                    ageTicks: tick - m.lastSeenTick,
                    status: "last_seen_not_current",
                })),
        };
    }
    receipts(tick, actions, results) {
        if (actions.length) {
            this.noCommandTurns = 0;
            this.lastCommandTick = tick;
        } else this.noCommandTurns++;
        for (let i = 0; i < actions.length; i++)
            this.event(
                tick,
                "action",
                `${actions[i].label}: ${
                    results?.[i]?.accepted
                        ? "queued (execution not yet confirmed)"
                        : `rejected ${
                              results?.[i]?.reason ?? "missing receipt"
                          }`
                }.`,
            );
    }
}
// Context-only strategic advice: no action filtering, scoring, or forced commands.
export function strategySummary(obs, frame, memory, tick, explored) {
    const v = obs.vectors,
        count = (names) =>
            frame.own.filter((u) => names.includes(u.name)).length;
    const yards = count(["NACNST", "GACNST"]),
        refineries = count(["NAREFN", "GAREFN"]),
        barracks = count(["NAHAND", "GAPILE"]);
    const powerSurplus = v.power_cap - v.power_used,
        ready = v.production
            .filter((q) => q.status === "Ready")
            .map((q) => q.queue);
    const signature = JSON.stringify(countsOf(frame.own));
    const previous = memory.strategicProgress;
    if (!previous) memory.strategicProgress = { tick, signature, explored };
    else {
        if (
            explored > previous.explored ||
            (!v.units_truncated && signature !== previous.signature)
        )
            previous.tick = tick;
        if (!v.units_truncated) previous.signature = signature;
        previous.explored = Math.max(previous.explored, explored);
    }
    if (
        !memory.creditHistory.length ||
        tick - memory.creditHistory.at(-1).tick >= 15
    )
        memory.creditHistory.push({ tick, credits: v.minerals });
    memory.creditHistory = memory.creditHistory.filter(
        (x) => tick - x.tick <= 900,
    );
    const first = memory.creditHistory[0],
        net = v.minerals - first.credits,
        span = Math.round((tick - first.tick) / 15);
    const priority = !yards
        ? "Deploy the construction vehicle to enable production."
        : !v.power_cap
        ? "Establish initial power so the economic buildings can be built."
        : !refineries
        ? "ECONOMY CRITICAL: build an ore refinery with harvester before extra power plants, walls, technology or mass infantry. Reserve 2000 credits for the refinery."
        : !v.worker_count
        ? "ECONOMY CRITICAL: restore a harvester; an economy without workers cannot mine ore."
        : v.is_low_power
        ? "Restore power to support the existing economy and factories."
        : !barracks
        ? "Develop military production while keeping harvesters working."
        : !v.army_count
        ? "Recruit combat units for scouting and defending the economy."
        : !frame.visible.length
        ? "Use available combat units to scout unknown areas and find the enemy; do not keep idle troops at home indefinitely."
        : "Engage visible vulnerable enemies with combat units; protect harvesters and the base, retreat if badly damaged.";
    return [
        "Strategic preference: active, economical play. Win through income, scouting, an army and attacks. Prefer a useful concrete action over waiting. Extra electricity alone produces no credits and cannot win.",
        `Current strategic priority: ${priority}`,
        `Economy assessment: refineries=${refineries}, harvesters=${
            v.worker_count
        }, credits=${v.minerals}, power surplus=${powerSurplus}. ${
            powerSurplus >= 100
                ? "Power is already sufficient: avoid another power plant unless expansion will need it."
                : ""
        } Net credit change over ${span}s=${net} (spending included; not gross mining income).`,
        `Completed building queues: ${
            ready.join(", ") || "none"
        }. Place completed buildings promptly: their cost is already paid; an unplaced building provides no benefit and blocks the queue.`,
        `Progress memory: ${Math.floor(
            (tick - memory.strategicProgress.tick) / 15,
        )}s since an observed own-composition change or newly explored terrain. Consecutive decisions with no command=${
            memory.noCommandTurns
        }; last command tick=${
            memory.lastCommandTick ?? "none"
        }. Accepted commands alone do not prove progress.`,
        "Waiting policy: wait only for ongoing useful production or movement, to save for a necessary economic building, or when no useful action is available. If troops are idle, prefer scouting, attacking a suitable visible target or defending against a real threat. When progress stalls, change the plan; repeated waiting does not create income, scouts or troops. Choose only the actions offered in each question; do not assume an unavailable action can be executed.",
    ];
}
export function contextParts(
    obs,
    catalog,
    frame,
    memory,
    tick,
    candidateSummary,
) {
    const v = obs.vectors,
        r = obs.world.data.shape[1],
        cells = r * r;
    const counts = Array.from(
        v.unit_counts?.data ?? v.unit_counts ?? [],
    ).flatMap((n, i) => (n ? [`${catalog.unit_types[i]}:${n}`] : []));
    const queueText = v.production
        .map(
            (q) =>
                `${q.queue}:${q.status},item=${
                    catalog.unit_types[q.item] ?? "none"
                },progress=${q.progress},size=${q.size}`,
        )
        .join("; ");
    let explored = 0,
        visibleCells = 0;
    const alerts = [];
    for (let cell = 0; cell < cells; cell++) {
        const vis = layerAt(obs, "visibility_map", cell);
        if (vis > 0) explored++;
        if (vis === 2) visibleCells++;
        const alert = layerAt(obs, "alerts", cell);
        if (alert && alerts.length < 6)
            alerts.push(`${cell % r},${Math.floor(cell / r)}:${alert}`);
    }
    const enemyCounts = countsOf(frame.visible);
    const essential = [
        "Red Alert 2. Goal: win by destroying the enemy base while preserving economy and combat forces. Only normal fog-of-war information is available. Unknown enemies are not absent enemies. Commands already issued continue during wait.",
        ...strategySummary(obs, frame, memory, tick, explored),
        "Public rules: construction vehicles must deploy into construction yards before base buildings can be produced. Power plants supply power; ore refineries and harvesters enable resource income; barracks produce infantry; war factories produce vehicles.",
        `Current construction yards: ${
            frame.own.filter((u) => ["NACNST", "GACNST"].includes(u.name))
                .length
        }. Current harvesters: ${
            v.worker_count
        }. Available production offers: ${
            v.production
                .flatMap((q) => q.available.map((i) => catalog.unit_types[i]))
                .join(", ") || "none"
        }.`,
        `Tick ${tick} (${(tick / 15).toFixed(1)} seconds). Credits ${
            v.minerals
        }. Power ${v.power_cap} generated/${
            v.power_used
        } used; low=${!!v.is_low_power}. Army ${v.army_count}; harvesters ${
            v.worker_count
        }, idle ${v.idle_worker_count}.`,
        `Own counts: ${counts.join(", ") || "none"}. Queues: ${
            queueText || "none"
        }.`,
        `Currently visible enemies: ${JSON.stringify(enemyCounts)}. ${
            frame.visible.length
                ? frame.visible
                      .slice(0, 6)
                      .map((u) => `${u.name} at ${u.x},${u.y} HP${u.health}`)
                      .join("; ")
                : "No enemies currently seen."
        }`,
        `Map grid ${r}x${r}. Explored ${Math.round(
            (explored / cells) * 100,
        )}%, visible ${Math.round(
            (visibleCells / cells) * 100,
        )}%. Own radar alerts ${
            alerts.join("; ") || "none"
        }. Unit rows omitted by engine=${v.units_truncated}.`,
        candidateSummary,
    ];
    const details = frame.own
        .slice()
        .sort((a, b) => a.health_ratio - b.health_ratio || a.tag - b.tag)
        .slice(0, 20)
        .map(
            (u) =>
                `Own ${u.name} #${u.tag} at ${u.x},${u.y}; HP ${u.health}, idle ${u.is_idle}, order ${u.order_id_0}, deployed ${u.is_deployed}.`,
        );
    return {
        essential,
        details,
        memory: frame.memory
            .slice(0, 8)
            .map(
                (m) =>
                    `Last seen ${m.name} at ${m.x},${m.y}, ${m.ageTicks} ticks ago; current position and survival unknown.`,
            ),
        logs: memory.events.slice(-8).map((e) => `t${e.tick}: ${e.text}`),
    };
}
