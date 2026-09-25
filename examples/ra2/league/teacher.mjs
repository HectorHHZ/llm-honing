// Training-data teacher only. Never used as a fallback for evaluated models.
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
export const TEACHER_VERSION = "normal-macro-v10";
export class MacroTeacher {
    constructor() {
        this.lastCombat = -150;
        this.lastGather = new Map();
        this.lastDeploy = -150;
        this.searched = new Map();
    }
    observe(actions, tick) {
        for (const action of actions) {
            if ([22, 23].includes(action.function)) this.lastCombat = tick;
            if (action.function === 28) this.lastDeploy = tick;
            if (action.function === 46)
                for (const tag of action.unit_tags)
                    this.lastGather.set(tag, tick);
        }
    }
    choose(policy, packet) {
        const choices = packet.groups.flatMap((g) => g.choices),
            leaves = choices.filter((c) => c.action),
            wait = choices.find((c) => !c.action);
        const own = policy.frame.own,
            v = policy.obs.vectors,
            r = policy.facade.res,
            rule = (u) => policy.meta.techno.get(u.name) ?? {};
        const completed = own.filter(
            (u) => rule(u).type !== 2 || u.build_progress === 100,
        );
        const count = (n) => own.filter((u) => u.name === n).length;
        const base = own.find((u) => rule(u).constructionYard) ??
            own[0] ?? { x: 0, y: 0 };
        const point = (c) => ({
            x: c.action.world % r,
            y: Math.floor(c.action.world / r),
        });
        const fn = (n) =>
            policy.catalog.functions.find((f) => f.name === n)?.id;
        const pick = (n) => leaves.find((c) => c.action.function === fn(n));
        const chooseFn = (id) => leaves.filter((c) => c.action.function === id);
        const queued = (q) => v.production.find((x) => x.queue === q);
        const emit = (c) => c ?? wait;
        const place = chooseFn(60).sort(
            (a, b) => distance(point(a), base) - distance(point(b), base),
        )[0];
        if (place) return place;
        const deploy = chooseFn(28).find((c) =>
            c.action.unit_tags.some((t) =>
                own.find(
                    (u) =>
                        u.tag === t &&
                        policy.meta.general.baseUnit.includes(u.name) &&
                        u.is_idle,
                ),
            ),
        );
        if (deploy && policy.tick - this.lastDeploy >= 90) return deploy;
        if (!count("NACNST")) return wait;
        if (!count("NAPOWR")) return emit(pick("Build_NAPOWR"));
        if (!count("NAREFN")) return emit(pick("Build_NAREFN"));
        // Check both current and next tasks before interrupting an automatic mining loop.
        const idle = new Set(
            completed
                .filter(
                    (u) =>
                        rule(u).harvester &&
                        u.is_idle &&
                        ![46, 47].includes(u.order_id_0) &&
                        ![46, 47].includes(u.order_id_1) &&
                        policy.tick - (this.lastGather.get(u.tag) ?? -150) >=
                            150,
                )
                .map((u) => u.tag),
        );
        const gather = chooseFn(46)
            .filter((c) => c.action.unit_tags.every((t) => idle.has(t)))
            .sort((a, b) => {
                const actor = (c) =>
                    own.find((u) => u.tag === c.action.unit_tags[0]);
                return (
                    distance(point(a), actor(a)) - distance(point(b), actor(b))
                );
            })[0];
        if (gather) return gather;
        const resume = chooseFn(63)[0];
        if (resume) return resume;
        if (v.is_low_power || v.power_cap - v.power_used < 20) {
            const c = pick("Build_NAPOWR");
            if (c) return c;
        }
        if (!count("NAHAND")) return emit(pick("Build_NAHAND"));
        if (!count("NAWEAP")) return emit(pick("Build_NAWEAP"));
        if (v.worker_count < 3 && (queued("vehicles")?.size ?? 0) < 1) {
            const c = pick("Train_HARV");
            if (c) return c;
        }
        if ((queued("vehicles")?.size ?? 0) < 2) {
            const c = pick("Train_HTNK");
            if (c) return c;
        }
        if (v.minerals > 2500 && v.worker_count < 5) {
            const c = pick("Build_NAREFN");
            if (c) return c;
        }
        if (
            v.minerals > 3000 &&
            count("E2") < 6 &&
            (queued("infantry")?.size ?? 0) < 2
        ) {
            const c = pick("Train_E2");
            if (c) return c;
        }
        if (policy.tick - this.lastCombat < 150) return wait;
        const army = own.filter(
            (u) => u.name === "HTNK" && u.health > 0 && !u.is_in_cargo,
        );
        const armed = (u) =>
            [rule(u).primary, rule(u).secondary]
                .filter(Boolean)
                .some(
                    (name) =>
                        Number(policy.meta.rules?.getWeapon(name)?.damage) > 0,
                );
        const priority = (u) =>
            u.name === "HTNK" ? 0 : u.name === "SHK" ? 1 : armed(u) ? 2 : 3;
        const threats = policy.frame.visible.filter(
            (u) =>
                distance(u, base) < 25 ||
                (armed(u) && army.some((a) => distance(a, u) <= 12)),
        );
        const targets = (
            threats.length
                ? threats
                : policy.frame.visible.filter((u) => rule(u).type === 2)
        )
            .slice()
            .sort(
                (a, b) =>
                    priority(a) - priority(b) ||
                    a.health - b.health ||
                    a.tag - b.tag,
            );
        const targetRank = new Map(targets.map((u, i) => [u.tag, i]));
        if (army.length >= 4 || threats.length) {
            const ids = new Set(targets.map((u) => u.tag));
            const attacks = chooseFn(23)
                .filter(
                    (c) =>
                        ids.has(c.action.target_unit_tag) &&
                        c.action.unit_tags.every((t) =>
                            army.some((u) => u.tag === t),
                        ),
                )
                .sort(
                    (a, b) =>
                        targetRank.get(a.action.target_unit_tag) -
                            targetRank.get(b.action.target_unit_tag) ||
                        b.action.unit_tags.length - a.action.unit_tags.length,
                );
            if (attacks[0]) {
                const target = targets.find(
                    (u) => u.tag === attacks[0].action.target_unit_tag,
                );
                // Reinforcements should keep travelling after a short-lived
                // target dies. Use only an already offered, observed point.
                const distant =
                    target &&
                    army.filter((u) => distance(u, target) > 12).length >
                        army.length / 2;
                const advance = distant
                    ? chooseFn(22)
                          .filter(
                              (c) =>
                                  c.action.unit_tags.every((t) =>
                                      army.some((u) => u.tag === t),
                                  ) && distance(point(c), target) < 2,
                          )
                          .sort(
                              (a, b) =>
                                  b.action.unit_tags.length -
                                  a.action.unit_tags.length,
                          )[0]
                    : null;
                return advance ?? attacks[0];
            }
            // A bounded individual-actor list can omit tank Attack_Unit leaves.
            // Use a currently offered tank squad AttackMove at a visible target.
            const assault = chooseFn(22)
                .filter(
                    (c) =>
                        c.action.unit_tags.every((t) =>
                            army.some((u) => u.tag === t),
                        ) &&
                        targets.some(
                            (target) => distance(point(c), target) < 2,
                        ),
                )
                .sort(
                    (a, b) =>
                        b.action.unit_tags.length - a.action.unit_tags.length,
                )[0];
            if (assault) return assault;
        }
        if (army.length >= 5) {
            const candidates = chooseFn(22).filter(
                (c) =>
                    !c.label.includes("(own base)") &&
                    c.action.unit_tags.every((t) =>
                        army.some((u) => u.tag === t),
                    ),
            );
            // A public location is searched only after our army actually reaches it.
            // No opponent start, privileged unit position or god-view data is used.
            const visits = policy.searchVisits ?? this.searched;
            if (!policy.searchVisits)
                for (const c of candidates) {
                    if (army.some((u) => distance(u, point(c)) < 8))
                        visits.set(c.action.world, policy.tick);
                }
            const rank = (c) => {
                if (visits.has(c.action.world)) return 3;
                if (c.label.includes("unexplored possible enemy base"))
                    return 0;
                if (c.label.includes("last seen")) return 1;
                return 2;
            };
            const scout = candidates.sort(
                (a, b) =>
                    rank(a) - rank(b) ||
                    // Revisit the least recently observed public location after
                    // the first sweep, instead of repeatedly choosing near home.
                    (visits.get(a.action.world) ?? -1) -
                        (visits.get(b.action.world) ?? -1) ||
                    b.action.unit_tags.length - a.action.unit_tags.length ||
                    distance(point(a), base) - distance(point(b), base),
            )[0];
            if (scout) {
                return scout;
            }
        }
        return wait;
    }
}
