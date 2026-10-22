/**
 * Event-aware inference scheduling for the pinned Laya RA2 v33 harness.
 *
 * This module never produces, filters, or changes a game command. Call evaluate
 * every engine step (3 ticks), using this seat's normal observation and an
 * IntentPolicy configured with decisionInterval=3 so that candidates exist on
 * event wakes. The scheduler independently enforces decisionInterval (default
 * 30). Use a separate instance for each seat and each match.
 *
 * Only a fully resolved ROOT intent-Wait may authorize suppressing later model
 * calls. A later target/parameter cancellation, rejected action, incomplete
 * trace, or unknown observation schema does not authorize suppression. We only
 * extend waits while production/a placed structure is busy, no army/enemy is
 * present, and there is no meaningful grounded action offered. Completion,
 * queue changes, own losses/status changes, visible-enemy changes, power
 * changes, new candidates and production stall/resumption wake a waiting
 * model. Combat and other non-wait decisions retain the normal base cadence.
 *
 * Ordinary credit/progress changes, descriptive text, map tensors, remembered
 * enemies and hidden/non-visible rows are not scheduling inputs. Construction
 * progress (0..100) is read only to detect completion or a >=stallTicks plateau;
 * individual percentage changes never wake the model. Continuous inference
 * suppression is bounded by maxWaitTicks since the last real model decision.
 */
export const SCHEDULER_VERSION = "laya-v33-wait-scheduler-v1";
const NON_OPPORTUNITIES = new Set([0, 29, 49, 61, 62, 80, 81]);
const QUEUE_STATUSES = new Set(["Idle", "Active", "Ready", "OnHold"]);
const finite = (x) => typeof x === "number" && Number.isFinite(x);
const flag = (x) => x === 0 || x === 1 || typeof x === "boolean";
const stable = (value) =>
    JSON.stringify(value, (_, v) =>
        v && typeof v === "object" && !Array.isArray(v)
            ? Object.fromEntries(
                  Object.entries(v).sort(([a], [b]) => a.localeCompare(b)),
              )
            : v,
    );
const sorted = (xs) => [...xs].sort();
const same = (a, b) => stable(a) === stable(b);

function inspect({ tick, observation, frame, packet } = {}) {
    const v = observation?.vectors;
    if (
        !Number.isSafeInteger(tick) ||
        tick < 0 ||
        !v ||
        !Array.isArray(frame?.rows) ||
        !Array.isArray(v.production) ||
        !finite(v.army_count) ||
        v.army_count < 0 ||
        !finite(v.power_cap) ||
        !finite(v.power_used) ||
        !flag(v.units_truncated) ||
        (v.is_low_power !== undefined && !flag(v.is_low_power)) ||
        !Array.isArray(packet?.groups)
    )
        return { error: "unknown_schema" };
    if (v.units_truncated) return { error: "truncated_observation" };
    if (observation.view && observation.view !== "normal")
        return { error: "non_normal_observation" };
    if (
        packet.groups.length > 1 ||
        packet.groups.some(
            (g) =>
                !g ||
                g.id !== "next_action" ||
                !Array.isArray(g.choices) ||
                g.choices.some((c) => !c || typeof c !== "object") ||
                g.choices.filter((c) => c.action === null).length !== 1 ||
                g.choices.some(
                    (c) =>
                        c.action !== null &&
                        !Number.isInteger(c.action?.function),
                ),
        )
    )
        return { error: "unknown_packet_schema" };
    const queues = v.production;
    if (
        queues.some(
            (q) =>
                !q ||
                typeof q.queue !== "string" ||
                !QUEUE_STATUSES.has(q.status) ||
                !Number.isInteger(q.size) ||
                q.size < 0 ||
                !Number.isInteger(q.item) ||
                !finite(q.progress) ||
                q.progress < 0 ||
                q.progress > 100,
        ) ||
        new Set(queues.map((q) => q.queue)).size !== queues.length
    )
        return { error: "unknown_queue_schema" };

    // Refilter rows ourselves. Never trust a caller's convenience `visible`
    // array, remembered enemies, or extra hidden fields in the observation.
    if (frame.rows.some((u) => !u || typeof u !== "object"))
        return { error: "unknown_unit_schema" };
    const own = frame.rows.filter((u) => u.alliance === 1);
    const visible = frame.rows.filter(
        (u) => u.alliance === 4 && u.display_type === 1 && !u.is_blip,
    );
    const rowValid = (u) =>
        Number.isSafeInteger(u.tag) &&
        finite(u.health) &&
        Number.isInteger(u.unit_type) &&
        finite(u.build_progress) &&
        u.build_progress >= 0 &&
        u.build_progress <= 100 &&
        flag(u.can_move) &&
        flag(u.is_idle) &&
        flag(u.is_powered) &&
        Number.isInteger(u.order_id_0) &&
        Number.isInteger(u.order_id_1);
    if (
        own.some((u) => !rowValid(u)) ||
        new Set(own.map((u) => u.tag)).size !== own.length ||
        visible.some(
            (u) =>
                !Number.isSafeInteger(u.tag) ||
                !finite(u.health) ||
                !Number.isInteger(u.unit_type) ||
                !finite(u.x) ||
                !finite(u.y),
        )
    )
        return { error: "unknown_unit_schema" };
    const leaves = packet.groups
        .flatMap((g) => g.choices)
        .filter((c) => c.action);
    const meaningful = leaves.filter(
        (c) => !NON_OPPORTUNITIES.has(c.action.function),
    );
    // Ignore labels, key allocation and candidate order: their wording changes
    // with credits/progress even when the grounded options are identical.
    const opportunities = sorted(
        new Set(
            meaningful.map(({ action: a }) =>
                stable({
                    function: a.function,
                    unit_tags: sorted(a.unit_tags ?? []),
                    target_unit_tag: a.target_unit_tag,
                    extra: a.extra,
                }),
            ),
        ),
    );
    const lowPower = !!v.is_low_power || v.power_used > v.power_cap;
    const unfolding = own.filter((u) => !u.can_move && u.build_progress < 100);
    const active = queues.filter((q) => q.status === "Active" && q.size > 0);
    const waitingOnProduction = active.length > 0 || unfolding.length > 0;
    return {
        tick,
        own,
        visible,
        queues,
        opportunities,
        lowPower,
        army: v.army_count,
        waitingOnProduction,
        active,
        unfolding,
        hasCandidates: packet.groups.length > 0,
        waitKey: packet.groups[0]?.choices.find((c) => c.action === null)?.key,
        queueState: queues
            .map((q) => ({
                queue: q.queue,
                status: q.status,
                item: q.item,
                size: q.size,
            }))
            .sort((a, b) => a.queue.localeCompare(b.queue)),
        ownStatus: own
            .map((u) => ({
                tag: u.tag,
                unit_type: u.unit_type,
                idle: !!u.is_idle,
                powered: !!u.is_powered,
                canMove: !!u.can_move,
                complete: u.build_progress === 100,
                deployed: u.is_deployed,
                cargo: u.is_in_cargo,
                order0: u.order_id_0,
                order1: u.order_id_1,
            }))
            .sort((a, b) => a.tag - b.tag),
        enemyState: visible
            .map((u) => ({
                tag: u.tag,
                unit_type: u.unit_type,
                x: u.x,
                y: u.y,
                health: u.health,
            }))
            .sort((a, b) => a.tag - b.tag),
    };
}

export class WaitScheduler {
    constructor({
        decisionInterval = 30,
        maxWaitTicks = 150,
        stallTicks = 150,
    } = {}) {
        for (const [key, value] of Object.entries({
            decisionInterval,
            maxWaitTicks,
            stallTicks,
        }))
            if (!Number.isSafeInteger(value) || value < 3 || value % 3)
                throw new Error(`${key} must be a positive multiple of 3`);
        if (maxWaitTicks < decisionInterval)
            throw new Error("maxWaitTicks must be >= decisionInterval");
        this.decisionInterval = decisionInterval;
        this.maxWaitTicks = maxWaitTicks;
        this.stallTicks = stallTicks;
        this.lastDecisionTick = null;
        this.last = null;
        this.waiting = false;
        this.pendingReceiptFailure = false;
        this.progress = new Map();
        this.stalled = new Set();
    }

    evaluate(input) {
        const state = inspect(input ?? {});
        if (state.error || (this.last && state.tick < this.last.tick)) {
            this.waiting = false;
            this.last = null;
            this.progress.clear();
            this.stalled.clear();
            return {
                infer: true,
                reason: state.error ?? "non_monotonic_tick",
                waiting: false,
                events: [],
            };
        }
        const previous = this.last,
            events = [];
        if (previous) {
            if (!same(state.queueState, previous.queueState))
                events.push("production_changed");
            if (!same(state.ownStatus, previous.ownStatus))
                events.push("own_units_changed");
            const oldHealth = new Map(
                previous.own.map((u) => [u.tag, u.health]),
            );
            if (
                state.own.some(
                    (u) =>
                        oldHealth.has(u.tag) && u.health < oldHealth.get(u.tag),
                )
            )
                events.push("own_damage");
            if (!same(state.enemyState, previous.enemyState))
                events.push("visible_enemy_changed");
            if (state.lowPower !== previous.lowPower)
                events.push("power_changed");
            if (state.army !== previous.army) events.push("army_changed");
            if (
                state.opportunities.some(
                    (x) => !previous.opportunities.includes(x),
                )
            )
                events.push("new_action_opportunity");
        }
        const progress = [
            ...state.active.map((q) => [
                `queue:${q.queue}:${q.item}`,
                q.progress,
            ]),
            ...state.unfolding.map((u) => [
                `placement:${u.tag}`,
                u.build_progress,
            ]),
        ];
        const liveKeys = new Set(progress.map(([key]) => key)),
            stalledNow = new Set();
        for (const [key, value] of progress) {
            const prior = this.progress.get(key);
            if (!prior || prior.value !== value)
                this.progress.set(key, { value, since: state.tick });
            if (state.tick - this.progress.get(key).since >= this.stallTicks)
                stalledNow.add(key);
        }
        for (const key of this.progress.keys())
            if (!liveKeys.has(key)) this.progress.delete(key);
        if ([...stalledNow].some((key) => !this.stalled.has(key)))
            events.push("production_stalled");
        if (
            [...this.stalled].some(
                (key) => liveKeys.has(key) && !stalledNow.has(key),
            )
        )
            events.push("production_resumed");
        this.stalled = stalledNow;
        this.last = structuredClone(state);
        const elapsed =
            this.lastDecisionTick === null
                ? Infinity
                : state.tick - this.lastDecisionTick;
        const reasons = [];
        if (!state.waitingOnProduction) reasons.push("no_busy_production");
        if (state.army > 0) reasons.push("army_present");
        if (state.visible.length) reasons.push("visible_enemy");
        if (state.lowPower) reasons.push("low_power");
        if (
            state.queues.some(
                (q) => q.status === "Ready" || q.status === "OnHold",
            )
        )
            reasons.push("queue_needs_action");
        if (state.opportunities.length) reasons.push("useful_action_available");
        if (stalledNow.size) reasons.push("stalled_production");
        state.skipEligible = reasons.length === 0;
        this.last.skipEligible = state.skipEligible;
        const base = {
            waiting: this.waiting,
            events,
            skipEligible: state.skipEligible,
            blockers: reasons,
            elapsedTicks: Number.isFinite(elapsed) ? elapsed : null,
        };
        if (!state.hasCandidates)
            return { ...base, infer: false, reason: "no_candidates" };
        if (this.pendingReceiptFailure) {
            this.pendingReceiptFailure = false;
            this.waiting = false;
            return {
                ...base,
                waiting: false,
                infer: true,
                reason: "receipt_rejected_or_missing",
            };
        }
        if (this.lastDecisionTick === null)
            return { ...base, infer: true, reason: "initial_decision" };
        if (this.waiting && events.length) {
            this.waiting = false;
            return {
                ...base,
                waiting: false,
                infer: true,
                reason: "event_wake",
            };
        }
        if (this.waiting && state.skipEligible) {
            if (elapsed >= this.maxWaitTicks)
                return { ...base, infer: true, reason: "wait_heartbeat" };
            return { ...base, infer: false, reason: "unchanged_busy_wait" };
        }
        // Non-wait/combat decisions keep the base cadence. Event wakes bypass
        // it only while a prior model Wait authorizes the suppression state.
        return {
            ...base,
            infer: elapsed >= this.decisionInterval,
            reason:
                elapsed >= this.decisionInterval
                    ? "base_cadence"
                    : "between_decisions",
        };
    }

    afterDecision({ tick, prediction, selected }) {
        if (!Number.isSafeInteger(tick) || tick < 0)
            throw new Error("Decision tick must be nonnegative integer");
        this.lastDecisionTick = tick;
        const path = prediction?.selectionPath,
            actions = selected?.actions,
            dropped = selected?.dropped;
        const root = Array.isArray(path) && path.length === 1 ? path[0] : null;
        // Root-Wait alone is not a sleep authorization during combat or while
        // useful actions are offered. Otherwise ordinary combat changes could
        // turn every 3-tick observation into an expensive event wake.
        this.waiting = !!(
            this.last?.tick === tick &&
            this.last.skipEligible === true &&
            root?.stageIndex === 1 &&
            root.group === "next_action" &&
            root.dimension === "intent" &&
            root.cancelled === true &&
            root.complete === true &&
            root.binding === null &&
            prediction?.resolvedAnswers?.next_action?.choice ===
                this.last.waitKey &&
            Array.isArray(actions) &&
            actions.length === 0 &&
            Array.isArray(dropped) &&
            dropped.length === 0
        );
        return {
            waiting: this.waiting,
            lastDecisionTick: this.lastDecisionTick,
        };
    }

    /** Call only for receipts of submitted commands; [] is a valid no-op. */
    receipts({ results }) {
        if (
            !Array.isArray(results) ||
            results.some((r) => r?.accepted !== true)
        ) {
            this.pendingReceiptFailure = true;
            this.waiting = false;
        }
    }
}
