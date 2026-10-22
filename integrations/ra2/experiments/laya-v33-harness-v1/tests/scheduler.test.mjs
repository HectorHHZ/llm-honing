import test from "node:test";
import assert from "node:assert/strict";
import { WaitScheduler } from "../harness/scheduler.mjs";

const unit = (extra = {}) => ({
    tag: 100,
    unit_type: 24,
    name: "NACNST",
    alliance: 1,
    health: 1000,
    display_type: 1,
    is_blip: 0,
    x: 20,
    y: 20,
    build_progress: 100,
    can_move: 0,
    is_idle: 1,
    is_powered: 1,
    order_id_0: 0,
    order_id_1: 0,
    ...extra,
});
function scene(tick = 0) {
    return {
        tick,
        observation: {
            view: "normal",
            vectors: {
                minerals: 9000,
                army_count: 0,
                power_cap: 150,
                power_used: 50,
                is_low_power: 0,
                units_truncated: 0,
                production: [
                    {
                        queue: "structures",
                        status: "Active",
                        size: 1,
                        item: 10,
                        progress: 1,
                        available: [10],
                        prices: [600],
                    },
                    {
                        queue: "infantry",
                        status: "Idle",
                        size: 0,
                        item: 0,
                        progress: 0,
                        available: [],
                        prices: [],
                    },
                ],
            },
        },
        frame: { rows: [unit()] },
        packet: {
            groups: [
                {
                    id: "next_action",
                    choices: [
                        {
                            key: "A",
                            label: "Continue production",
                            action: null,
                            cost: 0,
                        },
                        {
                            key: "L1",
                            action: { function: 61, extra: { item: 10 } },
                            label: "Cancel production at 1%",
                        },
                        {
                            key: "L2",
                            action: {
                                function: 62,
                                extra: { queue: "structures" },
                            },
                        },
                        { key: "L3", action: { function: 81 } },
                    ],
                },
            ],
        },
    };
}
const rootWait = (tick = 0) => ({
    tick,
    prediction: {
        selectionPath: [
            {
                stageIndex: 1,
                group: "next_action",
                dimension: "intent",
                cancelled: true,
                complete: true,
                binding: null,
                key: "A",
            },
        ],
        resolvedAnswers: { next_action: { choice: "A" } },
    },
    selected: { actions: [], selected: [], dropped: [] },
});
function waiting(options = {}) {
    const scheduler = new WaitScheduler(options);
    assert.equal(scheduler.evaluate(scene()).reason, "initial_decision");
    assert.equal(scheduler.afterDecision(rootWait()).waiting, true);
    return scheduler;
}
function evolve(tick, edit) {
    const state = scene(tick);
    edit?.(state);
    return state;
}

test("unchanged busy production suppresses repeated waits until bounded heartbeat", () => {
    const scheduler = waiting();
    for (let tick = 3; tick < 150; tick += 3) {
        const result = scheduler.evaluate(
            evolve(tick, (s) => {
                s.observation.vectors.production[0].progress = 1 + tick / 3;
                s.observation.vectors.minerals -= tick;
                s.packet.groups[0].choices[1].label = `Cancel at ${tick}%`;
                s.packet.parts = {
                    essential: [`tick ${tick}, credits ${9000 - tick}`],
                };
            }),
        );
        assert.equal(result.infer, false);
        assert.equal(result.reason, "unchanged_busy_wait");
    }
    const heartbeat = scheduler.evaluate(
        evolve(150, (s) => {
            s.observation.vectors.production[0].progress = 51;
        }),
    );
    assert.equal(heartbeat.infer, true);
    assert.equal(heartbeat.reason, "wait_heartbeat");
});

test("later target cancellation never counts as a root wait", () => {
    const scheduler = new WaitScheduler();
    scheduler.evaluate(scene());
    const decision = rootWait();
    decision.prediction.selectionPath = [
        {
            stageIndex: 1,
            group: "next_action",
            dimension: "intent",
            cancelled: false,
            complete: false,
            binding: { function: 60 },
        },
        {
            stageIndex: 2,
            group: "next_action",
            dimension: "target",
            cancelled: true,
            complete: true,
            binding: null,
        },
    ];
    assert.equal(scheduler.afterDecision(decision).waiting, false);
    assert.equal(scheduler.evaluate(scene(3)).reason, "between_decisions");
    assert.equal(scheduler.evaluate(scene(30)).reason, "base_cadence");
});

test("a filtered or malformed selection cannot authorize wait suppression", () => {
    for (const mutate of [
        (d) => {
            d.selected.dropped.push({ reason: "shared_budget" });
        },
        (d) => {
            d.selected.actions.push({ function: 60 });
        },
        (d) => {
            d.prediction.resolvedAnswers.next_action.choice = "L1";
        },
        (d) => {
            delete d.prediction.selectionPath;
        },
        (d) => {
            d.prediction.selectionPath[0].dimension = "target";
        },
    ]) {
        const scheduler = new WaitScheduler();
        scheduler.evaluate(scene());
        const decision = rootWait();
        mutate(decision);
        assert.equal(scheduler.afterDecision(decision).waiting, false);
        assert.equal(scheduler.evaluate(scene(30)).infer, true);
    }
});

test("queue completion, pause, item, size and queue deletion each wake immediately", () => {
    for (const mutate of [
        (q) => {
            q.status = "Ready";
            q.progress = 100;
        },
        (q) => {
            q.status = "OnHold";
        },
        (q) => {
            q.item = 11;
        },
        (q) => {
            q.size = 2;
        },
    ]) {
        const scheduler = waiting();
        const result = scheduler.evaluate(
            evolve(3, (s) => mutate(s.observation.vectors.production[0])),
        );
        assert.equal(result.infer, true);
        assert.ok(result.events.includes("production_changed"));
    }
    const scheduler = waiting();
    assert.equal(
        scheduler.evaluate(
            evolve(3, (s) => s.observation.vectors.production.pop()),
        ).reason,
        "event_wake",
    );
});

test("placement animation is busy; intermediate percentages do not wake, completion does", () => {
    const scheduler = new WaitScheduler();
    const state = evolve(0, (s) => {
        s.observation.vectors.production[0] = {
            queue: "structures",
            status: "Idle",
            size: 0,
            item: 0,
            progress: 0,
        };
        s.frame.rows[0].build_progress = 0;
    });
    scheduler.evaluate(state);
    scheduler.afterDecision(rootWait());
    state.tick = 3;
    state.frame.rows[0].build_progress = 20;
    assert.equal(scheduler.evaluate(state).reason, "unchanged_busy_wait");
    state.tick = 6;
    state.frame.rows[0].build_progress = 100;
    const result = scheduler.evaluate(state);
    assert.equal(result.reason, "event_wake");
    assert.ok(result.events.includes("own_units_changed"));
});

test("own loss, damage, idle/order change and power outage wake without waiting for cadence", () => {
    for (const mutate of [
        (s) => {
            s.frame.rows = [];
        },
        (s) => {
            s.frame.rows[0].health -= 25;
        },
        (s) => {
            s.frame.rows[0].is_idle = 0;
        },
        (s) => {
            s.frame.rows[0].order_id_0 = 28;
        },
        (s) => {
            s.observation.vectors.power_used = 200;
        },
        (s) => {
            s.observation.vectors.army_count = 1;
        },
    ])
        assert.equal(
            waiting().evaluate(evolve(3, mutate)).reason,
            "event_wake",
        );
});

test("new visible enemy wakes; visible combat prevents wait suppression", () => {
    const scheduler = waiting();
    const state = evolve(3, (s) =>
        s.frame.rows.push(
            unit({ tag: 200, alliance: 4, can_move: 1, name: "HTNK" }),
        ),
    );
    const wake = scheduler.evaluate(state);
    assert.equal(wake.reason, "event_wake");
    assert.ok(wake.events.includes("visible_enemy_changed"));
    scheduler.afterDecision(rootWait(3));
    state.tick = 33;
    const next = scheduler.evaluate(state);
    assert.equal(next.infer, true);
    assert.ok(next.blockers.includes("visible_enemy"));
});

test("an existing army never receives extended waiting intervals", () => {
    const scheduler = new WaitScheduler();
    const state = evolve(0, (s) => {
        s.observation.vectors.army_count = 4;
    });
    scheduler.evaluate(state);
    scheduler.afterDecision(rootWait());
    state.tick = 30;
    const result = scheduler.evaluate(state);
    assert.equal(result.reason, "base_cadence");
    assert.ok(result.blockers.includes("army_present"));
});

test("new recruitment, placement, resume, scout, repair and economic choices wake", () => {
    for (const fn of [2024, 60, 63, 22, 50, 46, 1012]) {
        const scheduler = waiting();
        const state = evolve(3, (s) =>
            s.packet.groups[0].choices.push({
                key: "L9",
                action: { function: fn },
            }),
        );
        const result = scheduler.evaluate(state);
        assert.equal(result.reason, "event_wake");
        assert.ok(result.events.includes("new_action_opportunity"));
        scheduler.afterDecision(rootWait(3));
        state.tick = 33;
        assert.equal(scheduler.evaluate(state).reason, "base_cadence");
    }
});

test("new actor/target of an existing action kind is a meaningful opportunity", () => {
    const scheduler = new WaitScheduler();
    const state = scene();
    state.packet.groups[0].choices.push({
        key: "L9",
        action: { function: 50, unit_tags: [100] },
    });
    scheduler.evaluate(state);
    scheduler.afterDecision(rootWait());
    state.tick = 3;
    state.packet.groups[0].choices.push({
        key: "L10",
        action: { function: 50, unit_tags: [101] },
    });
    assert.ok(
        scheduler.evaluate(state).events.includes("new_action_opportunity"),
    );
});

test("plain progress and credits do not wake, but a progress plateau and resumption do", () => {
    const scheduler = waiting({ maxWaitTicks: 300, stallTicks: 150 });
    assert.equal(scheduler.evaluate(scene(147)).infer, false);
    const stalled = scheduler.evaluate(scene(150));
    assert.equal(stalled.reason, "event_wake");
    assert.ok(stalled.events.includes("production_stalled"));
    scheduler.afterDecision(rootWait(150));
    assert.equal(scheduler.evaluate(scene(153)).reason, "between_decisions");
    const resumed = scheduler.evaluate(
        evolve(156, (s) => {
            s.observation.vectors.production[0].progress = 2;
        }),
    );
    assert.equal(resumed.reason, "between_decisions");
    assert.ok(resumed.events.includes("production_resumed"));
});

test("root Wait during combat or useful opportunities does not authorize d3 event wakes", () => {
    for (const prepare of [
        (s) => {
            s.observation.vectors.army_count = 3;
        },
        (s) => {
            s.frame.rows.push(unit({ tag: 200, alliance: 4, can_move: 1 }));
        },
        (s) => {
            s.packet.groups[0].choices.push({
                key: "L9",
                action: { function: 2024 },
            });
        },
    ]) {
        const scheduler = new WaitScheduler();
        const state = scene();
        prepare(state);
        scheduler.evaluate(state);
        assert.equal(scheduler.afterDecision(rootWait()).waiting, false);
        state.tick = 3;
        state.frame.rows[0].health -= 1;
        if (state.frame.rows[1]) state.frame.rows[1].x += 1;
        const intermediate = scheduler.evaluate(state);
        assert.equal(intermediate.infer, false);
        assert.equal(intermediate.reason, "between_decisions");
        state.tick = 30;
        assert.equal(scheduler.evaluate(state).reason, "base_cadence");
    }
});

test("hidden, remembered and blip-only enemies cannot influence scheduling", () => {
    const plain = waiting(),
        injected = waiting();
    const first = scene(3),
        second = scene(3);
    second.frame.rows.push(
        unit({ tag: 999, alliance: 4, display_type: 2, health: 1 }),
        unit({ tag: 998, alliance: 4, display_type: 1, is_blip: 1 }),
    );
    second.frame.visible = [unit({ alliance: 4, tag: 997 })];
    second.frame.memory = [{ tag: 996, health: 20, x: 99, y: 99 }];
    second.observation.enemySecret = { minerals: 100000, units: 500 };
    second.observation.world = { omniscient: [1, 2, 3] };
    assert.deepEqual(injected.evaluate(second), plain.evaluate(first));
});

test("each seat owns independent waiting and progress state", () => {
    const red = waiting(),
        blue = new WaitScheduler();
    assert.equal(red.evaluate(scene(30)).infer, false);
    const state = evolve(30, (s) => {
        s.frame.rows[0].tag = 987;
    });
    assert.equal(blue.evaluate(state).reason, "initial_decision");
    blue.afterDecision(rootWait(30));
    blue.receipts({ results: [{ accepted: false, reason: "invalid" }] });
    assert.equal(
        blue.evaluate({ ...state, tick: 33 }).reason,
        "receipt_rejected_or_missing",
    );
    assert.equal(red.evaluate(scene(33)).infer, false);
});

test("rejected/missing receipts force the next available inference; empty receipts do not", () => {
    for (const results of [[{ accepted: false }], [null], undefined]) {
        const scheduler = waiting();
        scheduler.receipts({ results });
        assert.equal(
            scheduler.evaluate(scene(3)).reason,
            "receipt_rejected_or_missing",
        );
    }
    const scheduler = waiting();
    scheduler.receipts({ results: [] });
    assert.equal(scheduler.evaluate(scene(3)).infer, false);
});

test("unknown schema and truncated observations fail open to inference", () => {
    for (const mutate of [
        (s) => {
            delete s.observation.vectors.army_count;
        },
        (s) => {
            s.observation.vectors.units_truncated = 1;
        },
        (s) => {
            s.observation.vectors.production[0].status = "Unknown";
        },
        (s) => {
            delete s.frame.rows[0].health;
        },
        (s) => {
            s.packet.groups[0].id = "new_protocol";
        },
        (s) => {
            s.observation.view = "privileged";
        },
        (s) => {
            s.frame.rows.push(null);
        },
        (s) => {
            s.observation.vectors.production.push(null);
        },
        (s) => {
            s.packet.groups = [null];
        },
        (s) => {
            s.packet.groups[0].choices.push(null);
        },
    ]) {
        const scheduler = waiting();
        const result = scheduler.evaluate(evolve(3, mutate));
        assert.equal(result.infer, true);
        assert.equal(result.waiting, false);
    }
});

test("missing input also fails open", () => {
    assert.equal(waiting().evaluate().reason, "unknown_schema");
    assert.equal(waiting().evaluate(null).infer, true);
});

test("changed current inputs cannot mutate a saved previous observation", () => {
    const scheduler = new WaitScheduler();
    const state = scene();
    scheduler.evaluate(state);
    scheduler.afterDecision(rootWait());
    state.tick = 3;
    state.frame.rows[0].health = 100;
    assert.ok(scheduler.evaluate(state).events.includes("own_damage"));
});

test("no busy production means ordinary cadence, not indefinite wait", () => {
    const scheduler = new WaitScheduler();
    const state = evolve(0, (s) => {
        s.observation.vectors.production = [];
    });
    scheduler.evaluate(state);
    scheduler.afterDecision(rootWait());
    state.tick = 30;
    assert.equal(scheduler.evaluate(state).reason, "base_cadence");
});

test("no candidates causes no model call and preserves a rejection wake for later", () => {
    const scheduler = waiting();
    scheduler.receipts({ results: [{ accepted: false }] });
    assert.equal(
        scheduler.evaluate(
            evolve(3, (s) => {
                s.packet.groups = [];
            }),
        ).reason,
        "no_candidates",
    );
    assert.equal(
        scheduler.evaluate(scene(6)).reason,
        "receipt_rejected_or_missing",
    );
});

test("invalid cadence options fail fast", () => {
    for (const options of [
        { decisionInterval: 1 },
        { maxWaitTicks: 10 },
        { stallTicks: 0 },
        { decisionInterval: 30, maxWaitTicks: 15 },
    ])
        assert.throws(() => new WaitScheduler(options));
});
