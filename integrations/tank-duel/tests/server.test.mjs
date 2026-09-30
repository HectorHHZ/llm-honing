import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { once } from "node:events";
import { createGameServer, configuration } from "../server.mjs";
import { adapter as classic } from "../web/classic/adapter.js";
import { adapter as auto } from "../web/auto-aim/adapter.js";
import { validHealth, validDecision } from "../web/shared/model-client.js";

async function serve(t, server) {
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    t.after(
        () =>
            new Promise((resolve) => {
                server.close(resolve);
                server.closeAllConnections();
            }),
    );
    return `http://127.0.0.1:${server.address().port}`;
}
function request(adapter, mode = "plans") {
    const game = new adapter.Game();
    adapter.initialize(game);
    game.bullets = Array.from({ length: 10 }, (_, i) => ({
        id: `shell-${i}`,
        owner: i % 2,
        x: 300 + i * 15,
        y: 300,
        vx: 200,
        vy: 100,
        bounces: 1,
        age: 1,
        trail: [],
    }));
    const obs = adapter.observe(game, 0, {
        controllerMode: mode,
        interval: 100,
        rtt: 120,
    });
    return {
        game,
        obs,
        body: {
            ...obs.request,
            id: "contract-test",
            player: 0,
            round_id: game.roundSerial,
        },
    };
}
const post = (url, body, headers = {}) =>
    fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...headers },
        body: typeof body === "string" ? body : JSON.stringify(body),
    });
async function fixture(t, { mutate = () => {}, delay = 0 } = {}) {
    const calls = [];
    const url = await serve(
        t,
        http.createServer(async (req, res) => {
            res.setHeader("Content-Type", "application/json");
            if (req.url === "/v1/health")
                return res.end(
                    JSON.stringify({
                        ready: true,
                        metadata: {
                            model: "test-model",
                            family: "fixture",
                            maxLen: 8192,
                        },
                    }),
                );
            assert.equal(req.url, "/v1/decide");
            const chunks = [];
            for await (const c of req) chunks.push(c);
            const body = JSON.parse(Buffer.concat(chunks));
            calls.push(body);
            const answers = Object.fromEntries(
                Object.entries(body.questions).map(([head, q]) => {
                    const keys = Object.keys(q.criteria),
                        choice = keys.at(-1);
                    return [
                        head,
                        {
                            choice,
                            probabilities: Object.fromEntries(
                                keys.map((k) => [k, k === choice ? 1 : 0]),
                            ),
                        },
                    ];
                }),
            );
            const reply = {
                id: body.id,
                answers,
                stateText: body.parts.essential.join("\n"),
                trimmed: [],
                tokenAudit: Object.fromEntries(
                    Object.keys(answers).map((k) => [
                        k,
                        { inputTokens: 123, stateTruncated: false },
                    ]),
                ),
            };
            mutate(reply, body);
            if (delay)
                await new Promise((resolve) => setTimeout(resolve, delay));
            res.end(JSON.stringify(reply));
        }),
    );
    return { url, calls };
}
test("both games and shared assets work without any model configuration", async (t) => {
    const url = await serve(t, createGameServer(configuration({})));
    for (const route of [
        "/",
        "/classic/",
        "/auto-aim/",
        "/shared/style.css",
        "/shared/game-client.js",
        "/classic/adapter.js",
        "/auto-aim/adapter.js",
    ])
        assert.equal((await fetch(url + route)).status, 200, route);
    const config = await (await fetch(url + "/api/config")).json();
    assert.deepEqual(config, { editions: { classic: [], "auto-aim": [] } });
    assert.equal((await fetch(url + "/api/classic/health")).status, 503);
    assert.equal((await fetch(url + "/%2e%2e%2fserver.mjs")).status, 403);
    assert.equal((await fetch(url + "/%ZZ")).status, 400);
    assert.equal((await fetch(url + "/.env")).status, 404);
});
for (const [adapter, mode] of [
    [classic, "plans"],
    [classic, "raw"],
    [auto, "raw"],
])
    test(`${adapter.edition}/${mode}: full engine context, HTTP decisions and native distributions reach both seats`, async (t) => {
        const model = await fixture(t),
            config = configuration({
                HONING_URL: model.url,
                MODEL_LABEL: "Fixture",
                HONING_TOKEN: "secret-only-on-server",
            }),
            url = await serve(t, createGameServer(config)),
            base = url + `/api/${adapter.edition}`;
        const publicConfig = await (await fetch(url + "/api/config")).text();
        assert.ok(!publicConfig.includes(model.url));
        assert.ok(!publicConfig.includes("secret"));
        const health = await (await fetch(base + "/health")).json();
        assert.ok(validHealth(health, adapter.protocols));
        assert.equal(health.metadata.independent_instances, false);
        const { game, obs, body } = request(adapter, mode);
        game.start();
        for (const player of [0, 1]) {
            body.player = player;
            const response = await post(base + "/decision", body);
            assert.equal(response.status, 200);
            const r = await response.json();
            assert.ok(validDecision(r, body, health));
            assert.ok(adapter.validControls(r, obs));
            assert.equal(r.state_truncated, false);
            const applied = adapter.apply(game, player, r, {
                controllerMode: mode,
                interval: 100,
            });
            assert.equal(typeof applied.controls.fire, "boolean");
        }
        assert.equal(model.calls.length, mode === "plans" ? 4 : 2);
        for (const call of model.calls) {
            assert.deepEqual(Object.keys(call.parts), [
                "essential",
                "details",
                "memory",
                "logs",
            ]);
            assert.deepEqual(call.parts.details, []);
            for (let i = 0; i < 10; i++)
                assert.ok(call.parts.essential[0].includes(`shell-${i}`));
        }
        if (mode === "plans")
            assert.match(model.calls[1].parts.essential[0], /SELECTED MOTION/);
    });
test("a conditional weapon choice is vetoed when the selected forecast self-hits", async (t) => {
    const model = await fixture(t, {
        mutate: (r) => {
            if (r.answers.weapon)
                r.answers.weapon = {
                    choice: "fire",
                    probabilities: { fire: 1, hold: 0 },
                };
        },
    });
    const url = await serve(
            t,
            createGameServer(configuration({ HONING_URL: model.url })),
        ),
        { body } = request(classic);
    for (const m of body.motions) m.shot.self_hit_ms = 100;
    const response = await post(url + "/api/classic/decision", body),
        r = await response.json();
    assert.equal(response.status, 200);
    assert.equal(r.raw_weapon, "fire");
    assert.equal(r.controls.fire, false);
    assert.equal(r.constraints.fire_vetoed, true);
});
for (const [name, mutate] of [
    [
        "missing probability",
        (r) => {
            delete r.answers.movement.probabilities.up;
        },
    ],
    [
        "non-normalized probability",
        (r) => {
            r.answers.weapon.probabilities.hold = 0.2;
        },
    ],
    [
        "wrong choice",
        (r) => {
            r.answers.weapon.choice = "fire";
        },
    ],
    [
        "truncated state",
        (r) => {
            r.stateText = r.stateText.slice(0, -1);
        },
    ],
    [
        "trimmed record",
        (r) => {
            r.trimmed = [{ section: "essential" }];
        },
    ],
    [
        "mismatched request",
        (r) => {
            r.id = "stale";
        },
    ],
])
    test(`reject ${name} without synthesizing a model decision`, async (t) => {
        const model = await fixture(t, { mutate }),
            url = await serve(
                t,
                createGameServer(configuration({ HONING_URL: model.url })),
            ),
            { body } = request(auto);
        assert.equal(
            (await post(url + "/api/auto-aim/decision", body)).status,
            502,
        );
    });
test("invalid controls, origin, protocol and oversized requests never reach the model", async (t) => {
    const model = await fixture(t),
        url = await serve(
            t,
            createGameServer(configuration({ HONING_URL: model.url })),
        ),
        base = url + "/api/auto-aim/decision",
        { body } = request(auto);
    assert.equal(
        (await post(base, body, { Origin: "https://other.example" })).status,
        403,
    );
    assert.equal(
        (await post(base, { ...body, protocol: "controls-v3" })).status,
        400,
    );
    assert.equal((await post(base, { ...body, player: 2 })).status, 400);
    assert.equal(
        (
            await post(base, {
                ...body,
                candidates: {
                    ...body.candidates,
                    movement: [{ id: "teleport", description: "Invalid" }],
                },
            })
        ).status,
        400,
    );
    assert.equal((await post(base, "broken")).status, 400);
    assert.equal((await post(base, "x".repeat(66000))).status, 413);
    assert.equal((await fetch(base)).status, 405);
    assert.equal(model.calls.length, 0);
});
test("one pending request per seat; other seat stays available", async (t) => {
    const model = await fixture(t, { delay: 80 }),
        url = await serve(
            t,
            createGameServer(configuration({ HONING_URL: model.url })),
        ),
        base = url + "/api/auto-aim/decision",
        { body } = request(auto);
    const first = post(base, body);
    while (model.calls.length === 0) await new Promise((r) => setTimeout(r, 5));
    assert.equal((await post(base, body)).status, 429);
    const other = post(base, { ...body, player: 1 });
    assert.equal((await first).status, 200);
    assert.equal((await other).status, 200);
    assert.equal((await post(base, body)).status, 200);
});
test("legacy tank services retain their identity and wire response", async (t) => {
    const { body } = request(auto),
        reply = {
            id: body.id,
            round_id: body.round_id,
            protocol: body.protocol,
            model: "legacy",
            agent_id: "legacy-a",
            state_truncated: false,
            controls: { move: "stay", fire: false },
            probabilities: {},
        };
    const model = await serve(
        t,
        http.createServer(async (req, res) => {
            for await (const _ of req) {
            }
            res.setHeader("Content-Type", "application/json");
            res.end(
                JSON.stringify(
                    req.url === "/health"
                        ? {
                              ready: true,
                              metadata: {
                                  model: "legacy",
                                  protocol: "controls-v2",
                                  instances: ["legacy-a", "legacy-b"],
                              },
                          }
                        : reply,
                ),
            );
        }),
    );
    const url = await serve(
            t,
            createGameServer(configuration({ AUTOAIM_MODEL_URL: model })),
        ),
        base = url + "/api/auto-aim";
    const health = await (await fetch(base + "/health")).json();
    assert.ok(validHealth(health, auto.protocols));
    assert.deepEqual(
        await (await post(base + "/decision", body)).json(),
        reply,
    );
});

test("mixed endpoints route each seat and protocol independently; registry never exposes credentials", async (t) => {
    const blue = await fixture(t),
        red = await fixture(t);
    const config = configuration({
        MODELS_JSON: JSON.stringify([
            {
                id: "gemma",
                label: "DiffusionGemma",
                version: "current",
                url: blue.url,
                editions: ["classic"],
                token: "blue-secret",
            },
            {
                id: "nano-v1",
                label: "NanoJev",
                version: "checkpoint-1",
                url: red.url,
                editions: ["classic"],
                token: "red-secret",
            },
        ]),
    });
    const base = await serve(t, createGameServer(config));
    const publicConfig = await (await fetch(base + "/api/config")).json();
    assert.deepEqual(publicConfig.editions.classic, [
        { id: "gemma", label: "DiffusionGemma", version: "current" },
        { id: "nano-v1", label: "NanoJev", version: "checkpoint-1" },
    ]);
    assert.ok(!JSON.stringify(publicConfig).includes("secret"));
    assert.ok(!JSON.stringify(publicConfig).includes(blue.url));
    const a = request(classic, "plans"),
        b = request(classic, "raw");
    b.body.player = 1;
    b.body.id = "red-test";
    const responses = await Promise.all([
        post(base + "/api/classic/decision?backend=gemma", a.body),
        post(base + "/api/classic/decision?backend=nano-v1", b.body),
    ]);
    assert.ok(responses.every((r) => r.status === 200));
    const [ar, br] = await Promise.all(responses.map((r) => r.json()));
    assert.equal(ar.protocol, "plans-v1");
    assert.equal(br.protocol, "controls-v3");
    assert.ok(classic.validControls(ar, a.obs));
    assert.ok(classic.validControls(br, b.obs));
    assert.equal(blue.calls.length, 2);
    assert.equal(red.calls.length, 1);
    assert.deepEqual(Object.keys(blue.calls[0].questions), ["motion"]);
    assert.deepEqual(Object.keys(red.calls[0].questions), [
        "drive",
        "turn",
        "weapon",
    ]);
    assert.ok(blue.calls.every((c) => c.id.startsWith(a.body.id)));
    assert.ok(red.calls.every((c) => c.id.startsWith(b.body.id)));
});
test("model registry rejects duplicate IDs, unsupported transports and unknown editions", () => {
    const entry = {
        id: "model",
        url: "http://localhost:1234",
        label: "Model",
        editions: ["classic"],
    };
    for (const list of [
        [entry, entry],
        [{ ...entry, transport: "unknown" }],
        [{ ...entry, editions: ["unknown"] }],
        [{ ...entry, url: "file:///tmp/model" }],
    ])
        assert.throws(() =>
            configuration({ MODELS_JSON: JSON.stringify(list) }),
        );
});
