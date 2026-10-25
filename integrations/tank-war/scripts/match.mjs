// Play one Tank War round between an agent and the planner opponent, headless and in lockstep, and record it.
//
// node scripts/match.mjs --seed=2001 --seat=blue --decision-server=http://127.0.0.1:8080 --out=runs/m1 \
//     [--harness-dir=DIR] [--seconds=30] [--interval-ms=200] [--opponent=planner] [--opponent-interval-ms=300] \
//     [--sample-seed=N] [--teacher]
//
// The game clock stops while a seat waits for its decision, so the result depends on the model's answers and
// never on inference latency: the same seed, harness, and answers replay the same round. The agent's seat is
// asked every --interval-ms of game time; the opponent every --opponent-interval-ms.
//
// --harness-dir holds the Honing harness: index.json lists every file as {path, kind}. tank/controller.json
// (settings) and tank/questions.json (instructions) configure the controller and the question text; every
// other file is shown to the model before the state as an operating note. Without it, config/ is used.
// --sample-seed draws each policy choice from the model's probabilities (an on-policy rollout) instead of
// taking the most probable one. --teacher plays the agent's seat with the planner teacher and needs no server.
//
// --out (which must not exist) receives manifest.json, decisions.jsonl (one row per question set the agent was
// asked, with the state, questions, answers, and executed choice), events.jsonl, and result.json; a run that
// stops on an error writes error.json and exits 1.
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile, appendFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { adapter } from "../web/game/adapter.js";
import { rng, blocked } from "../web/game/engine.js";
import { strategyFrom } from "../web/game/strategies.js";
import { plannerTeacher, TEACHER_MODEL } from "../web/game/teacher.js";
import {
    decide,
    healthFor,
    upstream,
    validateRequest,
    validateQuestions,
    DEFAULT_QUESTIONS,
} from "../model-bridge.mjs";

export const CONTROLLER_PATH = "tank/controller.json";
export const QUESTIONS_PATH = "tank/questions.json";
export const SEATS = ["blue", "red"];
export const OPPONENTS = ["planner"];
const ROOT = new URL("../", import.meta.url);
const STEP = 1 / 60;
const CONTROLLER_KEYS = {
    controllerMode: (v) => ["plans", "raw"].includes(v),
    contextVersion: (v) => ["full", "arena-v1"].includes(v),
    assistEnabled: (v) => typeof v === "boolean",
    assistProfile: (v) => ["light", "standard", "strong"].includes(v),
    fireGuardEnabled: (v) => typeof v === "boolean",
    strategy: () => true,
};
// What the opponent plays with: the published planned steering and the default dodge assist.
const OPPONENT_SEAT = Object.freeze({
    controllerMode: "plans",
    contextVersion: "full",
    decisionVersion: "baseline",
    assistEnabled: true,
    assistProfile: "light",
    fireGuardEnabled: false,
});
// Events where geometry or dodge assist overrode what the seat asked for.
const INTERVENTIONS = [
    "assist",
    "assist_fire_cancel",
    "plan_revalidate",
    "plan_veto",
    "plan_intent_retained",
];
const sha256 = (text) => createHash("sha256").update(text).digest("hex");

// The controller settings a harness asks for, checked; throws on anything a match would misread.
export function parseController(text) {
    let c;
    try {
        c = JSON.parse(text);
    } catch (e) {
        throw Error(`${CONTROLLER_PATH} is not JSON: ${e.message}`);
    }
    if (!c || typeof c !== "object" || Array.isArray(c))
        throw Error(`${CONTROLLER_PATH} must be a JSON object`);
    for (const [key, value] of Object.entries(c)) {
        if (!CONTROLLER_KEYS[key])
            throw Error(`${CONTROLLER_PATH} has unknown setting ${key}`);
        if (!CONTROLLER_KEYS[key](value))
            throw Error(
                `${CONTROLLER_PATH} setting ${key} has invalid value ${JSON.stringify(value)}`,
            );
    }
    const seat = {
        controllerMode: c.controllerMode ?? "plans",
        contextVersion: c.contextVersion ?? "full",
        assistEnabled: c.assistEnabled ?? true,
        assistProfile: c.assistProfile ?? "light",
        fireGuardEnabled: c.fireGuardEnabled ?? false,
    };
    if (seat.controllerMode === "plans") {
        if (seat.contextVersion !== "full")
            throw Error("Planned steering requires contextVersion full");
        if (seat.fireGuardEnabled)
            throw Error(
                "The raw shot guard does not apply to planned steering",
            );
        seat.strategy = strategyFrom(c.strategy ?? "baseline");
        seat.decisionVersion = seat.strategy.id;
    } else {
        if (c.strategy !== undefined)
            throw Error("A strategy applies to planned steering only");
        seat.decisionVersion = "raw";
    }
    return seat;
}

// The harness in a directory written by Honing's runner, or the integration's defaults.
export async function loadHarness(dir) {
    const files = dir
        ? JSON.parse(await readFile(join(dir, "index.json"), "utf8"))
        : [
              {
                  path: CONTROLLER_PATH,
                  kind: "settings",
                  source: "config/controller.json",
              },
              {
                  path: QUESTIONS_PATH,
                  kind: "instructions",
                  source: "config/questions.json",
              },
          ];
    if (!Array.isArray(files))
        throw Error("index.json must list the harness files");
    let seat = parseController("{}"),
        questions = DEFAULT_QUESTIONS;
    const notes = [],
        manifest = [];
    for (const file of files) {
        const content = await readFile(
            dir ? join(dir, file.path) : new URL(file.source, ROOT),
            "utf8",
        );
        let role = "note";
        if (file.path === CONTROLLER_PATH) {
            seat = parseController(content);
            role = "controller";
        } else if (file.path === QUESTIONS_PATH) {
            let parsed;
            try {
                parsed = JSON.parse(content);
            } catch (e) {
                throw Error(`${QUESTIONS_PATH} is not JSON: ${e.message}`);
            }
            questions = validateQuestions(parsed);
            role = "questions";
        } else notes.push(`OPERATING NOTES (${file.path}):\n${content.trim()}`);
        manifest.push({
            path: file.path,
            kind: file.kind,
            role,
            sha256: sha256(content),
        });
    }
    return { seat, questions, notes: notes.join("\n\n"), files: manifest };
}

// A choice drawn from the allowed options in proportion to their probabilities.
function sampler(seed) {
    const random = rng(seed);
    return (head, probabilities, allowed) => {
        const total = allowed.reduce((s, k) => s + probabilities[k], 0);
        if (!(total > 0)) return allowed[0];
        let x = random() * total;
        for (const k of allowed) if ((x -= probabilities[k]) < 0) return k;
        return allowed.at(-1);
    };
}

async function sources() {
    const files = [
        "web/game/engine.js",
        "web/game/planner.js",
        "web/game/adapter.js",
        "web/game/strategies.js",
        "web/game/teacher.js",
        "model-bridge.mjs",
        "scripts/match.mjs",
    ];
    return Object.fromEntries(
        await Promise.all(
            files.map(async (f) => [
                f,
                sha256(await readFile(new URL(f, ROOT))),
            ]),
        ),
    );
}

export async function playMatch({
    seed,
    seat,
    out,
    decisionServer,
    harnessDir,
    seconds = 30,
    intervalMs = 200,
    opponent = "planner",
    opponentIntervalMs = 300,
    sampleSeed,
    teacher = false,
    token,
}) {
    if (!Number.isInteger(seed) || seed < 0)
        throw Error("--seed must be a non-negative integer");
    if (!SEATS.includes(seat)) throw Error(`--seat must be one of ${SEATS}`);
    if (!OPPONENTS.includes(opponent))
        throw Error(`--opponent must be one of ${OPPONENTS}`);
    for (const [name, v] of [
        ["--seconds", seconds],
        ["--interval-ms", intervalMs],
        ["--opponent-interval-ms", opponentIntervalMs],
    ])
        if (!(Number.isFinite(v) && v > 0))
            throw Error(`${name} must be positive`);
    if (!teacher && !decisionServer)
        throw Error("--decision-server is required unless --teacher");
    if (existsSync(out)) throw Error(`--out ${out} already exists`);
    await mkdir(out, { recursive: true });
    const harness = await loadHarness(harnessDir).catch((error) => {
        // A harness the match cannot read is the harness's fault, not the run's.
        error.harness = true;
        throw error;
    });
    const me = SEATS.indexOf(seat),
        them = 1 - me;
    let model, ask;
    if (teacher) {
        if (harness.seat.controllerMode !== "plans")
            throw Object.assign(
                Error("The planner teacher plays planned steering only"),
                { harness: true },
            );
        model = { model: TEACHER_MODEL };
    } else {
        const backend = {
            url: decisionServer.replace(/\/$/, ""),
            token,
            transport: "honing",
        };
        model = (
            await healthFor(backend, "tank-war", AbortSignal.timeout(60000))
        ).metadata;
        ask = (request) =>
            upstream(
                backend,
                "/v1/decide",
                request,
                AbortSignal.timeout(300000),
            );
    }
    const kind = teacher
        ? "teacher-demonstration"
        : sampleSeed !== undefined
          ? "rollout"
          : "evaluation";
    await writeFile(
        join(out, "manifest.json"),
        JSON.stringify(
            {
                kind,
                edition: "tank-war",
                seed,
                seat,
                opponent,
                seconds,
                interval_ms: intervalMs,
                opponent_interval_ms: opponentIntervalMs,
                sample_seed: sampleSeed ?? null,
                decision_server: teacher ? null : decisionServer,
                model,
                harness: harness.files,
                agent_seat: { ...harness.seat },
                opponent_seat: OPPONENT_SEAT,
                sources: await sources(),
                node: process.version,
                created_at: new Date().toISOString(),
            },
            null,
            2,
        ),
    );

    const game = new adapter.Game(seed);
    adapter.initialize(game);
    game.timeLeft = seconds;
    const seats = [];
    seats[me] = { ...harness.seat, interval: intervalMs };
    seats[them] = { ...OPPONENT_SEAT, interval: opponentIntervalMs };
    seats.forEach((s, i) =>
        game.setPlayerSettings(i, {
            assistEnabled: s.assistEnabled,
            assistProfile: s.assistProfile,
            fireGuardEnabled: s.fireGuardEnabled,
        }),
    );
    game.start();
    const choose = sampleSeed !== undefined ? sampler(sampleSeed) : undefined;
    const next = [0, 0],
        decisions = [0, 0];
    let serial = 0;

    async function turn(player) {
        const s = seats[player],
            observed = adapter.observe(game, player, { ...s, rtt: 0 }),
            body = {
                ...observed.request,
                id: `${seat}-${seed}-${++serial}`,
                player,
                round_id: game.roundSerial,
                map_revision: game.maze.revision,
            };
        validateRequest(body, "tank-war");
        const agent = player === me;
        const response =
            agent && !teacher
                ? await decide(
                      {
                          model: model.model,
                          agentIds: ["agent", "agent"],
                          contextLimit: null,
                          ask,
                      },
                      body,
                      {
                          questions: harness.questions,
                          notes: harness.notes,
                          choose,
                      },
                  )
                : await decide(
                      {
                          model: TEACHER_MODEL,
                          agentIds: ["planner", "planner"],
                          contextLimit: null,
                          ask: plannerTeacher(observed),
                      },
                      body,
                      agent
                          ? {
                                questions: harness.questions,
                                notes: harness.notes,
                            }
                          : {},
                  );
        if (!adapter.validControls(response, observed))
            throw Error("The decision has invalid controls");
        const applied = adapter.apply(game, player, response, s);
        decisions[player]++;
        if (!agent) return;
        const executed =
            body.protocol === "plans-v1"
                ? {
                      motion: response.policy_motion_id,
                      weapon: response.weapon_choice,
                  }
                : response.chosen;
        const rows = response.honing_results.map((call) => ({
            time: Math.round(game.elapsed * 1000) / 1000,
            round: game.round,
            protocol: body.protocol,
            request_id: call.id,
            contextParts: {
                essential: [call.stateText],
                details: [],
                memory: [],
                logs: [],
            },
            questions: call.questions,
            answers: call.answers,
            chosen: Object.fromEntries(
                Object.keys(call.questions).map((head) => [
                    head,
                    executed[head],
                ]),
            ),
            constraints: response.constraints ?? null,
            applied,
        }));
        await appendFile(
            join(out, "decisions.jsonl"),
            rows.map((r) => JSON.stringify(r) + "\n").join(""),
        );
    }

    const limit = Math.ceil((seconds + 1) / STEP);
    for (let frame = 0; game.phase === "running" && frame < limit; frame++) {
        for (const player of [me, them])
            if (game.elapsed >= next[player]) {
                next[player] = game.elapsed + seats[player].interval / 1000;
                await turn(player);
            }
        game.step(STEP);
        for (const t of game.tanks)
            if (blocked(game.maze, t.x, t.y)) throw Error("Wall penetration");
    }
    if (game.phase === "running")
        throw Error("The round did not end within its time limit");
    const round = game.events.find((e) => e.type === "round"),
        kills = game.events.filter((e) => e.type === "kill"),
        winner = game.result === null ? null : SEATS[game.result];
    const result = {
        seed,
        seat,
        opponent,
        map: game.maze.name,
        map_revision: game.maze.revision,
        winner,
        reason: round.reason,
        outcome: Object.fromEntries(
            SEATS.map((s) => [s, winner === null ? 0 : winner === s ? 1 : -1]),
        ),
        ended: round.reason !== "timeout",
        truncated: round.reason === "timeout",
        sim_seconds: Math.round(game.elapsed * 1000) / 1000,
        decisions: Object.fromEntries(SEATS.map((s, i) => [s, decisions[i]])),
        shots: Object.fromEntries(SEATS.map((s, i) => [s, game.shots[i]])),
        self_hits: kills.filter((k) => k.self).map((k) => SEATS[k.victim]),
        enemy_hits: kills.filter((k) => !k.self).map((k) => SEATS[k.player]),
        interventions: Object.fromEntries(
            SEATS.map((s, i) => [
                s,
                game.events.filter(
                    (e) => e.player === i && INTERVENTIONS.includes(e.type),
                ).length,
            ]),
        ),
    };
    await writeFile(
        join(out, "events.jsonl"),
        game.events.map((e) => JSON.stringify(e) + "\n").join(""),
    );
    await writeFile(join(out, "result.json"), JSON.stringify(result, null, 2));
    return result;
}

function parse(argv) {
    const args = {};
    for (const a of argv) {
        const m = /^--([a-z-]+)(?:=(.*))?$/.exec(a);
        if (!m) throw Error(`Unrecognized argument ${a}`);
        args[m[1]] = m[2] ?? true;
    }
    const number = (k) => (args[k] === undefined ? undefined : Number(args[k]));
    const known = [
        "seed",
        "seat",
        "out",
        "decision-server",
        "harness-dir",
        "seconds",
        "interval-ms",
        "opponent",
        "opponent-interval-ms",
        "sample-seed",
        "teacher",
    ];
    const unknown = Object.keys(args).filter((k) => !known.includes(k));
    if (unknown.length) throw Error(`Unknown option --${unknown[0]}`);
    if (!args.out) throw Error("--out is required");
    return {
        seed: number("seed"),
        seat: args.seat,
        out: resolve(args.out),
        decisionServer: args["decision-server"],
        harnessDir: args["harness-dir"] && resolve(args["harness-dir"]),
        seconds: number("seconds"),
        intervalMs: number("interval-ms"),
        opponent: args.opponent,
        opponentIntervalMs: number("opponent-interval-ms"),
        sampleSeed: number("sample-seed"),
        teacher: args.teacher === true,
        token: process.env.HONING_TOKEN,
    };
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
    let options;
    try {
        options = parse(process.argv.slice(2));
        const defined = Object.fromEntries(
            Object.entries(options).filter(([, v]) => v !== undefined),
        );
        console.log(JSON.stringify(await playMatch(defined)));
    } catch (error) {
        console.error(error.stack || String(error));
        if (options?.out && existsSync(options.out))
            await writeFile(
                join(options.out, "error.json"),
                JSON.stringify(
                    {
                        error: String(error.message || error),
                        kind: error.harness ? "harness" : "match",
                    },
                    null,
                    2,
                ),
            );
        process.exitCode = 1;
    }
}
