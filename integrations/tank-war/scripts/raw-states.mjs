// Re-render, replay and label recorded rounds from their raw states (states.jsonl, written by match.mjs).
//
// node scripts/raw-states.mjs render < items.jsonl > requests.jsonl
//     One process renders a batch. Each input line is {"observed": <a decision line of states.jsonl>, "harness":
//     [{path, kind, content}], "motion": <optional motion id>}. Each output line is {"requests": [...]} or
//     {"error": "..."}. The requests are those decide() would send, as {id, parts, questions}, and parts is the
//     contextParts. The weapon request is about "motion", by default the motion the logged weapon question was
//     asked about. A line whose saved state no longer hashes to its state_sha256 gives an error.
// node scripts/raw-states.mjs roundtrip RUN...
//     Re-renders every decision under its logged harness and checks each row's contextParts and questions byte
//     for byte.
// node scripts/raw-states.mjs replay RUN [--to=K]
//     Plays the round again from its seed with the logged answers and choices fed back, and checks every
//     decision's state hash and requests against the log, up to decision K. Without --to it plays to the end and
//     also checks events.jsonl and result.json. Replay is bit exact on the platform that recorded the round. On
//     another OS, CPU or Node version the engine's floating point can differ in the last bits, and a mismatch
//     then names both platforms.
// node scripts/raw-states.mjs label RUN... [--out=DIR]
//     Writes labels.jsonl next to decisions.jsonl: the planner teacher's answer to every question of every row.
//     With --out it writes DIR/<the run's directory name>/labels.jsonl instead, so a frozen dataset's runs stay
//     unchanged.
//
// Each run command first checks the run's files against the SHA-256 its manifest records when the round ends, then
// every saved state against its logged state_sha256. Each prints one JSON line per run. Any mismatch exits 1 and
// names the first decision it found.
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { pathToFileURL } from "node:url";
import { adapter } from "../web/game/adapter.js";
import { restoreGame } from "../web/game/snapshot.js";
import { plannerTeacher, TEACHER_MODEL } from "../web/game/teacher.js";
import { decide, decisionRequests, validateRequest } from "../model-bridge.mjs";
import {
    harnessFrom,
    playRound,
    roundResult,
    stateHash,
    OUTPUTS,
    PLATFORM,
    SEATS,
    STATES_FILE,
    RAW_STATES_FORMAT,
    RAW_STATES_VERSION,
} from "./match.mjs";

export const LABELS_FILE = "labels.jsonl",
    LABELS_FORMAT = "tank-teacher-labels",
    LABELS_VERSION = 1;
const mazes = new Map();
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export class Mismatch extends Error {
    constructor(decision, message) {
        super(`decision ${decision}: ${message}`);
        this.decision = decision;
    }
}

// A decision line's raw state, refused when it no longer hashes to the state_sha256 logged with it. The hash is
// of the saved JSON, so it does not depend on the platform.
function savedState(observed) {
    const hash = stateHash(observed.game);
    if (hash !== observed.state_sha256)
        throw new Mismatch(
            observed.decision,
            `the saved state hashes to ${hash}, not to its logged state_sha256 ${observed.state_sha256}`,
        );
    return observed.game;
}

// The seat's observation of a logged state under a harness (from harnessFrom),
// and the request body the match builds from it. The harness's assist and shot
// guard settings replace the logged ones. The rest of the state is as logged.
export function observe(observed, harness) {
    const { seat } = harness,
        game = restoreGame(observed.game, mazes);
    Object.assign(game.playerSettings[observed.player], {
        assistEnabled: seat.assistEnabled,
        assistProfile: seat.assistProfile,
        fireGuardEnabled: seat.fireGuardEnabled,
    });
    const observation = adapter.observe(game, observed.player, {
        ...seat,
        interval: observed.interval_ms,
        rtt: observed.rtt_ms,
    });
    const body = {
        ...observation.request,
        id: observed.request_id,
        player: observed.player,
        round_id: game.roundSerial,
        map_revision: game.maze.revision,
    };
    validateRequest(body, "tank-war");
    return { observation, body };
}

// render(observed, harness) -> the requests, each with its contextParts. `harness` is the harness's files,
// [{path, kind, content}], as states.jsonl's header holds them.
export function render(observed, files, { motion = observed.motion } = {}) {
    savedState(observed);
    const harness = harnessFrom(files);
    const { body } = observe(observed, harness);
    return decisionRequests(body, {
        questions: harness.questions,
        notes: harness.notes,
        motion: motion ?? undefined,
    });
}

const lines = async (file) =>
    (await readFile(file, "utf8"))
        .split("\n")
        .filter(Boolean)
        .map((l) => JSON.parse(l));

// A recorded round: its manifest, harness, raw states, and decision rows grouped by decision. Every file the
// manifest hashes must still match, and every saved state must still hash to its logged state_sha256.
export async function readRun(run) {
    const manifest = JSON.parse(
        await readFile(join(run, "manifest.json"), "utf8"),
    );
    for (const name of OUTPUTS) {
        const recorded = manifest.outputs?.[name];
        if (typeof recorded !== "string")
            throw Error(
                `${run}/manifest.json records no SHA-256 of ${name}. The round did not finish, or an older match wrote it`,
            );
        const hash = createHash("sha256")
            .update(await readFile(join(run, name)))
            .digest("hex");
        if (hash !== recorded)
            throw Error(
                `${run}/${name} does not match the SHA-256 its manifest records. The run was changed`,
            );
    }
    const [header, ...states] = await lines(join(run, STATES_FILE));
    if (header?.format !== RAW_STATES_FORMAT)
        throw Error(`${run}/${STATES_FILE} has no raw-states header`);
    if (header.version !== RAW_STATES_VERSION)
        throw Error(
            `${run}/${STATES_FILE} is version ${header.version}, and this tool reads version ${RAW_STATES_VERSION}`,
        );
    const harness = harnessFrom(header.harness);
    if (!same(harness.files, manifest.harness))
        throw Error(
            `${run}: the harness in ${STATES_FILE} is not the one its manifest records`,
        );
    const rows = await lines(join(run, "decisions.jsonl")),
        calls = states.map(() => []);
    states.forEach((s, k) => {
        if (s.decision !== k)
            throw Error(
                `${run}: state line ${k + 1} is decision ${s.decision}`,
            );
        savedState(s);
    });
    rows.forEach((row, index) => {
        if (!calls[row.decision])
            throw Error(
                `${run}: row ${index} names decision ${row.decision}, which has no state`,
            );
        calls[row.decision].push({ ...row, index });
    });
    return { manifest, header, harness, states, calls, rows };
}

const platformName = ({ os, arch, node }) => `${os}-${arch} node ${node}`;

// What to add to a mismatch when the round was recorded on another platform.
export function platformNote(header) {
    if (!header.platform)
        return "The recording names no platform, and floating point may differ in the last bits between platforms.";
    if (same(header.platform, PLATFORM)) return null;
    return `Recorded on ${platformName(header.platform)}, replayed on ${platformName(PLATFORM)}: floating point may differ in the last bits.`;
}

// `check` run on a read run, with the platform note added to any mismatch it reports.
async function checked(run, check) {
    const recorded = await readRun(run);
    try {
        return await check(recorded);
    } catch (error) {
        const note = error instanceof Mismatch && platformNote(recorded.header);
        if (note) error.message += `. ${note}`;
        throw error;
    }
}

// Every decision re-rendered under the logged harness, checked against its rows.
export const roundTrip = (run) =>
    checked(run, (recorded) => roundTripRun(run, recorded));

async function roundTripRun(run, { header, states, calls, rows }) {
    for (const [k, observed] of states.entries()) {
        const requests = render(observed, header.harness);
        if (requests.length !== calls[k].length)
            throw new Mismatch(
                k,
                `${requests.length} requests rendered, ${calls[k].length} logged`,
            );
        requests.forEach((r, i) => {
            const row = calls[k][i];
            if (r.id !== row.request_id)
                throw new Mismatch(
                    k,
                    `request ${r.id} is not ${row.request_id}`,
                );
            if (!same(r.parts, row.contextParts))
                throw new Mismatch(k, `the contextParts of ${r.id} differ`);
            if (!same(r.questions, row.questions))
                throw new Mismatch(k, `the questions of ${r.id} differ`);
        });
    }
    return { run, decisions: states.length, rows: rows.length };
}

// The first path at which two encoded states differ, for a divergence report.
function difference(a, b, path = "state") {
    if (same(a, b)) return null;
    if (a && b && typeof a === "object" && typeof b === "object")
        for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
            const d = difference(a[k], b[k], `${path}.${k}`);
            if (d) return d;
        }
    return `${path} is ${JSON.stringify(a)}, logged ${JSON.stringify(b)}`;
}

// The round played again from its seed with the logged answers and choices fed back. Checks every decision up to
// `to`, or all of them and then the round's events and result.
export const replay = (run, { to } = {}) =>
    checked(run, (recorded) => replayRun(run, recorded, to));

async function replayRun(run, { manifest, harness, states, calls }, to) {
    const last = to ?? states.length - 1;
    if (!Number.isInteger(last) || last < 0 || last >= states.length)
        throw Error(
            `--to must be a decision of the run, 0 to ${states.length - 1}`,
        );
    const stop = Symbol("stop");
    const agent = async ({ body, state, decision }) => {
        const logged = states[decision];
        if (!logged)
            throw new Mismatch(
                decision,
                "the replay asks a decision the log does not have",
            );
        const hash = stateHash(state);
        if (hash !== logged.state_sha256)
            throw new Mismatch(
                decision,
                `state hash ${hash} is not the logged ${logged.state_sha256}: ${difference(state, logged.game) ?? "the state is as logged, but its hash is not"}`,
            );
        const rows = calls[decision];
        let i = 0;
        const response = await decide(
            {
                model: "replay",
                agentIds: ["agent", "agent"],
                contextLimit: null,
                async ask(sent) {
                    const row = rows[i++];
                    if (!row)
                        throw new Mismatch(
                            decision,
                            `request ${sent.id} was not logged`,
                        );
                    if (sent.id !== row.request_id)
                        throw new Mismatch(
                            decision,
                            `request ${sent.id} is logged as ${row.request_id}`,
                        );
                    if (!same(sent.parts, row.contextParts))
                        throw new Mismatch(
                            decision,
                            `the context of ${sent.id} differs from the log`,
                        );
                    if (!same(sent.questions, row.questions))
                        throw new Mismatch(
                            decision,
                            `the questions of ${sent.id} differ from the log`,
                        );
                    return {
                        id: sent.id,
                        answers: row.answers,
                        stateText: sent.parts.essential[0],
                        trimmed: [],
                        tokenAudit: {},
                    };
                },
            },
            body,
            {
                questions: harness.questions,
                notes: harness.notes,
                choose: (head) =>
                    rows.find((r) => Object.hasOwn(r.chosen, head)).chosen[
                        head
                    ],
            },
        );
        if (i !== rows.length)
            throw new Mismatch(
                decision,
                `${i} requests replayed, ${rows.length} logged`,
            );
        if (decision === to) throw stop;
        return response;
    };
    let round;
    try {
        round = await playRound({
            seed: manifest.seed,
            seat: manifest.seat,
            harness,
            seconds: manifest.seconds,
            intervalMs: manifest.interval_ms,
            opponentIntervalMs: manifest.opponent_interval_ms,
            agent,
            record: async () => {},
        });
    } catch (error) {
        if (error === stop) return { run, checked: last + 1, to: last };
        throw error;
    }
    const { game, decisions } = round,
        played = decisions[SEATS.indexOf(manifest.seat)];
    if (played !== states.length)
        throw new Mismatch(
            played,
            `the replay ended after ${played} decisions, and the log has ${states.length}`,
        );
    const events = game.events.map((e) => JSON.stringify(e) + "\n").join("");
    if (events !== (await readFile(join(run, "events.jsonl"), "utf8")))
        throw new Mismatch(played, "the replayed events.jsonl differs");
    const result = roundResult(game, {
        seed: manifest.seed,
        seat: manifest.seat,
        opponent: manifest.opponent,
        decisions,
    });
    if (
        JSON.stringify(result, null, 2) !==
        (await readFile(join(run, "result.json"), "utf8"))
    )
        throw new Mismatch(played, "the replayed result.json differs");
    return {
        run,
        checked: states.length,
        to: null,
        events: true,
        result: true,
    };
}

// The planner teacher's answer to every question of every row, written to labels.jsonl. On a teacher's own
// round the labels must be its recorded choices, and `recorded` counts the rows compared and those that matched.
export async function label(run, { out } = {}) {
    const { manifest, harness, states, calls, rows } = await readRun(run);
    if (harness.seat.controllerMode !== "plans")
        throw Error(
            `${run} played raw controls, and the planner teacher labels planned steering only`,
        );
    const labelled = [
        {
            format: LABELS_FORMAT,
            version: LABELS_VERSION,
            source: TEACHER_MODEL,
            harness: harness.files,
        },
    ];
    let questions = 0,
        matched = 0;
    for (const [k, observed] of states.entries()) {
        const motion = observed.motion ?? undefined,
            ask = plannerTeacher(observe(observed, harness).observation, {
                motion,
            });
        for (const row of calls[k]) {
            const reply = await ask({
                id: row.request_id,
                parts: row.contextParts,
                questions: row.questions,
            });
            const labels = Object.fromEntries(
                Object.entries(reply.answers).map(([head, a]) => [
                    head,
                    a.choice,
                ]),
            );
            questions += Object.keys(labels).length;
            if (same(labels, row.chosen)) matched++;
            labelled.push({
                row: row.index,
                decision: k,
                request_id: row.request_id,
                labels,
                ...(row.questions.weapon && { motion }),
            });
        }
    }
    const dir = out === undefined ? run : join(out, basename(run));
    await mkdir(dir, { recursive: true });
    await writeFile(
        join(dir, LABELS_FILE),
        labelled.map((l) => JSON.stringify(l) + "\n").join(""),
    );
    const teacher = manifest.kind === "teacher-demonstration";
    if (teacher && matched !== rows.length)
        throw Error(
            `${run}: the teacher's labels match ${matched} of its ${rows.length} recorded rows`,
        );
    return {
        run,
        rows: rows.length,
        questions,
        ...(teacher && { recorded: { compared: rows.length, matched } }),
    };
}

function renderBatch(input) {
    const out = [];
    for (const line of input.split("\n").filter((l) => l.trim())) {
        try {
            const item = JSON.parse(line);
            out.push({
                requests: render(item.observed, item.harness, {
                    motion: item.motion ?? item.observed.motion,
                }),
            });
        } catch (error) {
            out.push({ error: String(error.message || error) });
        }
    }
    return out;
}

async function main([command, ...args]) {
    const runs = args.filter((a) => !a.startsWith("--")),
        to = args.find((a) => a.startsWith("--to=")),
        out = args.find((a) => a.startsWith("--out="));
    if (command === "render") {
        let input = "";
        for await (const chunk of process.stdin) input += chunk;
        for (const line of renderBatch(input))
            process.stdout.write(JSON.stringify(line) + "\n");
        return;
    }
    const tools = { roundtrip: roundTrip, replay, label };
    if (!tools[command] || !runs.length)
        throw Error(
            "usage: raw-states.mjs render | roundtrip RUN... | replay RUN [--to=K] | label RUN... [--out=DIR]",
        );
    if (to && (command !== "replay" || runs.length !== 1))
        throw Error("--to applies to replay of one run");
    if (out && command !== "label") throw Error("--out applies to label");
    // Two runs with one directory name would write the same DIR/<name>/labels.jsonl.
    if (out && new Set(runs.map((r) => basename(r))).size !== runs.length)
        throw Error("--out needs runs with different directory names");
    const options = to
        ? { to: Number(to.slice(5)) }
        : out
          ? { out: out.slice(6) }
          : undefined;
    for (const run of runs)
        console.log(JSON.stringify(await tools[command](run, options)));
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href)
    main(process.argv.slice(2)).catch((error) => {
        console.error(error.message || error);
        process.exitCode = 1;
    });
