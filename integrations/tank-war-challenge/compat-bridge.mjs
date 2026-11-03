import { readFileSync } from "node:fs";
import {fileURLToPath, pathToFileURL} from 'node:url';
// HONING_RELEASE is the Honing checkout whose integrations/tank-war this bridge follows; by default, this one.
process.env.HONING_RELEASE ||= fileURLToPath(new URL('../..', import.meta.url)).replace(/\/$/, '');
const {validProbabilities}=await import(pathToFileURL(process.env.HONING_RELEASE+'/integrations/tank-war/web/shared/model-client.js'));
const DRIVE = ["forward", "reverse", "stop"],
    TURN = ["left", "right", "none"];
export const protocols = (edition) => ["plans-v1", "controls-v3"];
export class HttpError extends Error {
    constructor(status, message) {
        super(message);
        this.status = status;
    }
}
function requireValue(condition, message) {
    if (!condition) throw new HttpError(400, message);
}
const text = (value) =>
    typeof value === "string" && value.length > 0 && value.length <= 40000;
const argmax = (p, keys) =>
    keys.reduce((a, b) => (p[b] > p[a] ? b : a), keys[0]);
export function validateRequest(body, edition) {
    requireValue(
        body &&
            text(body.id) &&
            [0, 1].includes(body.player) &&
            Number.isInteger(body.round_id) &&
            body.round_id >= 1,
        "Invalid decision identity",
    );
    requireValue(
        protocols(edition).includes(body.protocol) && text(body.state),
        "Invalid protocol or state",
    );
    if (body.protocol === "plans-v1") {
        const expected = DRIVE.flatMap((d) => TURN.map((t) => `${d}_${t}`));
        requireValue(
            Array.isArray(body.motions) &&
                body.motions.length === 9 &&
                new Set(body.motions.map((m) => m?.id)).size === 9,
            "Nine unique motion plans are required",
        );
        requireValue(
            body.motions.every(
                (m) =>
                    expected.includes(m.id) &&
                    m.id === `${m.drive}_${m.turn}` &&
                    text(m.description) &&
                    typeof m.shot?.can_fire === "boolean" &&
                    ["self_hit_ms", "enemy_hit_ms"].every(
                        (k) =>
                            m.shot[k] === null ||
                            (typeof m.shot[k] === "number" &&
                                Number.isFinite(m.shot[k]) &&
                                m.shot[k] >= 0),
                    ) &&
                    Number.isFinite(m.shot.fire_after_ms) &&
                    m.shot.fire_after_ms >= 0,
            ),
            "Invalid motion plan",
        );
        requireValue(
            Array.isArray(body.admissible_motion_ids) &&
                body.admissible_motion_ids.length > 0 &&
                new Set(body.admissible_motion_ids).size ===
                    body.admissible_motion_ids.length &&
                body.admissible_motion_ids.every((k) => expected.includes(k)) &&
                text(body.weapon_state),
            "Invalid execution constraints",
        );
    } else {
        const heads = {
            drive: DRIVE,
            turn: TURN,
            weapon: ["fire", "hold"],
        };
        requireValue(
            body.candidates &&
                Object.keys(body.candidates).length ===
                    Object.keys(heads).length,
            "Invalid control heads",
        );
        for (const [head, allowed] of Object.entries(heads)) {
            const opts = body.candidates[head];
            requireValue(
                Array.isArray(opts) &&
                    opts.length >= 2 &&
                    opts.length <= allowed.length &&
                    new Set(opts.map((o) => o?.id)).size === opts.length &&
                    opts.every(
                        (o) => allowed.includes(o.id) && text(o.description),
                    ),
                "Invalid control candidates",
            );
            requireValue(
                opts.length === allowed.length,
                "Incomplete control candidates",
            );
        }
    }
}
export async function upstream(backend, route, body, signal) {
    const headers = {
        "Content-Type": "application/json",
    };
    if (backend.token) headers.Authorization = `Bearer ${backend.token}`;
    let response;
    try {
        response = await fetch(backend.url + route, {
            method: body ? "POST" : "GET",
            headers,
            body: body ? JSON.stringify(body) : undefined,
            signal,
            redirect: "error",
        });
    } catch {
        throw new HttpError(503, "Model service unavailable");
    }
    if (!response.ok)
        throw new HttpError(
            response.status === 429 ? 429 : 502,
            `Model service returned HTTP ${response.status}`,
        );
    try {
        return await response.json();
    } catch {
        throw new HttpError(502, "Model service returned invalid JSON");
    }
}
export async function healthFor(backend, edition, signal) {
    const h = await upstream(
        backend,
        backend.transport === "tank" ? "/health" : "/v1/health",
        undefined,
        signal,
    );
    if (h?.ready !== true || !h.metadata)
        throw new HttpError(503, "Model is not ready");
    if (backend.transport === "tank") return h;
    const m = h.metadata;
    if (typeof m.model !== "string" || !m.model)
        throw new HttpError(502, "Model metadata is missing its identity");
    // Seats have separate requests/context. This does not claim separate GPU weights.
    return {
        ready: true,
        metadata: {
            model: m.model,
            model_label: backend.label,
            instances: ["tank-war-a", "tank-war-b"],
            independent_instances: false,
            supported_protocols: protocols(edition),
            family: m.family,
            revision: m.revision,
            context_limit: m.maxLen ?? null,
            branches: Number.isInteger(h.maxBranches) ? h.maxBranches : 0,
            transport: "honing",
        },
    };
}
// The question text the model reads. config/questions.json is the default and
// a Honing harness replaces it (tank/questions.json).
export const DEFAULT_QUESTIONS = Object.freeze(
    JSON.parse(
        readFileSync(
            new URL('file://'+process.env.HONING_RELEASE+'/integrations/tank-war/config/questions.json'),
            "utf8",
        ),
    ),
);
export const QUESTION_KEYS = Object.freeze(Object.keys(DEFAULT_QUESTIONS));
export function validateQuestions(questions) {
    if (!questions || typeof questions !== "object" || Array.isArray(questions))
        throw Error("Question text must be a JSON object");
    const missing = QUESTION_KEYS.filter((k) => !text(questions[k])),
        unknown = Object.keys(questions).filter(
            (k) => !QUESTION_KEYS.includes(k),
        );
    if (missing.length)
        throw Error(`Question text is missing ${missing.join(", ")}`);
    if (unknown.length)
        throw Error(`Question text has unknown keys ${unknown.join(", ")}`);
    return questions;
}
const question = (instructions, options) => ({
    type: "choice",
    instructions,
    criteria: Object.fromEntries(options.map((o) => [o.id, o.description])),
});
export async function decideWithHoning(
    backend,
    edition,
    body,
    signal,
    options = {},
) {
    const health = await healthFor(backend, edition, signal);
    return decide(
        {
            model: health.metadata.model,
            agentIds: health.metadata.instances,
            contextLimit: health.metadata.context_limit,
            branches: health.metadata.branches,
            ask: (request) => upstream(backend, "/v1/decide", request, signal),
        },
        body,
        options,
    );
}
// One decision through any model that answers Honing's /v1/decide requests.
// `ask` sends one request. `questions` replaces the default question text,
// `notes` is shown to the model before the state, and `choose(head,
// probabilities, options)` samples the native policy before safety projection.
// Without a sampler, select the most probable admissible motion for play.
// `branches` is the server's maxBranches. When it covers every admissible
// motion, a plans-v1 decision sends the weapon question of each admissible
// motion as a branch of the motion request, one request in all, and keeps the
// selected motion's answer; the recorded calls are the two a decision of two
// requests records.
export async function decide(
    { model, agentIds, contextLimit, ask, branches = 0 },
    body,
    { questions: text = DEFAULT_QUESTIONS, notes = "", choose, legacyNanoJev = false } = {},
) {
    const calls = [],
        started = performance.now(),
        pick = choose ?? ((head, p, allowed) => argmax(p, allowed));
    let asked = 0;
    // The request for `questions` about `state`, as the model reads it.
    function request(state, questions, suffix) {
        if (notes) state = notes + "\n\n" + state;
        // Laya's native option pointers allow 48 tokens each. Preserve every
        // forecast in the state and use concise pointers, never truncate it.
        if (!legacyNanoJev) {
        state +=
            "\n\nOPTION FORECASTS:\n" +
            Object.entries(questions)
                .map(([head, q]) =>
                    Object.entries(q.criteria)
                        .map(
                            ([id, description]) =>
                                `${head}/${id}: ${description}`,
                        )
                        .join("\n"),
                )
                .join("\n");
        questions = Object.fromEntries(
            Object.entries(questions).map(([head, q]) => [
                head,
                {
                    ...q,
                    criteria: Object.fromEntries(
                        Object.keys(q.criteria).map((id) => [
                            id,
                            `${id}: use the ${head}/${id} forecast in state.`,
                        ]),
                    ),
                },
            ]),
        );
        }
        return {
            id: body.id + ":" + suffix,
            parts: {
                essential: [state],
                details: [],
                memory: [],
                logs: [],
            },
            questions,
        };
    }
    // The answers `reply` gives to `sent`, checked, and recorded as one call.
    function accept(sent, reply) {
        const state = sent.parts.essential[0],
            questions = sent.questions;
        if (
            reply?.id !== sent.id ||
            reply.stateText !== state ||
            !Array.isArray(reply.trimmed) ||
            reply.trimmed.length ||
            Object.values(reply.tokenAudit || {}).some(
                (a) => a.stateTruncated === true,
            )
        )
            throw new HttpError(
                502,
                "Model did not preserve complete decision context",
            );
        if (
            !reply.answers ||
            Object.keys(reply.answers).length !== Object.keys(questions).length
        )
            throw new HttpError(502, "Model returned incomplete answers");
        for (const [head, q] of Object.entries(questions)) {
            const a = reply.answers[head],
                keys = Object.keys(q.criteria);
            if (
                !validProbabilities(a?.probabilities, keys) ||
                !keys.includes(a.choice) ||
                a.probabilities[a.choice] <
                    Math.max(...keys.map((k) => a.probabilities[k]))
            )
                throw new HttpError(
                    502,
                    "Model returned invalid native probabilities or choice",
                );
        }
        const { branches: _, ...call } = reply;
        calls.push({
            ...call,
            questions,
        });
        return reply.answers;
    }
    async function predict(state, questions, suffix) {
        const sent = request(state, questions, suffix);
        asked++;
        return accept(sent, await ask(sent));
    }
    let controls,
        probabilities,
        extra = {};
    if (body.protocol === "plans-v1") {
        const motionQuestions = {
                motion: question(text.motion, body.motions),
            },
            weaponQuestions = {
                weapon: question(text.weapon, [
                    {
                        id: "fire",
                        description: text["weapon.fire"],
                    },
                    {
                        id: "hold",
                        description: text["weapon.hold"],
                    },
                ]),
            };
        // The weapon question's state after one motion: its shot forecast and whether firing is permitted.
        const weaponFor = (motionId) => {
            const motion = body.motions.find((m) => m.id === motionId),
                permitted =
                    motion.shot.can_fire &&
                    motion.shot.self_hit_ms === null &&
                    motion.shot.enemy_hit_ms !== null;
            return {
                permitted,
                state:
                    legacyNanoJev ? body.weapon_state + '\nSELECTED motion (already fixed): ' + motionId + '. ' + motion.description + '\nShot forecast for THIS motion: ' + JSON.stringify(motion.shot).replaceAll(',', ', ').replaceAll(':', ': ') : body.weapon_state +
                    `\nSELECTED MOTION ${motionId}: ${motion.description}\nShot forecast ${JSON.stringify(motion.shot)}. Fire permitted=${permitted}. Assess ONLY this selected motion.`,
            };
        };
        const together = branches >= body.admissible_motion_ids.length;
        let a, weaponSent, weaponReplies;
        if (together) {
            const sent = request(body.state, motionQuestions, "motion");
            weaponSent = Object.fromEntries(
                body.admissible_motion_ids.map((id) => [
                    id,
                    request(weaponFor(id).state, weaponQuestions, "weapon"),
                ]),
            );
            asked++;
            const reply = await ask({ ...sent, branches: weaponSent });
            a = accept(sent, reply);
            weaponReplies = reply.branches ?? {};
        } else a = await predict(body.state, motionQuestions, "motion");
        const raw = a.motion.choice;
        // Sample the policy's full distribution. Safety is an environment action
        // projection, not a second policy with unrecorded conditional probabilities.
        const policyMotion = choose
            ? pick(
                  "motion",
                  a.motion.probabilities,
                  body.motions.map((m) => m.id),
              )
            : argmax(a.motion.probabilities, body.admissible_motion_ids);
        if (!body.motions.some((m) => m.id === policyMotion))
            throw new HttpError(502, "Chosen policy motion is invalid");
        const motionId = body.admissible_motion_ids.includes(policyMotion)
                ? policyMotion
                : argmax(a.motion.probabilities, body.admissible_motion_ids),
            motion = body.motions.find((m) => m.id === motionId);
        if (!motion || !body.admissible_motion_ids.includes(motionId))
            throw new HttpError(502, "Chosen motion is not admissible");
        const { permitted, state: weaponState } = weaponFor(motionId);
        const w = together
            ? accept(weaponSent[motionId], weaponReplies[motionId])
            : await predict(weaponState, weaponQuestions, "weapon");
        const weapon = choose
            ? pick("weapon", w.weapon.probabilities, ["fire", "hold"])
            : w.weapon.choice;
        controls = {
            drive: motion.drive,
            turn: motion.turn,
            fire: permitted && weapon === "fire",
        };
        probabilities = {
            motion: a.motion.probabilities,
            weapon: w.weapon.probabilities,
            drive: Object.fromEntries(
                DRIVE.map((k) => [
                    k,
                    body.motions
                        .filter((m) => m.drive === k)
                        .reduce((s, m) => s + a.motion.probabilities[m.id], 0),
                ]),
            ),
            turn: Object.fromEntries(
                TURN.map((k) => [
                    k,
                    body.motions
                        .filter((m) => m.turn === k)
                        .reduce((s, m) => s + a.motion.probabilities[m.id], 0),
                ]),
            ),
        };
        extra = {
            motion_id: motionId,
            policy_motion_id: policyMotion,
            raw_motion_id: raw,
            raw_weapon: w.weapon.choice,
            weapon_choice: weapon,
            weapon_context: weaponState,
            constraints: {
                admissible_motion_ids: body.admissible_motion_ids,
                motion_changed: raw !== motionId,
                policy_motion_changed: policyMotion !== motionId,
                fire_permitted: permitted,
                fire_vetoed: weapon === "fire" && !permitted,
            },
        };
    } else {
        const answers = await predict(
            body.state,
            Object.fromEntries(
                Object.entries(body.candidates).map(([head, opts]) => [
                    head,
                    question(text[head], opts),
                ]),
            ),
            "controls",
        );
        const chosen = Object.fromEntries(
            Object.entries(answers).map(([head, a]) => [
                head,
                choose
                    ? pick(head, a.probabilities, Object.keys(a.probabilities))
                    : a.choice,
            ]),
        );
        controls = {
            drive: chosen.drive,
            turn: chosen.turn,
            fire: chosen.weapon === "fire",
        };
        probabilities = Object.fromEntries(
            Object.entries(answers).map(([head, a]) => [head, a.probabilities]),
        );
        extra = {
            chosen,
        };
    }
    const audits = calls.flatMap((c) => Object.values(c.tokenAudit || {}));
    const counted =
        audits.length > 0 &&
        audits.every(
            (a) => Number.isInteger(a.inputTokens) && a.inputTokens >= 0,
        );
    return {
        id: body.id,
        round_id: body.round_id,
        protocol: body.protocol,
        model,
        agent_id: agentIds[body.player],
        controls,
        probabilities,
        ...extra,
        inference_ms: performance.now() - started,
        input_tokens: counted
            ? Math.max(...audits.map((a) => a.inputTokens))
            : null,
        total_input_tokens: counted
            ? audits.reduce((s, a) => s + a.inputTokens, 0)
            : null,
        context_limit: contextLimit,
        state_truncated: false,
        structured_requests: asked,
        honing_results: calls,
    };
}
