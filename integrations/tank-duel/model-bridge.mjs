import { validProbabilities } from "./web/shared/model-client.js";
const DRIVE = ["forward", "reverse", "stop"],
    TURN = ["left", "right", "none"],
    MOVE = ["up", "down", "left", "right", "stay"];
export const protocols = (edition) =>
    edition === "classic" ? ["plans-v1", "controls-v3"] : ["controls-v2"];
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
        const heads =
            body.protocol === "controls-v2"
                ? { movement: MOVE, weapon: ["fire", "hold"] }
                : { drive: DRIVE, turn: TURN, weapon: ["fire", "hold"] };
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
                head === "movement" || opts.length === allowed.length,
                "Incomplete control candidates",
            );
        }
    }
}
export async function upstream(backend, route, body, signal) {
    const headers = { "Content-Type": "application/json" };
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
            instances: ["ricochet-a", "ricochet-b"],
            independent_instances: false,
            supported_protocols: protocols(edition),
            family: m.family,
            revision: m.revision,
            context_limit: m.maxLen ?? null,
            transport: "honing",
        },
    };
}
const instructions = {
    motion: "Choose joint driving and turning to survive and obtain a verified shot. Every shell is lethal. Prefer safe progress and small heading error. Do not fire yet.",
    drive: "Choose forward, reverse, or stop. The barrel is fixed to the hull; turn and weapon are separate choices. Avoid all shells.",
    turn: "Choose hull rotation. Use the verified firing direction or navigation waypoint in context. Left is negative angle, right is positive.",
    movement:
        "Choose movement now. Avoid every shell, including your own. Prefer no-hit routes, then approach the enemy. Shooting is independent.",
    weapon: "Decide whether to fire. Own ricochets are lethal. Use the selected motion forecast when present. Fire only with a safe valid shot; otherwise hold.",
};
const question = (head, options) => ({
    type: "choice",
    instructions: instructions[head],
    criteria: Object.fromEntries(options.map((o) => [o.id, o.description])),
});
export async function decideWithHoning(backend, edition, body, signal) {
    const health = await healthFor(backend, edition, signal),
        calls = [],
        started = performance.now();
    async function predict(state, questions, suffix) {
        // Laya's native option pointers allow 48 tokens each. Preserve every
        // forecast in the state and use concise pointers, never truncate it.
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
        const id = body.id + ":" + suffix,
            reply = await upstream(
                backend,
                "/v1/decide",
                {
                    id,
                    parts: {
                        essential: [state],
                        details: [],
                        memory: [],
                        logs: [],
                    },
                    questions,
                },
                signal,
            );
        if (
            reply?.id !== id ||
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
        calls.push(reply);
        return reply.answers;
    }
    let controls,
        probabilities,
        extra = {};
    if (body.protocol === "plans-v1") {
        const a = await predict(
            body.state,
            { motion: question("motion", body.motions) },
            "motion",
        );
        const raw = a.motion.choice,
            motionId = argmax(
                a.motion.probabilities,
                body.admissible_motion_ids,
            ),
            motion = body.motions.find((m) => m.id === motionId);
        const permitted =
            motion.shot.can_fire &&
            motion.shot.self_hit_ms === null &&
            motion.shot.enemy_hit_ms !== null;
        const weaponState =
            body.weapon_state +
            `\nSELECTED MOTION ${motionId}: ${motion.description}\nShot forecast ${JSON.stringify(motion.shot)}. Fire permitted=${permitted}. Assess ONLY this selected motion.`;
        const w = await predict(
            weaponState,
            {
                weapon: question("weapon", [
                    {
                        id: "fire",
                        description:
                            "Fire after this selected motion turns. Only useful if its new shot is safe and hits the enemy.",
                    },
                    {
                        id: "hold",
                        description:
                            "Execute this motion without firing. Choose this if the selected shot is unavailable or unsafe.",
                    },
                ]),
            },
            "weapon",
        );
        controls = {
            drive: motion.drive,
            turn: motion.turn,
            fire: permitted && w.weapon.choice === "fire",
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
            raw_motion_id: raw,
            raw_weapon: w.weapon.choice,
            weapon_context: weaponState,
            constraints: {
                admissible_motion_ids: body.admissible_motion_ids,
                motion_changed: raw !== motionId,
                fire_permitted: permitted,
                fire_vetoed: w.weapon.choice === "fire" && !permitted,
            },
        };
    } else {
        const answers = await predict(
            body.state,
            Object.fromEntries(
                Object.entries(body.candidates).map(([head, opts]) => [
                    head,
                    question(head, opts),
                ]),
            ),
            "controls",
        );
        controls =
            body.protocol === "controls-v2"
                ? {
                      move: answers.movement.choice,
                      fire: answers.weapon.choice === "fire",
                  }
                : {
                      drive: answers.drive.choice,
                      turn: answers.turn.choice,
                      fire: answers.weapon.choice === "fire",
                  };
        probabilities = Object.fromEntries(
            Object.entries(answers).map(([head, a]) => [head, a.probabilities]),
        );
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
        model: health.metadata.model,
        agent_id: health.metadata.instances[body.player],
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
        context_limit: health.metadata.context_limit,
        state_truncated: false,
        structured_requests: calls.length,
        honing_results: calls,
    };
}
