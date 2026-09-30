import {
    Game,
    DRIVE,
    TURNS,
    WEAPONS,
    PROTOCOL,
    MODEL_CONTROL_SECONDS,
    buildObservation,
    traceShot,
    responseDelay,
    controlLabel,
    avoidanceProfile,
    assistEnabledFor,
    assistLevelFor,
    shotGuardFor,
} from "./engine.js";
import { buildPlans, applyPlanDecision, PLAN_PROTOCOL } from "./planner.js";
import { buildArenaContext } from "./context.js";
import { contextRevision } from "../shared/seat-config.js";
import { validProbabilities } from "../shared/model-client.js";

export const adapter = {
    edition: "classic",
    Game,
    traceShot,
    heads: { drive: DRIVE, turn: TURNS, weapon: WEAPONS },
    protocols: [PROTOCOL, PLAN_PROTOCOL],
    initialize: (game) => game.setAssistEnabled(true),
    clear: (t) =>
        Object.assign(t, {
            drive: "stop",
            turn: "none",
            firing: false,
            pendingFire: false,
            commandUntil: 0,
            planned: false,
            avoidance: null,
            assistActive: false,
        }),
    setAssist: (game, on) => game.setAssistEnabled(on),
    assistance: (game, id) => ({
        enabled: assistEnabledFor(game, id),
        profile: assistLevelFor(game, id),
        parameters: avoidanceProfile(game, id),
        shot_guard: shotGuardFor(game, id),
    }),
    shortControls: "W/S drive · A/D turn · Space fire",
    keyboard: (mode) =>
        mode === "local"
            ? "Blue: W/S drive, A/D turn, Space fire · Red: ↑/↓ drive, ←/→ turn, Enter fire · P pause"
            : "W/S or ↑/↓ drive · A/D or ←/→ turn · Space fire · P pause",
    description: (mode) =>
        mode === "plans"
            ? "The model chooses a joint drive/turn plan, then decides whether to fire. Geometry limits unsafe plans, times hull rotation and checks the shot again before firing. The barrel stays fixed to the hull. All native probabilities and execution changes are recorded."
            : "Experimental raw controls: drive, turn and fire are independent model choices. The barrel stays fixed to the hull. Unsafe shots can hit you. Dodge assist and shot guard are explicit optional interventions.",
    assistNote: (game) =>
        game.assistEnabled
            ? `Dodge assist: ${avoidanceProfile(game).label} · reacts to threats within ${Math.round(avoidanceProfile(game).trigger * 1000)} ms. Interventions appear in match exports.`
            : "Dodge assist off. Your shells can hit you.",
    observe(
        game,
        id,
        { controllerMode, interval, rtt, contextVersion = "full" },
    ) {
        const controlDuration = Math.max(
                MODEL_CONTROL_SECONDS,
                interval / 1000 + 0.3,
            ),
            opts = {
                reactionDelay: responseDelay(rtt ?? 300),
                controlDuration,
            };
        if (controllerMode === "plans" && contextVersion !== "full")
            throw Error("Planned steering requires full tactical context");
        const revision = contextRevision({ controllerMode, contextVersion });
        const obs =
            controllerMode === "plans"
                ? buildPlans(game, id, opts)
                : contextVersion === "arena-v1"
                  ? buildArenaContext(game, id, opts)
                  : buildObservation(game, id, opts);
        return {
            ...obs,
            controlDuration,
            bulletCount: obs.bullets.length,
            text:
                `CONTEXT PROFILE: ${revision}\n` +
                obs.state +
                "\n\n" +
                (obs.actions
                    ? obs.actions
                          .map((a) => a.id + ": " + a.description)
                          .join("\n")
                    : Object.entries(obs.candidates)
                          .map(
                              ([head, opts]) =>
                                  head +
                                  ":\n" +
                                  opts
                                      .map((c) => c.id + ": " + c.description)
                                      .join("\n"),
                          )
                          .join("\n\n")),
            request: {
                ...(obs.request || {
                    protocol: PROTOCOL,
                    state: obs.state,
                    candidates: obs.candidates,
                }),
                context_revision: revision,
                avoidance_enabled: assistEnabledFor(game, id),
                avoidance_level: assistLevelFor(game, id),
                predicted_response_delay_ms: Math.round(
                    obs.reactionDelay * 1000,
                ),
            },
        };
    },
    validControls(r, obs) {
        const c = r.controls;
        if (
            !Object.hasOwn(DRIVE, c?.drive) ||
            !Object.hasOwn(TURNS, c?.turn) ||
            typeof c?.fire !== "boolean"
        )
            return false;
        if (!validProbabilities(r.probabilities?.weapon, Object.keys(WEAPONS)))
            return false;
        if (obs.actions)
            return (
                obs.admissible.includes(r.motion_id) &&
                r.motion_id === `${c.drive}_${c.turn}` &&
                validProbabilities(
                    r.probabilities?.motion,
                    obs.actions.map((a) => a.id),
                )
            );
        return (
            validProbabilities(r.probabilities?.drive, Object.keys(DRIVE)) &&
            validProbabilities(r.probabilities?.turn, Object.keys(TURNS))
        );
    },
    apply(game, id, r, { controllerMode, interval }) {
        if (controllerMode === "plans") return applyPlanDecision(game, id, r);
        game.setControls(
            id,
            r.controls,
            Math.max(MODEL_CONTROL_SECONDS, interval / 1000 + 0.3),
        );
        return { controls: r.controls };
    },
    action: (t) =>
        controlLabel({ drive: t.executedDrive, turn: t.executedTurn }),
    aimNote: (data) => {
        const aim = data?.observed?.aim?.best;
        if (data?.observed?.contextRevision === "arena-v1")
            return "Arena-only context: the model receives no aim hints.";
        return !data
            ? "Waiting for an aim observation."
            : aim
              ? `Hit direction ${aim.heading_deg}° · ${aim.turn === "none" ? "aligned" : aim.turn + " " + Math.abs(aim.delta_deg) + "°"} · ${aim.bounces ? "ricochet" : "direct"} · ${aim.hit_ms} ms`
              : "No verified shot. Move to find an opening.";
    },
    execution: (data) =>
        data.protocol === PLAN_PROTOCOL
            ? `Model ${data.raw_motion_id} / ${data.raw_weapon} → executed ${data.applied.motion_id} / ${data.applied.controls.fire ? "fire after turn" : "hold"}`
            : "Raw controls applied; any assist intervention is logged.",
    selected: (data, head) =>
        head === "weapon"
            ? data.applied.controls.fire
                ? "fire"
                : "hold"
            : data.applied.controls[head],
    manual(id, mode, held) {
        const arrows = id === 1 || mode === "human";
        return {
            drive:
                Number(
                    held(id ? "ArrowUp" : "KeyW") ||
                        (arrows && held("ArrowUp")),
                ) -
                Number(
                    held(id ? "ArrowDown" : "KeyS") ||
                        (arrows && held("ArrowDown")),
                ),
            turn:
                Number(
                    held(id ? "ArrowRight" : "KeyD") ||
                        (arrows && held("ArrowRight")),
                ) -
                Number(
                    held(id ? "ArrowLeft" : "KeyA") ||
                        (arrows && held("ArrowLeft")),
                ),
            fire: held(id ? "Enter" : "Space"),
        };
    },
};
