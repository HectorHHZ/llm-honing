// One predeclared, game-local intervention at a state chosen before the fault.
import { bulletForecast } from "../web/game/engine.js";

const LAYERS = new Set(["harness", "weights", "controller"]);
const TRIGGERS = new Set(["aim_fire", "incoming_shell", "either"]);

export function parseCriticalIntervention(text) {
    const fault = JSON.parse(text);
    if (
        !fault ||
        !LAYERS.has(fault.layer) ||
        !TRIGGERS.has(fault.trigger) ||
        typeof fault.instance !== "string" ||
        !/^[a-z0-9_-]+$/.test(fault.instance)
    )
        throw Error("Invalid critical intervention manifest");
    if (
        fault.layer === "harness" &&
        !["one_turn_questions", "one_turn_settings"].includes(fault.kind)
    )
        throw Error(
            "Critical harness fault must replace one turn's questions or settings",
        );
    if (fault.layer === "weights" && fault.kind !== "one_turn_checkpoint")
        throw Error(
            "Critical weights fault requires a real alternate checkpoint",
        );
    if (
        fault.layer === "controller" &&
        fault.kind !== "drop_fire_after_selection"
    )
        throw Error(
            "Critical controller fault must alter execution after selection",
        );
    if (fault.layer === "controller" && fault.trigger !== "aim_fire")
        throw Error("Drop-fire controller fault requires aim_fire trigger");
    if (
        fault.layer === "harness" &&
        typeof fault.alternate_harness_dir !== "string"
    )
        throw Error("Missing alternate harness directory");
    if (
        fault.layer === "weights" &&
        typeof fault.alternate_decision_server !== "string"
    )
        throw Error("Missing alternate checkpoint server");
    if (
        fault.decision_index !== undefined &&
        (!Number.isInteger(fault.decision_index) || fault.decision_index < 0)
    )
        throw Error("Critical decision index must be non-negative");
    return fault;
}

export function criticalCandidate(game, player, observed, trigger) {
    const tank = game.tanks[player];
    const enemy = game.tanks[1 - player];
    const distancePx = Math.hypot(tank.x - enemy.x, tank.y - enemy.y);
    const shot =
        distancePx <= 400 &&
        observed.request.motions?.some(
            (motion) =>
                motion.shot.can_fire &&
                motion.shot.enemy_hit_ms !== null &&
                motion.shot.self_hit_ms === null,
        );
    const incoming = game.bullets.some(
        (bullet) =>
            bullet.owner !== player &&
            bulletForecast(game, bullet, tank, 0.8).hitTime !== null,
    );
    if ((trigger === "aim_fire" || trigger === "either") && shot)
        return { kind: "aim_fire", distance_px: Math.round(distancePx) };
    if ((trigger === "incoming_shell" || trigger === "either") && incoming)
        return { kind: "incoming_shell", distance_px: Math.round(distancePx) };
    return null;
}
