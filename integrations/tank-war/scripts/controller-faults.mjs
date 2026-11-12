// Explicit, game-local fault injections. No generic Honing module knows tank
// turns, plans, or firing. The private trigger record is never a diagnoser input.
export const CONTROLLER_FAULTS = Object.freeze([
    "swap_selected_turn",
    "drop_fire_after_selection",
    "wrong_legal_projection",
]);

export function parseControllerFault(text) {
    const fault = JSON.parse(text);
    if (
        !fault ||
        fault.layer !== "controller" ||
        !CONTROLLER_FAULTS.includes(fault.kind) ||
        typeof fault.instance !== "string" ||
        !/^[a-z0-9_-]+$/.test(fault.instance)
    )
        throw Error("Invalid controller fault manifest");
    return { layer: fault.layer, kind: fault.kind, instance: fault.instance };
}

export function beforeControllerApply(response, observed, fault) {
    if (fault?.kind !== "wrong_legal_projection") return null;
    const chosen = response.motion_id;
    const other = observed.admissible.find((id) => id !== chosen);
    if (!other) return null;
    const plan = observed.actions.find((action) => action.id === other);
    if (!plan) throw Error("Admissible motion is missing its plan");
    return {
        response: {
            ...response,
            motion_id: other,
            controls: { ...plan.controls, fire: false },
        },
        before: { motion: chosen, controls: response.controls },
        after: { motion: other, controls: { ...plan.controls, fire: false } },
    };
}

export function afterControllerApply(game, player, applied, fault) {
    const tank = game.tanks[player];
    if (fault?.kind === "drop_fire_after_selection" && applied.controls.fire) {
        const before = { ...applied.controls };
        tank.pendingFire = false;
        tank.firing = false;
        applied.controls.fire = false;
        return { before, after: { ...applied.controls } };
    }
    if (
        fault?.kind === "swap_selected_turn" &&
        ["left", "right"].includes(applied.controls.turn)
    ) {
        const before = { ...applied.controls };
        const turn = applied.controls.turn === "left" ? "right" : "left";
        tank.turn = turn;
        tank.executedTurn = turn;
        applied.controls.turn = turn;
        return { before, after: { ...applied.controls } };
    }
    return null;
}
