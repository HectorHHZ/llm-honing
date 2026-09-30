import {
    W,
    H,
    SPEED,
    TURN_SPEED,
    RADIUS,
    PROTOCOL,
    avoidanceContext,
} from "./engine.js";

// A deliberately small observation profile, not a claim to reproduce an archived prompt.
export function buildArenaContext(
    game,
    id,
    { reactionDelay, controlDuration },
) {
    const round = (n) => Math.round(n * 10) / 10;
    const tank = (t) => ({
        x: round(t.x),
        y: round(t.y),
        heading_deg: round((t.angle * 180) / Math.PI),
        alive: t.alive,
    });
    const t = game.tanks[id];
    const bullets = game.bullets
        .filter((b) => !b.dead)
        .map((b) => ({
            id: b.id,
            owner: b.owner === id ? "self" : "enemy",
            x: round(b.x),
            y: round(b.y),
            vx: round(b.vx),
            vy: round(b.vy),
            bounces: b.bounces,
            remaining_s: round(Math.max(0, 8 - b.age)),
        }));
    const state =
        `CONTEXT arena-v1. Tank duel: one hit kills; ALL shells including your own are lethal. Walls reflect shells. ` +
        `NO AUTO-AIM. Barrel rotates with hull. Drive forward/reverse/stop, turn left/right/none and fire/hold independently. ` +
        `Coordinates: origin top-left; +x right, +y down; heading 0 right, 90 down; positive turn RIGHT. Arena ${W}x${H}, tank radius ${RADIUS}, shell radius 3, speed ${SPEED}px/s, turn ${round((TURN_SPEED * 180) / Math.PI)}deg/s. ` +
        `Me ${JSON.stringify(tank(t))}; enemy ${JSON.stringify(tank(game.tanks[1 - id]))}. ` +
        `Round ${game.round}, elapsed ${round(game.elapsed)}s, remaining ${round(game.timeLeft)}s. Ammo ${5 - bullets.filter((b) => b.owner === "self").length}/5, cooldown ${Math.round(t.cooldown * 1000)}ms. ` +
        `Executed ${t.executedDrive}/${t.executedTurn}, expires in ${Math.round(Math.max(0, t.commandUntil - game.elapsed) * 1000)}ms. Estimated response delay ${Math.round(reactionDelay * 1000)}ms; new command lasts ${Math.round(controlDuration * 1000)}ms then stops. FIRE is one immediate attempt, not a queued trigger. ` +
        avoidanceContext(game, id) +
        `Walls [x,y,w,h]: ${JSON.stringify(game.maze.walls.map((w) => [w.x, w.y, w.w, w.h]))}. ` +
        `All ${bullets.length} shells: ${JSON.stringify(bullets)}. No aim hints, navigation waypoints or trajectory forecasts are supplied in this profile. Choose from raw state.`;
    const candidates = {
        drive: [
            { id: "forward", description: "Drive forward along hull heading." },
            { id: "reverse", description: "Reverse opposite hull heading." },
            {
                id: "stop",
                description: "Stop translation; rotation is independent.",
            },
        ],
        turn: [
            {
                id: "left",
                description: "Rotate hull and fixed barrel counterclockwise.",
            },
            {
                id: "right",
                description: "Rotate hull and fixed barrel clockwise.",
            },
            { id: "none", description: "Keep hull and barrel heading." },
        ],
        weapon: [
            {
                id: "fire",
                description: "Attempt one shot along the current hull heading.",
            },
            { id: "hold", description: "Do not fire." },
        ],
    };
    return {
        state,
        candidates,
        bullets,
        reactionDelay,
        controlDuration,
        contextRevision: "arena-v1",
        request: { protocol: PROTOCOL, state, candidates },
    };
}
