import {
    Game,
    MOVES,
    WEAPONS,
    PROTOCOL,
    traceShot,
    buildObservation,
} from "./engine.js";
import { validProbabilities } from "../shared/model-client.js";
export const adapter = {
    edition: "auto-aim",
    Game,
    traceShot,
    heads: { movement: MOVES, weapon: WEAPONS },
    protocols: [PROTOCOL],
    initialize: () => {},
    clear: (t) =>
        Object.assign(t, {
            move: "stay",
            executedMove: "stay",
            firing: false,
            commandUntil: 0,
            safetyAt: 0,
            assistActive: false,
        }),
    setAssist(game, on) {
        game.assistEnabled = on;
        for (const t of game.tanks) {
            t.safetyAt = 0;
            t.safeMove = t.move;
            t.assistActive = false;
        }
    },
    assistance: (game) => ({
        enabled: game.assistEnabled,
        horizon_ms: 700,
        check_interval_ms: 50,
    }),
    shortControls: "WASD move · Space fire",
    keyboard: (mode) =>
        mode === "local"
            ? "Blue: WASD move, Space fire · Red: arrow keys move, Enter fire · P pause"
            : "WASD or arrow keys move · Space fire · P pause",
    description: () =>
        "The model chooses movement and firing independently. The turret turns automatically toward a verified direct or ricochet shot; the game checks that shot before firing. Every active shell, including your own, is included in model context.",
    assistNote: (game) =>
        game.assistEnabled
            ? "Dodge assist on · checks the next 700 ms every 50 ms. Interventions appear in match exports."
            : "Dodge assist off · automatic aiming stays on. Your shells can hit you.",
    observe(game, id, { rtt }) {
        const obs = buildObservation(game, id, {
            reactionDelay: Math.min(1.6, Math.max(1 / 60, (rtt ?? 120) / 1000)),
        });
        return {
            ...obs,
            bulletCount: obs.bullets.length,
            text:
                obs.state +
                "\n\n" +
                Object.entries(obs.candidates)
                    .map(
                        ([head, opts]) =>
                            head +
                            ":\n" +
                            opts
                                .map((c) => c.id + ": " + c.description)
                                .join("\n"),
                    )
                    .join("\n\n"),
            request: {
                protocol: PROTOCOL,
                state: obs.state,
                candidates: obs.candidates,
            },
        };
    },
    validControls(r, obs) {
        return (
            obs.candidates.movement.some((c) => c.id === r.controls?.move) &&
            typeof r.controls?.fire === "boolean" &&
            Object.entries(obs.candidates).every(([head, opts]) =>
                validProbabilities(
                    r.probabilities?.[head],
                    opts.map((c) => c.id),
                ),
            )
        );
    },
    apply(game, id, r, { interval }) {
        game.setControls(id, r.controls, Math.max(0.6, interval / 1000 + 0.2));
        return { controls: r.controls };
    },
    action: (t) => MOVES[t.executedMove],
    aimNote: (data) =>
        data?.observed?.shot
            ? "A verified firing direction is available."
            : "Turret is looking for a firing lane.",
    execution: () =>
        "Model movement + weapon request; automatic turret and shot validation.",
    selected: (data, head) =>
        head === "weapon"
            ? data.controls.fire
                ? "fire"
                : "hold"
            : data.controls.move,
    manual(id, mode, held) {
        const arrows = id === 1 || mode === "human";
        let move = "stay";
        for (const [key, arrow, direction] of [
            ["KeyW", "ArrowUp", "up"],
            ["KeyS", "ArrowDown", "down"],
            ["KeyA", "ArrowLeft", "left"],
            ["KeyD", "ArrowRight", "right"],
        ])
            if (held(id ? arrow : key) || (arrows && held(arrow)))
                move = direction;
        return { move, fire: held(id ? "Enter" : "Space") };
    },
};
