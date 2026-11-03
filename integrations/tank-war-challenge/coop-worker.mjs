import { parentPort, workerData } from "node:worker_threads";
import { CoopGame } from "./coop-game.mjs";
import { adapter } from "./web/classic/adapter.js";
const game = new CoopGame(workerData.players, workerData.seed);
const settings = {
    controllerMode: "plans",
    contextVersion: "full",
    decisionVersion: "baseline",
    interval: 100,
};
let pending = null,
    serial = 0,
    nextDecision = 0,
    rtt = 150,
    ended = false,
    last = performance.now(),
    accumulator = 0,
    lastSnapshot = 0;
const inputs = new Map();
parentPort.on("message", (m) => {
    if (m.type === "input")
        inputs.set(m.id, { ...m.controls, at: performance.now() });
    if (m.type === "drop") {
        game.drop(m.id);
        inputs.delete(m.id);
    }
    if (m.type === "next") {
        pending = null;
        ended = false;
        game.beginRound();
        inputs.clear();
    }
    if (m.type === "decision" && pending?.id === m.id) {
        const p = pending;
        pending = null;
        rtt = performance.now() - p.sent;
        nextDecision = performance.now() + (m.response ? 50 : 1000);
        const target = game.tanks.find((t) => t.id === p.target);
        if (
            m.response &&
            game.phase === "running" &&
            game.round === p.round &&
            target?.alive
        ) {
            try {
                const view = game.view(target);
                if (adapter.validControls(m.response, p.obs)) {
                    adapter.apply(view, 1, m.response, settings);
                    game.probabilities = m.response.probabilities;
                }
            } catch (e) {
                parentPort.postMessage({
                    type: "fault",
                    message: "Invalid model response",
                });
            }
        }
    }
});
setInterval(() => {
    const now = performance.now();
    accumulator += Math.min((now - last) / 1000, 0.1);
    last = now;
    while (accumulator >= 1 / 60) {
        const manual = Object.fromEntries(
            [...inputs].filter(([, c]) => now - c.at < 400),
        );
        game.step(1 / 60, manual);
        accumulator -= 1 / 60;
    }
    if (game.phase === "roundover" && !ended) {
        ended = true;
        pending = null;
        parentPort.postMessage({
            type: "roundover",
            round: game.round,
            result: game.result,
        });
    }
    if (game.phase === "running" && !pending && now >= nextDecision) {
        const target = game.target();
        if (target) {
            try {
                const obs = adapter.observe(game.view(target), 1, {
                    ...settings,
                    rtt: Math.min(rtt, 800),
                });
                const roster = game.tanks
                    .filter((t) => t.team === 0)
                    .map((t) => ({
                        id: t.id,
                        x: Math.round(t.x),
                        y: Math.round(t.y),
                        heading_deg: Math.round((t.angle * 180) / Math.PI),
                        alive: t.alive,
                    }));
                const extra = `\nCOOPERATIVE ARENA: you are the single bot. All human tanks are enemies. Selected target id=${target.id}. All humans [id,x,y,heading_deg,alive]: ${JSON.stringify(roster)}. Shot and motion forecasts refer to the selected target. All active shells are included in danger forecasts. Human teammates cannot hurt one another; all human shells can hit you. Your own ricochets remain lethal.\n`;
                const id = `coop-${game.seed}-${++serial}`;
                pending = {
                    id,
                    obs,
                    target: target.id,
                    round: game.round,
                    sent: now,
                };
                parentPort.postMessage({
                    type: "infer",
                    id,
                    round: game.round,
                    body: {
                        ...obs.request,
                        state: obs.request.state + extra,
                        weapon_state: obs.request.weapon_state + extra,
                        id,
                        player: 1,
                        round_id: game.round,
                        decision_interval_ms: 100,
                    },
                });
            } catch (e) {
                nextDecision = now + 1000;
                parentPort.postMessage({
                    type: "fault",
                    message: "Cannot build model context",
                });
            }
        }
    }
    if (now - lastSnapshot >= 50) {
        lastSnapshot = now;
        parentPort.postMessage({ type: "state", state: game.snapshot() });
    }
}, 1000 / 60);
