import { createRenderer } from "../shared/renderer.js";
import { traceShot } from "../classic/engine.js";
const $ = (id) => document.getElementById(id);
document.body.innerHTML = `<main class="shell tank-war human-challenge coop">
<header class="war-header"><h1>TANK WAR</h1></header>
<nav class="toolbar"><a href="/play/">Solo</a><strong>Team vs Bot</strong><button id="join">Join team</button><button id="leave" hidden>Leave</button></nav>
<div class="war-score"><div class="blue"><strong>Team</strong><b id="score-a">0</b></div><div class="war-clock"><span id="round">CO-OP</span><strong id="clock">60.0</strong></div><div class="red"><b id="score-b">0</b><strong>Bot</strong></div></div>
<div class="war-stage"><section class="war-arena"><div class="board-wrap"><canvas id="arena" width="900" height="630" tabindex="0" aria-label="Co-op tank arena"></canvas></div><p id="status" role="status">Join a shared arena. Up to 8 players vs one bot.</p></section>
<section class="agent-panel red" aria-label="Bot decisions"><div class="probabilities" id="probabilities"></div></section></div>
<p class="challenge-controls"><span><kbd>W</kbd><kbd>S</kbd> Drive</span><span><kbd>A</kbd><kbd>D</kbd> Turn</span><span><kbd>Space</kbd> Fire</span></p>
<p class="coop-rule">No teammate damage. Your own ricochets can still hit you. 3 rounds per turn.</p>
<section class="arena-lobby"><header><strong id="nickname">Team lobby</strong></header><div class="lobby-playing" id="roster"></div><ol class="lobby-queue" id="waiting"></ol></section></main>`;
const heads = {
    drive: { forward: "Forward", reverse: "Reverse", stop: "Stop" },
    turn: { left: "Left", right: "Right", none: "Straight" },
    weapon: { fire: "Fire", hold: "Hold" },
};
for (const [head, actions] of Object.entries(heads)) {
    const group = document.createElement("div");
    group.className = "decision-group";
    group.innerHTML = `<h3 class="decision-head">${head[0].toUpperCase() + head.slice(1)}</h3>`;
    for (const [key, label] of Object.entries(actions))
        group.innerHTML += `<div class="prob-row" data-head="${head}" data-action="${key}"><span>${label}</span><div class="prob-track"><div class="prob-fill"></div></div><span class="prob-value">—</span></div>`;
    $("probabilities").append(group);
}
const render = createRenderer($("arena"), traceShot);
let token = null,
    stream = null,
    state = null,
    previous = null,
    received = 0,
    seq = 0,
    sending = false,
    joining = false;
const keys = new Set();
const controls = () => ({
    drive:
        Number(keys.has("KeyW") || keys.has("ArrowUp")) -
        Number(keys.has("KeyS") || keys.has("ArrowDown")),
    turn:
        Number(keys.has("KeyD") || keys.has("ArrowRight")) -
        Number(keys.has("KeyA") || keys.has("ArrowLeft")),
    fire: keys.has("Space"),
});
async function post(action, body = {}) {
    const res = await fetch(`/api/coop/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, ...body }),
        signal: AbortSignal.timeout(5000),
    });
    const data = await res.json();
    if (!res.ok) {
        const error = new Error(data.error || "Connection failed");
        error.status = res.status;
        throw error;
    }
    return data;
}
function drawRoster(items, node, queued = false) {
    node.replaceChildren();
    items.forEach((p, i) => {
        const el = document.createElement(queued ? "li" : "span");
        el.className =
            (queued ? "" : "lobby-player ") + (p.is_you ? "is-you" : "");
        el.textContent = `${queued ? `${i + 1}. ` : ""}${p.nickname}${p.is_you ? " · You" : ""}`;
        node.append(el);
    });
}
function update(data) {
    if (data.state === "expired") {
        leave(false);
        $("status").textContent = "Your place expired. Join again.";
        return;
    }
    previous = state?.game;
    state = data;
    received = performance.now();
    $("nickname").textContent = data.nickname;
    drawRoster(data.players, $("roster"));
    drawRoster(data.waiting, $("waiting"), true);
    const g = data.game,
        me = g?.tanks.find((t) => t.id === data.id);
    $("status").textContent =
        data.fault ||
        (data.state === "queued"
            ? `Queue position ${data.position}`
            : data.state === "waiting"
              ? data.countdown
                  ? `Starting in ${data.countdown}…`
                  : "Waiting for the next team turn…"
              : g?.phase === "roundover"
                ? g.result === 0
                    ? "Team wins!"
                    : g.result === 1
                      ? "Bot wins!"
                      : "Draw."
                : me && !me.alive
                  ? "You are out. Watch your teammates until the next round."
                  : `${data.nickname} · ${g?.tanks.filter((t) => t.team === 0 && t.alive).length ?? 0} players remaining`);
    if (g) {
        $("round").textContent = `ROUND ${g.round} / 3`;
        $("clock").textContent = Math.max(0, g.timeLeft).toFixed(1);
        $("score-a").textContent = g.scores[0];
        $("score-b").textContent = g.scores[1];
    }
    for (const row of document.querySelectorAll(".prob-row")) {
        const p = g?.probabilities?.[row.dataset.head],
            v = p?.[row.dataset.action];
        row.querySelector(".prob-value").textContent =
            v == null ? "—" : `${Math.round(v * 100)}%`;
        row.querySelector(".prob-fill").style.width =
            `${v == null ? 0 : Math.max(0, Math.min(100, v * 100))}%`;
        row.classList.toggle(
            "selected",
            !!p && v === Math.max(...Object.values(p)),
        );
    }
}
async function sendInput() {
    if (!token || sending) return;
    sending = true;
    try {
        await post("input", {
            seq: ++seq,
            round: state?.game?.round,
            controls: controls(),
        });
    } catch (e) {
        if ([401, 403, 410].includes(e.status)) {
            leave(false);
            $("status").textContent = "Your place expired. Join again.";
        } else
            $("status").textContent = "Connection interrupted. Reconnecting…";
    } finally {
        sending = false;
    }
}
$("join").onclick = async () => {
    if (joining || token) return;
    joining = true;
    $("join").disabled = true;
    try {
        const data = await post("join");
        token = data.token;
        seq = 0;
        update(data);
        $("join").hidden = true;
        $("leave").hidden = false;
        stream = new EventSource("/api/coop/events");
        stream.onmessage = (e) => update(JSON.parse(e.data));
        stream.onerror = () => {
            $("status").textContent = "Connection interrupted. Reconnecting…";
        };
        $("arena").focus();
    } catch (e) {
        $("status").textContent = e.message;
    } finally {
        joining = false;
        $("join").disabled = false;
    }
};
function leave(notify = true) {
    const old = token;
    token = null;
    stream?.close();
    stream = null;
    keys.clear();
    if (notify && old)
        navigator.sendBeacon(
            "/api/coop/leave",
            new Blob([JSON.stringify({ token: old })], {
                type: "application/json",
            }),
        );
    $("join").hidden = false;
    $("leave").hidden = true;
    state = null;
    previous = null;
    $("status").textContent = "You left the team. Join again to play.";
    $("roster").replaceChildren();
    $("waiting").replaceChildren();
}
$("leave").onclick = () => leave();
window.addEventListener("pagehide", () => leave());
window.addEventListener("blur", () => {
    keys.clear();
    void sendInput();
});
document.addEventListener("visibilitychange", () => {
    if (document.hidden) leave();
});
window.addEventListener("keydown", (e) => {
    if (
        ![
            "KeyW",
            "KeyS",
            "KeyA",
            "KeyD",
            "Space",
            "ArrowUp",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
        ].includes(e.code) ||
        !token
    )
        return;
    e.preventDefault();
    keys.add(e.code);
    void sendInput();
});
window.addEventListener("keyup", (e) => {
    if (keys.delete(e.code)) {
        e.preventDefault();
        void sendInput();
    }
});
setInterval(() => {
    if (token) void sendInput();
}, 125);
function frame() {
    const g = state?.game;
    if (g) {
        let draw = g;
        if (previous?.round === g.round) {
            const mix = Math.min(1, (performance.now() - received) / 100);
            draw = {
                ...g,
                tanks: g.tanks.map((t) => {
                    const old = previous.tanks.find((p) => p.id === t.id);
                    if (!old || !t.alive) return t;
                    const da = Math.atan2(
                        Math.sin(t.angle - old.angle),
                        Math.cos(t.angle - old.angle),
                    );
                    return {
                        ...t,
                        x: old.x + (t.x - old.x) * mix,
                        y: old.y + (t.y - old.y) * mix,
                        angle: old.angle + da * mix,
                        label: t.id === state.id ? `${t.label} · You` : t.label,
                    };
                }),
            };
        }
        render(draw);
    }
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
