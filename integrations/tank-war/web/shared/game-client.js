import { AdaptiveCadence } from "./cadence.js";
import { mountTankSetup } from "../game/setup.js";
import { defaultSeats, seatProtocol } from "./seat-config.js";
import { mountShell } from "./shell.js";
import { createRenderer, COLORS } from "./renderer.js";
import {
    modelConfig,
    validHealth,
    validDecision,
    tokenLabel,
} from "./model-client.js";
export async function boot(adapter) {
    const { edition } = adapter;
    mountShell(edition);
    const $ = (id) => document.getElementById(id),
        canvas = $("arena"),
        game = new adapter.Game();
    adapter.initialize(game);
    const models = await modelConfig(edition);
    let seats = defaultSeats(models);
    let setup;
    {
        setup = mountTankSetup(models, game, (player, field) => {
            if (
                ["assistEnabled", "assistProfile", "fireGuardEnabled"].includes(
                    field,
                )
            ) {
                cancelSeat(player);
                game.emit("configuration", {
                    player,
                    field,
                    settings: {
                        ...seats[player],
                    },
                });
                addFeed(`${side(player)} ${field} updated.`, player);
                update();
            } else {
                cadenceModels.forEach((c) => c.reset());
                health = [null, null];
                healthSerial = healthSerial.map((n) => n + 1);
                error = "Connecting selected models";
                reset();
                void checkHealth();
            }
        });
        seats = setup.seats;
    }
    let mode = models.length ? "ai" : "local";
    let paused = false,
        health = [null, null],
        healthSerial = [0, 0],
        error = "",
        generation = 0,
        serial = 0,
        trajectory = false;
    const cadenceModels = [new AdaptiveCadence(), new AdaptiveCadence()];
    let pending = [null, null],
        seatEpoch = [0, 0],
        next = [0, 0],
        latest = [null, null],
        counts = [0, 0],
        dispatch = [[], []],
        record = [],
        feed = [];
    let cursor = 0,
        lastRound = game.roundSerial,
        lastFrame = performance.now(),
        accumulator = 0,
        uiAt = 0;
    const keys = new Set(),
        taps = new Set(),
        side = (i) => (i ? "Red" : "Blue"),
        isAI = (i) => mode === "ai" || (mode === "human" && i === 1);
    const ready = () => seats.every((_, i) => !isAI(i) || !!health[i]);
    const modelName = (i) =>
        models.find((m) => m.id === seats[i].backend)?.label ||
        health[i]?.metadata?.model_label ||
        "Model";
    const render = createRenderer(canvas, adapter.traceShot);
    $("mode").value = mode;
    if (!models.length)
        for (const option of $("mode").options)
            option.disabled = option.value !== "local";
    const panels = [0, 1].map((i) => {
        const p = $(`agent-${i}`);
        p.innerHTML = `<div class="agent-heading"><h2></h2><span class="agent-status">Ready</span></div><p class="aim-note model-note"></p><p class="execution-note model-note"></p><div class="probabilities"></div><div class="agent-metrics"><span>Decision <b class="latency">—</b> ms</span><span>Round trip <b class="rtt">—</b> ms</span><span>Actual interval <b class="cadence">—</b> ms</span><span><b class="count">0</b> decisions</span><span><b class="bullets">0</b> shells observed</span></div><details class="context-view"><summary>Model context <span class="tokens">—</span></summary><pre class="context">Waiting for a decision.</pre></details>`;
        for (const [head, labels] of Object.entries(adapter.heads)) {
            const group = document.createElement("div");
            group.className = "decision-group";
            const title = document.createElement("div");
            title.className = "decision-head";
            title.textContent = {
                movement: "Movement",
                drive: "Drive",
                turn: "Turn",
                weapon: "Weapon",
            }[head];
            group.append(title);
            for (const [key, label] of Object.entries(labels)) {
                const row = document.createElement("div");
                row.className = "prob-row";
                row.dataset.head = head;
                row.dataset.action = key;
                row.innerHTML =
                    '<span></span><div class="prob-track"><div class="prob-fill"></div></div><span class="prob-value">—</span>';
                row.firstChild.textContent = label;
                group.append(row);
            }
            p.querySelector(".probabilities").append(group);
        }
        return p;
    });
    function addFeed(message, player = null) {
        feed.unshift({
            message,
            player,
            time: game.elapsed,
        });
        feed = feed.slice(0, 4);
        $("events").replaceChildren(
            ...feed.map((e) => {
                const li = document.createElement("li"),
                    t = document.createElement("time"),
                    s = document.createElement("span");
                t.textContent =
                    String(Math.floor(e.time / 60)).padStart(2, "0") +
                    ":" +
                    String(Math.floor(e.time % 60)).padStart(2, "0");
                s.textContent = e.message;
                if (e.player !== null)
                    s.className = e.player ? "feed-red" : "feed-blue";
                li.append(t, s);
                return li;
            }),
        );
    }
    function cancelSeat(player) {
        seatEpoch[player]++;
        pending[player]?.abort();
        pending[player] = null;
        next[player] = 0;
        adapter.clear(game.tanks[player]);
    }
    function cancel() {
        generation++;
        [0, 1].forEach(cancelSeat);
        keys.clear();
        taps.clear();
    }
    function reset(random = false) {
        cancel();
        const seed = random ? Date.now() >>> 0 : game.seed;
        game.reset(
            random && adapter.nextSeed ? adapter.nextSeed(game, seed) : seed,
        );
        paused = false;
        latest = [null, null];
        counts = [0, 0];
        dispatch = [[], []];
        record = [];
        feed = [];
        cursor = 0;
        lastRound = game.roundSerial;
        accumulator = 0;
        addFeed("Fresh paper. Fresh match.");
        update();
    }
    function pause(value) {
        paused = value;
        if (value) cancel();
        else {
            if (!ready()) {
                paused = true;
                update();
                return;
            }
            lastFrame = performance.now();
            accumulator = 0;
        }
        update();
    }
    function start() {
        if (!ready()) return;
        if (game.phase === "matchover") reset();
        error = "";
        paused = false;
        game.start();
        lastFrame = performance.now();
        accumulator = 0;
        canvas.focus({
            preventScroll: true,
        });
        update();
    }
    $("start").onclick = start;
    $("pause").onclick = () => pause(!paused);
    $("reset").onclick = () => reset();
    $("new-maze").onclick = () => reset(true);
    $("mode").onchange = (e) => {
        mode = e.target.value;
        reset();
    };
    $("trajectory").onclick = () => {
        trajectory = !trajectory;
        $("trajectory").setAttribute("aria-pressed", String(trajectory));
        $("trajectory").querySelector("span").textContent = trajectory
            ? "On"
            : "Off";
    };
    $("details").ontoggle = () => {
        $("details").querySelector("summary span").textContent = $("details")
            .open
            ? "Close ↑"
            : "Open ↓";
    };
    $("fullscreen").onclick = async () => {
        try {
            if (document.fullscreenElement) await document.exitFullscreen();
            else await document.documentElement.requestFullscreen();
        } catch {
            addFeed("Fullscreen is unavailable in this browser.");
        }
    };
    $("export").onclick = () => {
        const data = {
            created_at: new Date().toISOString(),
            edition,
            physics_revision: "full-footprint-v1",
            rules_revision: "loop-ammo-dodge-v3",
            map_revision: game.maze.revision || "maze-v1",
            players: seats.map((s, i) => ({
                player: i,
                ...s,
                model: health[i]?.metadata,
                assistance: adapter.assistance(game, i),
            })),
            mode,
            seed: game.seed,
            scores: game.scores,
            shots: game.shots,
            events: game.events,
            decisions: record,
        };
        const url = URL.createObjectURL(
            new Blob([JSON.stringify(data, null, 2)], {
                type: "application/json",
            }),
        );
        const a = document.createElement("a");
        a.href = url;
        a.download = `tank-war-${Date.now()}.json`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    const controlKeys = new Set([
        "KeyW",
        "KeyA",
        "KeyS",
        "KeyD",
        "Space",
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "Enter",
    ]);
    addEventListener("keydown", (e) => {
        if (e.target.matches("select,input,button,summary")) return;
        if (e.code === "KeyP" && !e.repeat && game.phase === "running") {
            pause(!paused);
            return;
        }
        if (
            mode !== "ai" &&
            !paused &&
            game.phase === "running" &&
            controlKeys.has(e.code)
        ) {
            e.preventDefault();
            keys.add(e.code);
            if (!e.repeat) taps.add(e.code);
        }
    });
    addEventListener("keyup", (e) => keys.delete(e.code));
    addEventListener("blur", () => {
        keys.clear();
        taps.clear();
    });
    document.addEventListener("visibilitychange", () => {
        if (
            document.hidden &&
            ["running", "roundover"].includes(game.phase) &&
            !paused
        )
            pause(true);
    });
    async function checkHealth() {
        await Promise.all(
            seats.map(async (seat, player) => {
                const selected = seat.backend,
                    protocol = seatProtocol(edition, seat),
                    ticket = ++healthSerial[player];
                if (!selected) {
                    health[player] = null;
                    return;
                }
                try {
                    const r = await fetch(
                        `/api/${edition}/health?backend=${encodeURIComponent(selected)}`,
                        {
                            signal: AbortSignal.timeout(4000),
                        },
                    );
                    const h = await r.json();
                    if (!r.ok || !validHealth(h, [protocol]))
                        throw Error(
                            "Model is offline or does not support the selected decision version",
                        );
                    if (ticket !== healthSerial[player]) return;
                    health[player] = h;
                } catch {
                    if (ticket !== healthSerial[player]) return;
                    health[player] = null;
                    if (isAI(player)) {
                        error = `${side(player)} model offline or incompatible`;
                        if (["running", "roundover"].includes(game.phase))
                            pause(true);
                    }
                }
            }),
        );
        if (ready()) error = "";
        update();
    }
    async function decide(player, now) {
        if (
            pending[player] ||
            now < next[player] ||
            !isAI(player) ||
            !health[player] ||
            !game.tanks[player].alive
        )
            return;
        const settings = {
            ...seats[player],
            interval:
                seats[player].intervalMode === "auto"
                    ? cadenceModels[player].interval
                    : seats[player].interval,
        };
        const { backend, controllerMode, interval } = settings;
        const observed = adapter.observe(game, player, {
                ...settings,
                rtt:
                    settings.intervalMode === "auto"
                        ? cadenceModels[player].prediction
                        : latest[player]?.rtt,
            }),
            round = game.roundSerial,
            epoch = generation,
            playerEpoch = seatEpoch[player],
            expectedHealth = health[player];
        const body = {
            ...observed.request,
            id: `${edition}-${game.seed}-${++serial}`,
            player,
            round_id: round,
            source: "tank-war",
            decision_interval_ms: interval,
            bullet_count: observed.bulletCount,
            physics_revision: "full-footprint-v1",
            rules_revision: "loop-ammo-dodge-v3",
            map_revision: game.maze.revision || "maze-v1",
            configuration: settings,
        };
        const ctl = new AbortController(),
            sent = performance.now();
        pending[player] = ctl;
        next[player] = sent + interval;
        const times = dispatch[player];
        times.push(sent);
        if (times.length > 21) times.shift();
        const cadence =
            times.length > 1 ? (sent - times[0]) / (times.length - 1) : null;
        const timer = setTimeout(() => ctl.abort(), 5000);
        try {
            const r = await fetch(
                `/api/${edition}/decision?backend=${encodeURIComponent(backend)}`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(body),
                    signal: ctl.signal,
                },
            );
            const response = await r.json();
            if (
                epoch !== generation ||
                playerEpoch !== seatEpoch[player] ||
                round !== game.roundSerial ||
                paused ||
                game.phase !== "running"
            )
                return;
            if (r.status === 429) {
                cadenceModels[player].busy();
                next[player] = performance.now() + interval;
                return;
            }
            if (!r.ok) throw Error(response.error || `HTTP ${r.status}`);
            if (
                !validDecision(response, body, expectedHealth) ||
                !adapter.validControls(response, observed)
            )
                throw Error(
                    "Model response did not match the current decision",
                );
            const rtt = performance.now() - sent,
                stale = rtt > 1600;
            cadenceModels[player].observe(rtt);
            record.push({
                player,
                round: game.round,
                elapsed: game.elapsed,
                backend,
                model_metadata: expectedHealth.metadata,
                configuration: settings,
                request: body,
                response,
                rtt_ms: rtt,
                actual_interval_ms: cadence,
                cadence_calibration: cadenceModels[player].stats,
                applied: !stale,
            });
            if (record.length > 20000) record.shift();
            if (stale) {
                next[player] = 0;
                addFeed(
                    `${side(player)} decision expired; reading the arena again.`,
                    player,
                );
                return;
            }
            const applied = adapter.apply(game, player, response, {
                ...settings,
            });
            record.at(-1).execution = applied;
            counts[player]++;
            latest[player] = {
                ...response,
                applied,
                rtt,
                cadence,
                observed,
            };
        } catch (e) {
            if (epoch !== generation || playerEpoch !== seatEpoch[player])
                return;
            health[player] = null;
            error = `${side(player)} model disconnected`;
            addFeed(error + ". Match paused.");
            pause(true);
            console.error(e);
        } finally {
            clearTimeout(timer);
            if (pending[player] === ctl) pending[player] = null;
        }
    }
    function update() {
        $("connection-light").className = ready() ? "ready" : "error";
        $("connection-title").textContent =
            mode === "local"
                ? "Local play ready"
                : ready()
                  ? "Models connected"
                  : error || "Connect a model";
        $("name-a").textContent = mode === "ai" ? modelName(0) : "Player 1";
        $("name-b").textContent = mode === "local" ? "Player 2" : modelName(1);
        $("score-a").textContent = game.scores[0];
        $("score-b").textContent = game.scores[1];
        $("round-label").textContent =
            "ROUND " +
            String(game.round).padStart(2, "0") +
            (game.maze.name ? " · " + game.maze.name.toUpperCase() : "");
        $("clock").textContent = Math.max(0, game.timeLeft).toFixed(1);
        $("live-tag").textContent = paused
            ? "PAUSED"
            : {
                  running: "LIVE MATCH",
                  roundover: "ROUND OVER",
                  matchover: "MATCH OVER",
                  ready: "READY",
              }[game.phase];
        $("live-tag").className =
            "board-corner" +
            (game.phase === "running" && !paused ? " live" : "");
        $("pause").disabled = !["running", "roundover"].includes(game.phase);
        $("pause").textContent = paused ? "Resume" : "Pause";
        setup?.render(
            health,
            isAI,
            cadenceModels.map((c) => c.stats),
        );
        $("assist-note").textContent =
            mode === "local"
                ? "Your shells can hit you. Make every bounce count."
                : seats
                      .map(
                          (s, i) =>
                              `${side(i)}: ${
                                  !isAI(i)
                                      ? "Keyboard"
                                      : "Dodge " +
                                        (s.assistEnabled
                                            ? {
                                                  light: "Reactive",
                                                  standard: "Balanced",
                                                  strong: "Protective",
                                              }[s.assistProfile]
                                            : "Off")
                              }`,
                      )
                      .join(" · ");
        $("keyboard-help").hidden = mode === "ai";
        $("keyboard-help").textContent = adapter.keyboard(mode);
        $("controller-description").textContent = seats
            .map(
                (s, i) =>
                    `${side(i)}: ${adapter.description(s.controllerMode)}`,
            )
            .join("\n");
        $("connection-detail").textContent =
            seats
                .map(
                    (s, i) =>
                        `${side(i)} · ${modelName(i)} · ${health[i] ? "connected" : "offline"} · ${s.intervalMode === "auto" ? "Auto " + cadenceModels[i].interval : s.interval} ms scheduling floor`,
                )
                .join(" | ") + ". Actual latency is measured below.";
        const overlay =
            paused || game.phase === "ready" || game.phase === "matchover";
        $("overlay").hidden = !overlay;
        $("round-result").hidden = game.phase !== "roundover" || paused;
        $("result-title").textContent =
            game.result === null
                ? "A draw. Go again."
                : `${side(game.result)} takes the round.`;
        $("result-title").style.color =
            game.result === null ? "#343e3d" : COLORS[game.result];
        if (overlay) {
            const over = game.phase === "matchover";
            $("overlay-title").textContent = over
                ? `${side(game.scores[0] > game.scores[1] ? 0 : 1)} wins the match.`
                : paused
                  ? "Take a breath."
                  : "Make your next move.";
            $("overlay-description").textContent = over
                ? `${game.scores[0]} : ${game.scores[1]} · First to five`
                : paused
                  ? error || "Ready when you are."
                  : "One hit ends the round. Your own shells count.";
            $("start").textContent =
                (over
                    ? "Play again"
                    : paused
                      ? "Resume match"
                      : !ready()
                        ? "Waiting for model"
                        : "Start match") + " →";
            $("start").disabled = !ready();
            $("overlay-note").textContent =
                mode === "ai"
                    ? `${modelName(0)} vs ${modelName(1)}`
                    : mode === "human"
                      ? `You are Blue · ${adapter.shortControls}`
                      : "Blue: WASD + Space · Red: arrows + Enter";
        }
        for (let id = 0; id < 2; id++) {
            const p = panels[id],
                t = game.tanks[id],
                data = latest[id],
                ai = isAI(id),
                chip = $(`player-${id}`);
            p.querySelector("h2").textContent =
                `${side(id)} · ${ai ? modelName(id) : "Player"}`;
            p.querySelector(".agent-status").textContent = !t.alive
                ? "Knocked out"
                : paused
                  ? "Paused"
                  : game.phase !== "running"
                    ? "Ready"
                    : !ai
                      ? "Keyboard"
                      : pending[id]
                        ? "Thinking"
                        : "Acting";
            chip.querySelector(".action").textContent = !t.alive
                ? "Knocked out"
                : paused
                  ? "Paused"
                  : adapter.action(t) + (t.assistActive ? " (dodge)" : "");
            chip.querySelector(".weapon").textContent = t.fireBlocked
                ? "Shot blocked"
                : t.pendingFire
                  ? "Shot queued"
                  : t.firing
                    ? "Fire"
                    : "Hold";
            const ammo = 5 - game.bullets.filter((b) => b.owner === id).length;
            chip.querySelectorAll(".ammo-slots i").forEach(
                (slot, i) => (slot.className = i < ammo ? "" : "used"),
            );
            p.querySelector(".aim-note").textContent = ai
                ? adapter.aimNote(data)
                : "Keyboard controls";
            p.querySelector(".execution-note").textContent =
                ai && data
                    ? adapter.execution(data)
                    : "Waiting for a decision.";
            const vals = {
                latency: ai && data ? Math.round(data.inference_ms) : "—",
                rtt: ai && data ? Math.round(data.rtt) : "—",
                cadence: ai && data?.cadence ? Math.round(data.cadence) : "—",
                count: counts[id],
                bullets: ai && data ? data.observed.bulletCount : 0,
                tokens: ai && data ? tokenLabel(data) : "—",
                context:
                    ai && data
                        ? data.observed.text +
                          (data.weapon_context
                              ? "\n\nCONDITIONAL WEAPON CONTEXT:\n" +
                                data.weapon_context
                              : "")
                        : "Waiting for a decision.",
            };
            for (const [key, value] of Object.entries(vals))
                p.querySelector("." + key).textContent = value;
            p.querySelectorAll(".prob-row").forEach((row) => {
                const head = row.dataset.head,
                    key = row.dataset.action,
                    value = ai ? data?.probabilities?.[head]?.[key] : null;
                row.classList.toggle(
                    "selected",
                    !!data && ai && adapter.selected(data, head) === key,
                );
                row.querySelector(".prob-fill").style.width =
                    `${value == null ? 0 : Math.max(0, Math.min(100, value * 100))}%`;
                row.querySelector(".prob-value").textContent =
                    value == null ? "—" : Math.round(value * 100) + "%";
            });
        }
    }
    function frame(now) {
        const dt = Math.min((now - lastFrame) / 1000, 0.12);
        lastFrame = now;
        if (!paused) {
            accumulator += dt;
            while (accumulator >= 1 / 60) {
                const manual = {},
                    held = (k) => keys.has(k) || taps.has(k);
                if (mode !== "ai") manual[0] = adapter.manual(0, mode, held);
                if (mode === "local") manual[1] = adapter.manual(1, mode, held);
                game.step(1 / 60, manual);
                taps.clear();
                accumulator -= 1 / 60;
            }
            if (lastRound !== game.roundSerial) {
                cancel();
                latest = [null, null];
                lastRound = game.roundSerial;
                dispatch = [[], []];
                addFeed(`Round ${game.round}. New maze.`);
            }
            for (let i = cursor; i < game.events.length; i++) {
                const e = game.events[i];
                if (e.type === "kill")
                    addFeed(
                        e.self
                            ? `${side(e.victim)} caught their own ricochet.`
                            : `${side(e.player)} landed the shot.`,
                        e.player,
                    );
                if (e.type === "assist")
                    addFeed(
                        `${side(e.player)} dodge assist intervened.`,
                        e.player,
                    );
                if (e.type === "plan_veto")
                    addFeed(
                        `${side(e.player)} cancelled an unsafe shot.`,
                        e.player,
                    );
            }
            cursor = game.events.length;
            if (ready() && game.phase === "running") {
                void decide(0, now);
                void decide(1, now);
            }
        }
        if (now - uiAt > 100) {
            update();
            uiAt = now;
        }
        render(game, trajectory);
        requestAnimationFrame(frame);
    }
    update();
    void checkHealth();
    setInterval(checkHealth, 10000);
    requestAnimationFrame(frame);
}
