import { AdaptiveCadence } from "./cadence.js";
import { MatchAccess } from "./match-access.js";
import { renderLobby } from "./lobby.js";
import { mountClassicSetup } from "../classic/setup.js";
import {
    defaultSeats,
    seatProtocol,
    classicVersionFor,
    humanChallengeSeats,
} from "./seat-config.js";
import { mountShell } from "./shell.js";
import { createRenderer, COLORS } from "./renderer.js";
import { mountPointerControls, freeMoveControls } from "./pointer-controls.js";
import { createSounds } from "./sound.js";
import { createAnnouncer } from "./announcer.js";
import {
    modelConfig,
    validHealth,
    validDecision,
    tokenLabel,
} from "./model-client.js";

export async function boot(adapter, { humanChallenge = false } = {}) {
    const { edition } = adapter;
    mountShell(edition, { humanChallenge });
    const $ = (id) => document.getElementById(id),
        canvas = $("arena"),
        game = new adapter.Game();
    adapter.initialize(game);
    const sounds = humanChallenge ? createSounds() : null,
        announcer = humanChallenge && $("announcer") ? createAnnouncer($("announcer"), $("announcer-corner")) : null;
    // A round counts for the modifiers leaderboard if modifiers were on at any moment of it.
    let modsUsed = false;
    if (sounds) {
        game.onBounce = () => sounds.bounce();
        game.onDryFire = (t) => t.id === 0 && sounds.dryFire();
        // Browsers start audio only after a gesture; the first click or key press unlocks it.
        for (const type of ["pointerdown", "keydown"]) addEventListener(type, () => sounds.unlock(), { capture: true });
    }
    const models = await modelConfig(edition);
    const classic = edition === "classic";
    let seats = defaultSeats(models, classic);
    let setup;
    if (humanChallenge) {
        seats = humanChallengeSeats(models);
        seats.forEach((seat, i) => game.setPlayerSettings(i, seat));
    } else if (classic) {
        setup = mountClassicSetup(models, game, (player, field) => {
            if (
                ["assistEnabled", "assistProfile", "fireGuardEnabled"].includes(
                    field,
                )
            ) {
                cancelSeat(player);
                game.emit("configuration", {
                    player,
                    field,
                    settings: { ...seats[player] },
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
    } else {
        $("model-backend").replaceChildren(
            ...(models.length
                ? models
                : [{ id: "", label: "No model connected" }]
            ).map((m) => new Option(m.label, m.id)),
        );
        $("model-backend").disabled = !models.length;
    }
    let mode = humanChallenge ? "human" : models.length ? "ai" : "local";
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
    const desktopInput = matchMedia("(hover: hover) and (pointer: fine)");
    let controlScheme = "steer";
    const keyboardEnabled = () => !humanChallenge || !desktopInput.matches || controlScheme !== "mouse";
    const pointer = humanChallenge && classic ? mountPointerControls({
        canvas, zone: $("joystick-zone"), stick: $("joystick"), knob: $("joystick-knob"), fire: $("touch-fire"),
        tank: () => game.tanks[0],
        enabled: () => (!desktopInput.matches || controlScheme !== "keyboard") && !paused && game.phase === "running" && game.tanks[0].alive &&
            (!admission || admission.hasRound(game.roundSerial)),
    }) : null;
    if (humanChallenge) {
        const applyControlScheme = () => {
            pointer?.reset(); keys.clear(); taps.clear();
            // The start dialog and the left panel show the same choice; either may change it at any time.
            for (const input of document.querySelectorAll('input[name="control-scheme"], input[name="panel-scheme"]'))
                input.checked = input.value === controlScheme;
            document.querySelector("main").dataset.controlScheme = desktopInput.matches ? controlScheme : "touch";
            document.querySelector(".pointer-hint").textContent = controlScheme === "hybrid"
                ? "WASD / arrows to move · Mouse to aim · Click to fire"
                : controlScheme === "steer"
                  ? "Mouse turns the hull · W forward · S back · Click or Space to fire · Esc to pause"
                  : "Move mouse to steer · Click to fire";
        };
        $("control-picker").addEventListener("change", (event) => {
            if (!["keyboard", "steer"].includes(event.target.value)) return;
            controlScheme = event.target.value;
            applyControlScheme();
        });
        desktopInput.addEventListener("change", () => { applyControlScheme(); update(); });
        applyControlScheme();
        const panel = $("mod-panel");
        panel.addEventListener("change", (event) => {
            if (event.target.name !== "panel-scheme") return;
            controlScheme = event.target.value;
            applyControlScheme();
            event.target.blur();
            canvas.focus({ preventScroll: true });
        });
        // The player's modifiers (game.mods[0], read by the engine), kept in this browser between visits.
        // shellLife is in seconds; the range slider's last step (31) is Infinity, which JSON keeps as null.
        // What "Turn on" starts from: a tougher tank (5 HP) that crushes walls and tanks, not invincible.
        const MOD_DEFAULTS = { noCooldown: false, maxShells: 5, invincible: false, hp: 5, shellRadius: 3, shellLife: 4.5, superTank: true };
        let stored = {};
        try { stored = JSON.parse(localStorage.getItem("tank-war-mods-v2") || "{}"); } catch {}
        if (stored.shellLife === null) stored.shellLife = Infinity;
        delete stored.selfImmune;
        const mods = { ...MOD_DEFAULTS, ...stored };
        // Modifiers start off on every visit: the panel shows "Too Hard!", and only a confirmation turns them on.
        const enableMods = (on) => {
            game.mods = on ? { 0: mods } : {};
            if (on) modsUsed = true;
            $("mod-cheats").hidden = !on;
            $("too-hard").hidden = on;
            $("mod-confirm").hidden = true;
        };
        enableMods(false);
        $("too-hard").addEventListener("click", () => { $("mod-confirm").hidden = false; $("mod-confirm-yes").focus(); });
        $("mod-confirm-yes").addEventListener("click", () => { enableMods(true); announcer?.say("POWER UP!", { spoken: "Power up!", tone: "cool", inCorner: true }); canvas.focus({ preventScroll: true }); });
        $("mod-confirm-no").addEventListener("click", () => { $("mod-confirm").hidden = true; canvas.focus({ preventScroll: true }); });
        $("mod-off").addEventListener("click", () => { enableMods(false); canvas.focus({ preventScroll: true }); });
        const modFields = [["noCooldown", "mod-no-cooldown"], ["maxShells", "mod-max-shells", 1, 60],
            ["invincible", "mod-invincible"], ["hp", "mod-hp", 1, 99], ["superTank", "mod-super-tank"],
            ["shellRadius", "mod-shell", 2, 24], ["shellLife", "mod-life", 1, 31]];
        const showLabels = () => {
            $("mod-shell-value").textContent = `${mods.shellRadius} px`;
            $("mod-life-value").textContent = mods.shellLife === Infinity ? "∞" : `${mods.shellLife} s`;
        };
        const showMods = () => {
            for (const [key, id] of modFields) {
                const field = $(id);
                if (field.type === "checkbox") field.checked = mods[key];
                else field.value = mods[key] === Infinity ? field.max : mods[key];
            }
            showLabels();
        };
        const readMods = () => {
            for (const [key, id, min, max] of modFields) {
                const field = $(id);
                if (field.type === "checkbox") mods[key] = field.checked;
                else if (key === "shellLife") mods[key] = Number(field.value) >= max ? Infinity : Number(field.value);
                else if (field.value !== "") mods[key] = Math.min(max, Math.max(min, Math.round(Number(field.value)) || min));
            }
            showLabels();
            try { localStorage.setItem("tank-war-mods-v2", JSON.stringify(mods)); } catch {}
        };
        panel.addEventListener("input", (event) => { if (event.target.name !== "panel-scheme") readMods(); });
        panel.addEventListener("change", (event) => {
            if (event.target.name === "panel-scheme") return;
            readMods(); showMods();
            if (event.target.type !== "number") event.target.blur();
        });
        $("mod-reset").addEventListener("click", () => { Object.assign(mods, MOD_DEFAULTS); showMods(); readMods(); $("mod-reset").blur(); });
        showMods();
        // Sound effects, on by default; the choice and volume are kept in this browser.
        let sound = { on: true, announcer: true, volume: 60 };
        try { sound = { ...sound, ...JSON.parse(localStorage.getItem("tank-war-sound") || "{}") }; } catch {}
        const applySound = () => {
            $("sound-on").checked = sound.on;
            $("announcer-on").checked = sound.announcer;
            announcer?.set({ enabled: sound.announcer, volume: sound.volume / 100 });
            $("sound-volume").value = sound.volume;
            $("sound-volume-value").textContent = `${sound.volume}%`;
            sounds.set({ muted: !sound.on, volume: sound.volume / 100 });
            try { localStorage.setItem("tank-war-sound", JSON.stringify(sound)); } catch {}
        };
        $("sound-on").addEventListener("change", (event) => { sound.on = event.target.checked; applySound(); sounds.unlock(); sounds.shot(0); event.target.blur(); });
        $("announcer-on").addEventListener("change", (event) => { sound.announcer = event.target.checked; applySound(); announcer?.say("READY?", { spoken: "Ready?" }); event.target.blur(); });
        $("sound-volume").addEventListener("input", (event) => { sound.volume = Number(event.target.value); applySound(); });
        $("sound-volume").addEventListener("change", (event) => { sounds.unlock(); sounds.shot(0); event.target.blur(); });
        applySound();
    }
    // The leaderboards (fair rounds and rounds with modifiers), top right; refreshed after each round.
    let board = "fair", boards = { fair: [], modified: [] };
    function renderLeaderboard() {
        const rows = boards[board] ?? [], list = $("board-rows");
        if (!list) return;
        list.replaceChildren(...rows.map((row, i) => {
            const item = document.createElement("li");
            item.innerHTML = `<b>${i + 1}</b><span class="board-name"></span><span class="board-score">${row.wins}<small>W</small> ${row.losses}<small>L</small></span>`;
            item.querySelector(".board-name").textContent = row.name;
            if (row.name === admission?.nickname) item.classList.add("you");
            return item;
        }));
        $("board-empty").hidden = rows.length > 0;
        for (const tab of document.querySelectorAll(".board-tabs [data-board]"))
            tab.setAttribute("aria-selected", String(tab.dataset.board === board));
    }
    async function loadLeaderboard() {
        if (!humanChallenge) return;
        try {
            const r = await fetch("/api/leaderboard", { cache: "no-store" });
            if (r.ok) boards = await r.json();
        } catch {}
        renderLeaderboard();
    }
    if (humanChallenge) {
        for (const tab of document.querySelectorAll(".board-tabs [data-board]"))
            tab.addEventListener("click", () => { board = tab.dataset.board; renderLeaderboard(); tab.blur(); });
    }
    // The player's name, bottom right: shown, edited in place, sent to the server and remembered here.
    let wantedName = "";
    try { wantedName = localStorage.getItem("tank-war-name") || ""; } catch {}
    let renaming = false, refusedName = "";
    function showName() {
        const shown = $("name-show");
        if (!shown || !$("name-form").hidden) return;
        const text = `${admission?.nickname || wantedName || "Pick a name"} ✎`;
        if (shown.textContent !== text) shown.textContent = text;
        if (admission?.nickname && wantedName && admission.nickname !== wantedName && !renaming && refusedName !== wantedName) {
            renaming = true;
            admission.rename(wantedName)
                .catch((error) => { refusedName = wantedName; nameError(error.message); })
                .finally(() => { renaming = false; renderLeaderboard(); });
        }
    }
    function nameError(message) {
        $("name-error").textContent = message;
        $("name-error").hidden = !message;
    }
    if (humanChallenge) {
        $("name-show").addEventListener("click", () => {
            $("name-input").value = admission?.nickname || wantedName;
            $("name-form").hidden = false;
            $("name-show").hidden = true;
            nameError("");
            $("name-input").focus();
            $("name-input").select();
        });
        const closeName = () => { $("name-form").hidden = true; $("name-show").hidden = false; canvas.focus({ preventScroll: true }); showName(); };
        $("name-input").addEventListener("keydown", (event) => {
            if (event.key === "Escape") { event.stopPropagation(); closeName(); }
        });
        $("name-form").addEventListener("submit", (event) => {
            event.preventDefault();
            const name = $("name-input").value.replace(/\s+/g, " ").trim();
            if (!/^[\p{L}\p{N} _.'!-]{1,20}$/u.test(name)) return nameError("Use 1 to 20 letters, digits or spaces.");
            wantedName = name;
            refusedName = "";
            try { localStorage.setItem("tank-war-name", name); } catch {}
            nameError("");
            closeName();
        });
    }
    const modelName = (i) =>
        models.find((m) => m.id === seats[i].backend)?.label ||
        health[i]?.metadata?.model_label ||
        "Model";
    const render = createRenderer(canvas, adapter.traceShot, {
        autoAim: edition === "auto-aim",
    });
    $("mode").value = mode;
    if (!models.length)
        for (const option of $("mode").options)
            option.disabled = option.value !== "local";
    const panels = [0, 1].map((i) => {
        const p = $(`agent-${i}`);
        if (!p) return null;
        p.innerHTML = classic
            ? `<div class="probabilities"></div>`
            : `<div class="agent-heading"><h2></h2><span class="agent-status">Ready</span></div><p class="aim-note model-note"></p><p class="execution-note model-note"></p><div class="probabilities"></div><div class="agent-metrics"><span>Decision <b class="latency">—</b> ms</span><span>Round trip <b class="rtt">—</b> ms</span><span>Actual interval <b class="cadence">—</b> ms</span><span><b class="count">0</b> decisions</span><span><b class="bullets">0</b> shells observed</span></div><details class="context-view"><summary>Model context <span class="tokens">—</span></summary><pre class="context">Waiting for a decision.</pre></details>`;
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
    let restartAfterQueue = false, roundStarting = false;
    const admission = humanChallenge
        ? new MatchAccess({
              onChange: () => update(),
              onActive: () => {
                  if (document.hidden || !ready()) {
                      admission.leave();
                      update();
                      return;
                  }
                  if (restartAfterQueue) { reset(true, true); restartAfterQueue = false; }
                  void start();
              },
              onQueued: () => {
                  cancel();
                  paused = true;
                  restartAfterQueue = true;
                  error = "";
                  update();
              },
              onLost: (message) => {
                  error = message;
                  if (["running", "roundover"].includes(game.phase))
                      pause(true);
                  update();
              },
          })
        : null;
    // The leaderboard and name chip read admission, so they start once it exists.
    if (humanChallenge) {
        void loadLeaderboard();
        setInterval(loadLeaderboard, 30000);
        showName();
    }
    function addFeed(message, player = null) {
        feed.unshift({ message, player, time: game.elapsed });
        feed = feed.slice(0, 4);
        if (!$("events")) return;
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
        pointer?.reset();
    }
    function reset(random = false, keepAdmission = false) {
        cancel();
        if (!keepAdmission) admission?.leave();
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
        if (!value && admission && !admission.active) {
            start();
            return;
        }
        paused = value;
        if (value) {
            cancel();
            admission?.leave();
        } else {
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
    async function start() {
        if (admission?.waiting) {
            admission.leave();
            update();
            return;
        }
        if (!ready()) return;
        if (game.phase === "matchover") reset();
        error = "";
        if (admission && !admission.active) {
            void admission.join();
            return;
        }
        if (admission && !admission.hasRound(game.roundSerial)) {
            if (roundStarting) return;
            roundStarting = true;
            let allowed;
            try { allowed = await admission.beginRound(game.roundSerial); }
            finally { roundStarting = false; }
            if (!allowed) return;
        }
        paused = false;
        game.start();
        lastFrame = performance.now();
        accumulator = 0;
        canvas.focus({ preventScroll: true });
        update();
    }
    $("start").onclick = start;
    $("pause").onclick = () => pause(!paused);
    $("reset").onclick = () => reset();
    $("new-maze").onclick = () => reset(true);
    $("mode").onchange = (e) => {
        if (humanChallenge) return;
        mode = e.target.value;
        reset();
    };
    if (!classic) {
        $("interval").onchange = (e) => {
            seats.forEach((s) => (s.interval = Number(e.target.value)));
        };
        $("model-backend").onchange = (e) => {
            seats.forEach((s) => (s.backend = e.target.value));
            health = [null, null];
            healthSerial = healthSerial.map((n) => n + 1);
            error = "Connecting model";
            reset();
            void checkHealth();
        };
        $("assist").onclick = () => {
            adapter.setAssist(game, !game.assistEnabled);
            seats.forEach((s) => (s.assistEnabled = game.assistEnabled));
            addFeed(
                `Dodge assist ${game.assistEnabled ? "enabled" : "disabled"}.`,
            );
            update();
        };
    }
    if (!classic) {
        $("trajectory").onclick = () => {
            trajectory = !trajectory;
            $("trajectory").setAttribute("aria-pressed", String(trajectory));
            $("trajectory").querySelector("span").textContent = trajectory
                ? "On"
                : "Off";
        };
        $("details").ontoggle = () => {
            $("details").querySelector("summary span").textContent = $(
                "details",
            ).open
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
    }
    if ($("export"))
        $("export").onclick = () => {
            const data = {
                created_at: new Date().toISOString(),
                edition,
                physics_revision: "full-footprint-v1",
                rules_revision: classic ? "loop-ammo-dodge-v3" : "auto-aim-v1",
                map_revision: game.maze.revision || "maze-v1",
                players: seats.map((s, i) => ({
                    player: i,
                    ...s,
                    version: classic ? classicVersionFor(s)?.id : undefined,
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
            a.download = `${classic ? "tank-war" : "ricochet-" + edition}-${Date.now()}.json`;
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
        if (e.code === "Escape" && humanChallenge && !$("mod-confirm").hidden) {
            $("mod-confirm").hidden = true;
            return;
        }
        if (e.code === "Escape" && !e.repeat && game.phase === "running" && !paused && e.target.id !== "name-input") {
            e.preventDefault();
            pause(true);
            return;
        }
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
            if (!keyboardEnabled()) return;
            if (!["hybrid", "steer"].includes(controlScheme) && !["Space", "Enter"].includes(e.code)) pointer?.releaseMouse();
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
        if (document.hidden && admission?.waiting) admission.leave(true);
        if (
            document.hidden &&
            ["running", "roundover"].includes(game.phase) &&
            !paused
        )
            pause(true);
    });
    addEventListener("pagehide", () => admission?.leave(true));
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
                        { signal: AbortSignal.timeout(4000) },
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
                        error = humanChallenge
                            ? "Opponent unavailable. Reconnecting…"
                            : `${side(player)} model offline or incompatible`;
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
            (admission && !admission.hasRound(game.roundSerial)) ||
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
            ...(admission ? {round_ticket: admission.roundTicket} : {}),
            source: "ricochet",
            decision_interval_ms: interval,
            bullet_count: observed.bulletCount,
            physics_revision: "full-footprint-v1",
            rules_revision: classic ? "loop-ammo-dodge-v3" : "auto-aim-v1",
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
                        ...(admission
                            ? { "X-Match-Token": admission.token }
                            : {}),
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
            if (
                r.status === 409 &&
                ["decision_expired", "decision_superseded"].includes(
                    response.code,
                )
            ) {
                next[player] = 0;
                return;
            }
            if (admission && response.code === "round_required") {
                error = "This round expired. Join the queue again.";
                pause(true);
                return;
            }
            if (
                admission &&
                (r.status === 401 ||
                    r.status === 410 ||
                    response.code === "match_queued")
            ) {
                error = "Your place expired. Resume to join again.";
                pause(true);
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
            latest[player] = { ...response, applied, rtt, cadence, observed };
        } catch (e) {
            if (epoch !== generation || playerEpoch !== seatEpoch[player])
                return;
            health[player] = null;
            error = humanChallenge
                ? "Connection lost. Reconnecting…"
                : `${side(player)} model disconnected`;
            addFeed(error + ". Match paused.");
            pause(true);
            console.error(e);
        } finally {
            clearTimeout(timer);
            if (pending[player] === ctl) pending[player] = null;
        }
    }
    function update() {
        if (humanChallenge) renderLobby($("arena-lobby"), admission);
        $("connection-light").className = ready() ? "ready" : "error";
        $("connection-title").textContent = humanChallenge
            ? ready()
                ? "Ready"
                : error || "Connecting…"
            : mode === "local"
              ? "Local play ready"
              : ready()
                ? "Models connected"
                : error || "Connect a model";
        const contestant = (i) =>
            classic
                ? `${modelName(i)} · ${classicVersionFor(seats[i])?.id.toUpperCase() || ""}`
                : modelName(i);
        $("name-a").textContent = humanChallenge
            ? admission?.nickname || "You"
            : mode === "ai"
              ? contestant(0)
              : "Player 1";
        $("name-b").textContent = humanChallenge
            ? "Opponent"
            : mode === "local"
              ? "Player 2"
              : contestant(1);
        $("score-a").textContent = game.scores[0];
        $("score-b").textContent = game.scores[1];
        $("round-label").textContent =
            "ROUND " +
            String(game.round).padStart(2, "0") +
            (admission?.active ? ` · ${admission.roundsUsed}/${admission.maxRounds}` : "") +
            (!classic && game.maze.name
                ? " · " + game.maze.name.toUpperCase()
                : "");
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
        if (!classic) {
            $("assist").disabled = mode === "local";
            $("assist").setAttribute(
                "aria-pressed",
                String(mode !== "local" && game.assistEnabled),
            );
            $("assist").querySelector("span").textContent =
                mode !== "local" && game.assistEnabled ? "On" : "Off";
        }
        if (!classic) {
            $("assist-note").textContent =
                mode === "local"
                    ? "Your shells can hit you. Make every bounce count."
                    : classic
                      ? seats
                            .map(
                                (s, i) =>
                                    `${side(i)}: ${!isAI(i) ? "Keyboard" : "Dodge " + (s.assistEnabled ? { light: "Reactive", standard: "Balanced", strong: "Protective" }[s.assistProfile] : "Off")}`,
                            )
                            .join(" · ")
                      : adapter.assistNote(game);
            $("keyboard-help").hidden = mode === "ai";
            $("keyboard-help").textContent = adapter.keyboard(mode);
            $("controller-description").textContent = classic
                ? seats
                      .map(
                          (s, i) =>
                              `${side(i)}: ${adapter.description(s.controllerMode)}`,
                      )
                      .join("\n")
                : adapter.description();
            $("connection-detail").textContent =
                seats
                    .map(
                        (s, i) =>
                            `${side(i)} · ${modelName(i)} · ${health[i] ? "connected" : "offline"} · ${s.intervalMode === "auto" ? "Auto " + cadenceModels[i].interval : s.interval} ms scheduling floor`,
                    )
                    .join(" | ") + ". Actual latency is measured below.";
        }
        const overlay =
            admission?.waiting ||
            paused ||
            game.phase === "ready" ||
            game.phase === "matchover";
        $("overlay").hidden = !overlay;
        $("round-result").hidden = game.phase !== "roundover" || paused;
        $("result-title").textContent =
            game.result === null
                ? "A draw. Go again."
                : `${side(game.result)} takes the round.`;
        $("result-title").style.color =
            game.result === null ? "#343e3d" : COLORS[game.result];
        if (humanChallenge) $("control-picker").hidden =
            !desktopInput.matches || !overlay || !!admission?.waiting || roundStarting;
        if (overlay) {
            const over = game.phase === "matchover";
            $("overlay-title").textContent = over
                ? `${side(game.scores[0] > game.scores[1] ? 0 : 1)} wins the match.`
                : paused
                  ? classic
                      ? "Paused"
                      : "Take a breath."
                  : classic
                    ? "Ready?"
                    : "Make your next move.";
            $("overlay-description").hidden = classic && !error && !over;
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
                        ? humanChallenge
                            ? "Connecting"
                            : "Waiting for model"
                        : "Start match") + " →";
            $("start").disabled = !ready();
            if (humanChallenge && error && !over) {
                $("overlay-description").hidden = false;
                $("overlay-description").textContent = error;
            }
            if (admission?.waiting) {
                $("overlay-title").textContent =
                    admission.state === "joining"
                        ? "Joining…"
                        : `Queue · #${admission.position}`;
                $("overlay-description").hidden = admission.state === "joining";
                $("overlay-description").textContent =
                    `${restartAfterQueue ? (admission.roundsUsed >= admission.maxRounds ? "Your 3 rounds are complete. " : "Your turn has ended. ") : ""}You're #${admission.position} in line. Play starts when a place is available.`;
                $("start").textContent = "Cancel";
                $("start").disabled = false;
                $("live-tag").textContent = "WAITING";
                $("pause").disabled = true;
            }
            if (!classic)
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
            if (!classic) {
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
            }
            const ammo = 5 - game.bullets.filter((b) => b.owner === id).length;
            chip.querySelectorAll(".ammo-slots i").forEach(
                (slot, i) => (slot.className = i < ammo ? "" : "used"),
            );
            if (!p) continue;
            if (!classic) {
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
                    cadence:
                        ai && data?.cadence ? Math.round(data.cadence) : "—",
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
            }
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
        if (!paused && admission?.active && game.phase === "running" && !admission.hasRound(game.roundSerial) && !roundStarting)
            void start();
        const roundBlocked = admission && (admission.finishing || roundStarting ||
            (game.phase === "running" && !admission.hasRound(game.roundSerial)));
        if (!paused && !roundBlocked) {
            accumulator += dt;
            while (accumulator >= 1 / 60) {
                const manual = {},
                    held = (k) => keys.has(k) || taps.has(k);
                if (mode !== "ai") {
                    manual[0] = adapter.manual(0, mode, (key) => keyboardEnabled() && held(key));
                    if (humanChallenge) manual[0].fastHumanTurn = true;
                    const input = pointer?.sample();
                    if (humanChallenge && desktopInput.matches && controlScheme === "hybrid") {
                        manual[0] = { ...freeMoveControls(held), freeMove: true, aimPoint: input?.aimPoint,
                            drive: 0, turn: 0, fire: !!input?.fire, fastHumanTurn: true };
                    } else if (humanChallenge && desktopInput.matches && controlScheme === "steer") {
                        // The mouse only turns the hull toward the cursor; W/S (or the arrows) drive along it.
                        const drive = Number(held("KeyW") || held("ArrowUp")) - Number(held("KeyS") || held("ArrowDown"));
                        manual[0] = { drive, turn: input?.active ? input.turn : 0, fire: !!input?.fire || held("Space"),
                            fastHumanTurn: true, responsiveTurn: true };
                    } else if (input?.active) {
                        manual[0].drive = input.drive;
                        manual[0].turn = input.turn;
                        manual[0].responsiveTurn = true;
                    }
                    if (input?.fire) manual[0].fire = true;
                }
                if (mode === "local") manual[1] = adapter.manual(1, mode, held);
                game.step(1 / 60, manual);
                taps.clear();
                accumulator -= 1 / 60;
                if (admission && (game.roundSerial !== lastRound || game.phase === "roundover")) break;
            }
            if (humanChallenge) showName();
            const mods = game.mods?.[0], hpLeft = $("mod-hp-left");
            if (mods && hpLeft) {
                const text = mods.invincible ? "Invincible: hits do no damage"
                    : `Hits left this round: ${Math.max(0, mods.hp - game.tanks[0].hits)} of ${mods.hp}`;
                if (hpLeft.textContent !== text) hpLeft.textContent = text;
            }
            if (lastRound !== game.roundSerial) {
                modsUsed = !!game.mods?.[0];
                sounds?.roundStart();
                if (mode === "human") {
                    announcer?.say(`ROUND ${game.round}`, { spoken: `Round ${game.round}`, interrupt: true });
                    announcer?.say("FIGHT!", { spoken: "Fight!", tone: "hot", delayMs: 1100 });
                }
                cancel();
                latest = [null, null];
                lastRound = game.roundSerial;
                dispatch = [[], []];
                addFeed(`Round ${game.round}. New maze.`);
            }
            for (let i = cursor; i < game.events.length; i++) {
                const e = game.events[i];
                if (announcer && mode === "human") {
                    if (e.type === "kill" && e.crush)
                        announcer.say("CRUSHED!", { spoken: "Crushed!", tone: "hot", interrupt: true });
                    else if (e.type === "kill")
                        announcer.say(e.self && e.victim === 0 ? "OOPS!" : "K.O.!", { spoken: e.self && e.victim === 0 ? "Oops!" : "K. O.!", tone: "hot", interrupt: true });
                    if (e.type === "near_miss" && e.victim === 0 && e.player === 1)
                        announcer.say("CLOSE ONE!", { spoken: "Close one!", tone: "cool", cooldownMs: 2500, inCorner: true });
                    if (e.type === "wall_break" && e.player === 0)
                        announcer.say("CRUSH!", { spoken: "Crush!", tone: "gold", cooldownMs: 3000, inCorner: true });
                    if (e.type === "round") {
                        const over = Math.max(...game.scores) >= 5, won = e.winner === 0;
                        const line = e.reason === "draw" ? ["DOUBLE K.O.!", "Double K. O.!", "hot"]
                            : e.reason === "timeout" ? ["TIME UP!", "Time up!", "gold"]
                            : over ? (won ? ["YOU WIN!", "You win!", "gold"] : ["GAME OVER", "Game over", "hot"])
                            : won && game.tanks[0].hits === 0 ? ["PERFECT!", "Perfect!", "gold"]
                            : won ? ["NICE SHOT!", "Nice shot!", "cool"] : ["YOU LOSE", "You lose", "hot"];
                        announcer.say(line[0], { spoken: line[1], tone: line[2], delayMs: e.reason === "timeout" ? 0 : 1300 });
                    }
                }
                if (sounds) {
                    if (e.type === "shot") sounds.shot(e.player);
                    if (e.type === "hit") game.mods?.[e.victim]?.invincible ? sounds.shield() : sounds.armor();
                    if (e.type === "kill") sounds.explosion();
                    if (e.type === "kill" && e.crush) sounds.crumble();
                    if (e.type === "wall_break") sounds.crumble();
                    if (e.type === "round") e.winner === 0 ? sounds.win() : e.winner === 1 ? sounds.lose() : sounds.draw();
                }
                if (e.type === "kill")
                    addFeed(
                        e.self
                            ? `${side(e.victim)} caught their own ricochet.`
                            : e.crush
                              ? `${side(e.player)} ran the other tank over.`
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
            if (admission?.active && game.phase === "roundover" && admission.roundTicket && !admission.finishing) {
                cancel();
                void admission.finishRound({ winner: game.result, modified: modsUsed }).then(() => setTimeout(loadLeaderboard, 400));
            }
            if (ready() && game.phase === "running" && (!admission || admission.hasRound(game.roundSerial))) {
                void decide(0, now);
                void decide(1, now);
            }
            if (game.phase === "matchover" && admission?.active)
                admission.leave();
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
