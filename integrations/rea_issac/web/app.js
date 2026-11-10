const $ = (id) => document.getElementById(id);
const canvas = $("game"),
    ctx = canvas.getContext("2d");
const W = 640,
    H = 416;
let meta,
    state,
    previous,
    stamp = 0,
    interval = 33,
    paused = false,
    floor,
    floorKey;
let keys = new Set(),
    taps = new Set(),
    bombPressed = false,
    journal = [],
    seen = new Set(),
    particles = [];
let labelTimer,
    toastTimer,
    lastHud = "",
    fault = false;
const esc = (text) =>
    String(text).replace(
        /[&<>"']/g,
        (x) =>
            ({
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#39;",
            })[x],
    );
const time = (tick) =>
    `${String(Math.floor(tick / 30 / 60)).padStart(2, "0")}:${String(
        Math.floor(tick / 30) % 60,
    ).padStart(2, "0")}`;
const hash = (n) => {
    n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
    n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
    return (n ^ (n >>> 16)) >>> 0;
};

async function api(path, body) {
    const response = await fetch(
        path,
        body === undefined
            ? { cache: "no-store" }
            : {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(body),
              },
    );
    const data = await response.json();
    if (!response.ok)
        throw Error(data.error || "The local game server did not respond.");
    return data;
}
function toast(text) {
    $("toast").textContent = text;
    $("toast").classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $("toast").classList.add("hidden"), 2300);
}
function accept(next) {
    const fresh = !meta || meta.run !== next.run;
    const entered =
        !state || state.room.id !== next.observation.room.id || fresh;
    previous = entered ? next.observation : state;
    meta = next;
    state = next.observation;
    stamp = performance.now();
    if (fresh) {
        seen.clear();
        journal = [];
        particles = [];
    }
    if (entered) {
        $("room-label").textContent = state.room.name;
        $("room-label").classList.add("show");
        clearTimeout(labelTimer);
        labelTimer = setTimeout(
            () => $("room-label").classList.remove("show"),
            1700,
        );
    }
    for (const event of state.events) {
        if (seen.has(event.id)) continue;
        seen.add(event.id);
        if (event.room !== state.room.id || state.tick - event.tick > 10)
            continue;
        if (
            event.kind === "pickup" ||
            event.kind === "clear" ||
            event.kind === "enter"
        ) {
            journal.unshift({ tick: event.tick, text: event.text });
            journal = journal.slice(0, 5);
            if (event.kind !== "enter") toast(event.text);
        }
        if (["splash", "defeat", "explosion", "hurt"].includes(event.kind)) {
            const count =
                event.kind === "explosion"
                    ? 30
                    : event.kind === "defeat"
                    ? 12
                    : 5;
            for (let i = 0; i < count; i++) {
                const n = hash(
                        Number(event.id.replace(/\D/g, "")) * 71 + i * 29,
                    ),
                    a = (n / 2 ** 32) * Math.PI * 2;
                particles.push({
                    x: event.x,
                    y: event.y,
                    vx: Math.cos(a) * (event.kind === "explosion" ? 3 : 1.3),
                    vy:
                        Math.sin(a) * (event.kind === "explosion" ? 3 : 1.3) -
                        0.5,
                    life: 1,
                    color:
                        event.kind === "explosion"
                            ? "#e4bc68"
                            : event.kind === "defeat"
                            ? "#b76d65"
                            : event.hostile
                            ? "#de7772"
                            : "#b1dbe4",
                });
            }
        }
    }
    hud();
}
function hud() {
    const p = state.player;
    $("room-name").textContent = state.room.name;
    $("clock").textContent = time(state.tick);
    $(
        "chambers",
    ).innerHTML = `${state.rooms_visited} / ${state.total_rooms}<span>EXPLORED</span>`;
    $("seed-label").textContent = `SEED ${state.seed}`;
    $("coins").textContent = p.coins;
    $("bombs").textContent = p.bombs;
    $("power").textContent = p.damage.toFixed(1);
    $("tears").textContent = (30 / p.shot_interval).toFixed(1) + "/s";
    $("pace").textContent = p.speed;
    $("kills").textContent = `${state.metrics.kills} defeated`;
    const healthKey = `${p.health}/${p.max_health}/${p.items.join(",")}`;
    if (healthKey !== lastHud) {
        $("hearts").innerHTML = Array.from(
            { length: Math.ceil(p.max_health / 2) },
            (_, i) =>
                `<span class="heart">♥<span style="width:${
                    Math.max(0, Math.min(2, p.health - i * 2)) * 50
                }%">♥</span></span>`,
        ).join("");
        $("hearts").setAttribute(
            "aria-label",
            `${p.health} of ${p.max_health} half hearts`,
        );
        $("inventory").innerHTML = p.items.length
            ? p.items
                  .map(
                      (item) =>
                          `<span class="item" title="${esc(
                              state.items[item][1],
                          )}"><b>◇</b>${esc(state.items[item][0])}</span>`,
                  )
                  .join("")
            : '<p class="empty-bag">A light bag. A long way down.</p>';
        lastHud = healthKey;
    }
    $("mode").innerHTML = `<i></i>${
        meta.controller === "human"
            ? "YOU ARE PLAYING"
            : meta.controller === "baseline"
            ? "BASELINE IS PLAYING"
            : "REPLAY"
    }`;
    $("bot").textContent =
        meta.controller === "baseline" ? "◉ Take control" : "▷ Watch baseline";
    $("bot").disabled = meta.controller === "replay";
    $("speed").disabled = meta.controller === "human";
    $("journal").innerHTML = journal.length
        ? journal
              .map(
                  (e) =>
                      `<p><time>${time(e.tick)}</time><span>${esc(
                          e.text,
                      )}</span></p>`,
              )
              .join("")
        : "<p>Every descent begins with a door.</p>";
    map();
    overlay();
}
function map() {
    const rooms = state.map,
        ids = new Map(rooms.map((r) => [r.id, r]));
    let svg =
        '<svg viewBox="0 0 200 140" role="img" aria-label="Discovered floor map">';
    for (const r of rooms)
        for (const d of [
            [1, 0],
            [0, 1],
        ]) {
            const other = ids.get(
                `${r.position[0] + d[0]},${r.position[1] + d[1]}`,
            );
            if (other && (r.visited || other.visited))
                svg += `<path d="M${100 + r.position[0] * 53} ${
                    70 + r.position[1] * 43
                } l${d[0] * 53} ${
                    d[1] * 43
                }" stroke="#6b5a3c" stroke-width="5"/>`;
        }
    for (const r of rooms) {
        const x = 100 + r.position[0] * 53,
            y = 70 + r.position[1] * 43,
            current = r.id === state.room.id;
        const color = current ? "#dec48e" : r.visited ? "#736347" : "#343026";
        svg += `<rect x="${x - 17}" y="${
            y - 12
        }" width="34" height="24" rx="3" fill="${color}" stroke="${
            current ? "#f6dfac" : r.visited ? "#9d8557" : "#5b503b"
        }" stroke-width="1.5"/>`;
        const symbol = current
            ? "•"
            : r.kind === "boss"
            ? "♛"
            : r.kind === "treasure"
            ? "◆"
            : r.kind === "shop"
            ? "¢"
            : r.cleared
            ? "·"
            : "";
        svg += `<text x="${x}" y="${y + 5}" text-anchor="middle" fill="${
            current ? "#322719" : "#e8d6a9"
        }" font-size="15">${symbol}</text>`;
    }
    $("map").innerHTML = svg + "</svg>";
}
function overlay() {
    let title = "",
        copy = "",
        eyebrow = "",
        button = "";
    if (state.status === "won") {
        title = "The hollow falls silent.";
        copy = `${state.metrics.kills} foes defeated · ${
            state.rooms_visited
        } chambers explored · ${time(state.tick)}`;
        eyebrow = "DESCENT COMPLETE";
        button = "New descent";
    } else if (state.status === "lost") {
        title = "A light goes out.";
        copy = `You reached ${state.rooms_visited} chambers. The next descent may be kinder.`;
        eyebrow = "THE PILGRIM HAS FALLEN";
        button = "Try again";
    } else if (state.status === "truncated") {
        title = "Time to surface.";
        copy =
            "This descent has reached its time limit. Your replay is ready to save.";
        eyebrow = "DESCENT ENDED";
        button = "New descent";
    } else if (meta.playback_complete) {
        title = "A story remembered.";
        copy = "Replay complete. Every recorded action has been shown.";
        eyebrow = "REPLAY COMPLETE";
        button = "New descent";
    } else if (paused) {
        title = "Take a breath.";
        copy = "The hollow can wait.";
        eyebrow = "PAUSED";
        button = "Continue";
    }
    $("overlay").classList.toggle("hidden", !title);
    if (title) {
        $("end-title").textContent = title;
        $("end-copy").textContent = copy;
        $("end-eyebrow").textContent = eyebrow;
        $("end-action").textContent = button;
    }
    $("pause").innerHTML = paused
        ? "▷ <span>Resume</span>"
        : "Ⅱ <span>Pause</span>";
}
function pause(value) {
    paused = value;
    keys.clear();
    taps.clear();
    bombPressed = false;
    if (state) overlay();
}
async function reset() {
    const seed = Number($("seed").value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) {
        toast("Choose a seed between 0 and 4294967295.");
        return;
    }
    accept(await api("/api/reset", { seed }));
    pause(false);
    $("viewport").focus();
}
$("pause").onclick = () => pause(!paused);
$("new-run").onclick = () => reset().catch((e) => toast(e.message));
$("end-action").onclick = () => {
    if (state.status === "running" && !meta.playback_complete) pause(false);
    else reset().catch((e) => toast(e.message));
};
$("bot").onclick = async () => {
    try {
        accept(
            await api("/api/controller", {
                controller:
                    meta.controller === "baseline" ? "human" : "baseline",
            }),
        );
        pause(false);
    } catch (e) {
        toast(e.message);
    }
};
$("export").onclick = async () => {
    try {
        const data = await api("/api/export");
        const url = URL.createObjectURL(
            new Blob([JSON.stringify(data)], { type: "application/json" }),
        );
        const a = document.createElement("a");
        a.href = url;
        a.download = `rea-issac-s${state.seed}-t${state.tick}.json`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        toast("Replay saved.");
    } catch (e) {
        toast(e.message);
    }
};
$("import").onclick = () => $("replay-file").click();
$("replay-file").onchange = async (e) => {
    try {
        const file = e.target.files[0];
        if (!file) return;
        pause(true);
        const record = JSON.parse(await file.text());
        accept(await api("/api/replay", record));
        pause(false);
        toast("Replay verified. Playing it back.");
    } catch (e) {
        toast(e.message);
    } finally {
        $("replay-file").value = "";
    }
};
window.addEventListener("keydown", (e) => {
    if (
        ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)
    )
        return;
    const code = e.code;
    if (
        ["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(
            code,
        )
    )
        e.preventDefault();
    if (code === "KeyP" && !e.repeat) {
        pause(!paused);
        return;
    }
    if (code === "Space" && !keys.has(code)) bombPressed = true;
    keys.add(code);
    taps.add(code);
});
window.addEventListener("keyup", (e) => keys.delete(e.code));
window.addEventListener("blur", () => keys.clear());
document.addEventListener("visibilitychange", () => {
    if (document.hidden) pause(true);
});

function ellipse(c, x, y, rx, ry, color) {
    c.fillStyle = color;
    c.beginPath();
    c.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    c.fill();
}
function rect(c, x, y, w, h, color, r = 0) {
    c.fillStyle = color;
    c.beginPath();
    if (r) c.roundRect(x, y, w, h, r);
    else c.rect(x, y, w, h);
    c.fill();
}
function path(c, points, color) {
    c.fillStyle = color;
    c.beginPath();
    c.moveTo(...points[0]);
    for (const p of points.slice(1)) c.lineTo(...p);
    c.closePath();
    c.fill();
}
function makeFloor(room) {
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const g = c.getContext("2d");
    rect(g, 0, 0, W, H, "#28231c");
    rect(g, 28, 36, W - 56, H - 72, "#655642");
    for (let y = 38, row = 0; y < H - 38; y += 29, row++)
        for (let x = 30 - (row % 2) * 18; x < W - 29; x += 37) {
            const n = hash(x * 101 + y * 83 + room.decoration_seed),
                v = n % 15;
            rect(
                g,
                x + 1,
                y + 1,
                35,
                27,
                `rgb(${91 + v},${78 + v},${59 + v})`,
                1,
            );
            rect(g, x + 2, y + 2, 33, 1, "#b39b6b16");
            if (n % 4 === 0) {
                g.strokeStyle = "#332d2335";
                g.lineWidth = 0.7;
                g.beginPath();
                g.moveTo(x + 9, y + 5);
                g.lineTo(x + 12, y + 13);
                g.lineTo(x + 7, y + 19);
                g.stroke();
            }
            for (let k = 0; k < 3; k++)
                rect(
                    g,
                    x + 3 + (hash(n + k) % 29),
                    y + 3 + (hash(n + k + 4) % 22),
                    1,
                    1,
                    "#d1b57810",
                );
        }
    for (let y = 0; y < H; y += 22)
        for (let x = 0; x < W; x += 40) {
            if (x > 12 && x < W - 43 && y > 20 && y < H - 27) continue;
            const v = hash(x * 83 + y * 41 + 3) % 9;
            rect(
                g,
                x + (y % 44 ? -18 : 0),
                y,
                38,
                20,
                `rgb(${62 + v},${55 + v},${42 + v})`,
                2,
            );
            rect(g, x + 3 + (y % 44 ? -18 : 0), y + 2, 32, 2, "#a18a5f24");
        }
    rect(g, 28, 34, W - 56, 3, "#c4a4713b");
    rect(g, 28, H - 38, W - 56, 5, "#211d17b0");
    rect(g, 25, 38, 5, H - 76, "#c4a47120");
    rect(g, W - 31, 38, 5, H - 76, "#211d17a0");
    const shade = g.createRadialGradient(CX(), CY(), 45, CX(), CY(), 350);
    shade.addColorStop(0, "#1b170d00");
    shade.addColorStop(1, "#0c0b08b0");
    g.fillStyle = shade;
    g.fillRect(0, 0, W, H);
    for (const x of [85, 555]) {
        const light = g.createRadialGradient(x, 47, 0, x, 47, 110);
        light.addColorStop(0, "#efad4730");
        light.addColorStop(1, "#efad4700");
        g.fillStyle = light;
        g.fillRect(x - 110, 0, 220, 170);
    }
    return c;
}
const CX = () => W / 2,
    CY = () => H / 2;
function door(d, t) {
    ctx.save();
    ctx.translate(d.x, d.y);
    ctx.rotate(
        { north: 0, east: Math.PI / 2, south: Math.PI, west: -Math.PI / 2 }[
            d.direction
        ],
    );
    const trim =
        d.style === "boss"
            ? "#a86d67"
            : d.style === "treasure"
            ? "#c8a352"
            : d.style === "shop"
            ? "#97a67b"
            : "#8f7a54";
    rect(ctx, -28, -19, 56, 37, "#2c261d", 3);
    rect(ctx, -25, -19, 50, 34, trim, 3);
    rect(ctx, -20, -18, 40, 32, "#171713", 3);
    if (!d.open) {
        for (let x = -14; x <= 14; x += 9) rect(ctx, x, -17, 3, 30, "#716b56");
        rect(ctx, -6, -1, 12, 10, "#b79b58", 2);
        ellipse(ctx, 0, 3, 1.5, 2, "#403520");
    } else {
        const glow = ctx.createLinearGradient(0, -13, 0, 34);
        glow.addColorStop(0, "#d9b56728");
        glow.addColorStop(1, "#d9b56700");
        ctx.fillStyle = glow;
        ctx.fillRect(-18, 0, 36, 35);
        rect(ctx, -19, 12, 38, 3, "#c5b08378");
    }
    ctx.fillStyle = trim;
    ctx.font = "11px Georgia";
    ctx.textAlign = "center";
    ctx.fillText(
        d.style === "boss"
            ? "♛"
            : d.style === "treasure"
            ? "◆"
            : d.style === "shop"
            ? "¢"
            : "·",
        0,
        -22,
    );
    ctx.restore();
}
function rock(r) {
    const [x, y, w, h] = r;
    ellipse(ctx, x + w / 2, y + h - 1, w * 0.65, 6, "#14130e70");
    path(
        ctx,
        [
            [x + 3, y + 5],
            [x + w - 6, y],
            [x + w, y + 9],
            [x + w - 2, y + h - 3],
            [x + 1, y + h],
            [x - 2, y + 13],
        ],
        "#6c6653",
    );
    path(
        ctx,
        [
            [x + 3, y + 5],
            [x + w - 6, y],
            [x + w - 4, y + 8],
            [x + 10, y + 13],
            [x - 2, y + 13],
        ],
        "#9c9277",
    );
    path(
        ctx,
        [
            [x + 10, y + 13],
            [x + w - 4, y + 8],
            [x + w - 2, y + h - 3],
            [x + 5, y + h - 4],
        ],
        "#79725e",
    );
    ctx.strokeStyle = "#4c493b";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + 12, y + 4);
    ctx.lineTo(x + 15, y + 13);
    ctx.lineTo(x + 11, y + 19);
    ctx.stroke();
}
function torch(x, t) {
    rect(ctx, x - 5, 35, 10, 12, "#28231c", 2);
    rect(ctx, x - 4, 35, 8, 3, "#9b8051");
    const n = Math.sin(t * 0.5 + x) * 1.4;
    ellipse(ctx, x, 29, 4 + n * 0.3, 8 + n, "#e6a447");
    ellipse(ctx, x, 30, 2, 5, "#ffe4a0");
    rect(ctx, x - 1, 34, 2, 6, "#b68146");
}
function pos(body, old, alpha) {
    const p = old?.find((e) => e.id === body.id);
    return p
        ? {
              ...body,
              x: p.x + (body.x - p.x) * alpha,
              y: p.y + (body.y - p.y) * alpha,
          }
        : body;
}
function pilgrim(p, t) {
    ctx.save();
    ctx.translate(p.x, p.y);
    const bob = Math.abs(p.vx) + Math.abs(p.vy) ? Math.sin(t * 0.5) * 1 : 0;
    ellipse(ctx, 1, 6, 11, 4, "#16140f80");
    if (t < p.invulnerable_until && Math.floor(t / 3) % 2 === 0)
        ctx.globalAlpha = 0.45;
    rect(ctx, -7, 3, 5, 5, "#4a4438", 2);
    rect(ctx, 3, 3, 5, 5, "#4a4438", 2);
    ellipse(ctx, 0, -1 + bob, 10, 10, "#b2ad91");
    path(
        ctx,
        [
            [-8, -1 + bob],
            [8, -1 + bob],
            [11, 6 + bob],
            [-10, 6 + bob],
        ],
        "#c9c6ad",
    );
    ellipse(ctx, 0, -13 + bob, 13, 13, "#e8e3c9");
    ellipse(ctx, 0, -11 + bob, 10, 9, "#c9b69b");
    ellipse(ctx, -6, -13 + bob, 4, 4, "#e9dfc5");
    ellipse(ctx, 6, -13 + bob, 4, 4, "#e9dfc5");
    const eye = p.facing_x * 2;
    ellipse(ctx, -4 + eye, -12 + bob, 2.3, 3, "#292a25");
    ellipse(ctx, 4 + eye, -12 + bob, 2.3, 3, "#292a25");
    ellipse(ctx, -3.5 + eye, -13 + bob, 0.7, 1, "#fff7dc");
    ellipse(ctx, 4.5 + eye, -13 + bob, 0.7, 1, "#fff7dc");
    rect(ctx, -6 + eye, -8 + bob, 2, 4, "#94c1c5", 1);
    rect(ctx, 3 + eye, -8 + bob, 2, 4, "#94c1c5", 1);
    rect(ctx, -1, -5 + bob, 3, 1, "#7c6350");
    ctx.restore();
}
function enemy(e, t) {
    const r = e.radius;
    ellipse(ctx, e.x, e.y + r * 0.45, r * 0.95, 5, "#17130f80");
    ctx.save();
    ctx.translate(e.x, e.y);
    const charging = e.kind !== "chaser" && e.next_shot - e.age < 17;
    if (charging) {
        ctx.strokeStyle = `rgba(224,104,80,${0.35 + Math.sin(t * 0.6) * 0.15})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.ellipse(0, 3, r + 8, r * 0.7 + 5, 0, 0, Math.PI * 2);
        ctx.stroke();
    }
    const flash = t < e.hit_until,
        bob = Math.sin(t * 0.12 + e.phase) * 1.2;
    if (e.kind === "fly") {
        const wing = 4 + Math.abs(Math.sin(t * 1.4)) * 7;
        ellipse(ctx, -12, -7, 11, wing, "#b9beb970");
        ellipse(ctx, 12, -7, 11, wing, "#b9beb970");
        ellipse(ctx, 0, -3 + bob, 11, 12, flash ? "#eee0bd" : "#655268");
        ellipse(ctx, -4, -6 + bob, 3.4, 4, "#de9c9e");
        ellipse(ctx, 4, -6 + bob, 3.4, 4, "#de9c9e");
        rect(ctx, -1, -1 + bob, 2, 5, "#28202b");
    } else if (e.kind === "boss") {
        ellipse(ctx, 0, 0, r, r * 0.8, flash ? "#eadabb" : "#4f4649");
        ellipse(
            ctx,
            0,
            -13 + bob,
            r * 0.86,
            r * 0.8,
            flash ? "#fff0ca" : "#aa9b81",
        );
        path(
            ctx,
            [
                [-21, -21],
                [-30, -37],
                [-14, -29],
            ],
            "#d8c8a5",
        );
        path(
            ctx,
            [
                [21, -21],
                [30, -37],
                [14, -29],
            ],
            "#d8c8a5",
        );
        ellipse(ctx, -10, -18 + bob, 7, 8, "#29232a");
        ellipse(ctx, 10, -18 + bob, 7, 8, "#29232a");
        ellipse(ctx, -10, -16 + bob, 2, 3, charging ? "#f3a86d" : "#bd6968");
        ellipse(ctx, 10, -16 + bob, 2, 3, charging ? "#f3a86d" : "#bd6968");
        ellipse(ctx, 0, -1 + bob, 12, 8, "#34242a");
        for (let x = -8; x <= 8; x += 5)
            rect(ctx, x, -7 + bob, 3, 5, "#d5c5a1");
    } else {
        ellipse(
            ctx,
            0,
            1,
            r,
            r * 0.72,
            flash ? "#e9d7b2" : e.kind === "chaser" ? "#80565a" : "#a66b59",
        );
        ellipse(
            ctx,
            0,
            -9 + bob,
            r * 0.9,
            r * 0.85,
            flash ? "#fff0c8" : e.kind === "chaser" ? "#b6998a" : "#c4987c",
        );
        ellipse(ctx, -4, -11 + bob, 3, 4, "#3c2c2a");
        ellipse(ctx, 4, -11 + bob, 3, 4, "#3c2c2a");
        if (e.kind === "spitter")
            ellipse(
                ctx,
                0,
                -3 + bob,
                charging ? 7 : 4,
                charging ? 6 : 4,
                "#4e2b28",
            );
        else {
            rect(ctx, -4, -4 + bob, 8, 3, "#493333", 1);
            rect(ctx, -2, -4 + bob, 2, 2, "#e2d3b2");
        }
        rect(ctx, -r * 0.7, r * 0.5, 5, 4, "#443b33", 2);
        rect(ctx, r * 0.35, r * 0.5, 5, 4, "#443b33", 2);
    }
    ctx.restore();
    if (e.health < e.max_health && e.kind !== "boss") {
        rect(ctx, e.x - 13, e.y + r + 8, 26, 3, "#252018", 1);
        rect(
            ctx,
            e.x - 13,
            e.y + r + 8,
            26 * Math.max(0, e.health / e.max_health),
            3,
            "#bb8570",
            1,
        );
    }
}
function pickup(p, t) {
    const y = p.y + Math.sin(t * 0.09 + Number(p.id.replace(/\D/g, ""))) * 2;
    ellipse(ctx, p.x, p.y + 7, 8, 3, "#211b1350");
    ctx.save();
    ctx.translate(p.x, y);
    if (p.kind === "coin") {
        ellipse(ctx, 0, 0, 6, 7, "#9c783b");
        ellipse(ctx, 0, -1, 5, 6, "#d3b26b");
        rect(ctx, -1, -5, 2, 7, "#a07a38");
        rect(ctx, -3, -4, 2, 1, "#f5d68c");
    } else if (p.kind === "heart") {
        ctx.fillStyle = "#d47f75";
        ctx.beginPath();
        ctx.moveTo(0, 6);
        ctx.bezierCurveTo(-14, -2, -6, -12, 0, -5);
        ctx.bezierCurveTo(6, -12, 14, -2, 0, 6);
        ctx.fill();
        ellipse(ctx, -3, -5, 2, 1, "#f0b29d");
    } else {
        rect(ctx, -13, 6, 26, 9, "#5b5440", 2);
        rect(ctx, -14, 5, 28, 3, "#9b8b65", 1);
        const glow = ctx.createRadialGradient(0, -5, 1, 0, -5, 31);
        glow.addColorStop(0, "#d3ac5940");
        glow.addColorStop(1, "#d3ac5900");
        ctx.fillStyle = glow;
        ctx.fillRect(-31, -35, 62, 62);
        ctx.strokeStyle = "#dfc18b";
        ctx.lineWidth = 2;
        if (p.item === "glass") {
            ctx.beginPath();
            ctx.arc(0, -4, 7, 0, Math.PI * 2);
            ctx.stroke();
            ctx.strokeStyle = "#abc9c5";
            ctx.beginPath();
            ctx.arc(-1, -5, 4, Math.PI, Math.PI * 1.6);
            ctx.stroke();
        } else if (p.item === "ribbon") {
            path(
                ctx,
                [
                    [-9, -10],
                    [0, -6],
                    [-8, 3],
                ],
                "#9cb3c5",
            );
            path(
                ctx,
                [
                    [9, -10],
                    [0, -6],
                    [8, 3],
                ],
                "#9cb3c5",
            );
            ellipse(ctx, 0, -5, 3, 4, "#c5d1d4");
        } else if (p.item === "boots") {
            rect(ctx, -7, -11, 5, 13, "#b88959", 1);
            rect(ctx, 2, -11, 5, 13, "#b88959", 1);
            rect(ctx, -8, -1, 7, 4, "#c59a67", 1);
            rect(ctx, 1, -1, 7, 4, "#c59a67", 1);
        } else {
            ellipse(ctx, 0, -3, 8, 9, "#b7a17a");
            rect(ctx, -5, -14, 10, 4, "#d9c59b", 1);
            rect(ctx, -3, -14, 6, 2, "#584b34");
        }
        if (p.price) {
            ctx.fillStyle = "#cbae6d";
            ctx.font = "9px Georgia";
            ctx.textAlign = "center";
            ctx.fillText(`${p.price} ¢`, 0, 27);
        }
    }
    ctx.restore();
}
function render(now) {
    requestAnimationFrame(render);
    if (!state) return;
    const t = state.tick,
        alpha = Math.min(1, (now - stamp) / Math.max(interval, 1));
    ctx.setTransform(2, 0, 0, 2, 0, 0);
    if (floorKey !== state.room.id + "-" + state.seed) {
        floor = makeFloor(state.room);
        floorKey = state.room.id + "-" + state.seed;
    }
    ctx.drawImage(floor, 0, 0);
    for (const stain of state.room.stains) {
        ellipse(
            ctx,
            stain[0],
            stain[1] + 3,
            stain[2] * 1.2,
            stain[2] * 0.45,
            "#603a324d",
        );
        for (let i = 0; i < 4; i++)
            ellipse(
                ctx,
                stain[0] + (hash(stain[3] + i) % 29) - 14,
                stain[1] + (hash(stain[3] + i + 5) % 13) - 6,
                2,
                1,
                "#63372e60",
            );
    }
    for (const d of state.doors) door(d, t);
    torch(85, t);
    torch(555, t);
    for (const r of state.room.rocks) rock(r);
    for (const p of state.room.pickups) pickup(p, t);
    for (const b of state.bombs) {
        ellipse(ctx, b.x, b.y + 5, 8, 3, "#1a171270");
        ellipse(
            ctx,
            b.x,
            b.y,
            7,
            8,
            b.explode_at - t < 12 && t % 5 < 3 ? "#df9472" : "#4d5557",
        );
        rect(ctx, b.x - 2, b.y - 11, 4, 4, "#9c986f");
        ellipse(ctx, b.x + 1, b.y - 12, 1.4 + Math.sin(t), 2, "#f1c267");
        if (b.explode_at - t < 20) {
            ctx.strokeStyle = "#df9c6650";
            ctx.beginPath();
            ctx.arc(b.x, b.y, 100, 0, Math.PI * 2);
            ctx.stroke();
        }
    }
    const actors = [
        ...state.room.enemies.map((e) => ({
            ...pos(e, previous?.room.enemies, alpha),
            isPlayer: false,
        })),
        {
            ...pos(state.player, previous ? [previous.player] : [], alpha),
            isPlayer: true,
        },
    ].sort((a, b) => a.y - b.y);
    for (const e of actors)
        if (e.isPlayer) pilgrim(e, t);
        else enemy(e, t);
    for (const raw of state.bullets) {
        const b = pos(raw, previous?.bullets, alpha);
        ellipse(ctx, b.x, b.y + 5, 3, 1.4, "#1a191040");
        ctx.save();
        ctx.translate(b.x, b.y);
        ctx.rotate(Math.atan2(b.vy, b.vx));
        ellipse(
            ctx,
            0,
            0,
            b.radius + 1,
            b.radius,
            b.hostile ? "#b64f4d" : "#75aebd",
        );
        ellipse(
            ctx,
            0,
            -0.7,
            b.radius - 1,
            b.radius - 1,
            b.hostile ? "#e19b87" : "#c0e6e6",
        );
        ellipse(ctx, -1, -1.2, 1, 1, "#fff3d9");
        ctx.restore();
    }
    for (const p of particles) {
        if (!paused && state.status === "running") {
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.025;
            p.life -= 0.035;
        }
        ctx.globalAlpha = Math.max(0, p.life);
        ellipse(ctx, p.x, p.y, 1.7, 1.3, p.color);
    }
    particles = particles.filter((p) => p.life > 0);
    ctx.globalAlpha = 1;
    const boss = state.room.enemies.find((e) => e.kind === "boss");
    if (boss) {
        rect(ctx, 205, 21, 230, 5, "#221b1a", 2);
        rect(
            ctx,
            206,
            22,
            228 * Math.max(0, boss.health / boss.max_health),
            3,
            "#bd7e6c",
            2,
        );
        ctx.fillStyle = "#d8bfa0";
        ctx.font = "8px Georgia";
        ctx.textAlign = "center";
        ctx.fillText("THE WARDEN", 320, 16);
    }
    if (state.room.kind === "start" && state.tick < 450) {
        ctx.fillStyle = "#dbc8a64f";
        ctx.font = "11px Georgia";
        ctx.textAlign = "center";
        ctx.fillText("A door opens. A descent begins.", 320, 293);
    }
}
async function loop() {
    const began = performance.now();
    let wait = 100;
    try {
        if (
            state &&
            !paused &&
            state.status === "running" &&
            !meta.playback_complete &&
            !fault
        ) {
            const held = (code) => keys.has(code) || taps.has(code);
            const action = {
                move_x: Number(held("KeyD")) - Number(held("KeyA")),
                move_y: Number(held("KeyS")) - Number(held("KeyW")),
                shoot_x: Number(held("ArrowRight")) - Number(held("ArrowLeft")),
                shoot_y: Number(held("ArrowDown")) - Number(held("ArrowUp")),
                bomb: bombPressed,
            };
            bombPressed = false;
            taps.clear();
            const before = state.tick;
            const next = await api("/api/step", {
                action,
                frames: meta.controller === "baseline" ? 3 : 1,
            });
            const elapsed = Math.max(1, next.observation.tick - before);
            const speed =
                meta.controller === "human" ? 1 : Number($("speed").value);
            interval = ((elapsed / 30) * 1000) / speed;
            accept(next);
            wait = interval;
        } else if (state && paused && !fault) {
            accept(await api("/api/state"));
            wait = 150;
        }
    } catch (e) {
        fault = true;
        pause(true);
        toast(e.message);
    }
    setTimeout(loop, Math.max(0, wait - (performance.now() - began)));
}
try {
    accept(await api("/api/state"));
    requestAnimationFrame(render);
    loop();
} catch (e) {
    toast(e.message);
    $("room-name").textContent = "Server unavailable";
}
