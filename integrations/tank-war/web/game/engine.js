import {
    TANK_RADIUS,
    moveTankPosition,
    advanceShell,
} from "../shared/collision.js";
export const W = 900,
    H = 630,
    CELL = 90,
    COLS = 10,
    ROWS = 7,
    RADIUS = TANK_RADIUS;
export const DRIVE = { forward: "Forward", reverse: "Reverse", stop: "Stop" };
export const TURNS = { left: "Left", right: "Right", none: "Straight" };
export const WEAPONS = { fire: "Fire", hold: "Hold" };
export const DRIVE_INPUT = { forward: 1, reverse: -1, stop: 0 },
    TURN_INPUT = { left: -1, right: 1, none: 0 };
export const SPEED = 125,
    TURN_SPEED = 2.5,
    PROTOCOL = "controls-v3";
export const CONTEXT_REVISION = "state-v7-avoidance-levels",
    MODEL_CONTROL_SECONDS = 0.6;
export const SHELL_LIFETIME = 4.5,
    FIRE_COOLDOWN = 0.3;
export const responseDelay = (ms) => clamp(ms / 1000, 1 / 60, 1.6);
export const COLORS = ["#357a9b", "#c4654b"];
export const controlLabel = (c) =>
    [DRIVE[c.drive], c.turn === "none" ? null : TURNS[c.turn]]
        .filter(Boolean)
        .join(" · ");
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
export const angleDelta = (a, b) =>
    Math.atan2(Math.sin(a - b), Math.cos(a - b));
export function rng(seed) {
    let s = seed >>> 0;
    return () => {
        s += 0x6d2b79f5;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
// Start with an open grid and insert mirrored walls while retaining connected
// loops. Unlike the original spanning-tree maze, no cell becomes a dead end.
export function makeMaze(seed) {
    const random = rng(seed),
        key = (a, b) => [Math.min(a, b), Math.max(a, b)].join("-"),
        links = Array.from({ length: COLS * ROWS }, () => []),
        candidates = [];
    for (let i = 0; i < COLS * ROWS; i++) {
        for (const j of [
            i % COLS < COLS - 1 ? i + 1 : -1,
            i < COLS * (ROWS - 1) ? i + COLS : -1,
        ]) {
            if (j < 0) continue;
            links[i].push(j);
            links[j].push(i);
            candidates.push([i, j]);
        }
    }
    const boundaries = [
        { x: 0, y: 0, w: W, h: 7 },
        { x: 0, y: H - 7, w: W, h: 7 },
        { x: 0, y: 0, w: 7, h: H },
        { x: W - 7, y: 0, w: 7, h: H },
    ];
    const wallFor = ([a, b]) =>
        b - a === 1
            ? {
                  x: ((a % COLS) + 1) * CELL - 4,
                  y: Math.floor(a / COLS) * CELL - 4,
                  w: 8,
                  h: CELL + 8,
              }
            : {
                  x: (a % COLS) * CELL - 4,
                  y: (Math.floor(a / COLS) + 1) * CELL - 4,
                  w: CELL + 8,
                  h: 8,
              };
    for (let i = candidates.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }
    const maze = {
        walls: boundaries,
        links,
        seed,
        name: `Loop Maze ${String((seed >>> 0) % 997).padStart(3, "0")}`,
        revision: "loop-mazes-v3",
    };
    // Paired walls can exceed this soft density target by one segment.
    const removed = new Set(),
        target = [28, 32, 36][(seed >>> 0) % 3];
    for (const [a, b] of candidates) {
        const pair = [
            ...new Map(
                [
                    [a, b],
                    [COLS * ROWS - 1 - b, COLS * ROWS - 1 - a],
                ].map((edge) => [key(...edge), edge]),
            ).values(),
        ];
        if (pair.some(([x, y]) => removed.has(key(x, y)))) continue;
        const walls = pair.map(wallFor);
        // Keep an inward-facing exit from both spawn regions.
        if (
            walls.some((w) =>
                [11, 58].some((i) => {
                    const p = center(i),
                        direction = i === 11 ? 1 : -1;
                    for (let d = 0; d <= 150; d += 3)
                        if (circleWall(p.x + direction * d, p.y, RADIUS, w))
                            return true;
                    return false;
                }),
            )
        )
            continue;
        for (const [x, y] of pair) {
            links[x] = links[x].filter((n) => n !== y);
            links[y] = links[y].filter((n) => n !== x);
        }
        const seen = new Set([0]),
            queue = [0];
        for (let i = 0; i < queue.length; i++)
            for (const n of links[queue[i]])
                if (!seen.has(n)) {
                    seen.add(n);
                    queue.push(n);
                }
        const safe =
            seen.size === COLS * ROWS &&
            links.every((l) => l.length >= 2) &&
            Array.from({ length: COLS * ROWS }, (_, i) => i).every(
                (i) =>
                    i === 11 || i === 58 || route(maze, 11, 58, i).length > 0,
            );
        if (!safe) {
            for (const [x, y] of pair) {
                links[x].push(y);
                links[y].push(x);
            }
            continue;
        }
        for (const [x, y] of pair) removed.add(key(x, y));
        maze.walls.push(...walls);
        if (removed.size >= target) break;
    }
    // Stable order makes tied navigation choices independent of rejected walls.
    links.forEach((l) => l.sort((a, b) => a - b));
    return maze;
}
export const center = (i) => ({
    x: ((i % COLS) + 0.5) * CELL,
    y: (Math.floor(i / COLS) + 0.5) * CELL,
});
export const cellOf = (t) =>
    clamp(Math.floor(t.y / CELL), 0, ROWS - 1) * COLS +
    clamp(Math.floor(t.x / CELL), 0, COLS - 1);
export function route(maze, start, end, avoid = -1) {
    const q = [start],
        prev = new Map([[start, null]]);
    for (let n = 0; n < q.length; n++) {
        const i = q[n];
        if (i === end) break;
        for (const j of maze.links[i])
            if (!prev.has(j) && j !== avoid) {
                prev.set(j, i);
                q.push(j);
            }
    }
    if (!prev.has(end)) return [];
    const out = [];
    for (let i = end; i !== start; i = prev.get(i)) out.push(i);
    return out.reverse();
}
export function circleWall(x, y, r, wall) {
    if (
        x <= wall.x - r ||
        x >= wall.x + wall.w + r ||
        y <= wall.y - r ||
        y >= wall.y + wall.h + r
    )
        return false;
    const dx = x - clamp(x, wall.x, wall.x + wall.w),
        dy = y - clamp(y, wall.y, wall.y + wall.h);
    return dx * dx + dy * dy < r * r;
}
export const blocked = (maze, x, y, r = RADIUS) =>
    maze.walls.some((w) => circleWall(x, y, r, w));
export const bulletHits = (b, t) =>
    !(b.owner === t.id && b.age < 0.15 && b.bounces === 0) &&
    distance(b, t) < RADIUS + 3;
const shellFrom = (t) => ({
    x: t.x + Math.cos(t.angle) * 23,
    y: t.y + Math.sin(t.angle) * 23,
    vx: Math.cos(t.angle) * 290,
    vy: Math.sin(t.angle) * 290,
    age: 0,
    bounces: 0,
    owner: t.id,
});
export function advanceBullet(b, dt, walls, onBounce = () => {}) {
    return advanceShell(b, dt, walls, onBounce);
}
export function traceShot(maze, tank, angle, maxTime = 2.8, target = null) {
    const b = shellFrom({ ...tank, angle });
    if (blocked(maze, b.x, b.y, 4))
        return {
            points: [{ x: tank.x, y: tank.y }],
            hit: false,
            selfHit: false,
            muzzleBlocked: true,
            bounces: 0,
            time: 0,
            firstBounceTime: null,
        };
    const points = [{ x: b.x, y: b.y }];
    let selfHit = false,
        hit = false,
        firstBounceTime = null;
    for (
        let k = 0;
        k < Math.ceil(Math.min(maxTime, SHELL_LIFETIME) * 240);
        k++
    ) {
        const prev = b.bounces;
        advanceBullet(b, 1 / 240, maze.walls);
        if (firstBounceTime === null && b.bounces) firstBounceTime = b.age;
        if (k % 10 === 0 || prev !== b.bounces) points.push({ x: b.x, y: b.y });
        if (bulletHits(b, tank)) {
            selfHit = true;
            break;
        }
        if (target && bulletHits(b, target)) {
            hit = true;
            break;
        }
    }
    return {
        points,
        hit,
        selfHit,
        muzzleBlocked: false,
        bounces: b.bounces,
        time: b.age,
        firstBounceTime,
        angle,
    };
}
// Cheap, conservative segment test for candidate generation. Every reported aim
// is then verified with traceShot's actual ricochet/self-hit physics.
function clearShotSegment(maze, a, b) {
    const dx = b.x - a.x,
        dy = b.y - a.y;
    for (const w of maze.walls) {
        let lo = 0,
            hi = 1;
        for (const [p, d, min, max] of [
            [a.x, dx, w.x - 3, w.x + w.w + 3],
            [a.y, dy, w.y - 3, w.y + w.h + 3],
        ]) {
            if (Math.abs(d) < 1e-9) {
                if (p <= min || p >= max) {
                    lo = 2;
                    break;
                }
                continue;
            }
            const u = (min - p) / d,
                v = (max - p) / d;
            lo = Math.max(lo, Math.min(u, v));
            hi = Math.min(hi, Math.max(u, v));
        }
        if (lo < hi && hi > 1e-6 && lo < 1 - 1e-6) return false;
    }
    return true;
}
export function findAimDirections(
    maze,
    tank,
    enemy,
    { horizon = 3, limit = 3 } = {},
) {
    const candidates = [],
        seen = new Set();
    const add = (image, bounce = null) => {
        const angle = Math.atan2(image.y - tank.y, image.x - tank.x),
            key = Math.round(angle * 1e6);
        if (
            seen.has(key) ||
            distance(tank, image) > 290 * horizon + 23 + RADIUS + 3
        )
            return;
        seen.add(key);
        const muzzle = {
            x: tank.x + Math.cos(angle) * 23,
            y: tank.y + Math.sin(angle) * 23,
        };
        if (blocked(maze, muzzle.x, muzzle.y, 4)) return;
        if (bounce) {
            if (
                distance(tank, bounce) < 24 ||
                !clearShotSegment(maze, muzzle, bounce) ||
                !clearShotSegment(maze, bounce, enemy)
            )
                return;
        } else if (!clearShotSegment(maze, muzzle, enemy)) return;
        candidates.push({
            angle,
            cost:
                Math.abs(angleDelta(angle, tank.angle)) / TURN_SPEED +
                distance(tank, image) / 290,
        });
    };
    add(enemy);
    for (const w of maze.walls) {
        for (const x of [w.x - 3, w.x + w.w + 3]) {
            const image = { x: 2 * x - enemy.x, y: enemy.y },
                u = (x - tank.x) / (image.x - tank.x),
                y = tank.y + u * (image.y - tank.y);
            if (u > 0 && u < 1 && y > w.y && y < w.y + w.h)
                add(image, { x, y });
        }
        for (const y of [w.y - 3, w.y + w.h + 3]) {
            const image = { x: enemy.x, y: 2 * y - enemy.y },
                u = (y - tank.y) / (image.y - tank.y),
                x = tank.x + u * (image.x - tank.x);
            if (u > 0 && u < 1 && x > w.x && x < w.x + w.w)
                add(image, { x, y });
        }
    }
    const options = [];
    for (const c of candidates.sort((a, b) => a.cost - b.cost).slice(0, 12)) {
        const shot = traceShot(maze, tank, c.angle, horizon, enemy);
        if (!shot.hit || shot.selfHit || shot.muzzleBlocked) continue;
        const delta = angleDelta(c.angle, tank.angle);
        options.push({
            angle: c.angle,
            heading_deg: +((c.angle * 180) / Math.PI).toFixed(1),
            delta_deg: +((delta * 180) / Math.PI).toFixed(1),
            turn:
                Math.abs(delta) < TURN_SPEED / 120
                    ? "none"
                    : delta > 0
                      ? "right"
                      : "left",
            turn_ms: Math.round((Math.abs(delta) / TURN_SPEED) * 1000),
            hit_ms: Math.round(shot.time * 1000),
            bounces: shot.bounces,
            kind: shot.bounces ? "ricochet" : "direct",
            self_hit: false,
        });
        if (options.length >= limit) break;
    }
    return {
        origin: {
            x: tank.x,
            y: tank.y,
            heading_deg: +((tank.angle * 180) / Math.PI).toFixed(1),
        },
        options,
        best: options[0] ?? null,
        assumption:
            "Enemy stationary; rotate in place, then shoot. Driving changes this firing solution.",
        search: "direct and one-bounce paths within 3 seconds; not exhaustive",
    };
}
export function aimContext(aim, label = "AIM AT CURRENT POSITION") {
    return (
        `${label}: ${aim.best ? "verified hit directions [heading_deg,delta_deg,turn,turn_ms,hit_ms,bounces]=" + JSON.stringify(aim.options.map((a) => [a.heading_deg, a.delta_deg, a.turn, a.turn_ms, a.hit_ms, a.bounces])) : "no verified direct/one-bounce hit direction; move toward the navigable waypoint and reassess"}. ` +
        `Angles are barrel headings, NOT the enemy bearing. Positive delta means RIGHT, negative LEFT. Rotate the hull toward the first verified direction; stop turning once aligned. ${aim.assumption} This is an aiming hint, not permission to fire now. `
    );
}
export function bulletForecast(game, bullet, p, horizon = 0.8) {
    const b = { ...bullet };
    let min = Infinity,
        closestTime = null,
        hitTime = null;
    for (let k = 0; k <= Math.ceil(horizon * 60); k++) {
        if (b.age >= SHELL_LIFETIME) break;
        // The short muzzle grace is identical to physics, not immunity to friendly fire.
        if (!(b.owner === p.id && b.age < 0.15 && b.bounces === 0)) {
            const d = distance(b, p),
                time = k / 60;
            if (d < min) {
                min = d;
                closestTime = time;
            }
            if (d < RADIUS + 3 && hitTime === null) hitTime = time;
        }
        advanceBullet(b, 1 / 60, game.maze.walls);
    }
    return { min, closestTime, hitTime };
}
export function dangerAt(game, p, horizon = 0.8) {
    let min = 999,
        near = 0;
    for (const bullet of game.bullets) {
        const forecast = bulletForecast(game, bullet, p, horizon);
        min = Math.min(min, forecast.min);
        if (forecast.min < 34) near++;
    }
    return { min, near };
}
export function movePosition(maze, p, dx, dy, dt, other = null) {
    moveTankPosition(maze, p, dx, dy, dt, other, SPEED);
}
function pathDistance(game, p, enemy) {
    const path = route(game.maze, cellOf(p), cellOf(enemy));
    return path.length
        ? distance(p, center(path[0])) +
              (path.length - 1) * CELL +
              distance(center(cellOf(enemy)), enemy)
        : distance(p, enemy);
}
export function tankMotion(maze, t, drive, turn, dt, other = null) {
    if (turn)
        t.angle = angleDelta(
            t.angle + clamp(turn, -1, 1) * (t.turnSpeed ?? TURN_SPEED) * dt,
            0,
        );
    movePosition(
        maze,
        t,
        Math.cos(t.angle) * clamp(drive, -1, 1),
        Math.sin(t.angle) * clamp(drive, -1, 1),
        dt,
        other,
    );
}
export function forecastCurrentControls(game, t, time) {
    if (
        assistEnabledFor(game, t.id) &&
        t.assistActive &&
        t.avoidance &&
        game.elapsed + time < t.avoidance.expiresAt
    )
        return { ...t.avoidance.controls };
    const original = t.assistActive && t.avoidance;
    return game.elapsed + time <= t.commandUntil
        ? {
              drive: original ? t.drive : t.executedDrive,
              turn:
                  game.elapsed + time <= (t.turnUntil ?? Infinity)
                      ? original
                          ? t.turn
                          : t.executedTurn
                      : "none",
          }
        : { drive: "stop", turn: "none" };
}
export function forecastMovement(
    game,
    id,
    controls,
    {
        horizon = 0.9,
        reactionDelay = 0.12,
        controlDuration = Infinity,
        turnDuration = Infinity,
    } = {},
) {
    const t = game.tanks[id],
        enemy = game.tanks[1 - id],
        p = { x: t.x, y: t.y, angle: t.angle },
        bullets = game.bullets.map((b) => ({ ...b }));
    let clearance = 999,
        hitTime = null,
        travel = 0;
    const dt = 1 / 120;
    for (let k = 0; k < Math.ceil(horizon / dt); k++) {
        const time = k * dt,
            before = { ...p };
        const command =
            time < reactionDelay
                ? forecastCurrentControls(game, t, time)
                : time < reactionDelay + controlDuration
                  ? {
                        drive: controls.drive,
                        turn:
                            time < reactionDelay + turnDuration
                                ? controls.turn
                                : "none",
                    }
                  : { drive: "stop", turn: "none" };
        tankMotion(
            game.maze,
            p,
            DRIVE_INPUT[command.drive],
            TURN_INPUT[command.turn],
            dt,
            enemy,
        );
        travel += distance(p, before);
        for (const b of bullets) {
            if (b.age >= SHELL_LIFETIME) continue;
            advanceBullet(b, dt, game.maze.walls);
            if (b.owner === id && b.age < 0.15 && b.bounces === 0) continue;
            const margin = distance(b, p) - (RADIUS + 3);
            clearance = Math.min(clearance, margin);
            if (margin < 0 && hitTime === null) hitTime = time + dt;
        }
    }
    return {
        ...controls,
        x: p.x,
        y: p.y,
        angle: p.angle,
        clearance,
        hitTime,
        travel,
        progress: pathDistance(game, t, enemy) - pathDistance(game, p, enemy),
    };
}
// Proposed shot, including motion during inference and the first new control step.
// Uses the same 60 Hz hull update and four shell substeps as Game.step.
export function forecastShot(
    game,
    id,
    controls,
    {
        horizon = 1.2,
        reactionDelay = 0.12,
        controlDuration = Infinity,
        turnDuration = Infinity,
        fireDelay = 0,
    } = {},
) {
    const t = game.tanks[id],
        enemy = game.tanks[1 - id],
        p = { ...t },
        dt = 1 / 60;
    let b = null,
        attempted = false,
        selfHitTime = null,
        enemyHitTime = null,
        firstBounceTime = null,
        launchTime = null,
        launchAngle = null;
    for (
        let k = 0;
        k < Math.ceil((reactionDelay + fireDelay + horizon) / dt);
        k++
    ) {
        const time = (k + 1) * dt,
            ready = k * dt + 1e-8 >= reactionDelay;
        const c = ready
            ? k * dt < reactionDelay + controlDuration
                ? {
                      drive: controls.drive,
                      turn:
                          k * dt < reactionDelay + turnDuration
                              ? controls.turn
                              : "none",
                  }
                : { drive: "stop", turn: "none" }
            : forecastCurrentControls(game, t, time);
        tankMotion(
            game.maze,
            p,
            DRIVE_INPUT[c.drive],
            TURN_INPUT[c.turn],
            dt,
            enemy,
        );
        if (k * dt + 1e-8 >= reactionDelay + fireDelay && !attempted) {
            attempted = true;
            const shell = shellFrom(p),
                ammo = game.bullets.filter(
                    (b) => b.owner === id && b.age + time < SHELL_LIFETIME,
                ).length;
            if (
                t.cooldown > time ||
                ammo >= 5 ||
                blocked(game.maze, shell.x, shell.y, 4)
            )
                break;
            b = shell;
            launchTime = k * dt;
            launchAngle = p.angle;
        }
        if (!b) continue;
        for (let s = 0; s < 4; s++) {
            advanceBullet(b, dt / 4, game.maze.walls);
            if (firstBounceTime === null && b.bounces) firstBounceTime = b.age;
            if (bulletHits(b, p)) {
                selfHitTime = b.age;
                break;
            }
            if (bulletHits(b, enemy)) {
                enemyHitTime = b.age;
                break;
            }
        }
        if (selfHitTime !== null || enemyHitTime !== null) break;
    }
    return {
        ...controls,
        canFire: b !== null,
        selfHitTime,
        enemyHitTime,
        firstBounceTime,
        launchTime,
        launchAngle,
    };
}
export const AVOIDANCE_PROFILES = Object.freeze({
    light: Object.freeze({
        label: "Reactive",
        interval: 0.06,
        horizon: 0.65,
        trigger: 0.48,
        maxDuration: 0.4,
        cooldown: 0.28,
        minHold: 0.1,
        releaseMargin: 3,
        instantRefresh: false,
    }),
    standard: Object.freeze({
        label: "Balanced",
        interval: 0.05,
        horizon: 0.8,
        trigger: 0.62,
        maxDuration: 0.5,
        cooldown: 0.2,
        minHold: 0.12,
        releaseMargin: 5,
        instantRefresh: false,
    }),
    strong: Object.freeze({
        label: "Protective",
        interval: 1 / 30,
        horizon: 0.95,
        trigger: 0.78,
        maxDuration: 0.65,
        cooldown: 0.12,
        minHold: 0.14,
        releaseMargin: 7,
        instantRefresh: true,
    }),
});
export const assistEnabledFor = (game, id) =>
    game.playerSettings?.[id]?.assistEnabled ?? game.assistEnabled;
export const assistLevelFor = (game, id) =>
    game.playerSettings?.[id]?.assistProfile ?? game.assistProfile ?? "light";
export const shotGuardFor = (game, id) =>
    game.playerSettings?.[id]?.fireGuardEnabled ?? game.fireGuardEnabled;
export const avoidanceProfile = (game, id) =>
    AVOIDANCE_PROFILES[assistLevelFor(game, id)];
const stopped = { drive: "stop", turn: "none" };
const sameMotion = (a, b) => a.drive === b.drive && a.turn === b.turn;
// Shared shell paths keep nine local forecasts cheap. Physics uses exactly the
// same 60 Hz hull motion, four shell substeps, bounce rules and muzzle grace.
export function buildAvoidance(game, id) {
    const t = game.tanks[id],
        enemy = game.tanks[1 - id],
        profile = avoidanceProfile(game, id),
        dt = 1 / 60,
        steps = Math.round(profile.horizon / dt);
    const remaining = t.avoidance
        ? Math.max(0, t.avoidance.expiresAt - game.elapsed)
        : profile.maxDuration;
    const originalAt = (time) =>
        game.elapsed + time <= t.commandUntil + 1e-8
            ? {
                  drive: t.drive,
                  turn:
                      game.elapsed + time <= (t.turnUntil ?? Infinity) + 1e-8
                          ? t.turn
                          : "none",
              }
            : stopped;
    const shells = game.bullets.filter((b) => !b.dead).map((b) => ({ ...b })),
        frames = [];
    for (let k = 0; k < steps; k++) {
        const samples = [];
        for (const b of shells) {
            if (b.age >= SHELL_LIFETIME) continue;
            for (let s = 0; s < 4; s++) {
                advanceBullet(b, dt / 4, game.maze.walls);
                if (!(b.owner === id && b.age < 0.15 && b.bounces === 0))
                    samples.push({
                        id: b.id,
                        owner: b.owner,
                        x: b.x,
                        y: b.y,
                        time: k * dt + ((s + 1) * dt) / 4,
                    });
            }
        }
        frames.push(samples);
    }
    const evaluate = (controls, bounded = false) => {
        const p = { ...t };
        let hitTime = null,
            threat = null,
            clearance = 999,
            travel = 0;
        for (let k = 0; k < steps; k++) {
            const time = k * dt,
                before = { x: p.x, y: p.y };
            // Light/standard forecasts include the mandatory return to the model's
            // original command after the short burst; they do not assume endless escape.
            const c =
                bounded || time + 1e-8 >= remaining
                    ? originalAt(time)
                    : controls;
            tankMotion(
                game.maze,
                p,
                DRIVE_INPUT[c.drive],
                TURN_INPUT[c.turn],
                dt,
                enemy,
            );
            travel += distance(p, before);
            for (const b of frames[k]) {
                const margin = distance(b, p) - (RADIUS + 3);
                clearance = Math.min(clearance, margin);
                if (margin < 0 && (hitTime === null || b.time < hitTime)) {
                    hitTime = b.time;
                    threat = { id: b.id, owner: b.owner };
                }
            }
        }
        return {
            controls: { ...controls },
            hitTime,
            threat,
            clearance,
            travel,
            x: p.x,
            y: p.y,
            angle: p.angle,
        };
    };
    const requested =
        game.elapsed <= t.commandUntil
            ? {
                  drive: t.drive,
                  turn:
                      game.elapsed > (t.turnUntil ?? Infinity) + 1e-8
                          ? "none"
                          : t.turn,
              }
            : { ...stopped };
    const proposed = evaluate(requested, true);
    const options = Object.keys(DRIVE).flatMap((drive) =>
        Object.keys(TURNS).map((turn) => evaluate({ drive, turn })),
    );
    return { requested, proposed, options };
}
export function chooseAvoidance(game, id) {
    const t = game.tanks[id],
        profile = avoidanceProfile(game, id),
        report = buildAvoidance(game, id),
        { proposed, requested, options } = report;
    if (
        !t.avoidance &&
        (proposed.hitTime === null || proposed.hitTime > profile.trigger)
    )
        return { ...report, choice: null };
    // Hysteresis reduces wobble, while the hard burst deadline still takes priority.
    const holding = t.avoidance && game.elapsed < t.avoidance.holdUntil;
    if (
        proposed.hitTime === null &&
        (!t.avoidance ||
            (proposed.clearance >= profile.releaseMargin && !holding))
    )
        return { ...report, choice: null };
    const previous =
        t.avoidance &&
        options.find((f) => sameMotion(f.controls, t.avoidance.controls));
    const preference = (f) =>
        Math.min(f.clearance, 60) -
        (f.controls.drive !== requested.drive ? 8 : 0) -
        (f.controls.turn !== requested.turn ? 4 : 0) -
        (f.controls.turn === "none" ? 0 : 2);
    options.sort(
        (a, b) =>
            (b.hitTime === null) - (a.hitTime === null) ||
            (b.hitTime ?? profile.horizon) - (a.hitTime ?? profile.horizon) ||
            preference(b) - preference(a),
    );
    const best =
        previous?.hitTime === null &&
        previous.clearance >= profile.releaseMargin
            ? previous
            : options[0];
    const improves =
        best.hitTime === null ||
        (proposed.hitTime !== null && best.hitTime > proposed.hitTime + 0.025);
    // If already safe, never force a minimum hold along a newly dangerous path.
    const choice =
        improves && (t.avoidance || proposed.hitTime !== null) ? best : null;
    return { ...report, choice };
}
export function avoidanceContext(game, id) {
    const t = game.tanks[id],
        p = avoidanceProfile(game, id),
        ms = (n) => Math.round(n * 1000);
    return (
        `Shot guard=${shotGuardFor(game, id) ? "ON" : "OFF"}. ` +
        `Local bullet avoidance=${assistEnabledFor(game, id) ? "ON" : "OFF"}; active=${!!t.assistActive}; requested=${t.drive}/${t.turn}, actual=${t.executedDrive}/${t.executedTurn}. ` +
        (assistEnabledFor(game, id)
            ? `Level=${assistLevelFor(game, id)}; checks every ${ms(p.interval)}ms, forecast ${ms(p.horizon)}ms, intervenes only for collision within ${ms(p.trigger)}ms. ` +
              `Maximum burst=${Number.isFinite(p.maxDuration) ? ms(p.maxDuration) + "ms" : "unlimited"}, cooldown=${ms(p.cooldown)}ms; remaining cooldown=${ms(Math.max(0, (t.assistCooldownUntil ?? 0) - game.elapsed))}ms. ` +
              `It checks ALL existing shells, uses normal hull physics, and cancels pending fire on path override. Response-position forecasts resume your original command after a burst expires. Future interventions are unknown. `
            : "") +
        `Last avoidance threat=${t.lastAvoidance?.threat?.id ?? "none"}; forecast hit=${t.lastAvoidance?.hit_ms == null ? "none" : t.lastAvoidance.hit_ms + "ms"}. `
    );
}
export function buildObservation(
    game,
    id,
    { reactionDelay = 0.3, controlDuration = MODEL_CONTROL_SECONDS } = {},
) {
    const t = game.tanks[id],
        e = game.tanks[1 - id],
        shot = traceShot(game.maze, t, t.angle, 2.8, e),
        ammo = 5 - game.bullets.filter((b) => b.owner === id).length;
    const aim = findAimDirections(game.maze, t, e);
    const round = (n) => Number(n.toFixed(1));
    const degrees = (a) => round((angleDelta(a, 0) * 180) / Math.PI);
    const bullets = game.bullets.map((b) => {
        const f = bulletForecast(game, b, t);
        return {
            id: b.id,
            owner: b.owner === id ? "self" : "enemy",
            x: round(b.x),
            y: round(b.y),
            vx: round(b.vx),
            vy: round(b.vy),
            bounces: b.bounces,
            life_s: round(Math.max(0, SHELL_LIFETIME - b.age)),
            closest_px: Number.isFinite(f.min) ? round(f.min) : null,
            closest_s:
                f.closestTime === null
                    ? null
                    : Number(f.closestTime.toFixed(3)),
            hit_s: f.hitTime === null ? null : Number(f.hitTime.toFixed(3)),
        };
    });
    const forecasts = Object.keys(DRIVE).flatMap((drive) =>
        Object.keys(TURNS).map((turn) =>
            forecastMovement(
                game,
                id,
                { drive, turn },
                {
                    reactionDelay,
                    controlDuration,
                    horizon: reactionDelay + controlDuration,
                },
            ),
        ),
    );
    const firingForecasts = Object.keys(DRIVE).flatMap((drive) =>
        Object.keys(TURNS).map((turn) =>
            forecastShot(
                game,
                id,
                { drive, turn },
                { reactionDelay, controlDuration, horizon: 3 },
            ),
        ),
    );
    const threatened =
        forecasts.find((f) => f.drive === "stop" && f.turn === "none")
            .hitTime !== null;
    const path = route(game.maze, cellOf(t), cellOf(e)),
        waypoint = path.length ? center(path[0]) : e;
    const waypointError = degrees(
        Math.atan2(waypoint.y - t.y, waypoint.x - t.x) - t.angle,
    );
    const enemyError = degrees(Math.atan2(e.y - t.y, e.x - t.x) - t.angle);
    const risk = (f) =>
        `existing_shell_hit=${f.hitTime === null ? "none_in_horizon" : Math.round(f.hitTime * 1000) + "ms"} (does NOT include firing), travel ${Math.round(f.travel)}px, route progress ${Math.round(f.progress)}px`;
    const drive = Object.keys(DRIVE).map((key) => ({
        id: key,
        description:
            `${key === "forward" ? "Drive FORWARD along the hull heading" : key === "reverse" ? "REVERSE opposite the hull heading, keeping the cannon facing forward" : "Stop translation; turning and firing remain independent"}. ` +
            forecasts
                .filter((f) => f.drive === key)
                .map((f) => `${f.turn} turn: ${risk(f)}`)
                .join("; "),
    }));
    const turn = Object.keys(TURNS).map((key) => ({
        id: key,
        description:
            `${key === "none" ? "Keep current hull and barrel angle" : `Rotate hull AND fixed barrel ${key === "left" ? "LEFT (counterclockwise)" : "RIGHT (clockwise)"} at 143 degrees/s`}. ` +
            `${key === "none" ? "" : `Route waypoint is ${waypointError < 0 ? "LEFT" : "RIGHT"} by ${Math.abs(waypointError)} degrees. `}` +
            forecasts
                .filter((f) => f.turn === key)
                .map((f) => `${f.drive}: ${risk(f)}`)
                .join("; "),
    }));
    const canShoot =
        ammo > 0 &&
        t.cooldown <= 0 &&
        !blocked(
            game.maze,
            t.x + Math.cos(t.angle) * 23,
            t.y + Math.sin(t.angle) * 23,
            4,
        );
    const shotOutcome = shot.selfHit
        ? `SELF DESTRUCTION after ${Math.round(shot.time * 1000)} ms if stationary`
        : shot.hit
          ? "predicted enemy hit"
          : shot.muzzleBlocked
            ? "muzzle blocked"
            : "no predicted enemy hit";
    const weapon = [
        { id: "fire", description: "Fire one shell." },
        { id: "hold", description: "Do not fire a shell." },
    ];
    const enemyBearing = Math.atan2(e.y - t.y, e.x - t.x),
        enemyDistance = distance(t, e);
    let frontWallDistance = null,
        enemyVisible = true;
    for (let d = 0; d <= Math.hypot(W, H); d += 2) {
        if (
            blocked(
                game.maze,
                t.x + Math.cos(t.angle) * d,
                t.y + Math.sin(t.angle) * d,
                3,
            )
        ) {
            frontWallDistance = d;
            break;
        }
    }
    for (let d = 0; d <= enemyDistance; d += 2)
        if (
            blocked(
                game.maze,
                t.x + Math.cos(enemyBearing) * d,
                t.y + Math.sin(enemyBearing) * d,
                3,
            )
        ) {
            enemyVisible = false;
            break;
        }
    const localWalls = game.maze.walls.filter(
        (w) =>
            Math.hypot(
                t.x - clamp(t.x, w.x, w.x + w.w),
                t.y - clamp(t.y, w.y, w.y + w.h),
            ) <= 135,
    );
    const state =
        avoidanceContext(game, id) +
        aimContext(aim) +
        `SHOT FACTS: ${shotOutcome}. Enemy straight-line visibility=${enemyVisible}; barrel ray meets wall at ${frontWallDistance === null ? "none" : frontWallDistance + "px from tank center"}. Weapon mechanically ready now=${canShoot}; readiness does NOT imply safety or enemy hit. ` +
        `Tank duel. One hit destroys a tank. ALL bullets, including my own, are lethal. Walls reflect bullets. ` +
        `Three independent controls: drive forward/reverse/stop, turn left/right/none, fire/hold. You can drive, turn and fire simultaneously. No sideways movement. NO AUTO-AIM: barrel is fixed to hull. ` +
        `Coordinates px, origin top-left, +x right, +y down; angle degrees: 0 right, 90 down, -90 up. Arena ${W}x${H}; speed ${SPEED} px/s; turn speed 143 deg/s; tank radius ${RADIUS}, shell radius 3. ` +
        `My position (${round(t.x)},${round(t.y)}), hull/barrel ${degrees(t.angle)} deg; enemy (${round(e.x)},${round(e.y)}), heading ${degrees(e.angle)} deg. Enemy relative angle ${enemyError} deg; positive means turn RIGHT, negative LEFT. ` +
        `Route waypoint (${round(waypoint.x)},${round(waypoint.y)}); relative angle ${waypointError} deg. This is observation only; YOU must steer. ` +
        `Current executed drive ${t.executedDrive}, turn ${t.executedTurn}, remaining command ${Math.round(Math.max(0, t.commandUntil - game.elapsed) * 1000)} ms. Last requested weapon=${t.firing ? "fire" : "hold"}; pending shot=${!!t.pendingFire}; last shot blocked by guard=${!!t.fireBlocked}. ` +
        `Ammo ${ammo}/5, cooldown ${Math.round(t.cooldown * 1000)} ms. Current barrel trajectory (stationary): enemy_hit=${shot.hit}, self_hit=${shot.selfHit}; ${shotOutcome}. ` +
        `Weapon goal: survive AND hit the opponent. HOLD when there is self-hit risk or no useful firing lane; first steer to aim. FIRE is one immediate attempt, never a held trigger or queued shot. ` +
        `${threatened ? "URGENT: stop/none after the response is predicted to collide with an existing shell." : enemyVisible ? "Enemy is visible: align the hull before firing." : "Enemy is behind walls: first navigate via the waypoint. Enemy bearing alone is not a firing lane."} ` +
        `Existing-shell forecasts EXCLUDE any shell you choose to fire. No existing bullets does NOT make firing safe. ` +
        `Timing: predicted response delay ${Math.round(reactionDelay * 1000)} ms; old command runs only until its expiry; new drive+turn lasts ${Math.round(controlDuration * 1000)} ms then STOPS. Later decisions are unknown. Predictions are observations, not controls; choose compatible drive, turn and weapon. ` +
        `Local wall rectangles within 135px [x,y,w,h]: ${JSON.stringify(localWalls.map((w) => [w.x, w.y, w.w, w.h]))}. Other walls are included in trajectory forecasts. ` +
        `NEW SHELL forecasts: shell launched once after response and first movement/turn step, with the above command expiry; horizon 3 seconds after launch. Enemy stationary. null means no event within horizon, NOT guaranteed safety. Rows [drive,turn,can_fire,self_hit_ms,enemy_hit_ms,first_bounce_ms]:\n` +
        JSON.stringify(
            firingForecasts.map((f) => [
                f.drive,
                f.turn,
                f.canFire,
                ...[f.selfHitTime, f.enemyHitTime, f.firstBounceTime].map(
                    (v) => (v === null ? null : Math.round(v * 1000)),
                ),
            ]),
        ) +
        `\nRemaining round ${Math.ceil(game.timeLeft)} s.\n` +
        `All ${bullets.length} active bullets, none omitted. Rows [id,owner,x,y,vx,vy,bounces,life_s,closest_px,closest_s,hit_s]:\n` +
        JSON.stringify(bullets.map((b) => Object.values(b))) +
        `\nPer-bullet closest/hit fields assume standing still for 0.8 s; movement-option predictions follow the whole MOVING path. Owner never implies safety.`;
    return {
        state,
        candidates: { drive, turn, weapon },
        forecasts,
        firingForecasts,
        shot,
        ammo,
        bullets,
        aim,
        threatened,
        contextRevision: CONTEXT_REVISION,
        reactionDelay,
        controlDuration,
        frontWallDistance,
        enemyVisible,
        localWalls,
    };
}

export class Game {
    constructor(seed = 20260928) {
        this.seed = seed;
        this.round = 0;
        this.scores = [0, 0];
        this.events = [];
        this.effects = [];
        this.tracks = [];
        this.shots = [0, 0];
        this.kills = [0, 0];
        this.roundSerial = 0;
        this.assistEnabled = false;
        this.assistProfile = "light";
        this.fireGuardEnabled = false;
        this.playerSettings = [{}, {}];
        this.nextRound();
    }
    nextRound() {
        this.round++;
        this.roundSerial++;
        this.maze = makeMaze(this.seed + this.round * 7919);
        this.bullets = [];
        this.effects = [];
        this.tracks = [];
        this.timeLeft = 60;
        this.phase = "ready";
        this.result = null;
        this.elapsed = 0;
        this.transition = 0;
        const a = this.round % 2 === 1 ? 11 : 58,
            b = this.round % 2 === 1 ? 58 : 11;
        this.tanks = [a, b].map((p, id) => ({
            id,
            ...center(p),
            angle: p === 11 ? 0 : Math.PI,
            alive: true,
            cooldown: 0,
            drive: "stop",
            turn: "none",
            executedDrive: "stop",
            executedTurn: "none",
            firing: false,
            commandUntil: 0,
            travel: 0,
            recoil: 0,
            lastTrack: 0,
            safetyUntil: 0,
            assistActive: false,
        }));
    }
    start() {
        if (this.phase === "ready") this.phase = "running";
    }
    reset(seed = this.seed) {
        this.seed = seed;
        this.round = 0;
        this.scores = [0, 0];
        this.shots = [0, 0];
        this.kills = [0, 0];
        this.events = [];
        this.nextRound();
    }
    setPlayerSettings(id, settings) {
        if (![0, 1].includes(id)) throw Error("Unknown player");
        if (
            settings.assistProfile !== undefined &&
            !Object.hasOwn(AVOIDANCE_PROFILES, settings.assistProfile)
        )
            throw Error("Unknown avoidance level");
        for (const key of ["assistEnabled", "fireGuardEnabled"])
            if (
                settings[key] !== undefined &&
                typeof settings[key] !== "boolean"
            )
                throw Error("Invalid player setting");
        Object.assign(this.playerSettings[id], settings);
        const t = this.tanks[id];
        t.avoidance = null;
        t.assistActive = false;
        t.safetyUntil = 0;
        t.assistCooldownUntil = 0;
        // Releasing a dodge immediately restores the valid model command in forecasts too.
        t.executedDrive = this.elapsed <= t.commandUntil ? t.drive : "stop";
        t.executedTurn =
            this.elapsed <= Math.min(t.commandUntil, t.turnUntil ?? Infinity)
                ? t.turn
                : "none";
    }
    setAssistEnabled(enabled) {
        this.playerSettings.forEach((s) => delete s.assistEnabled);
        this.assistEnabled = !!enabled;
        for (const t of this.tanks) {
            t.avoidance = null;
            t.assistActive = false;
            t.safetyUntil = 0;
            t.assistCooldownUntil = 0;
        }
    }
    setAssistProfile(profile) {
        if (!Object.hasOwn(AVOIDANCE_PROFILES, profile))
            throw Error("Unknown avoidance level");
        this.assistProfile = profile;
        this.playerSettings.forEach((s) => delete s.assistProfile);
        this.setAssistEnabled(this.assistEnabled);
    }
    emit(type, data = {}) {
        this.events.push({
            type,
            time: this.elapsed,
            round: this.round,
            ...data,
        });
    }
    setControls(id, controls, ttl = 0.6) {
        if (
            !Object.hasOwn(DRIVE, controls.drive) ||
            !Object.hasOwn(TURNS, controls.turn) ||
            typeof controls.fire !== "boolean"
        )
            throw Error("Invalid tank controls");
        const t = this.tanks[id];
        t.drive = controls.drive;
        t.turn = controls.turn;
        t.firing = controls.fire;
        t.pendingFire = controls.fire;
        t.fireBlocked = false;
        t.commandUntil = this.elapsed + ttl;
        t.travel = 0;
        if (avoidanceProfile(this, id).instantRefresh) t.safetyUntil = 0;
        t.planned = false;
        t.turnUntil = Infinity;
        t.pendingFireAt = this.elapsed;
    }
    setPlan(id, plan) {
        this.setControls(id, plan.controls, plan.duration);
        const t = this.tanks[id];
        t.planned = true;
        t.turnUntil = this.elapsed + plan.turnDuration;
        t.pendingFireAt = this.elapsed + plan.fireDelay;
        this.emit("plan", {
            player: id,
            motion_id: plan.motionId,
            controls: { ...plan.controls },
            turn_ms: Math.round(plan.turnDuration * 1000),
            fire_after_ms: Math.round(plan.fireDelay * 1000),
            duration_ms: Math.round(plan.duration * 1000),
        });
    }
    fire(t) {
        if (
            !t.alive ||
            t.cooldown > 0 ||
            this.bullets.filter((b) => b.owner === t.id).length >= 5
        )
            return false;
        const a = t.angle,
            x = t.x + Math.cos(a) * 23,
            y = t.y + Math.sin(a) * 23;
        if (blocked(this.maze, x, y, 4)) return false;
        this.bullets.push({
            x,
            y,
            vx: Math.cos(a) * 290,
            vy: Math.sin(a) * 290,
            age: 0,
            bounces: 0,
            owner: t.id,
            trail: [],
            id: `${this.roundSerial}-${t.id}-${this.shots[t.id]}`,
        });
        this.shots[t.id]++;
        t.cooldown = FIRE_COOLDOWN;
        t.recoil = 1;
        this.effects.push({
            type: "muzzle",
            x,
            y,
            angle: a,
            owner: t.id,
            life: 0.14,
            max: 0.14,
        });
        this.emit("shot", { player: t.id });
        return true;
    }
    move(t, drive, turn, dt) {
        const before = { x: t.x, y: t.y };
        tankMotion(this.maze, t, drive, turn, dt, this.tanks[1 - t.id]);
        t.travel += distance(before, t);
    }
    control(t, dt) {
        const active = this.elapsed <= t.commandUntil;
        if (!active) {
            t.drive = "stop";
            t.turn = "none";
            t.firing = false;
        }
        const requested = {
            drive: t.drive,
            turn: this.elapsed > t.turnUntil + 1e-8 ? "none" : t.turn,
        };
        let execute = requested;
        const profile = avoidanceProfile(this, t.id);
        let releaseReason = "safe_path";
        const release = (reason) => {
            if (t.avoidance) {
                t.avoidance = null;
                t.assistCooldownUntil = this.elapsed + profile.cooldown;
                releaseReason = reason;
            }
        };
        if (t.avoidance && this.elapsed + 1e-8 >= t.avoidance.expiresAt)
            release("burst_limit");
        if (
            assistEnabledFor(this, t.id) &&
            this.bullets.length &&
            this.elapsed + 1e-8 >= (t.assistCooldownUntil ?? 0)
        ) {
            const signature = this.bullets
                .map((b) => `${b.id}:${b.bounces}`)
                .join("|");
            if (
                this.elapsed + 1e-8 >= t.safetyUntil ||
                (profile.instantRefresh && signature !== t.safetySignature)
            ) {
                const { proposed, choice } = chooseAvoidance(this, t.id);
                const previous = t.avoidance;
                if (choice) {
                    const expiresAt =
                        previous?.expiresAt ??
                        this.elapsed + profile.maxDuration;
                    t.avoidance = {
                        controls: choice.controls,
                        startedAt: previous?.startedAt ?? this.elapsed,
                        expiresAt,
                        holdUntil:
                            previous &&
                            sameMotion(previous.controls, choice.controls)
                                ? previous.holdUntil
                                : Math.min(
                                      expiresAt,
                                      this.elapsed + profile.minHold,
                                  ),
                    };
                } else release("safe_path");
                if (choice && proposed.hitTime !== null)
                    t.lastAvoidance = {
                        threat: proposed.threat,
                        hit_ms: Math.round(proposed.hitTime * 1000),
                        clearance_px: Math.round(choice.clearance),
                    };
                t.safetyUntil = this.elapsed + profile.interval;
                t.safetySignature = signature;
            }
            if (t.avoidance) execute = t.avoidance.controls;
        } else
            release(assistEnabledFor(this, t.id) ? "no_bullets" : "disabled");
        const assisted = !!t.avoidance;
        if (
            assisted &&
            (!t.assistActive ||
                t.executedDrive !== execute.drive ||
                t.executedTurn !== execute.turn)
        ) {
            t.assistCount = (t.assistCount ?? 0) + 1;
            this.emit("assist", {
                player: t.id,
                level: assistLevelFor(this, t.id),
                requested: { ...requested },
                executed: { ...execute },
                burst_remaining_ms: Number.isFinite(t.avoidance.expiresAt)
                    ? Math.round((t.avoidance.expiresAt - this.elapsed) * 1000)
                    : null,
                ...t.lastAvoidance,
                reason: `Predicted shell collision in ${t.lastAvoidance?.hit_ms ?? "?"} ms`,
            });
        }
        if (!assisted && t.assistActive)
            this.emit("assist_end", {
                player: t.id,
                reason: releaseReason,
                cooldown_ms: Math.round(profile.cooldown * 1000),
                executed: { ...execute },
            });
        t.assistActive = assisted;
        t.executedDrive = execute.drive;
        t.executedTurn = execute.turn;
        // Any path override invalidates the weapon decision conditioned on the old plan.
        // Discard it immediately, including shots waiting for a future turn deadline.
        if (assisted && t.pendingFire) {
            t.pendingFire = false;
            t.fireBlocked = true;
            this.emit("assist_fire_cancel", {
                player: t.id,
                requested: { ...requested },
                executed: { ...execute },
            });
        }
        let veto = false;
        const trigger =
            active && t.pendingFire && this.elapsed > t.pendingFireAt + 1e-8;
        if (trigger && t.planned) {
            const forecast = forecastShot(this, t.id, execute, {
                reactionDelay: 0,
                horizon: 3,
                controlDuration: Math.max(0, t.commandUntil - this.elapsed),
                turnDuration: Math.max(0, t.turnUntil - this.elapsed),
            });
            if (
                !forecast.canFire ||
                forecast.selfHitTime !== null ||
                forecast.enemyHitTime === null
            ) {
                veto = true;
                t.fireBlocked = true;
                this.emit("plan_veto", {
                    player: t.id,
                    reason: !forecast.canFire
                        ? "weapon_unavailable"
                        : forecast.selfHitTime !== null
                          ? "self_hit"
                          : "no_enemy_hit",
                    forecast,
                });
            }
        }
        if (trigger && shotGuardFor(this, t.id)) {
            const forecast = forecastShot(this, t.id, execute, {
                reactionDelay: 0,
                horizon: 0.8,
            });
            if (forecast.selfHitTime !== null) {
                veto = true;
                t.fireBlocked = true;
                this.emit("fire_guard", {
                    player: t.id,
                    requested: true,
                    executed: false,
                    self_hit_ms: Math.round(forecast.selfHitTime * 1000),
                    controls: { ...execute },
                });
            }
        }
        this.move(t, DRIVE_INPUT[execute.drive], TURN_INPUT[execute.turn], dt);
        if (trigger && !veto) this.fire(t);
        if (trigger || !active) t.pendingFire = false;
    }
    step(dt, manual = {}) {
        dt = Math.min(dt, 1 / 30);
        for (const e of this.effects) e.life -= dt;
        this.effects = this.effects.filter((e) => e.life > 0);
        if (this.phase === "roundover") {
            this.transition -= dt;
            if (this.transition <= 0) {
                if (Math.max(...this.scores) >= 5) this.phase = "matchover";
                else {
                    this.nextRound();
                    this.start();
                }
            }
            return;
        }
        if (this.phase !== "running") return;
        this.elapsed += dt;
        this.timeLeft -= dt;
        for (const t of this.tanks) {
            t.cooldown = Math.max(0, t.cooldown - dt);
            t.recoil = Math.max(0, t.recoil - dt * 7);
            if (!t.alive) continue;
            // Public human controls receive the selected turn boost. Model turns stay unchanged.
            t.turnSpeed =
                TURN_SPEED *
                (manual[t.id]?.responsiveTurn ? 1.68 : 1) *
                (manual[t.id]?.fastHumanTurn ? 2 : 1);
            if (manual[t.id]) {
                const c = manual[t.id];
                t.drive =
                    c.drive > 0 ? "forward" : c.drive < 0 ? "reverse" : "stop";
                t.turn = c.turn > 0 ? "right" : c.turn < 0 ? "left" : "none";
                t.executedDrive = t.drive;
                t.executedTurn = t.turn;
                t.firing = !!c.fire;
                t.assistActive = false;
                if (c.freeMove) {
                    const dx = Number.isFinite(c.moveX)
                        ? clamp(c.moveX, -1, 1)
                        : 0;
                    const dy = Number.isFinite(c.moveY)
                        ? clamp(c.moveY, -1, 1)
                        : 0;
                    const before = { x: t.x, y: t.y };
                    t.hullAngle ??= t.angle;
                    if (dx || dy) t.hullAngle = Math.atan2(dy, dx);
                    movePosition(
                        this.maze,
                        t,
                        dx,
                        dy,
                        dt,
                        this.tanks[1 - t.id],
                    );
                    t.travel += distance(before, t);
                    if (
                        Number.isFinite(c.aimPoint?.x) &&
                        Number.isFinite(c.aimPoint?.y) &&
                        Math.hypot(c.aimPoint.x - t.x, c.aimPoint.y - t.y) >
                            0.001
                    )
                        t.angle = Math.atan2(
                            c.aimPoint.y - t.y,
                            c.aimPoint.x - t.x,
                        );
                } else {
                    delete t.hullAngle;
                    this.move(t, c.drive || 0, c.turn || 0, dt);
                }
                if (c.fire) this.fire(t);
            } else this.control(t, dt);
            t.lastTrack += dt;
            if (t.lastTrack > 0.11) {
                this.tracks.push({
                    x: t.x,
                    y: t.y,
                    angle: t.hullAngle ?? t.angle,
                    age: 0,
                });
                t.lastTrack = 0;
            }
        }
        for (const track of this.tracks) track.age += dt;
        this.tracks = this.tracks.filter((t) => t.age < 9).slice(-300);
        const deaths = new Set();
        for (const b of this.bullets) {
            b.trail.push({ x: b.x, y: b.y });
            if (b.trail.length > 12) b.trail.shift();
            const substeps = 4;
            for (let k = 0; k < substeps; k++) {
                advanceBullet(b, dt / substeps, this.maze.walls, (p) => {
                    this.effects.push({
                        type: "spark",
                        x: p.x,
                        y: p.y,
                        owner: p.owner,
                        life: 0.22,
                        max: 0.22,
                    });
                });
                for (const t of this.tanks) {
                    if (!t.alive || deaths.has(t.id)) continue;
                    if (bulletHits(b, t)) {
                        deaths.add(t.id);
                        b.dead = true;
                        this.emit("kill", {
                            player: b.owner,
                            victim: t.id,
                            self: b.owner === t.id,
                        });
                        this.kills[b.owner] += b.owner !== t.id ? 1 : 0;
                        break;
                    }
                }
                if (b.dead) break;
            }
        }
        this.bullets = this.bullets.filter(
            (b) => !b.dead && b.age < SHELL_LIFETIME,
        );
        for (const id of deaths) {
            const t = this.tanks[id];
            t.alive = false;
            this.effects.push({
                type: "explosion",
                x: t.x,
                y: t.y,
                owner: id,
                life: 1.2,
                max: 1.2,
            });
        }
        if (deaths.size || this.timeLeft <= 0) {
            const alive = this.tanks.filter((t) => t.alive);
            const winner =
                deaths.size && alive.length === 1 ? alive[0].id : null;
            if (winner !== null) this.scores[winner]++;
            this.result = winner;
            this.phase = "roundover";
            this.transition = 2.6;
            this.emit("round", {
                winner,
                reason:
                    this.timeLeft <= 0
                        ? "timeout"
                        : deaths.size === 2
                          ? "draw"
                          : "hit",
            });
        }
    }
}
