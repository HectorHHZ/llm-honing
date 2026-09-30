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
    RADIUS = TANK_RADIUS,
    SPEED = 125;
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
export function makeMaze(seed) {
    const random = rng(seed),
        links = Array.from({ length: COLS * ROWS }, () => []),
        visited = new Set([0]),
        stack = [0],
        edges = new Set();
    const key = (a, b) => [Math.min(a, b), Math.max(a, b)].join("-");
    const neighbors = (i) =>
        [
            i % COLS > 0 ? i - 1 : -1,
            i % COLS < COLS - 1 ? i + 1 : -1,
            i >= COLS ? i - COLS : -1,
            i < COLS * (ROWS - 1) ? i + COLS : -1,
        ].filter((j) => j >= 0);
    const connect = (a, b) => {
        if (edges.has(key(a, b))) return;
        edges.add(key(a, b));
        links[a].push(b);
        links[b].push(a);
    };
    while (stack.length) {
        const i = stack.at(-1),
            next = neighbors(i).filter((j) => !visited.has(j));
        if (!next.length) {
            stack.pop();
            continue;
        }
        const j = next[Math.floor(random() * next.length)];
        connect(i, j);
        visited.add(j);
        stack.push(j);
    }
    // Additional loops create sight lines and prevent a single-corridor chase.
    for (let i = 0; i < links.length; i++)
        for (const j of neighbors(i))
            if (j > i && random() < 0.3) connect(i, j);
    const walls = [
        { x: 0, y: 0, w: W, h: 7 },
        { x: 0, y: H - 7, w: W, h: 7 },
        { x: 0, y: 0, w: 7, h: H },
        { x: W - 7, y: 0, w: 7, h: H },
    ];
    for (let y = 0; y < ROWS; y++)
        for (let x = 0; x < COLS; x++) {
            const i = y * COLS + x;
            if (x < COLS - 1 && !edges.has(key(i, i + 1)))
                walls.push({
                    x: (x + 1) * CELL - 4,
                    y: y * CELL - 4,
                    w: 8,
                    h: CELL + 8,
                });
            if (y < ROWS - 1 && !edges.has(key(i, i + COLS)))
                walls.push({
                    x: x * CELL - 4,
                    y: (y + 1) * CELL - 4,
                    w: CELL + 8,
                    h: 8,
                });
        }
    return { walls, links, seed };
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
    for (let k = 0; k < Math.ceil(Math.min(maxTime, 8) * 240); k++) {
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
export function bulletForecast(game, bullet, p, horizon = 0.8) {
    const b = { ...bullet };
    let min = Infinity,
        closestTime = null,
        hitTime = null;
    for (let k = 0; k <= Math.ceil(horizon * 60); k++) {
        if (b.age >= 8) break;
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
export function pathDistance(game, p, enemy) {
    const path = route(game.maze, cellOf(p), cellOf(enemy));
    return path.length
        ? distance(p, center(path[0])) +
              (path.length - 1) * CELL +
              distance(center(cellOf(enemy)), enemy)
        : distance(p, enemy);
}
