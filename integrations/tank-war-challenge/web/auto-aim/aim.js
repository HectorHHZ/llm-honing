import { distance, blocked, traceShot, angleDelta, RADIUS } from "./physics.js";
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
                Math.abs(angleDelta(angle, tank.angle)) / 8 +
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
                Math.abs(delta) < 8 / 120
                    ? "none"
                    : delta > 0
                      ? "right"
                      : "left",
            turn_ms: Math.round((Math.abs(delta) / 8) * 1000),
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
