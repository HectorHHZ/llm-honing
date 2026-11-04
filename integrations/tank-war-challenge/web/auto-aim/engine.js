import {
    W,
    H,
    RADIUS,
    SPEED,
    makeMaze,
    center,
    distance,
    angleDelta,
    blocked,
    traceShot,
    advanceBullet,
    bulletHits,
    bulletForecast,
    movePosition,
    pathDistance,
} from "./physics.js";
import { findAimDirections } from "./aim.js";
export { W, H, traceShot, angleDelta };
export const PROTOCOL = "controls-v2";
export const MOVES = {
    up: "Up",
    down: "Down",
    left: "Left",
    right: "Right",
    stay: "Stay",
};
export const VECTORS = {
    up: [0, -1],
    down: [0, 1],
    left: [-1, 0],
    right: [1, 0],
    stay: [0, 0],
};
export const WEAPONS = { fire: "Fire", hold: "Hold" },
    COLORS = ["#357a9b", "#c4654b"];
// Reconstruct the v2 movement forecast and compact observation from archived requests.
export function forecastMovement(
    game,
    id,
    move,
    { horizon = 0.9, reactionDelay = 0.12 } = {},
) {
    const t = game.tanks[id],
        enemy = game.tanks[1 - id],
        p = { x: t.x, y: t.y },
        bullets = game.bullets.map((b) => ({ ...b }));
    let clearance = 999,
        hitTime = null,
        travel = 0;
    const dt = 1 / 120;
    for (let k = 0; k < Math.ceil(horizon / dt); k++) {
        const time = k * dt,
            before = { ...p };
        const key =
            time < reactionDelay
                ? game.elapsed + time <= t.commandUntil
                    ? t.executedMove
                    : "stay"
                : move;
        movePosition(game.maze, p, ...VECTORS[key], dt, enemy);
        travel += distance(before, p);
        for (const b of bullets) {
            if (b.age >= 8) continue;
            advanceBullet(b, dt, game.maze.walls);
            if (b.owner === id && b.age < 0.15 && b.bounces === 0) continue;
            const margin = distance(b, p) - (RADIUS + 3);
            clearance = Math.min(clearance, margin);
            if (margin < 0 && hitTime === null) hitTime = time + dt;
        }
    }
    return {
        move,
        x: p.x,
        y: p.y,
        clearance,
        hitTime,
        travel,
        progress: pathDistance(game, t, enemy) - pathDistance(game, p, enemy),
    };
}
export function bestShot(game, id) {
    const t = game.tanks[id],
        enemy = game.tanks[1 - id];
    const aim = findAimDirections(
        game.maze,
        { ...t, angle: t.turretAngle },
        enemy,
        { horizon: 2.8, limit: 3 },
    );
    return aim.options.sort((a, b) => a.hit_ms - b.hit_ms)[0] ?? null;
}
export function buildObservation(game, id, { reactionDelay = 0.12 } = {}) {
    const t = game.tanks[id],
        e = game.tanks[1 - id],
        shot = bestShot(game, id),
        ammo = 5 - game.bullets.filter((b) => b.owner === id).length,
        round = (n) => Number(n.toFixed(1));
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
            life_s: round(Math.max(0, 8 - b.age)),
            closest_px: Number.isFinite(f.min) ? round(f.min) : null,
            closest_s:
                f.closestTime === null
                    ? null
                    : Number(f.closestTime.toFixed(3)),
            hit_s: f.hitTime === null ? null : Number(f.hitTime.toFixed(3)),
        };
    });
    const forecasts = Object.keys(MOVES).map((m) =>
        forecastMovement(game, id, m, { reactionDelay }),
    );
    const threatened =
        forecasts.find((f) => f.move === "stay").hitTime !== null;
    // The archived live client omitted directions blocked by walls, always retaining stay.
    let usable = forecasts.filter((f) => f.move === "stay" || f.travel > 3);
    if (usable.length < 2) usable = forecasts;
    const movement = usable.map((f) => ({
        id: f.move,
        description: `${f.move === "stay" ? "Stay at current position." : `Move ${f.move.toUpperCase()} immediately.`} ${f.hitTime === null ? "SAFE over forecast; no bullet collision" : `DANGER: bullet collision after ${Math.round(f.hitTime * 1000)} ms`}; minimum bullet clearance ${Math.round(f.clearance)} px; travel ${Math.round(f.travel)} px${f.move !== "stay" && f.travel < 20 ? " (wall or tank limits movement)" : ""}; enemy-route progress ${Math.round(f.progress)} px. Firing is independent.`,
    }));
    const weapon = [
        {
            id: "fire",
            description:
                shot && ammo > 0
                    ? `Fire while continuing chosen movement. A ${shot.bounces ? "ricochet" : "direct"} shot can hit enemy; ${ammo} shells available.`
                    : "Fire request has no valid shot or no ammunition; cannon waits for a valid shot.",
        },
        {
            id: "hold",
            description:
                "Do not launch a new shell; continue chosen movement and keep aiming.",
        },
    ];
    const state = `Tank duel. One hit destroys a tank. ALL bullets, including my own, are lethal. Walls reflect bullets. Movement and shooting are independent: I can move up/down/left/right while firing. Moving starts immediately without turning first; turret aims separately. Coordinates px, origin top-left, +x right, +y down; arena ${W}x${H}; speed ${SPEED} px/s; tank radius ${RADIUS}, shell radius 3. My position (${round(t.x)},${round(t.y)}); enemy (${round(e.x)},${round(e.y)}). My current executed movement ${t.executedMove}, model-requested movement ${t.move}, weapon ${t.firing ? "fire" : "hold"}. Ammo ${ammo}/5, cannon cooldown ${Math.round(t.cooldown * 1000)} ms. ${shot ? "A" : "No"} valid shot is available. ${threatened ? "URGENT: staying still is predicted to be hit. MOVE to escape." : "Choose safe movement that approaches the enemy."} Each movement option is simulated for 900 ms along the entire moving path, with ${Math.round(reactionDelay * 1000)} ms response delay, walls, tank collision and shell lifetime. Positive route progress approaches the enemy. A negative clearance or DANGER means predicted death. Choose a no-hit direction when available, even if it moves away from the enemy. Remaining round ${Math.ceil(game.timeLeft)} s.\nAll ${bullets.length} active bullets, none omitted. Rows [id,owner,x,y,vx,vy,bounces,life_s,closest_px,closest_s,hit_s]:\n${JSON.stringify(bullets.map((b) => Object.values(b)))}\nPer-bullet closest/hit fields assume standing still for 0.8 s; movement-option predictions follow the whole MOVING path. Owner never implies safety.`;
    return {
        state,
        candidates: { movement, weapon },
        forecasts,
        bullets,
        shot,
        ammo,
        threatened,
        reactionDelay,
    };
}
export class Game {
    constructor(seed = 20260928) {
        this.seed = seed;
        this.round = 0;
        this.roundSerial = 0;
        this.scores = [0, 0];
        this.shots = [0, 0];
        this.kills = [0, 0];
        this.events = [];
        this.assistEnabled = false;
        this.nextRound();
    }
    nextRound() {
        this.round++;
        this.roundSerial++;
        this.maze = makeMaze(this.seed + this.round * 7919);
        this.bullets = [];
        this.effects = [];
        this.tracks = [];
        this.elapsed = 0;
        this.timeLeft = 60;
        this.phase = "ready";
        this.result = null;
        this.transition = 0;
        this.tanks = [this.round % 2 ? 11 : 58, this.round % 2 ? 58 : 11].map(
            (p, id) => ({
                id,
                ...center(p),
                angle: id ? Math.PI : 0,
                turretAngle: id ? Math.PI : 0,
                alive: true,
                cooldown: 0,
                move: "stay",
                executedMove: "stay",
                firing: false,
                commandUntil: 0,
                recoil: 0,
                aimAt: 0,
                aim: null,
                safetyAt: 0,
                assistActive: false,
                lastTrack: 0,
            }),
        );
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
    emit(type, data = {}) {
        this.events.push({
            type,
            round: this.round,
            time: this.elapsed,
            ...data,
        });
    }
    setControls(id, c, ttl = 0.6) {
        if (!Object.hasOwn(MOVES, c.move) || typeof c.fire !== "boolean")
            throw Error("Invalid controls-v2 response");
        Object.assign(this.tanks[id], {
            move: c.move,
            firing: c.fire,
            commandUntil: this.elapsed + ttl,
            safetyAt: 0,
        });
    }
    fire(t) {
        if (
            !t.alive ||
            t.cooldown > 0 ||
            this.bullets.filter((b) => b.owner === t.id).length >= 5
        )
            return false;
        // Re-read current geometry before every shot; the model requests fire, auto-aim supplies the lane.
        const shot = traceShot(
            this.maze,
            t,
            t.turretAngle,
            2.8,
            this.tanks[1 - t.id],
        );
        if (!shot.hit || shot.selfHit || shot.muzzleBlocked) return false;
        const a = t.turretAngle,
            x = t.x + Math.cos(a) * 23,
            y = t.y + Math.sin(a) * 23;
        this.bullets.push({
            id: `${this.roundSerial}-${t.id}-${this.shots[t.id]}`,
            x,
            y,
            vx: Math.cos(a) * 290,
            vy: Math.sin(a) * 290,
            age: 0,
            bounces: 0,
            owner: t.id,
            trail: [],
        });
        this.shots[t.id]++;
        t.cooldown = 0.48;
        t.recoil = 1;
        this.effects.push({
            type: "muzzle",
            x,
            y,
            owner: t.id,
            life: 0.14,
            max: 0.14,
        });
        this.emit("shot", { player: t.id, angle: a });
        return true;
    }
    control(t, dt, manual) {
        let requested =
                manual?.move ??
                (this.elapsed <= t.commandUntil ? t.move : "stay"),
            fire = manual?.fire ?? (this.elapsed <= t.commandUntil && t.firing),
            execute = requested;
        if (this.assistEnabled && !manual && this.bullets.length) {
            if (this.elapsed >= t.safetyAt) {
                const proposed = forecastMovement(this, t.id, requested, {
                    reactionDelay: 0,
                    horizon: 0.7,
                });
                t.safeMove = requested;
                if (proposed.hitTime !== null) {
                    const options = Object.keys(MOVES).map((m) =>
                        forecastMovement(this, t.id, m, {
                            reactionDelay: 0,
                            horizon: 0.7,
                        }),
                    );
                    options.sort(
                        (a, b) =>
                            (b.hitTime === null) - (a.hitTime === null) ||
                            (b.hitTime ?? 1) - (a.hitTime ?? 1) ||
                            b.clearance - a.clearance ||
                            b.progress - a.progress,
                    );
                    const best = options[0];
                    if (
                        best.hitTime === null ||
                        (best.hitTime ?? 0) > proposed.hitTime + 0.08
                    )
                        t.safeMove = best.move;
                }
                t.safetyAt = this.elapsed + 0.05;
            }
            execute = t.safeMove;
        }
        const assisted = execute !== requested;
        if (assisted && (!t.assistActive || t.executedMove !== execute))
            this.emit("assist", { player: t.id, requested, executed: execute });
        t.assistActive = assisted;
        t.executedMove = execute;
        const [dx, dy] = VECTORS[execute];
        movePosition(this.maze, t, dx, dy, dt, this.tanks[1 - t.id]);
        if (dx || dy) t.angle = Math.atan2(dy, dx);
        if (this.elapsed >= t.aimAt) {
            t.aim = bestShot(this, t.id);
            t.aimAt = this.elapsed + 0.05;
        }
        const target =
                t.aim?.angle ??
                Math.atan2(
                    this.tanks[1 - t.id].y - t.y,
                    this.tanks[1 - t.id].x - t.x,
                ),
            delta = angleDelta(target, t.turretAngle);
        t.turretAngle = angleDelta(
            t.turretAngle +
                Math.sign(delta) * Math.min(Math.abs(delta), 8 * dt),
            0,
        );
        if (
            fire &&
            t.aim &&
            Math.abs(angleDelta(t.aim.angle, t.turretAngle)) < 0.07
        )
            this.fire(t);
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
            this.control(t, dt, manual[t.id]);
            t.lastTrack += dt;
            if (t.lastTrack > 0.12) {
                this.tracks.push({ x: t.x, y: t.y, angle: t.angle, age: 0 });
                t.lastTrack = 0;
            }
        }
        for (const t of this.tracks) t.age += dt;
        this.tracks = this.tracks.filter((t) => t.age < 6).slice(-200);
        const deaths = new Set();
        for (const b of this.bullets) {
            b.trail.push({ x: b.x, y: b.y });
            if (b.trail.length > 12) b.trail.shift();
            for (let k = 0; k < 4; k++) {
                advanceBullet(b, dt / 4, this.maze.walls, (p) =>
                    this.effects.push({
                        type: "spark",
                        x: p.x,
                        y: p.y,
                        owner: p.owner,
                        life: 0.22,
                        max: 0.22,
                    }),
                );
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
                        if (b.owner !== t.id) this.kills[b.owner]++;
                        break;
                    }
                }
                if (b.dead) break;
            }
        }
        this.bullets = this.bullets.filter((b) => !b.dead && b.age < 8);
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
            const alive = this.tanks.filter((t) => t.alive),
                winner = deaths.size && alive.length === 1 ? alive[0].id : null;
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
