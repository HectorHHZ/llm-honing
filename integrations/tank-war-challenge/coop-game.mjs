import {
    Game,
    makeMaze,
    center,
    blocked,
    distance,
    RADIUS,
    tankMotion,
    bulletHits,
    advanceBullet,
    SHELL_LIFETIME,
} from "./web/classic/engine.js";

// Humans are blue (team 0); bot is red (team 1). Ricochets can hit their shooter.
export const canDamage = (bullet, tank) =>
    bullet.owner === tank.id || bullet.team !== tank.team;
export class CoopGame extends Game {
    constructor(players, seed = Date.now() >>> 0) {
        super(seed);
        this.players = players;
        this.round = 0;
        this.scores = [0, 0];
        this.beginRound();
    }
    beginRound() {
        super.nextRound();
        this.maze = makeMaze(this.seed + this.round * 7919);
        const template = this.tanks[0];
        const used = [];
        const spawn = (right) => {
            const cells = Array.from({ length: 70 }, (_, i) => center(i))
                .filter((p) => !blocked(this.maze, p.x, p.y, RADIUS + 2))
                .sort(
                    (a, b) =>
                        (right ? b.x - a.x : a.x - b.x) ||
                        Math.abs(a.y - 315) - Math.abs(b.y - 315),
                );
            const p = cells.find((p) => used.every((q) => distance(p, q) > 65));
            if (!p) throw Error("No safe spawn");
            used.push(p);
            return p;
        };
        const bot = {
            ...template,
            ...spawn(true),
            id: 1,
            team: 1,
            label: "Bot",
            angle: Math.PI,
        };
        this.tanks = [
            bot,
            ...this.players.map((p, i) => ({
                ...template,
                ...spawn(false),
                id: p.id,
                team: 0,
                label: p.nickname,
                angle: 0,
            })),
        ];
        this.shots = Array(10).fill(0);
        this.kills = Array(10).fill(0);
        this.playerSettings = [
            { assistEnabled: false },
            {
                assistEnabled: true,
                assistProfile: "strong",
                fireGuardEnabled: true,
            },
        ];
        this.phase = "running";
        this.probabilities = null;
    }
    emit(type, data = {}) {
        super.emit(type, data);
        if (this.events.length > 100) this.events.shift();
    }
    target() {
        const bot = this.tanks.find((t) => t.id === 1);
        const live = this.tanks.filter((t) => t.team === 0 && t.alive);
        // Hysteresis avoids changing targets on every small movement.
        const old = live.find((t) => t.id === this.targetId);
        const nearest = live.sort(
            (a, b) => distance(bot, a) - distance(bot, b),
        )[0];
        const chosen =
            old && distance(bot, old) < distance(bot, nearest) + 90
                ? old
                : nearest;
        this.targetId = chosen?.id;
        return chosen;
    }
    view(target = this.target()) {
        if (!target) return null;
        const view = Object.create(this);
        view.tanks = [{ ...target, id: 0 }, this.tanks.find((t) => t.id === 1)];
        view.bullets = this.bullets.map((b) => ({
            ...b,
            owner: b.owner === 1 ? 1 : 0,
        }));
        view.move = (t, d, r, dt) => this.move(t, d, r, dt);
        view.fire = (t) => this.fire(t);
        view.emit = (type, data) => this.emit(type, data);
        return view;
    }
    move(t, drive, turn, dt) {
        const before = { x: t.x, y: t.y };
        tankMotion(this.maze, t, drive, turn, dt);
        if (
            this.tanks.some(
                (o) => o !== t && o.alive && distance(t, o) < RADIUS * 2,
            )
        ) {
            t.x = before.x;
            t.y = before.y;
        }
        t.travel += distance(before, t);
    }
    fire(t) {
        if (!super.fire(t)) return false;
        this.bullets.at(-1).team = t.team;
        return true;
    }
    drop(id) {
        const t = this.tanks.find((t) => t.id === id);
        if (t) t.alive = false;
        this.players = this.players.filter((p) => p.id !== id);
    }
    step(dt, manual = {}) {
        if (this.phase !== "running") return;
        dt = Math.min(dt, 1 / 30);
        this.elapsed += dt;
        this.timeLeft -= dt;
        for (const e of this.effects) e.life -= dt;
        this.effects = this.effects.filter((e) => e.life > 0);
        const view = this.view();
        for (const t of this.tanks) {
            t.cooldown = Math.max(0, t.cooldown - dt);
            t.recoil = Math.max(0, t.recoil - dt * 7);
            if (!t.alive) continue;
            if (t.team === 0) {
                const c = manual[t.id] || { drive: 0, turn: 0, fire: false };
                this.move(t, c.drive, c.turn, dt);
                if (c.fire) this.fire(t);
            } else if (view) Game.prototype.control.call(view, t, dt);
            t.lastTrack += dt;
            if (t.lastTrack > 0.11) {
                this.tracks.push({ x: t.x, y: t.y, angle: t.angle, age: 0 });
                t.lastTrack = 0;
            }
        }
        for (const t of this.tracks) t.age += dt;
        this.tracks = this.tracks.filter((t) => t.age < 6).slice(-300);
        const deaths = new Set();
        for (const b of this.bullets) {
            b.trail.push({ x: b.x, y: b.y });
            if (b.trail.length > 8) b.trail.shift();
            for (let k = 0; k < 4 && !b.dead; k++) {
                advanceBullet(b, dt / 4, this.maze.walls);
                for (const t of this.tanks) {
                    if (!t.alive || deaths.has(t.id) || !canDamage(b, t))
                        continue;
                    if (bulletHits(b, t)) {
                        deaths.add(t.id);
                        b.dead = true;
                        break;
                    }
                }
            }
        }
        this.bullets = this.bullets.filter(
            (b) => !b.dead && b.age < SHELL_LIFETIME,
        );
        for (const id of deaths) {
            const t = this.tanks.find((t) => t.id === id);
            t.alive = false;
            this.effects.push({
                type: "explosion",
                x: t.x,
                y: t.y,
                owner: id,
                team: t.team,
                life: 1.2,
                max: 1.2,
            });
        }
        const humans = this.tanks.some((t) => t.team === 0 && t.alive),
            bot = this.tanks.find((t) => t.id === 1).alive;
        if (!humans || !bot || this.timeLeft <= 0) {
            this.result = !bot && humans ? 0 : bot && !humans ? 1 : null;
            if (this.result !== null) this.scores[this.result]++;
            this.phase = "roundover";
        }
    }
    snapshot() {
        return {
            seed: this.seed,
            round: this.round,
            phase: this.phase,
            result: this.result,
            timeLeft: this.timeLeft,
            elapsed: this.elapsed,
            scores: this.scores,
            maze: this.maze,
            tanks: this.tanks.map((t) => ({
                id: t.id,
                team: t.team,
                label: t.label,
                x: t.x,
                y: t.y,
                angle: t.angle,
                alive: t.alive,
                recoil: t.recoil,
                assistActive: t.assistActive,
            })),
            bullets: this.bullets,
            effects: this.effects.map((e) => ({
                ...e,
                team: e.team ?? (e.owner === 1 ? 1 : 0),
            })),
            tracks: this.tracks,
            probabilities: this.probabilities,
        };
    }
}
