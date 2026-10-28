import { TANK_RADIUS } from "./collision.js";
export const COLORS = ["#357a9b", "#c4654b"];
export function createRenderer(
    canvas,
    traceShot,
    { width = 900, height = 630, autoAim = false } = {},
) {
    const ctx = canvas.getContext("2d");
    function line(x1, y1, x2, y2, color, weight = 1.4, seed = 0) {
        ctx.strokeStyle = color;
        ctx.lineWidth = weight;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        const bend = Math.sin(seed * 3.71 + x1 * 0.2 + y2 * 0.1) * 0.5;
        ctx.quadraticCurveTo(
            (x1 + x2) / 2 + bend,
            (y1 + y2) / 2 - bend,
            x2,
            y2,
        );
        ctx.stroke();
    }
    function box(x, y, w, h, fill, stroke, weight = 1.4) {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 2.5);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = weight;
        ctx.stroke();
    }
    // The current game's modifiers, set by render(): a Super Tank is drawn as a heavier, spiked machine.
    let mods = null;
    function superAura(t) {
        const now = performance.now() / 1000,
            pulse = 0.5 + 0.5 * Math.sin(now * 6);
        ctx.save();
        ctx.translate(t.x, t.y);
        const glow = ctx.createRadialGradient(0, 0, TANK_RADIUS - 4, 0, 0, TANK_RADIUS + 12);
        glow.addColorStop(0, `rgba(255, 106, 0, ${0.35 + 0.2 * pulse})`);
        glow.addColorStop(1, "rgba(255, 31, 90, 0)");
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(0, 0, TANK_RADIUS + 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.rotate(now * 1.6);
        ctx.fillStyle = "#ff3d1f";
        for (let i = 0; i < 10; i++) {
            ctx.rotate(Math.PI / 5);
            ctx.beginPath();
            ctx.moveTo(TANK_RADIUS + 2, -2.4);
            ctx.lineTo(TANK_RADIUS + 7 + 2 * pulse, 0);
            ctx.lineTo(TANK_RADIUS + 2, 2.4);
            ctx.fill();
        }
        ctx.restore();
    }
    function tank(t) {
        if (!t.alive) return;
        const color = COLORS[t.team ?? t.id],
            heavy = !!mods?.[t.id]?.superTank,
            ink = "#2c2620",
            fire = "#ff5a1f";
        if (heavy) superAura(t);
        if (t.assistActive) {
            ctx.save();
            ctx.strokeStyle = "#ab8037";
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 4]);
            ctx.beginPath();
            ctx.arc(t.x, t.y, TANK_RADIUS + 5, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
        }
        ctx.save();
        ctx.translate(t.x, t.y);
        // Ink must stay within the physical footprint at every hull/turret angle.
        ctx.beginPath();
        ctx.arc(0, 0, TANK_RADIUS - 0.5, 0, Math.PI * 2);
        ctx.clip();
        ctx.save();
        ctx.rotate(t.hullAngle ?? t.angle);
        if (heavy) {
            // Heavy treads, dark armor with the team stripe, and a spiked dozer blade on the front.
            for (const s of [-1, 1]) {
                box(-16, s * 11 - 3.5, 32, 7, ink, fire, 1.4);
                for (let x = -12; x < 14; x += 4) line(x, s * 11 - 2.5, x, s * 11 + 2.5, "#ffb000", 0.9, x);
            }
            box(-13, -9, 25, 18, "#4a2c22", fire, 1.8);
            line(-11, 0, 9, 0, color, 2.4);
            ctx.beginPath();
            ctx.moveTo(13, -12);
            ctx.lineTo(17.5, -8);
            ctx.lineTo(17.5, 8);
            ctx.lineTo(13, 12);
            ctx.closePath();
            ctx.fillStyle = ink;
            ctx.fill();
            ctx.strokeStyle = fire;
            ctx.lineWidth = 1.4;
            ctx.stroke();
            ctx.fillStyle = "#ffb000";
            for (const y of [-6, 0, 6]) {
                ctx.beginPath();
                ctx.moveTo(17.5, y - 2.2);
                ctx.lineTo(21.3, y);
                ctx.lineTo(17.5, y + 2.2);
                ctx.fill();
            }
        } else {
            for (const s of [-1, 1]) {
                box(-15, s * 11 - 3, 30, 6, color + "18", color);
                for (let x = -11; x < 13; x += 5)
                    line(x, s * 11 - 2, x, s * 11 + 2, color + "88", 0.8, x);
            }
            box(-13, -9, 26, 18, color + "24", color, 1.6);
            line(-10, -5, -10, 5, color + "88", 1);
        }
        ctx.restore();
        ctx.rotate(autoAim ? t.turretAngle : t.angle);
        if (heavy) {
            box(-7.5, -7.5, 15, 15, ink, fire, 1.6);
            box(3 - t.recoil * 2, -3, 17, 6, ink, fire, 1.6);
            ctx.fillStyle = "#ffb000";
            ctx.beginPath();
            for (let i = 0; i < 8; i++) {
                const r = i % 2 ? 1.6 : 4.2, a = (i * Math.PI) / 4;
                ctx[i ? "lineTo" : "moveTo"](Math.cos(a) * r, Math.sin(a) * r);
            }
            ctx.closePath();
            ctx.fill();
        } else {
            box(-7, -7, 14, 14, "#fffcf4", color);
            box(3 - t.recoil * 2, -2.5, 17, 5, "#fffcf4", color, 1.4);
        }
        ctx.restore();
        ctx.fillStyle = heavy ? "#d4002a" : color;
        ctx.font = "italic 11px Georgia,serif";
        ctx.textAlign = "center";
        const label = t.label ?? (t.id ? "02" : "01");
        const halfLabel = ctx.measureText(label).width / 2 + 3;
        const labelX = t.label ? Math.max(halfLabel, Math.min(width - halfLabel, t.x)) : t.x;
        ctx.fillText(label, labelX, Math.max(12, t.y - TANK_RADIUS - 7));
    }
    return function render(game, trajectories = false) {
        mods = game.mods ?? null;
        const dpr = Math.min(devicePixelRatio || 1, 2),
            w = Math.max(1, Math.round(canvas.clientWidth * dpr)),
            h = Math.round((w * height) / width);
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        ctx.setTransform(w / width, 0, 0, h / height, 0, 0);
        ctx.fillStyle = "#fffcf4";
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = "#94a38a26";
        for (let x = 22; x < width; x += 22)
            for (let y = 22; y < height; y += 22) ctx.fillRect(x, y, 0.9, 0.9);
        for (const t of game.tracks) {
            ctx.save();
            ctx.translate(t.x, t.y);
            ctx.rotate(t.angle);
            ctx.fillStyle = `rgba(101,120,105,${0.1 * Math.max(0, 1 - t.age / 6)})`;
            ctx.fillRect(-1, -12, 2, 3);
            ctx.fillRect(-1, 9, 2, 3);
            ctx.restore();
        }
        if (trajectories)
            for (const t of game.tanks) {
                if (!t.alive) continue;
                const shot = traceShot(
                    game.maze,
                    t,
                    autoAim ? t.turretAngle : t.angle,
                    1.4,
                );
                ctx.save();
                ctx.beginPath();
                shot.points.forEach((p, i) =>
                    i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y),
                );
                ctx.setLineDash([3, 7]);
                ctx.strokeStyle = COLORS[t.id] + "70";
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.restore();
            }
        for (const wall of game.maze.walls) {
            // Fill the exact collision rectangle. Decorative strokes are clipped to it.
            ctx.save();
            ctx.beginPath();
            ctx.rect(wall.x, wall.y, wall.w, wall.h);
            ctx.clip();
            ctx.fillStyle = "#75847732";
            ctx.fillRect(wall.x, wall.y, wall.w, wall.h);
            const v = wall.h > wall.w;
            line(
                v ? wall.x + wall.w / 2 : wall.x,
                v ? wall.y : wall.y + wall.h / 2,
                v ? wall.x + wall.w / 2 : wall.x + wall.w,
                v ? wall.y + wall.h : wall.y + wall.h / 2,
                "#566659",
                3,
            );
            ctx.restore();
        }
        for (const b of game.bullets) {
            const color = COLORS[b.team ?? b.owner];
            ctx.strokeStyle = color + "50";
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            b.trail.forEach((p, i) =>
                i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y),
            );
            ctx.stroke();
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(b.x, b.y, b.r ?? 3, 0, Math.PI * 2);
            ctx.fill();
        }
        game.tanks.forEach(tank);
        // The modifier panel's invincibility: a dashed shield around that tank.
        for (const t of game.tanks) {
            if (!t.alive || !game.mods?.[t.id]?.invincible) continue;
            ctx.save();
            ctx.strokeStyle = COLORS[t.team ?? t.id] + "aa";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([5, 4]);
            ctx.beginPath();
            ctx.arc(t.x, t.y, TANK_RADIUS + 8, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
        }
        for (const e of game.effects) {
            const age = 1 - e.life / e.max,
                color = COLORS[e.team ?? e.owner];
            ctx.save();
            ctx.globalAlpha = 1 - age;
            for (let i = 0; i < 10; i++) {
                const a = (i * Math.PI) / 5,
                    r = (e.type === "explosion" ? 32 : 12) * (1 + age);
                line(
                    e.x + Math.cos(a) * r * 0.55,
                    e.y + Math.sin(a) * r * 0.55,
                    e.x + Math.cos(a) * r,
                    e.y + Math.sin(a) * r,
                    color,
                    1.2,
                    i,
                );
            }
            if (e.type === "explosion") {
                ctx.font = "italic 16px Georgia,serif";
                ctx.fillStyle = color;
                ctx.textAlign = "center";
                ctx.fillText("POW!", e.x, e.y + 5);
            }
            ctx.restore();
        }
    };
}
