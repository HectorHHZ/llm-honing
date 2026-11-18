(function (root) {
  "use strict";

  const { WIDTH, HEIGHT, FLOORS } = root.EmberVault;

  function plate(ctx, x, y, w, h, radius, fill, stroke) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, radius);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  function circle(ctx, x, y, radius, color) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  function line(ctx, x, y, toX, toY, color, width = 1) {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(toX, toY);
    ctx.lineWidth = width;
    ctx.strokeStyle = color;
    ctx.stroke();
  }

  function ambient(ctx, run, seconds) {
    const background = ctx.createLinearGradient(0, 0, 0, HEIGHT);
    background.addColorStop(0, "#252027");
    background.addColorStop(1, "#17161d");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    const floor = ctx.createRadialGradient(506, 215, 20, 506, 215, 610);
    floor.addColorStop(0, "#353037");
    floor.addColorStop(1, "#231e25");
    plate(ctx, 29, 29, 902, 482, 8, floor, "#534047");

    for (let y = 34, row = 0; y < 502; y += 40, row += 1) {
      for (let x = 34, col = 0; x < 926; x += 40, col += 1) {
        const n = (row * 91 + col * 53 + run.seed) % 11;
        ctx.fillStyle = n > 7 ? "#f8d1b108" : "#0e0b101c";
        ctx.fillRect(x, y, Math.min(38, 926 - x), Math.min(38, 506 - y));
        if (n % 3 === 0) {
          circle(ctx, x + 4, y + 4, 1, "#c1946e38");
        }
      }
    }

    const warm = ctx.createRadialGradient(481, 269, 30, 481, 269, 480);
    warm.addColorStop(0, "#d7712a10");
    warm.addColorStop(1, "#d7712a00");
    ctx.fillStyle = warm;
    ctx.fillRect(30, 30, 900, 480);
    for (const y of [43, HEIGHT - 43]) {
      line(ctx, 43, y, WIDTH - 43, y, "#9e685845", 2);
      for (let x = 48; x < WIDTH - 45; x += 18) {
        line(ctx, x, y - 3, x + 7, y + 3, "#e4a46755", 2);
      }
    }

    ctx.font = "700 10px ui-monospace, monospace";
    ctx.textBaseline = "middle";
    ctx.letterSpacing = "2px";
    ctx.fillStyle = "#bb9e9f";
    ctx.fillText("VAULT / 0" + run.floor + "  " + FLOORS[run.floor - 1].toUpperCase(), 48, 18);
    ctx.textAlign = "right";
    ctx.fillText("EMBER DIVISION  /  " + String(run.seed).padStart(4, "0"), WIDTH - 47, HEIGHT - 18);
    ctx.textAlign = "left";
    ctx.letterSpacing = "0px";
    for (let i = 0; i < 24; i += 1) {
      const x = 47 + ((i * 159 + seconds * (4 + i % 5)) % 865);
      const y = 55 + ((i * 91 + 390 - (seconds * (8 + i % 7) % 390)) % 390);
      circle(ctx, x, y, 0.8 + i % 3 * 0.35, "#ffb66a48");
    }
  }

  function cover(ctx, wall) {
    const { x, y, w, h } = wall;
    plate(ctx, x + 5, y + 8, w, h, 5, "#110b13a0");
    plate(ctx, x, y, w, h, 4, "#51404a", "#956a5e");
    plate(ctx, x + 5, y + 5, w - 10, h - 14, 3, "#3f313d", "#765661");
    line(ctx, x + 4, y + 4, x + w - 4, y + 4, "#b17d6b", 2);
    line(ctx, x + 5, y + h - 5, x + w - 5, y + h - 5, "#271e2a", 3);
    for (let offset = 12; offset <= w - 12; offset += 14) {
      line(ctx, x + offset, y + 10, x + offset, y + h - 16, "#826364", 2);
      circle(ctx, x + offset, y + h - 6, 1.4, "#d29d74");
    }
  }

  function gate(ctx, run, seconds) {
    const { x, y, open } = run.gate;
    const light = open ? "#7fe1b0" : "#ee845d";
    const halo = ctx.createRadialGradient(x, y, 2, x, y, 68);
    halo.addColorStop(0, open ? "#51e99c65" : "#ec694b34");
    halo.addColorStop(1, "#00000000");
    ctx.fillStyle = halo;
    ctx.fillRect(x - 70, y - 70, 140, 140);
    plate(ctx, x - 28, y - 32, 56, 64, 8, "#241c26", light);
    plate(ctx, x - 21, y - 25, 42, 50, 5, open ? "#185345" : "#4a2934", "#8d6255");
    for (const dx of [-24, 24]) {
      line(ctx, x + dx, y - 19, x + dx, y + 19, light, 4);
    }
    if (open) {
      const offset = 3 + Math.sin(seconds * 4) * 4;
      line(ctx, x - 10 + offset, y - 10, x + 1 + offset, y, "#b0f6ce", 3);
      line(ctx, x + 1 + offset, y, x - 10 + offset, y + 10, "#b0f6ce", 3);
    } else {
      line(ctx, x - 14, y - 5, x + 14, y - 5, "#b95c4f", 2);
      line(ctx, x - 14, y + 5, x + 14, y + 5, "#b95c4f", 2);
    }
    ctx.textAlign = "center";
    ctx.font = "800 10px ui-monospace, monospace";
    ctx.fillStyle = light;
    ctx.fillText(open ? "EXIT OPEN" : "SEALED", x, y + 46);
    ctx.textAlign = "left";
  }

  function drawEnemy(ctx, enemy, seconds) {
    const boss = enemy.kind === "furnace";
    const hue = boss ? "#fb633e" : enemy.kind === "sentry" ? "#e77982" : "#ee9467";
    if (enemy.wind) {
      const pulse = 0.6 + 0.4 * Math.sin(seconds * 27);
      ctx.save();
      ctx.globalAlpha = pulse;
      ctx.strokeStyle = hue;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(enemy.x, enemy.y, enemy.r + 8, 0, Math.PI * 2);
      ctx.stroke();
      if (boss || enemy.kind === "sentry") {
        ctx.setLineDash([9, 8]);
        line(ctx, enemy.x, enemy.y, enemy.x + Math.cos(enemy.angle) * 245,
          enemy.y + Math.sin(enemy.angle) * 245, "#ff8977b0", 1.5);
      }
      ctx.restore();
    }
    circle(ctx, enemy.x + 3, enemy.y + 6, enemy.r + 2, "#110b13a8");
    ctx.save();
    ctx.translate(enemy.x, enemy.y);
    ctx.rotate(enemy.angle);
    const r = enemy.r;
    ctx.beginPath();
    for (let i = 0, sides = boss ? 8 : enemy.kind === "sentry" ? 4 : 6; i < sides; i += 1) {
      const angle = Math.PI * 2 * i / sides;
      const x = Math.cos(angle) * r;
      const y = Math.sin(angle) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = enemy.flash ? "#fff0e2" : "#543044";
    ctx.fill();
    ctx.strokeStyle = hue;
    ctx.lineWidth = boss ? 3.5 : 2.5;
    ctx.stroke();
    if (boss) {
      for (const dy of [-17, 17]) plate(ctx, -15, dy - 4, 27, 8, 2, "#bb4e44", "#ff8e62");
    } else if (enemy.kind === "charger") {
      line(ctx, 4, -7, 18, -14, hue, 3);
      line(ctx, 4, 7, 18, 14, hue, 3);
    }
    plate(ctx, r * 0.21, -3, r * 0.68, 6, 2, hue);
    circle(ctx, -r * 0.28, 0, boss ? 8 : 4.5, enemy.wind ? "#fff3b2" : "#ffb38d");
    ctx.restore();
    if (enemy.hp < enemy.maxHp || boss) {
      const w = boss ? 80 : 32;
      plate(ctx, enemy.x - w / 2, enemy.y - r - 13, w, 4, 2, "#2c1721");
      plate(ctx, enemy.x - w / 2, enemy.y - r - 13, w * enemy.hp / enemy.maxHp, 4, 2, hue);
    }
  }

  function drawPlayer(ctx, player) {
    circle(ctx, player.x + 2, player.y + 6, player.r + 3, "#0d0a129c");
    if (player.safeTicks || player.dashTicks) circle(ctx, player.x, player.y, player.r + 6, "#95edff3f");
    ctx.save();
    ctx.translate(player.x, player.y);
    ctx.rotate(player.angle);
    plate(ctx, -13, -13, 24, 8, 3, "#333543", "#adc0c1");
    plate(ctx, -13, 5, 24, 8, 3, "#333543", "#adc0c1");
    plate(ctx, -12, -9, 23, 18, 5, "#d4dfd6", "#fbefd2");
    plate(ctx, 1, -3, 23, 6, 2, "#fff0d4", "#c89561");
    circle(ctx, -3, 0, 5, "#466b74");
    circle(ctx, -3, 0, 2.5, "#b6f5e6");
    ctx.restore();
  }

  function effects(ctx, run, trail) {
    for (const event of trail) {
      const age = run.tick - event.tick;
      if (age < 0 || age > 22) continue;
      const burst = event.type === "burst";
      const color = event.type === "repair" ? "#89e9ae" :
        event.type === "hurt" ? "#ff7859" : event.type === "dash" ? "#b6f6ed" : "#ffbf7d";
      ctx.save();
      ctx.globalAlpha = 1 - age / 23;
      if (event.type === "shot") {
        const tipX = event.x + Math.cos(event.angle) * 27;
        const tipY = event.y + Math.sin(event.angle) * 27;
        if (age < 4) circle(ctx, tipX, tipY, 7 - age, "#fff2ac");
      } else {
        const radius = 8 + age * (burst ? 2.1 : 1.1);
        ctx.beginPath();
        ctx.arc(event.x, event.y, radius, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = burst ? 3 : 1.4;
        ctx.stroke();
        if (burst) {
          for (let i = 0; i < 9; i += 1) {
            const angle = i * 2.4;
            circle(ctx, event.x + Math.cos(angle) * radius, event.y + Math.sin(angle) * radius, 2, color);
          }
        }
      }
      ctx.restore();
    }
  }

  function paint(ctx, run, seconds = 0, trail = [], pointer = null) {
    ambient(ctx, run, seconds);
    for (const wall of run.walls) cover(ctx, wall);
    gate(ctx, run, seconds);
    for (const pickup of run.pickups) {
      const y = pickup.y + Math.sin(seconds * 4 + pickup.x) * 3;
      circle(ctx, pickup.x, y, 14, "#64dfad33");
      plate(ctx, pickup.x - 8, y - 8, 16, 16, 4, "#173c34", "#82dfb0");
      line(ctx, pickup.x - 4, y, pickup.x + 4, y, "#b6ffd5", 2);
      line(ctx, pickup.x, y - 4, pickup.x, y + 4, "#b6ffd5", 2);
    }
    for (const shot of run.shots) {
      const hue = shot.friendly ? "#ffd08b" : "#ff745b";
      line(ctx, shot.x - shot.vx * 1.65, shot.y - shot.vy * 1.65, shot.x, shot.y, hue, shot.friendly ? 3.2 : 4.3);
      circle(ctx, shot.x, shot.y, shot.friendly ? 2.1 : 3.1, "#fff0c3");
    }
    for (const enemy of run.enemies) drawEnemy(ctx, enemy, seconds);
    drawPlayer(ctx, run.player);
    effects(ctx, run, trail);
    if (pointer) {
      const { x, y } = pointer;
      ctx.strokeStyle = "#f8d2a1";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.stroke();
      line(ctx, x - 15, y, x - 6, y, "#f8d2a1");
      line(ctx, x + 6, y, x + 15, y, "#f8d2a1");
      line(ctx, x, y - 15, x, y - 6, "#f8d2a1");
      line(ctx, x, y + 6, x, y + 15, "#f8d2a1");
    }
  }

  root.EmberVaultView = Object.freeze({ paint });
})(globalThis);
