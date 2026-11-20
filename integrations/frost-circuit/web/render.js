(function (root) {
  "use strict";

  const { COLS, ROWS, danger } = root.FrostCircuit;
  const TILE = 52;
  const MARGIN = 20;
  const WIDTH = COLS * TILE + MARGIN * 2;
  const HEIGHT = ROWS * TILE + MARGIN * 2;

  function box(ctx, x, y, w, h, radius, fill, stroke, width = 1) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, radius);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.stroke();
    }
  }

  function line(ctx, x1, y1, x2, y2, color, width = 1) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = width;
    ctx.strokeStyle = color;
    ctx.stroke();
  }

  function dot(ctx, x, y, radius, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  function diamond(ctx, x, y, radius, color, stroke) {
    ctx.beginPath();
    ctx.moveTo(x, y - radius);
    ctx.lineTo(x + radius * 0.78, y);
    ctx.lineTo(x, y + radius);
    ctx.lineTo(x - radius * 0.78, y);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }

  const center = at => ({
    x: MARGIN + (at.x + 0.5) * TILE,
    y: MARGIN + (at.y + 0.5) * TILE,
  });

  function floor(ctx, run, time) {
    const backdrop = ctx.createRadialGradient(410, 260, 30, 410, 260, 600);
    backdrop.addColorStop(0, "#263c52");
    backdrop.addColorStop(1, "#111f33");
    ctx.fillStyle = backdrop;
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    const warned = danger(run);
    for (let y = 0; y < ROWS; y += 1) {
      for (let x = 0; x < COLS; x += 1) {
        const index = y * COLS + x;
        const px = MARGIN + x * TILE;
        const py = MARGIN + y * TILE;
        const tint = (x * 23 + y * 37 + run.seed + run.floor * 79) % 12;
        if (!run.tiles[index]) {
          if (tint > 2) dot(ctx, px + 26, py + 26, 1.2, "#80b2c333");
          if (tint === 6) line(ctx, px + 6, py + 19, px + 20, py + 9, "#91c9de1c", 2);
          continue;
        }
        box(ctx, px + 1, py + 3, 51, 50, 3, "#0b1928");
        box(ctx, px + 1, py + 1, 49, 48, 2, tint > 5 ? "#304e63" : "#2b455c", "#506e82");
        line(ctx, px + 6, py + 46, px + 45, py + 46, "#152b3f");
        line(ctx, px + 4, py + 4, px + 46, py + 4, "#567589");
        dot(ctx, px + 7, py + 7, 1.2, "#8ab1bd");
        dot(ctx, px + 43, py + 43, 1.2, "#7ba3b2");
        if (tint > 8) {
          line(ctx, px + 11, py + 35, px + 19, py + 28, "#6393a44b");
          line(ctx, px + 19, py + 28, px + 31, py + 33, "#6393a44b");
        }
        if (warned.has(index)) {
          box(ctx, px + 2, py + 2, 47, 46, 2, "#b331405e", "#ff8075", 1.6);
          ctx.save();
          ctx.beginPath();
          ctx.rect(px + 3, py + 3, 45, 44);
          ctx.clip();
          for (let i = -TILE; i < TILE; i += 11) {
            line(ctx, px + i, py + TILE, px + i + TILE, py, "#f78c8730", 3);
          }
          ctx.restore();
          ctx.fillStyle = "#ffc0a8";
          ctx.font = "800 13px ui-monospace, monospace";
          ctx.fillText("!", px + 37, py + 17);
        }
      }
    }
    const snow = (time * 13) % (HEIGHT - 40);
    for (let i = 0; i < 29; i += 1) {
      const x = 22 + ((i * 251 + time * 4) % (WIDTH - 44));
      const y = 22 + ((i * 123 + snow * (0.5 + (i % 3) / 3)) % (HEIGHT - 44));
      dot(ctx, x, y, i % 2 ? 0.8 : 1.35, "#d3f7ff43");
    }
  }

  function landmarks(ctx, run, time) {
    const exit = center(run.exit);
    const gateColor = run.hasCore ? "#a6f2ba" : "#b1c8d9";
    box(ctx, exit.x - 19, exit.y - 19, 38, 38, 8, run.hasCore ? "#235d55" : "#223e54", gateColor, 2);
    box(ctx, exit.x - 11, exit.y - 12, 22, 24, 4, run.hasCore ? "#367e62" : "#2c5069", gateColor);
    if (run.hasCore) {
      line(ctx, exit.x - 5, exit.y - 4, exit.x + 5, exit.y, "#d7ffdd", 2);
      line(ctx, exit.x + 5, exit.y, exit.x - 5, exit.y + 4, "#d7ffdd", 2);
    } else {
      box(ctx, exit.x - 5, exit.y - 1, 10, 8, 2, "#9bb8c6");
      ctx.strokeStyle = "#9bb8c6";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(exit.x, exit.y - 2, 3, Math.PI, 0);
      ctx.stroke();
    }
    if (!run.hasCore) {
      const core = center(run.core);
      const sealed = run.enemies.some(enemy => enemy.kind === "warden");
      dot(ctx, core.x, core.y, 22 + Math.sin(time * 4) * 2, sealed ? "#f7968d39" : "#92f9fb34");
      diamond(ctx, core.x, core.y, 15, sealed ? "#f3a77f" : "#95f9f9", "#edfaff");
      diamond(ctx, core.x - 2, core.y - 3, 6, "#e5ffff", "#c1f4ff");
    }
    for (const item of run.items) {
      const { x, y } = center(item);
      const heal = item.kind === "medkit";
      box(ctx, x - 11, y - 11, 22, 22, 4, heal ? "#1a6157" : "#3d4f6b", heal ? "#a9f0c8" : "#bbefff");
      if (heal) {
        line(ctx, x - 6, y, x + 6, y, "#d3ffd7", 3);
        line(ctx, x, y - 6, x, y + 6, "#d3ffd7", 3);
      } else {
        line(ctx, x + 2, y - 7, x - 4, y + 1, "#d0fcff", 2.5);
        line(ctx, x - 4, y + 1, x + 4, y + 1, "#d0fcff", 2.5);
        line(ctx, x + 4, y + 1, x - 1, y + 7, "#d0fcff", 2.5);
      }
    }
  }

  function drone(ctx, enemy) {
    const { x, y } = center(enemy);
    dot(ctx, x + 2, y + 8, 17, "#121f32a0");
    const highlight = enemy.kind === "warden" ? "#fdad84" : "#ff8277";
    if (enemy.kind === "crawler") {
      for (const dy of [-8, 8]) {
        line(ctx, x - 9, y + dy, x - 17, y + dy * 1.65, highlight, 3.5);
        line(ctx, x + 9, y + dy, x + 17, y + dy * 1.65, highlight, 3.5);
      }
      box(ctx, x - 11, y - 11, 22, 22, 7, "#843f58", highlight, 2);
      dot(ctx, x, y - 3, 4, "#ffe0a0");
    } else if (enemy.kind === "sentry") {
      box(ctx, x - 14, y - 12, 28, 24, 4, "#71364f", highlight, 2);
      box(ctx, x - 7, y - 5, 14, 10, 2, "#ff8e78", "#ffe2b6");
      line(ctx, x - 16, y + 14, x + 16, y + 14, highlight, 3);
    } else {
      diamond(ctx, x, y, 23, "#7b3c55", highlight);
      diamond(ctx, x, y, 12, "#df6d65", "#ffe2b6");
      dot(ctx, x, y, 4, "#fff0b8");
      for (const dx of [-17, 17]) dot(ctx, x + dx, y, 3, "#ffae89");
    }
    if (enemy.hp < enemy.maxHp || enemy.kind === "warden") {
      box(ctx, x - 17, y - 23, 34, 4, 2, "#24182b");
      box(ctx, x - 17, y - 23, 34 * enemy.hp / enemy.maxHp, 4, 2, highlight);
    }
  }

  function rover(ctx, player) {
    const { x, y } = center(player);
    dot(ctx, x, y, 22, "#7be4e839");
    dot(ctx, x + 2, y + 7, 16, "#122838");
    box(ctx, x - 15, y - 9, 30, 22, 6, "#5e9eb7", "#e0fbff", 2);
    box(ctx, x - 10, y - 14, 20, 23, 5, "#c6edf1", "#f1ffff");
    box(ctx, x - 7, y - 9, 14, 7, 3, "#234f71", "#81cfe5");
    line(ctx, x - 5, y - 5, x + 5, y - 5, "#79fafa", 2);
    for (const dx of [-12, 12]) dot(ctx, x + dx, y + 6, 3, "#b9fdff");
  }

  function paint(ctx, run, time = 0, hover = null, effect = 0) {
    floor(ctx, run, time);
    landmarks(ctx, run, time);
    for (const enemy of run.enemies) drone(ctx, enemy);
    rover(ctx, run.player);
    if (hover && run.tiles[hover.y * COLS + hover.x] &&
        Math.abs(hover.x - run.player.x) + Math.abs(hover.y - run.player.y) === 1) {
      const { x, y } = center(hover);
      ctx.strokeStyle = "#d1feff";
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 23, y - 23, 46, 46);
    }
    if (effect > 0) {
      ctx.save();
      ctx.globalAlpha = effect * 0.72;
      for (const event of run.events) {
        if (event.type === "blast") {
          for (const cell of event.cells) {
            const { x, y } = center(cell);
            box(ctx, x - 22, y - 22, 44, 44, 5, "#ff726e74", "#ffb594", 2);
          }
        } else if (["pulse", "strike", "core", "destroy"].includes(event.type)) {
          const { x, y } = center(event);
          ctx.beginPath();
          ctx.arc(x, y, (1 - effect) * (event.type === "pulse" ? 106 : 32) + 11, 0, 2 * Math.PI);
          ctx.lineWidth = 4;
          ctx.strokeStyle = event.type === "pulse" ? "#a5ffff" : "#fff1bd";
          ctx.stroke();
        }
      }
      ctx.restore();
    }
  }

  function tileAt(x, y) {
    return { x: Math.floor((x - MARGIN) / TILE), y: Math.floor((y - MARGIN) / TILE) };
  }

  root.FrostCircuitView = Object.freeze({ WIDTH, HEIGHT, paint, tileAt });
})(globalThis);
