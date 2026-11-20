(function (root) {
  "use strict";

  const WIDTH = 960;
  const HEIGHT = 540;
  const TICKS_PER_SECOND = 30;
  const DEMO_SEED = 2;
  const BORDER = 33;
  const FLOORS = ["Intake", "Smelter", "Furnace"];
  const UPGRADES = Object.freeze({
    repeater: { name: "Repeater", detail: "Fire two ticks sooner." },
    overcharge: { name: "Overcharge", detail: "Each shot deals +1 damage." },
    plating: { name: "Heat shield", detail: "+25 max hull and repair 35." },
    coolant: { name: "Coolant", detail: "Dash recharges 16 ticks faster." },
    repair: { name: "Field rebuild", detail: "Repair up to 65 hull." },
  });
  const AXIS = { type: "number", minimum: -1, maximum: 1 };
  const AIM = { type: "number" };
  const BUTTON = { type: "boolean" };
  const TICK_INPUT_FIELDS = { dx: AXIS, dy: AXIS, aimX: AIM, aimY: AIM, fire: BUTTON, dash: BUTTON };
  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function randomFor(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let bits = Math.imul(state ^ (state >>> 15), state | 1);
      bits ^= bits + Math.imul(bits ^ (bits >>> 7), bits | 61);
      return ((bits ^ (bits >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffle(items, random) {
    for (let i = items.length - 1; i > 0; i -= 1) {
      const j = Math.floor(random() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
    return items;
  }

  function blocked(walls, x, y, radius) {
    if (x - radius < BORDER || x + radius > WIDTH - BORDER ||
        y - radius < BORDER || y + radius > HEIGHT - BORDER) return true;
    return walls.some(wall => {
      const nearX = clamp(x, wall.x, wall.x + wall.w);
      const nearY = clamp(y, wall.y, wall.y + wall.h);
      return Math.hypot(x - nearX, y - nearY) <= radius;
    });
  }

  function clearPath(walls, from, to, radius = 0) {
    const checks = Math.ceil(distance(from, to) / 9);
    for (let i = 1; i <= checks; i += 1) {
      const fraction = i / checks;
      if (blocked(walls, from.x + (to.x - from.x) * fraction,
        from.y + (to.y - from.y) * fraction, radius)) return false;
    }
    return true;
  }

  function slide(run, body, dx, dy) {
    if (!blocked(run.walls, body.x + dx, body.y, body.r)) body.x += dx;
    if (!blocked(run.walls, body.x, body.y + dy, body.r)) body.y += dy;
  }

  function makeFloor(run) {
    const random = randomFor(run.seed ^ Math.imul(run.floor, 0x9e3779b9));
    run.walls = [];
    for (const x of [280, 450, 630]) {
      for (const y of [120, 355]) {
        if (random() < 0.18) continue;
        run.walls.push({
          x: x + Math.floor(random() * 65),
          y: y + Math.floor(random() * 42) - 21,
          w: 64 + Math.floor(random() * 55),
          h: 42 + Math.floor(random() * 26),
        });
      }
    }
    if (run.floor !== 1 && random() < 0.7) {
      run.walls.push({ x: 468, y: 226, w: 41, h: 88 });
    }
    run.gate = { x: 868, y: 270, open: false };
    Object.assign(run.player, {
      x: 92, y: 270, angle: 0, dashTicks: 0, dashCooldown: 0, safeTicks: 0, fireCooldown: 0,
    });
    run.shots = [];
    run.pickups = [];
    run.upgrades = [];
    run.enemies = [];
    const locations = shuffle([
      { x: 414, y: 87 }, { x: 680, y: 82 }, { x: 820, y: 102 },
      { x: 360, y: 263 }, { x: 610, y: 274 }, { x: 762, y: 232 },
      { x: 270, y: 464 }, { x: 520, y: 455 }, { x: 716, y: 470 },
      { x: 813, y: 416 },
    ], random);
    const lineup = [
      ["skitter", "sentry", "skitter", "skitter", "sentry"],
      ["charger", "skitter", "sentry", "charger", "sentry", "skitter", "skitter"],
      ["furnace", "sentry", "skitter", "charger"],
    ][run.floor - 1];
    lineup.forEach((kind, index) => {
      const stats = {
        skitter: { hp: 2, r: 14, speed: 2.25 },
        sentry: { hp: 3, r: 17, speed: 0.8 },
        charger: { hp: 4, r: 20, speed: 1.55 },
        furnace: { hp: 29, r: 33, speed: 1.25 },
      }[kind];
      const spawn = kind === "furnace" ? { x: 783, y: 272 } : locations.find(point =>
        !blocked(run.walls, point.x, point.y, stats.r + 5) &&
        run.enemies.every(other => distance(point, other) > 74));
      if (!spawn) throw new Error("Could not place a vault enemy");
      run.enemies.push({
        ...spawn, ...stats, id: index, kind, maxHp: stats.hp, angle: Math.PI,
        cooldown: 18 + Math.floor(random() * 35), wind: 0, flash: 0,
      });
    });
    run.floorEnemyCount = run.enemies.length;
    run.status = "playing";
  }

  function createRun(seed = DEMO_SEED) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      throw new RangeError("Seed must be an unsigned 32-bit integer");
    }
    const run = {
      seed, floor: 1, tick: 0, kills: 0, events: [],
      player: {
        r: 15, hp: 100, maxHp: 100, damage: 1, firePeriod: 7, dashPeriod: 54,
        dashX: 0, dashY: 0,
      },
    };
    makeFloor(run);
    return run;
  }

  function shoot(run, actor, angle, speed, damage, friendly) {
    const vx = Math.cos(angle) * speed;
    const vy = Math.sin(angle) * speed;
    // Start at the actor center so a muzzle next to cover cannot skip the wall.
    run.shots.push({ x: actor.x, y: actor.y, vx, vy, friendly, damage, life: friendly ? 69 : 126 });
    if (friendly) run.events.push({ type: "shot", x: actor.x, y: actor.y, angle });
  }

  function hurtPlayer(run, damage) {
    const player = run.player;
    if (player.safeTicks > 0) return;
    player.hp = Math.max(0, player.hp - damage);
    player.safeTicks = 12;
    run.events.push({ type: "hurt", x: player.x, y: player.y, damage });
    if (player.hp === 0) run.status = "lost";
  }

  function movePlayer(run, input) {
    const player = run.player;
    player.safeTicks = Math.max(0, player.safeTicks - 1);
    player.dashCooldown = Math.max(0, player.dashCooldown - 1);
    player.fireCooldown = Math.max(0, player.fireCooldown - 1);
    const dx = Number.isFinite(input.dx) ? clamp(input.dx, -1, 1) : 0;
    const dy = Number.isFinite(input.dy) ? clamp(input.dy, -1, 1) : 0;
    const length = Math.hypot(dx, dy);
    if (Number.isFinite(input.aimX) && Number.isFinite(input.aimY)) {
      player.angle = Math.atan2(input.aimY - player.y, input.aimX - player.x);
    } else if (input.fire) {
      const closest = run.enemies.reduce((best, enemy) =>
        !best || distance(player, enemy) < distance(player, best) ? enemy : best, null);
      if (closest) player.angle = Math.atan2(closest.y - player.y, closest.x - player.x);
    } else if (length > 0) {
      player.angle = Math.atan2(dy, dx);
    }
    if (input.dash && player.dashCooldown === 0) {
      player.dashTicks = 6;
      player.dashCooldown = player.dashPeriod;
      player.safeTicks = Math.max(player.safeTicks, 8);
      player.dashX = length ? dx / length : Math.cos(player.angle);
      player.dashY = length ? dy / length : Math.sin(player.angle);
      run.events.push({ type: "dash", x: player.x, y: player.y });
    }
    if (player.dashTicks > 0) {
      player.dashTicks -= 1;
      slide(run, player, player.dashX * 13, player.dashY * 13);
    } else if (length > 0) {
      slide(run, player, 5.55 * dx / Math.max(1, length), 5.55 * dy / Math.max(1, length));
    }
    if (input.fire && player.fireCooldown === 0) {
      shoot(run, player, player.angle, 13.2, player.damage, true);
      player.fireCooldown = player.firePeriod;
    }
  }

  function moveEnemies(run) {
    const player = run.player;
    for (const enemy of run.enemies) {
      const gap = distance(enemy, player);
      const targetAngle = Math.atan2(player.y - enemy.y, player.x - enemy.x);
      enemy.flash = Math.max(0, enemy.flash - 1);
      enemy.cooldown = Math.max(0, enemy.cooldown - 1);
      if (enemy.wind > 0) {
        enemy.wind -= 1;
        if (enemy.wind !== 0) continue;
        if (enemy.kind === "sentry") {
          shoot(run, enemy, enemy.angle, 6.0, 11, false);
          enemy.cooldown = 50;
        } else if (enemy.kind === "furnace") {
          for (let i = 0; i < 10; i += 1) {
            shoot(run, enemy, enemy.angle + (i + 0.5) * Math.PI / 5, 5.15, 13, false);
          }
          shoot(run, enemy, enemy.angle, 7.1, 14, false);
          enemy.cooldown = enemy.hp < 13 ? 25 : 43;
        } else {
          if (gap < enemy.r + player.r + 21) hurtPlayer(run, enemy.kind === "charger" ? 16 : 9);
          enemy.cooldown = 21;
        }
        continue;
      }
      enemy.angle = targetAngle;
      const range = enemy.kind === "furnace" ? 550 : enemy.kind === "sentry" ? 490 : enemy.r + player.r + 13;
      if (enemy.cooldown === 0 && gap < range && clearPath(run.walls, enemy, player)) {
        enemy.wind = enemy.kind === "furnace" ? 22 : enemy.kind === "sentry" ? 17 : 12;
        continue;
      }
      const standOff = enemy.kind === "sentry" ? 255 : enemy.kind === "furnace" ? 143 : enemy.r + player.r;
      if (gap > standOff || !clearPath(run.walls, enemy, player)) {
        const dx = Math.cos(targetAngle) * enemy.speed;
        const dy = Math.sin(targetAngle) * enemy.speed;
        const x = enemy.x;
        const y = enemy.y;
        slide(run, enemy, dx, dy);
        if (x === enemy.x && y === enemy.y) {
          const flank = enemy.id % 2 === 0 ? 1 : -1;
          slide(run, enemy, dy * flank, -dx * flank);
        }
      }
    }
  }

  function advanceShots(run) {
    run.shots = run.shots.filter(shot => {
      shot.life -= 1;
      const segments = Math.ceil(Math.hypot(shot.vx, shot.vy) / 5);
      for (let segment = 0; segment < segments; segment += 1) {
        shot.x += shot.vx / segments;
        shot.y += shot.vy / segments;
        if (blocked(run.walls, shot.x, shot.y, 1)) return false;
        if (shot.friendly) {
          const hit = run.enemies.find(enemy => enemy.hp > 0 && distance(shot, enemy) <= enemy.r + 2);
          if (hit) {
            hit.hp -= shot.damage;
            hit.flash = 4;
            run.events.push({ type: "hit", x: hit.x, y: hit.y });
            return false;
          }
        } else if (distance(shot, run.player) <= run.player.r + 3) {
          hurtPlayer(run, shot.damage);
          return false;
        }
      }
      return shot.life > 0;
    });
    run.enemies = run.enemies.filter(enemy => {
      if (enemy.hp > 0) return true;
      run.kills += 1;
      run.events.push({ type: "burst", x: enemy.x, y: enemy.y });
      if (enemy.id % 3 === 1) run.pickups.push({ x: enemy.x, y: enemy.y, heal: 19 });
      return false;
    });
  }

  function step(run, input = {}) {
    run.events = [];
    if (run.status === "upgrade") {
      if (!run.upgrades.includes(input.upgrade)) return run;
      const player = run.player;
      switch (input.upgrade) {
        case "repeater": player.firePeriod = Math.max(3, player.firePeriod - 2); break;
        case "overcharge": player.damage += 1; break;
        case "plating": player.maxHp += 25; player.hp += 35; break;
        case "coolant": player.dashPeriod = Math.max(22, player.dashPeriod - 16); break;
        case "repair": player.hp += 65; break;
      }
      player.hp = Math.min(player.hp, player.maxHp);
      run.floor += 1;
      makeFloor(run);
      return run;
    }
    if (run.status !== "playing") return run;

    run.tick += 1;
    movePlayer(run, input);
    moveEnemies(run);
    advanceShots(run);
    if (run.status === "lost") return run;

    run.pickups = run.pickups.filter(pickup => {
      if (distance(pickup, run.player) > 26 || run.player.hp === run.player.maxHp) return true;
      run.player.hp = Math.min(run.player.maxHp, run.player.hp + pickup.heal);
      run.events.push({ type: "repair", x: pickup.x, y: pickup.y });
      return false;
    });
    if (!run.enemies.length && !run.gate.open) {
      run.gate.open = true;
      run.shots = run.shots.filter(shot => shot.friendly);
      run.events.push({ type: "clear", x: run.gate.x, y: run.gate.y });
    }
    if (run.gate.open && distance(run.player, run.gate) < 33) {
      if (run.floor === FLOORS.length) {
        run.status = "won";
      } else {
        run.status = "upgrade";
        run.upgrades = shuffle(Object.keys(UPGRADES), randomFor(run.seed + run.floor * 347)).slice(0, 3);
      }
    }
    return run;
  }

  // The demo steers around cover on a small navigation grid. The simulation
  // and its legal inputs are shared with keyboard and pointer play.
  function waypoint(run, target, needsShot = false) {
    const start = run.player;
    if (clearPath(run.walls, start, target, start.r)) return target;
    const cell = 40;
    const cols = WIDTH / cell;
    const rows = Math.floor(HEIGHT / cell);
    const point = i => ({ x: (i % cols) * cell + 20, y: Math.floor(i / cols) * cell + 30 });
    const accessible = Array.from({ length: cols * rows }, (_, i) => {
      const p = point(i);
      return !blocked(run.walls, p.x, p.y, start.r + 1);
    });
    const sources = accessible.map((canWalk, i) => ({ canWalk, i, gap: distance(point(i), start) }))
      .filter(entry => entry.canWalk && entry.gap < 102 && clearPath(run.walls, start, point(entry.i), start.r))
      .sort((a, b) => a.gap - b.gap);
    const origin = sources[0]?.i;
    if (origin === undefined) return target;
    const queue = [origin];
    const first = new Map([[origin, origin]]);
    let best = origin;
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const current = queue[cursor];
      const position = point(current);
      if (needsShot && clearPath(run.walls, position, target)) {
        best = current;
        break;
      }
      if (distance(position, target) < distance(point(best), target)) best = current;
      const col = current % cols;
      const row = Math.floor(current / cols);
      for (const next of [
        col > 0 ? current - 1 : -1,
        col < cols - 1 ? current + 1 : -1,
        row > 0 ? current - cols : -1,
        row < rows - 1 ? current + cols : -1,
      ]) {
        if (!accessible[next] || first.has(next) ||
            !clearPath(run.walls, point(current), point(next), start.r)) continue;
        first.set(next, current === origin ? next : first.get(current));
        queue.push(next);
      }
    }
    const near = point(first.get(best));
    return clearPath(run.walls, start, near, start.r) ? near : point(origin);
  }

  function demoInput(run) {
    if (run.status === "upgrade") {
      return { upgrade: ["overcharge", "repeater", "plating", "repair", "coolant"]
        .find(name => run.upgrades.includes(name)) };
    }
    const player = run.player;
    const ordered = [...run.enemies].sort((a, b) => distance(player, a) - distance(player, b));
    const visible = ordered.find(enemy => clearPath(run.walls, player, enemy));
    const target = visible ?? ordered[0] ?? run.gate;
    const gap = distance(player, target);
    let move = target;
    if (!visible) {
      move = waypoint(run, target, ordered.length > 0);
    } else if (gap < (visible.kind === "furnace" ? 176 : 123)) {
      move = { x: player.x * 2 - target.x, y: player.y * 2 - target.y };
    } else if (gap < 320) {
      move = player;
    }

    let dx = move.x - player.x;
    let dy = move.y - player.y;
    let dash = false;
    const threat = run.shots.find(shot => !shot.friendly && distance(shot, player) < 120 &&
      Array.from({ length: 10 }, (_, i) => i + 1).some(i =>
        Math.hypot(shot.x + shot.vx * i - player.x, shot.y + shot.vy * i - player.y) < 30));
    if (threat) {
      const sign = blocked(run.walls, player.x - threat.vy * 11, player.y + threat.vx * 11, player.r) ? -1 : 1;
      dx = -threat.vy * sign;
      dy = threat.vx * sign;
      dash = distance(threat, player) < 62;
    }
    const length = Math.max(1, Math.hypot(dx, dy));
    if (length > 2 && blocked(run.walls, player.x + dx / length * 8, player.y + dy / length * 8, player.r)) {
      const around = waypoint(run, visible ? target : move);
      dx = around.x - player.x;
      dy = around.y - player.y;
    }
    const scale = Math.max(1, Math.hypot(dx, dy));
    return { dx: dx / scale, dy: dy / scale, aimX: target.x, aimY: target.y, fire: Boolean(visible), dash };
  }

  function observe(run) {
    return structuredClone(run);
  }

  function actionSpace(run) {
    if (run.status === "upgrade") {
      return { kind: "upgrade", choices: run.upgrades.map(id => ({ id, ...UPGRADES[id] })) };
    }
    if (run.status !== "playing") return null;
    return {
      kind: "tick_input",
      fields: TICK_INPUT_FIELDS,
      // movePlayer reduces both cooldowns before checking the input.
      fire_ready: run.player.fireCooldown <= 1,
      dash_ready: run.player.dashCooldown <= 1,
    };
  }

  function observeForAI(run) {
    return structuredClone({
      schema_version: 1,
      game: "ember-vault",
      seed: String(run.seed),
      status: run.status,
      done: run.status === "won" || run.status === "lost",
      clock: { unit: "tick", value: run.tick, ticks_per_second: TICKS_PER_SECOND },
      floor: run.floor,
      floor_name: FLOORS[run.floor - 1],
      kills: run.kills,
      map: {
        kind: "arena",
        units: "pixels",
        width: WIDTH,
        height: HEIGHT,
        bounds: { left: BORDER, top: BORDER, right: WIDTH - BORDER, bottom: HEIGHT - BORDER },
        walls: run.walls,
      },
      gate: run.gate,
      player: run.player,
      enemies: run.enemies,
      projectiles: run.shots,
      pickups: run.pickups,
      events: run.events,
      action_space: actionSpace(run),
    });
  }

  function stepForAI(run, input) {
    if (run.status !== "playing" && run.status !== "upgrade") throw new RangeError("the game has ended");
    if (!input || typeof input !== "object" || Array.isArray(input)) throw new TypeError("expected an action object");
    const action = { ...input };
    if (run.status === "upgrade") {
      if (Object.keys(action).length !== 1 || !run.upgrades.includes(action.upgrade)) {
        throw new RangeError("expected an offered upgrade as the only action field");
      }
    } else {
      for (const [name, value] of Object.entries(action)) {
        if (!Object.hasOwn(TICK_INPUT_FIELDS, name)) throw new RangeError("unknown action field: " + name);
        const { type, minimum, maximum } = TICK_INPUT_FIELDS[name];
        if (typeof value !== type || (type === "number" && !Number.isFinite(value))) {
          throw new TypeError(name + " must be a " + type);
        }
        if (minimum !== undefined && (value < minimum || value > maximum)) {
          throw new RangeError(`${name} must be between ${minimum} and ${maximum}`);
        }
      }
      if (Object.hasOwn(action, "aimX") !== Object.hasOwn(action, "aimY")) {
        throw new RangeError("aimX and aimY must be supplied together");
      }
    }
    step(run, action);
    return observeForAI(run);
  }

  const api = Object.freeze({
    WIDTH, HEIGHT, TICKS_PER_SECOND, DEMO_SEED, FLOORS, UPGRADES,
    createRun, step, observe, demoInput, observeForAI, stepForAI,
  });
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.EmberVault = api;
})(globalThis);
