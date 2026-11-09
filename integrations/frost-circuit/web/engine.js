(function (root) {
  "use strict";

  const COLS = 17;
  const ROWS = 11;
  const DEMO_SEED = 1729;
  const FLOORS = ["Relay", "Cold storage", "Deep reactor"];
  const DIRECTIONS = Object.freeze({
    north: [0, -1], east: [1, 0], south: [0, 1], west: [-1, 0],
  });
  const UPGRADES = Object.freeze({
    blade: { name: "Diamond edge", detail: "Melee attacks deal +1 damage." },
    shield: { name: "Insulated shell", detail: "+3 max hull. Restore all hull." },
    pulse: { name: "Flux battery", detail: "+1 max charge and refill Pulse." },
  });

  const key = (x, y) => y * COLS + x;
  const point = index => ({ x: index % COLS, y: Math.floor(index / COLS) });
  const gap = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
  const same = (a, b) => a.x === b.x && a.y === b.y;
  const shifted = (at, [dx, dy]) => ({ x: at.x + dx, y: at.y + dy });

  function randomFor(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let bits = Math.imul(state ^ (state >>> 15), state | 1);
      bits ^= bits + Math.imul(bits ^ (bits >>> 7), bits | 61);
      return ((bits ^ (bits >>> 14)) >>> 0) / 4294967296;
    };
  }

  function walkable(run, x, y) {
    return x > 0 && x < COLS - 1 && y > 0 && y < ROWS - 1 && run.tiles[key(x, y)] === 1;
  }

  function enemyAt(run, at) {
    return run.enemies.find(enemy => enemy.hp > 0 && same(enemy, at));
  }

  function carve(tiles, from, to, horizontalFirst) {
    let { x, y } = from;
    tiles[key(x, y)] = 1;
    for (const axis of horizontalFirst ? ["x", "y"] : ["y", "x"]) {
      while ((axis === "x" ? x : y) !== to[axis]) {
        if (axis === "x") x += Math.sign(to.x - x);
        else y += Math.sign(to.y - y);
        tiles[key(x, y)] = 1;
      }
    }
  }

  function makeFloor(run) {
    const random = randomFor(run.seed ^ Math.imul(run.floor, 0x9e3779b9));
    run.tiles = Array(COLS * ROWS).fill(0);
    const rooms = [];
    for (const baseY of [2, 7]) {
      for (const baseX of [2, 7, 12]) {
        const x = baseX + Math.floor(random() * 2);
        const y = baseY + Math.floor(random() * 2);
        rooms.push({ x, y });
        const right = random() > 0.42 ? 2 : 1;
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= right; dx += 1) run.tiles[key(x + dx, y + dy)] = 1;
        }
      }
    }
    for (const [a, b] of [[0, 1], [1, 2], [3, 4], [4, 5]]) carve(run.tiles, rooms[a], rooms[b], random() < 0.5);
    const link = Math.floor(random() * 3);
    carve(run.tiles, rooms[link], rooms[link + 3], random() < 0.5);
    const secondLink = (link + 1 + Math.floor(random() * 2)) % 3;
    if (random() < 0.78) carve(run.tiles, rooms[secondLink], rooms[secondLink + 3], random() < 0.5);

    Object.assign(run.player, rooms[0]);
    run.exit = rooms[2];
    run.core = rooms[5];
    run.hasCore = false;
    run.enemies = [];
    run.items = [];
    const available = run.tiles.map((value, index) => value === 1 ? point(index) : null)
      .filter(tile => tile && gap(tile, run.player) >= 4 && !same(tile, run.exit) && !same(tile, run.core));
    function place(target) {
      const choices = available.filter(tile => gap(tile, target) <= 3);
      const pool = choices.length ? choices : available;
      const tile = pool[Math.floor(random() * pool.length)];
      available.splice(available.indexOf(tile), 1);
      return tile;
    }
    const kinds = [
      ["crawler", "sentry", "crawler"],
      ["crawler", "sentry", "crawler", "sentry", "crawler"],
      ["crawler", "sentry", "warden", "sentry", "crawler"],
    ][run.floor - 1];
    const sites = [rooms[1], rooms[3], rooms[5], rooms[4], rooms[2]];
    kinds.forEach((kind, id) => {
      const hp = kind === "warden" ? 7 : kind === "sentry" ? 3 : 2;
      run.enemies.push({ ...place(sites[id]), kind, id, hp, maxHp: hp, intent: [], cooldown: 0 });
    });
    for (const [kind, room] of [["medkit", rooms[1]], ["cell", rooms[3]], ["medkit", rooms[4]], ["cell", rooms[5]]]) {
      run.items.push({ ...place(room), kind });
    }
    run.status = "playing";
    run.events = [{ type: "floor", text: FLOORS[run.floor - 1] + " online." }];
  }

  function createRun(seed = DEMO_SEED) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      throw new RangeError("Seed must be an unsigned 32-bit integer");
    }
    const run = {
      seed, floor: 1, turn: 0, kills: 0,
      player: { hp: 10, maxHp: 10, damage: 1, charge: 2, maxCharge: 2 },
    };
    makeFloor(run);
    return run;
  }

  function damageEnemy(run, enemy, damage) {
    enemy.hp = Math.max(0, enemy.hp - damage);
    run.events.push({ type: "strike", x: enemy.x, y: enemy.y, text: enemy.kind + " takes " + damage + " damage." });
    if (enemy.hp === 0) {
      run.kills += 1;
      run.events.push({ type: "destroy", x: enemy.x, y: enemy.y, text: enemy.kind + " disabled." });
    }
  }

  function beam(run, enemy, player) {
    if (enemy.x !== player.x && enemy.y !== player.y) return [];
    if (same(enemy, player) || gap(enemy, player) > 7) return [];
    const vector = [Math.sign(player.x - enemy.x), Math.sign(player.y - enemy.y)];
    const cells = [];
    let cell = shifted(enemy, vector);
    while (walkable(run, cell.x, cell.y) && cells.length < 7) {
      cells.push(cell);
      cell = shifted(cell, vector);
    }
    return cells.some(cell => same(cell, player)) ? cells : [];
  }

  function enemyTurn(run) {
    const player = run.player;
    for (const enemy of run.enemies) {
      if (enemy.hp <= 0) continue;
      if (enemy.intent.length) {
        run.events.push({ type: "blast", cells: enemy.intent });
        if (enemy.intent.some(tile => same(tile, player))) {
          const damage = enemy.kind === "warden" ? 3 : enemy.kind === "sentry" ? 2 : 1;
          player.hp = Math.max(0, player.hp - damage);
          run.events.push({ type: "hurt", x: player.x, y: player.y, text: "Hull lost " + damage + "." });
        }
        enemy.intent = [];
        enemy.cooldown = 2;
        continue;
      }
      const range = gap(enemy, player);
      if (enemy.cooldown > 0) enemy.cooldown -= 1;
      if (enemy.cooldown === 0) {
        if (enemy.kind === "sentry") {
          enemy.intent = beam(run, enemy, player);
        } else if (enemy.kind === "warden" && range <= 3) {
          const cross = Math.abs(enemy.x - player.x) > Math.abs(enemy.y - player.y)
            ? [[0, -1], [0, 1]] : [[-1, 0], [1, 0]];
          const sides = cross.map(vector => shifted(player, vector));
          const escape = Object.values(DIRECTIONS).map(vector => shifted(player, vector)).some(tile =>
            walkable(run, tile.x, tile.y) && !enemyAt(run, tile) && !sides.some(side => same(tile, side)));
          enemy.intent = [{ x: player.x, y: player.y }, ...(escape ? sides.filter(tile =>
            walkable(run, tile.x, tile.y) && !same(tile, enemy)) : [])];
        } else if (enemy.kind === "crawler" && range === 1) {
          enemy.intent = [{ x: player.x, y: player.y }];
        }
      }
      if (enemy.intent.length || enemy.kind === "sentry" || range > 6) continue;
      const options = Object.values(DIRECTIONS).map(vector => shifted(enemy, vector))
        .filter(tile => walkable(run, tile.x, tile.y) && !enemyAt(run, tile) && !same(tile, player) &&
          gap(tile, player) < range);
      options.sort((a, b) => gap(a, player) - gap(b, player));
      if (options.length) Object.assign(enemy, options[0]);
    }
    run.enemies = run.enemies.filter(enemy => enemy.hp > 0);
    if (player.hp === 0) run.status = "lost";
  }

  function step(run, action) {
    if (run.status === "upgrade") {
      if (!Object.hasOwn(UPGRADES, action)) return run;
      const player = run.player;
      if (action === "blade") player.damage += 1;
      if (action === "shield") {
        player.maxHp += 3;
        player.hp = player.maxHp;
      }
      if (action === "pulse") {
        player.maxCharge += 1;
        player.charge = player.maxCharge;
      }
      run.floor += 1;
      makeFloor(run);
      return run;
    }
    if (run.status !== "playing") return run;

    const player = run.player;
    const vector = Object.hasOwn(DIRECTIONS, action) ? DIRECTIONS[action] : null;
    const next = vector ? shifted(player, vector) : player;
    if (vector && !walkable(run, next.x, next.y)) return run;
    if (!vector && action !== "wait" && (action !== "pulse" || player.charge === 0)) return run;
    run.events = [];
    run.turn += 1;
    if (action === "pulse") {
      player.charge -= 1;
      run.events.push({ type: "pulse", x: player.x, y: player.y, text: "Pulse discharged." });
      for (const enemy of run.enemies) {
        if (gap(enemy, player) > 2) continue;
        damageEnemy(run, enemy, 2);
        enemy.intent = [];
        enemy.cooldown = 2;
      }
    } else if (vector) {
      const hit = enemyAt(run, next);
      if (hit) damageEnemy(run, hit, player.damage);
      else Object.assign(player, next);
    }

    run.items = run.items.filter(item => {
      if (!same(item, player)) return true;
      if (item.kind === "medkit" && player.hp < player.maxHp) {
        player.hp = Math.min(player.hp + 4, player.maxHp);
        run.events.push({ type: "heal", x: player.x, y: player.y, text: "Medkit repaired hull." });
        return false;
      }
      if (item.kind === "cell" && player.charge < player.maxCharge) {
        player.charge += 1;
        run.events.push({ type: "charge", x: player.x, y: player.y, text: "Pulse charge restored." });
        return false;
      }
      return true;
    });
    enemyTurn(run);
    if (run.status === "lost") return run;

    if (!run.hasCore && same(player, run.core) && !run.enemies.some(enemy => enemy.kind === "warden")) {
      run.hasCore = true;
      run.events.push({ type: "core", x: player.x, y: player.y, text: "Core recovered. Airlock open." });
    }
    if (run.hasCore && same(player, run.exit)) {
      run.status = run.floor === FLOORS.length ? "won" : "upgrade";
    }
    return run;
  }

  function danger(run) {
    return new Set(run.enemies.flatMap(enemy => enemy.intent.map(tile => key(tile.x, tile.y))));
  }

  function legalActions(run) {
    if (run.status === "upgrade") return Object.keys(UPGRADES);
    if (run.status !== "playing") return [];
    return [
      ...Object.keys(DIRECTIONS).filter(action => {
        const tile = shifted(run.player, DIRECTIONS[action]);
        return walkable(run, tile.x, tile.y);
      }),
      "wait",
      ...(run.player.charge > 0 ? ["pulse"] : []),
    ];
  }

  function pathTo(run, target, hazards) {
    const start = key(run.player.x, run.player.y);
    const queue = [start];
    const first = new Map([[start, null]]);
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const cell = queue[cursor];
      const from = point(cell);
      if (same(from, target)) return first.get(cell);
      for (const [action, vector] of Object.entries(DIRECTIONS)) {
        const to = shifted(from, vector);
        const index = key(to.x, to.y);
        if (!walkable(run, to.x, to.y) || first.has(index) ||
            (cell === start && hazards.has(index))) continue;
        first.set(index, first.get(cell) ?? action);
        queue.push(index);
      }
    }
    return null;
  }

  function demoAction(run) {
    if (run.status === "upgrade") return run.player.hp < 6 ? "shield" : "blade";
    if (run.status !== "playing") return "wait";
    const player = run.player;
    const nearby = run.enemies.filter(enemy => gap(enemy, player) <= 2);
    const hazards = danger(run);
    const threatened = hazards.has(key(player.x, player.y));
    if (player.charge > 0 && nearby.length &&
        (threatened || nearby.length > 1 || (nearby[0].kind === "warden" && gap(nearby[0], player) <= 1))) return "pulse";

    const warden = run.enemies.find(enemy => enemy.kind === "warden");
    const medkit = run.items.find(item => item.kind === "medkit" && gap(item, player) <= 5);
    const adjacent = nearby.find(enemy => gap(enemy, player) === 1);
    const target = adjacent && !threatened ? adjacent :
      medkit && player.hp <= 4 && !nearby.length ? medkit :
      run.hasCore ? run.exit : warden ?? run.core;
    if (threatened) {
      const escapes = legalActions(run).filter(action => {
        if (!Object.hasOwn(DIRECTIONS, action)) return false;
        const tile = shifted(player, DIRECTIONS[action]);
        return !enemyAt(run, tile) && !hazards.has(key(tile.x, tile.y));
      });
      escapes.sort((a, b) => gap(shifted(player, DIRECTIONS[a]), target) -
        gap(shifted(player, DIRECTIONS[b]), target));
      if (escapes.length) return escapes[0];
    }
    return pathTo(run, target, hazards) ?? pathTo(run, target, new Set()) ?? "wait";
  }

  function observe(run) {
    return { ...structuredClone(run), legalActions: legalActions(run) };
  }

  const api = Object.freeze({
    COLS, ROWS, DEMO_SEED, FLOORS, DIRECTIONS, UPGRADES, createRun, step, observe, legalActions, danger, demoAction,
  });
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.FrostCircuit = api;
})(globalThis);
