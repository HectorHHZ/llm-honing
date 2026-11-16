(function () {
  "use strict";

  const { WIDTH, HEIGHT, TICKS_PER_SECOND, DEMO_SEED, FLOORS, UPGRADES, createRun, step, demoInput } = globalThis.EmberVault;
  const { paint } = globalThis.EmberVaultView;
  const byId = id => document.getElementById(id);
  const canvas = byId("game");
  const ctx = canvas.getContext("2d");
  const seedField = byId("seed");
  const tickMs = 1000 / TICKS_PER_SECOND;
  const keys = new Set();
  const route = FLOORS.map((name, index) => {
    const item = document.createElement("li");
    item.textContent = "0" + (index + 1) + " " + name;
    byId("route").append(item);
    return item;
  });
  const movement = ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowLeft", "ArrowDown", "ArrowRight"];
  let run = createRun();
  let mode = "menu";
  let paused = false;
  let pointer = null;
  let firing = false;
  let queuedShot = false;
  let dash = false;
  let trail = [];
  let inputs = [];
  let panel = "";
  let previous = 0;
  let accumulated = 0;
  let upgradeAt = 0;

  function clearInput() {
    keys.clear();
    firing = false;
    queuedShot = false;
    dash = false;
    pointer = null;
  }

  function setRun(seed, controller) {
    run = createRun(seed);
    seedField.value = seed;
    mode = controller;
    paused = false;
    inputs = [];
    trail = [];
    accumulated = 0;
    clearInput();
    sync();
    if (mode === "human") canvas.focus();
  }

  function apply(input) {
    const floor = run.floor;
    const wasUpgrading = run.status === "upgrade";
    step(run, input);
    inputs.push(input);
    if (floor !== run.floor) trail = [];
    trail = [...trail.filter(event => run.tick - event.tick <= 22),
      ...run.events.map(event => ({ ...event, tick: run.tick }))];
    if (!wasUpgrading && run.status === "upgrade") {
      upgradeAt = performance.now();
      clearInput();
    }
  }

  function statePanel() {
    return mode === "menu" ? "brief" :
      paused ? "pause-card" :
      run.status === "upgrade" ? "upgrade-card" :
      run.status === "playing" ? "" : "finish";
  }

  function sync() {
    const current = statePanel();
    byId("mode").textContent = mode === "menu" ? "Standby" : paused ? "Paused" :
      run.status === "won" ? "Extracted" : run.status === "lost" ? "Run lost" :
      mode === "demo" ? "Live demo" : "Live run";
    byId("pause").hidden = mode === "menu" || run.status !== "playing";
    byId("pause").textContent = paused ? "Resume" : "Pause";
    byId("reset").hidden = mode === "menu";
    byId("takeover").hidden = mode !== "demo" || !["playing", "upgrade"].includes(run.status);
    byId("overlay").hidden = !current;
    if (panel !== current) {
      panel = current;
      for (const id of ["brief", "pause-card", "upgrade-card", "finish"]) byId(id).hidden = id !== current;
      if (current === "upgrade-card") {
        const buttons = run.upgrades.map(name => {
          const button = document.createElement("button");
          button.type = "button";
          const title = document.createElement("strong");
          title.textContent = UPGRADES[name].name;
          const copy = document.createElement("span");
          copy.textContent = UPGRADES[name].detail;
          button.append(title, copy);
          button.addEventListener("click", () => {
            apply({ upgrade: name });
            sync();
            if (mode === "human") canvas.focus();
          });
          return button;
        });
        byId("upgrades").replaceChildren(...buttons);
      }
    }
    byId("upgrade-label").textContent = mode === "demo" ? "Demo chooses in a moment" : "Vault cleared";
    byId("finish-title").textContent = run.status === "won" ? "Vault secured." : "Rover down.";
    byId("finish-summary").textContent = `${run.kills} hostiles disabled · Vault ${run.floor} of 3 · Seed ${run.seed}`;
    byId("objective").textContent = run.gate.open ? "Reach the green gate" :
      run.floor === 3 ? "Defeat the Furnace" : "Clear the " + FLOORS[run.floor - 1].toLowerCase();
    byId("objective-detail").textContent = run.gate.open ? "The exit on the right is open." :
      `${run.enemies.length} of ${run.floorEnemyCount} hostiles remain.`;
    route.forEach((item, index) => {
      item.className = index + 1 === run.floor ? "current" : index + 1 < run.floor ? "complete" : "";
      if (index + 1 === run.floor) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    });
    byId("hull").textContent = `${run.player.hp} / ${run.player.maxHp}`;
    byId("health").max = run.player.maxHp;
    byId("health").value = run.player.hp;
    byId("dash-state").textContent = run.player.dashCooldown ? "Charging" : "Ready";
    byId("dash-meter").max = run.player.dashPeriod;
    byId("dash-meter").value = run.player.dashPeriod - run.player.dashCooldown;
    byId("kills").textContent = `${run.kills} disabled`;
    byId("timer").textContent = (run.tick / TICKS_PER_SECOND).toFixed(1) + " s";
    byId("export").disabled = inputs.length === 0;
  }

  function togglePause() {
    if (mode === "menu" || run.status !== "playing") return;
    paused = !paused;
    clearInput();
    accumulated = 0;
    sync();
    if (!paused && mode === "human") canvas.focus();
  }

  function active() {
    return mode === "human" && !paused && run.status === "playing";
  }

  function direction(primary, alternative) {
    return Number(keys.has(primary) || keys.has(alternative));
  }

  function humanInput() {
    const auto = keys.has("Space") && !firing && !queuedShot;
    const input = {
      dx: direction("KeyD", "ArrowRight") - direction("KeyA", "ArrowLeft"),
      dy: direction("KeyS", "ArrowDown") - direction("KeyW", "ArrowUp"),
      fire: firing || queuedShot || auto,
      dash,
    };
    if (pointer && !auto) Object.assign(input, { aimX: pointer.x, aimY: pointer.y });
    dash = false;
    queuedShot = false;
    return input;
  }

  function aim(event) {
    const rect = canvas.getBoundingClientRect();
    pointer = {
      x: (event.clientX - rect.left) / rect.width * WIDTH,
      y: (event.clientY - rect.top) / rect.height * HEIGHT,
    };
  }

  byId("start").addEventListener("click", () => {
    if (seedField.reportValidity()) setRun(seedField.valueAsNumber, "human");
  });
  byId("demo").addEventListener("click", () => setRun(DEMO_SEED, "demo"));
  byId("random-seed").addEventListener("click", () => setRun(Math.floor(Math.random() * 0x100000000), "menu"));
  byId("reset").addEventListener("click", () => setRun(run.seed, "menu"));
  byId("briefing").addEventListener("click", () => setRun(run.seed, "menu"));
  byId("retry").addEventListener("click", () => setRun(run.seed, "human"));
  byId("pause").addEventListener("click", togglePause);
  byId("resume").addEventListener("click", togglePause);
  byId("takeover").addEventListener("click", () => {
    mode = "human";
    clearInput();
    sync();
    if (active()) canvas.focus();
  });
  byId("export").addEventListener("click", () => {
    const file = new Blob([JSON.stringify({ format: "ember-vault-v1", seed: run.seed, inputs })], { type: "application/json" });
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = `ember-vault-${run.seed}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  canvas.addEventListener("pointermove", event => { if (active()) aim(event); });
  canvas.addEventListener("pointerdown", event => {
    if (!active() || event.button !== 0) return;
    aim(event);
    firing = true;
    queuedShot = true;
    canvas.setPointerCapture(event.pointerId);
    canvas.focus();
  });
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
    canvas.addEventListener(event, () => { firing = false; });
  }
  canvas.addEventListener("pointerleave", () => { if (!firing) pointer = null; });

  window.addEventListener("keydown", event => {
    if (event.target.matches("input, textarea, select")) return;
    if (event.code === "KeyP" && !event.repeat) togglePause();
    if (!active() || (event.code === "Space" && event.target.matches("button"))) return;
    if (movement.includes(event.code) || event.code === "Space") {
      event.preventDefault();
      keys.add(event.code);
    } else if (event.code.startsWith("Shift") && !event.repeat) {
      dash = true;
    }
  });
  window.addEventListener("keyup", event => keys.delete(event.code));
  window.addEventListener("blur", clearInput);
  document.addEventListener("visibilitychange", () => {
    clearInput();
    if (document.hidden && mode === "human" && !paused && run.status === "playing") togglePause();
  });
  document.querySelectorAll("[data-hold]").forEach(button => {
    const code = button.dataset.hold;
    button.addEventListener("pointerdown", event => {
      if (!active()) return;
      event.preventDefault();
      button.setPointerCapture(event.pointerId);
      if (code.startsWith("Shift")) dash = true;
      else keys.add(code);
    });
    for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
      button.addEventListener(event, () => keys.delete(code));
    }
  });

  function frame(now) {
    const dt = previous ? Math.min(100, now - previous) : 0;
    previous = now;
    if (mode !== "menu" && !paused && !document.hidden) {
      if (run.status === "playing") {
        accumulated += dt;
        while (accumulated >= tickMs && run.status === "playing") {
          accumulated -= tickMs;
          apply(mode === "demo" ? demoInput(run) : humanInput());
        }
        sync();
      } else if (mode === "demo" && run.status === "upgrade" && now - upgradeAt > 1400) {
        apply(demoInput(run));
        accumulated = 0;
        sync();
      }
    }
    paint(ctx, run, now / 1000, trail, active() ? pointer : null);
    requestAnimationFrame(frame);
  }

  setRun(DEMO_SEED, "menu");
  requestAnimationFrame(frame);
})();
