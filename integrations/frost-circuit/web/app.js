(function () {
  "use strict";

  const { DEMO_SEED, FLOORS, DIRECTIONS, UPGRADES, createRun, step, demoAction, legalActions, danger } = globalThis.FrostCircuit;
  const { WIDTH, HEIGHT, paint, tileAt } = globalThis.FrostCircuitView;
  const byId = id => document.getElementById(id);
  const canvas = byId("game");
  const ctx = canvas.getContext("2d");
  const seedField = byId("seed");
  const route = FLOORS.map((name, index) => {
    const item = document.createElement("li");
    item.textContent = "0" + (index + 1) + " " + name;
    byId("route").append(item);
    return item;
  });
  const actionKeys = {
    ArrowUp: "north", KeyW: "north", ArrowRight: "east", KeyD: "east", ArrowDown: "south",
    KeyS: "south", ArrowLeft: "west", KeyA: "west", Space: "pulse", Period: "wait",
  };
  let run = createRun();
  let mode = "menu";
  let paused = false;
  let nextAuto = 0;
  let lastAction = 0;
  let actions = [];
  let notices = [];
  let hover = null;
  let panel = "";

  function active() {
    return mode === "human" && run.status === "playing";
  }

  function setRun(seed, controller) {
    run = createRun(seed);
    seedField.value = seed;
    mode = controller;
    paused = false;
    nextAuto = performance.now() + 480;
    lastAction = 0;
    actions = [];
    notices = ["Explore the station and find the core."];
    hover = null;
    updateLog();
    sync();
    if (active()) canvas.focus();
  }

  function apply(action) {
    if (!legalActions(run).includes(action)) return;
    step(run, action);
    actions.push(action);
    lastAction = performance.now();
    const messages = run.events.filter(event => event.text).map(event => event.text);
    if (messages.length) {
      notices = [...messages.reverse(), ...notices].slice(0, 3);
      updateLog();
    }
    sync();
  }

  function updateLog() {
    const lines = notices.map(text => {
      const line = document.createElement("li");
      line.textContent = text.charAt(0).toUpperCase() + text.slice(1);
      return line;
    });
    byId("log").replaceChildren(...lines);
  }

  function sync() {
    const current = mode === "menu" ? "brief" : run.status === "upgrade" ? "upgrade-card" :
      run.status === "playing" ? "" : "finish";
    byId("overlay").hidden = !current;
    if (panel !== current) {
      panel = current;
      for (const id of ["brief", "upgrade-card", "finish"]) byId(id).hidden = id !== current;
      if (current === "upgrade-card") {
        byId("upgrades").replaceChildren(...Object.entries(UPGRADES).map(([action, upgrade]) => {
          const button = document.createElement("button");
          const name = document.createElement("strong");
          name.textContent = upgrade.name;
          const detail = document.createElement("span");
          detail.textContent = upgrade.detail;
          button.append(name, detail);
          button.addEventListener("click", () => {
            apply(action);
            if (active()) canvas.focus();
          });
          return button;
        }));
      }
    }
    byId("sector").textContent = "SECTOR 0" + run.floor;
    byId("turn").textContent = `Turn ${run.turn}${mode === "demo" ? paused ? " · Demo paused" : " · Live demo" : ""}`;
    byId("pause").hidden = mode !== "demo" || !["playing", "upgrade"].includes(run.status);
    byId("pause").textContent = paused ? "Resume demo" : "Pause demo";
    byId("takeover").hidden = byId("pause").hidden;
    byId("reset").hidden = mode === "menu";
    byId("upgrade-label").textContent = mode === "demo" && !paused ? "Demo chooses in a moment" : "Airlock reached";
    byId("finish-title").textContent = run.status === "won" ? "Extraction complete." : "Signal lost.";
    byId("finish-summary").textContent = `${run.turn} turns · ${run.kills} drones disabled · Seed ${run.seed}`;

    const warden = run.enemies.some(enemy => enemy.kind === "warden");
    byId("objective").textContent = run.hasCore ? "Reach the airlock" : warden ? "Disable the Warden" : "Recover the core";
    byId("objective-detail").textContent = run.hasCore ? "The green airlock is unlocked." :
      warden ? "The Warden seals this sector’s core." : "Move onto the glowing cyan diamond.";
    route.forEach((item, index) => {
      item.className = index + 1 < run.floor || (index + 1 === run.floor && run.hasCore) ? "complete" :
        index + 1 === run.floor ? "current" : "";
      if (index + 1 === run.floor) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    });
    byId("hull").textContent = `${run.player.hp} / ${run.player.maxHp}`;
    byId("health").max = run.player.maxHp;
    byId("health").value = run.player.hp;
    byId("charge").textContent = `${run.player.charge} / ${run.player.maxCharge}`;
    byId("warning").textContent = danger(run).size ? "Red tiles strike on your next turn." : "";
    byId("pulse").disabled = !active() || !run.player.charge;
    byId("wait").disabled = !active();
    byId("export").disabled = !actions.length;
  }

  function pointerTile(event) {
    const rect = canvas.getBoundingClientRect();
    return tileAt((event.clientX - rect.left) / rect.width * WIDTH, (event.clientY - rect.top) / rect.height * HEIGHT);
  }

  byId("start").addEventListener("click", () => {
    if (seedField.reportValidity()) setRun(seedField.valueAsNumber, "human");
  });
  byId("demo").addEventListener("click", () => setRun(DEMO_SEED, "demo"));
  byId("random-seed").addEventListener("click", () => setRun(Math.floor(Math.random() * 0x100000000), "menu"));
  byId("reset").addEventListener("click", () => setRun(run.seed, "menu"));
  byId("briefing").addEventListener("click", () => setRun(run.seed, "menu"));
  byId("retry").addEventListener("click", () => setRun(run.seed, "human"));
  byId("takeover").addEventListener("click", () => {
    mode = "human";
    paused = false;
    sync();
    if (active()) canvas.focus();
  });
  byId("pause").addEventListener("click", () => {
    paused = !paused;
    nextAuto = performance.now() + 360;
    sync();
  });
  byId("pulse").addEventListener("click", () => { if (active()) apply("pulse"); });
  byId("wait").addEventListener("click", () => { if (active()) apply("wait"); });
  byId("export").addEventListener("click", () => {
    const file = new Blob([JSON.stringify({ format: "frost-circuit-v1", seed: run.seed, actions })], { type: "application/json" });
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = `frost-circuit-${run.seed}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  canvas.addEventListener("pointermove", event => { hover = active() ? pointerTile(event) : null; });
  canvas.addEventListener("pointerleave", () => { hover = null; });
  canvas.addEventListener("click", event => {
    if (!active()) return;
    const tile = pointerTile(event);
    const next = Object.entries(DIRECTIONS).find(([, [dx, dy]]) =>
      run.player.x + dx === tile.x && run.player.y + dy === tile.y);
    if (next) apply(next[0]);
  });
  window.addEventListener("keydown", event => {
    if (!active() || event.repeat || !Object.hasOwn(actionKeys, event.code) ||
        event.target.matches("input, textarea, select") || (event.code === "Space" && event.target.matches("button"))) return;
    event.preventDefault();
    apply(actionKeys[event.code]);
  });

  function frame(now) {
    if (mode === "demo" && !paused && !document.hidden && now >= nextAuto &&
        ["playing", "upgrade"].includes(run.status)) {
      apply(demoAction(run));
      nextAuto = now + (run.status === "upgrade" ? 1300 : 360);
    }
    const effect = lastAction ? Math.max(0, 1 - (now - lastAction) / 320) : 0;
    paint(ctx, run, now / 1000, active() ? hover : null, effect);
    requestAnimationFrame(frame);
  }

  setRun(DEMO_SEED, "menu");
  requestAnimationFrame(frame);
})();
