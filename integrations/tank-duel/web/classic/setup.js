import {
    defaultSeats,
    normalizeSeat,
    matchupPreset,
    contextRevision,
    INTERVALS,
} from "../shared/seat-config.js";
const storageKey = "ricochet-classic-seats-v1";
export function mountClassicSetup(models, game, changed) {
    const $ = (id) => document.getElementById(id);
    let saved;
    try {
        saved = JSON.parse(localStorage.getItem(storageKey));
    } catch {}
    const seats = defaultSeats(models).map((s, i) =>
        normalizeSeat(saved?.[i], models, s),
    );
    const preset = matchupPreset(models);
    $("matchup-preset").disabled = !preset;
    function persist() {
        try {
            localStorage.setItem(storageKey, JSON.stringify(seats));
        } catch {}
    }
    function sync(player = null) {
        seats.forEach((s, i) => {
            if (player === null || player === i) game.setPlayerSettings(i, s);
            for (const k of [
                "backend",
                "controllerMode",
                "contextVersion",
                "interval",
                "assistProfile",
            ])
                $(`seat-${i}-${k}`).value = s[k];
            $(`seat-${i}-contextVersion`).disabled =
                s.controllerMode === "plans";
            $(`seat-${i}-fireGuardEnabled`).disabled =
                s.controllerMode === "plans";
            $(`seat-${i}-assistProfile`).disabled = !s.assistEnabled;
            for (const k of ["assistEnabled", "fireGuardEnabled"]) {
                $(`seat-${i}-${k}`).setAttribute("aria-pressed", String(s[k]));
                $(`seat-${i}-${k}`).querySelector("span").textContent = s[k]
                    ? "On"
                    : "Off";
            }
            $(`seat-${i}-description`).textContent =
                s.controllerMode === "plans"
                    ? "Joint motion → conditional shot. Aim hints + forecasts. Geometry filters plans and checks shots, even with Dodge Off."
                    : s.contextVersion === "arena-v1"
                      ? "Independent drive / turn / fire. Positions, walls and every shell; no aim hints or trajectory forecasts. New minimal profile."
                      : "Independent drive / turn / fire. Full tactical context with aim hints and shell forecasts. No joint-plan filtering.";
        });
    }
    seats.forEach((s, i) => {
        const box = $(`seat-config-${i}`);
        box.innerHTML = `<div class="seat-heading"><h3>${i ? "02 / Red" : "01 / Blue"}</h3><span id="seat-${i}-status">Connecting…</span></div>
        <div class="seat-fields"><label>Model / checkpoint<select id="seat-${i}-backend" aria-label="${i ? "Red" : "Blue"} model"></select></label>
        <label>Decision version<select id="seat-${i}-controllerMode" aria-label="${i ? "Red" : "Blue"} decision version"><option value="plans">Current · planned v1</option><option value="raw">Independent · controls v3</option></select></label>
        <label>Context<select id="seat-${i}-contextVersion" aria-label="${i ? "Red" : "Blue"} context"><option value="full">Full tactical</option><option value="arena-v1">Arena only · v1</option></select></label>
        <label>Decision interval<select id="seat-${i}-interval" aria-label="${i ? "Red" : "Blue"} decision interval">${INTERVALS.map((n) => `<option value="${n}">${n} ms</option>`).join("")}</select></label></div>
        <div class="seat-assist"><button id="seat-${i}-assistEnabled" aria-label="${i ? "Red" : "Blue"} dodge assist">Dodge assist <span></span></button>
        <select id="seat-${i}-assistProfile" aria-label="${i ? "Red" : "Blue"} dodge strength"><option value="light">Light</option><option value="standard">Standard</option><option value="strong">Strong</option></select>
        <button id="seat-${i}-fireGuardEnabled" aria-label="${i ? "Red" : "Blue"} shot guard">Shot guard <span></span></button></div>
        <p class="seat-description" id="seat-${i}-description"></p><details class="seat-revision"><summary>Model & context identity</summary><p id="seat-${i}-revision"></p></details>`;
        $(`seat-${i}-backend`).replaceChildren(
            ...(models.length
                ? models
                : [{ id: "", label: "No model connected" }]
            ).map(
                (m) =>
                    new Option(
                        m.label + (m.version ? " · " + m.version : ""),
                        m.id,
                    ),
            ),
        );
        $(`seat-${i}-backend`).disabled = !models.length;
        for (const k of [
            "backend",
            "controllerMode",
            "contextVersion",
            "interval",
            "assistProfile",
        ])
            $(`seat-${i}-${k}`).onchange = (e) => {
                const value =
                    k === "interval" ? Number(e.target.value) : e.target.value;
                Object.assign(
                    s,
                    normalizeSeat({ ...s, [k]: value }, models, s),
                );
                sync(i);
                persist();
                changed(i, k);
            };
        for (const k of ["assistEnabled", "fireGuardEnabled"])
            $(`seat-${i}-${k}`).onclick = () => {
                s[k] = !s[k];
                sync(i);
                persist();
                changed(i, k);
            };
    });
    $("matchup-preset").onclick = () => {
        if (!preset) return;
        seats.forEach((s, i) => Object.assign(s, preset[i]));
        sync();
        persist();
        changed(null, "preset");
    };
    sync();
    return {
        seats,
        render(health, isAI) {
            seats.forEach((s, i) => {
                const m = health[i]?.metadata;
                $(`seat-${i}-status`).textContent = !isAI(i)
                    ? "Keyboard seat"
                    : m
                      ? "Connected"
                      : "Offline / incompatible";
                $(`seat-${i}-revision`).textContent =
                    `${m?.model || "Awaiting model identity"} · checkpoint: ${m?.revision || "not reported by service"} · adapter: ${m?.code_sha256 || m?.adapterContractRevision || "not reported"} · context: ${contextRevision(s)}. Model/checkpoint options use separate configured endpoints.`;
            });
        },
    };
}
