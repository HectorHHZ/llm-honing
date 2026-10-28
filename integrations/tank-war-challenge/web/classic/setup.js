import {
    defaultSeats,
    normalizeSeat,
    CLASSIC_VERSIONS,
    classicVersionFor,
    selectClassicVersion,
} from "../shared/seat-config.js";
const storageKey = "ricochet-classic-seats-v1";
export function mountClassicSetup(models, game, changed) {
    const $ = (id) => document.getElementById(id);
    let saved;
    try {
        saved = JSON.parse(localStorage.getItem(storageKey));
    } catch {}
    const seats = defaultSeats(models).map((s, i) => {
        const restored = normalizeSeat(saved?.[i], models, s);
        return {
            ...selectClassicVersion(
                restored,
                classicVersionFor(restored)?.id || "v3",
                models,
            ),
            intervalMode: "auto",
            assistProfile: "strong",
            fireGuardEnabled: false,
        };
    });
    function persist() {
        try {
            localStorage.setItem(storageKey, JSON.stringify(seats));
        } catch {}
    }
    function sync(player = null) {
        seats.forEach((s, i) => {
            if (player === null || player === i) game.setPlayerSettings(i, s);
            $(`seat-${i}-backend`).value = s.backend;
            $(`seat-${i}-version`).value = classicVersionFor(s)?.id || "";
        });
    }
    seats.forEach((s, i) => {
        const side = i ? "Red" : "Blue";
        const box = $(`seat-config-${i}`);
        box.innerHTML = `<div class="seat-fields">
            <label>Model<select id="seat-${i}-backend" aria-label="${side} model"></select></label>
            <label>Version<select id="seat-${i}-version" aria-label="${side} version">${CLASSIC_VERSIONS.map(
                (v) => `<option value="${v.id}">${v.id.toUpperCase()}</option>`,
            ).join("")}</select></label>
        </div>`;
        $(`seat-${i}-backend`).replaceChildren(
            ...(models.length
                ? models
                : [{ id: "", label: "No model connected" }]
            ).map((m) => new Option(m.label, m.id)),
        );
        $(`seat-${i}-backend`).disabled = !models.length;
        $(`seat-${i}-version`).onchange = (e) => {
            Object.assign(s, selectClassicVersion(s, e.target.value, models));
            sync(i);
            persist();
            changed(i, "decisionVersion");
        };
        $(`seat-${i}-backend`).onchange = (e) => {
            Object.assign(
                s,
                normalizeSeat({ ...s, backend: e.target.value }, models, s),
            );
            sync(i);
            persist();
            changed(i, "backend");
        };
    });
    sync();
    persist();
    return { seats };
}
