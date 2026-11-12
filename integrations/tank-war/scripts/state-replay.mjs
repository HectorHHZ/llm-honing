// The engine owns only data fields; retaining their JSON form lets a saved
// decision state be re-rendered without relying on a later model response.
// Non-finite turn deadlines matter to the engine, so JSON's null conversion
// must not silently change them.
import { createHash } from "node:crypto";
import { Game } from "../web/game/engine.js";

const NUMBER_MARKER = "__tank_war_nonfinite_number__";

function encode(_key, value) {
    if (typeof value !== "number" || Number.isFinite(value)) return value;
    return { [NUMBER_MARKER]: String(value) };
}

function decode(_key, value) {
    if (!value || typeof value !== "object" || !(NUMBER_MARKER in value))
        return value;
    if (value[NUMBER_MARKER] === "Infinity") return Infinity;
    if (value[NUMBER_MARKER] === "-Infinity") return -Infinity;
    if (value[NUMBER_MARKER] === "NaN") return NaN;
    throw Error("Invalid non-finite number in saved tank state");
}

export function snapshotGame(game) {
    return JSON.parse(JSON.stringify(game, encode));
}

export function restoreGame(snapshot) {
    if (!snapshot || !Number.isInteger(snapshot.seed) || snapshot.seed < 0)
        throw Error("Saved tank state has no valid seed");
    const game = new Game(snapshot.seed);
    Object.assign(game, JSON.parse(JSON.stringify(snapshot), decode));
    return game;
}

export function stateHash(snapshot) {
    return createHash("sha256").update(JSON.stringify(snapshot)).digest("hex");
}
