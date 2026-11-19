import { Game, makeMaze } from "./engine.js";

// The raw state behind a decision: every Game field the controller reads when it
// builds a seat's observation (both tanks with their command, assist and shot
// history, every shell, the clock, the round and the seats' settings), as plain
// JSON. Effects, tracks and the event log only draw the game or record it, and a
// shell's trail only draws it, so they are left out. The maze is rebuilt from its
// seed. JSON has no Infinity, NaN or negative zero, and the engine uses Infinity
// for a command with no turn deadline, so such numbers are written as
// {"$number": "Infinity"}. Every other number round-trips exactly.
const DRAWN = new Set(["events", "effects", "tracks"]);

function encode(value) {
    if (typeof value === "number")
        return Number.isFinite(value) && !Object.is(value, -0)
            ? value
            : { $number: Object.is(value, -0) ? "-0" : String(value) };
    if (Array.isArray(value)) return value.map(encode);
    if (value && typeof value === "object") {
        const out = {};
        for (const [k, v] of Object.entries(value))
            if (v !== undefined) out[k] = encode(v);
        return out;
    }
    return value;
}

function decode(value) {
    if (Array.isArray(value)) return value.map(decode);
    if (value && typeof value === "object") {
        const keys = Object.keys(value);
        if (keys.length === 1 && keys[0] === "$number")
            return Number(value.$number);
        return Object.fromEntries(keys.map((k) => [k, decode(value[k])]));
    }
    return value;
}

export function snapshotGame(game) {
    const fields = {};
    for (const [k, v] of Object.entries(game))
        if (!DRAWN.has(k) && k !== "maze") fields[k] = v;
    fields.bullets = game.bullets.map(({ trail, ...shell }) => shell);
    fields.maze = { seed: game.maze.seed, revision: game.maze.revision };
    return encode(fields);
}

// A Game in the snapshot's state. `mazes` caches built mazes by seed, which is
// safe because the engine never changes a maze after building it.
export function restoreGame(snapshot, mazes = new Map()) {
    const { maze, bullets, ...fields } = decode(snapshot);
    if (!mazes.has(maze.seed)) mazes.set(maze.seed, makeMaze(maze.seed));
    const built = mazes.get(maze.seed);
    if (built.revision !== maze.revision)
        throw Error(
            `The state's maze is ${maze.revision}, but this engine builds ${built.revision}`,
        );
    return Object.assign(Object.create(Game.prototype), fields, {
        maze: built,
        bullets: bullets.map((shell) => ({ ...shell, trail: [] })),
        events: [],
        effects: [],
        tracks: [],
    });
}

// The snapshot as JSON with every object's keys sorted, so equal states give
// equal text however their fields were added. A state's hash is this text's.
export function canonicalJSON(value) {
    if (Array.isArray(value)) return `[${value.map(canonicalJSON).join(",")}]`;
    if (value && typeof value === "object")
        return `{${Object.keys(value)
            .sort()
            .map((k) => `${JSON.stringify(k)}:${canonicalJSON(value[k])}`)
            .join(",")}}`;
    return JSON.stringify(value);
}
