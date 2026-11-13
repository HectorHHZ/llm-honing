// Precompute finite muzzle lines for a recorded Tank War episode using the
// game's own ricochet physics. These are aiming guides, not fired bullets.
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { traceShot } from "../web/game/engine.js";
import { stateHash } from "./state-replay.mjs";

function parseArgs(argv) {
    const args = Object.fromEntries(
        argv.map((argument) => {
            const match = /^--(run|out|max-distance)=(.+)$/.exec(argument);
            if (!match) throw Error(`Unknown argument ${argument}`);
            return [match[1], match[2]];
        }),
    );
    if (!args.run || !args.out) throw Error("--run and --out are required");
    const maxDistance = Number(args["max-distance"] ?? 400);
    if (!(Number.isFinite(maxDistance) && maxDistance > 0 && maxDistance <= 1200))
        throw Error("--max-distance must be within (0, 1200] pixels");
    return { run: resolve(args.run), out: resolve(args.out), maxDistance };
}

const { run, out, maxDistance } = parseArgs(process.argv.slice(2));
const manifest = JSON.parse(await readFile(join(run, "manifest.json"), "utf8"));
const engine = fileURLToPath(new URL("../web/game/engine.js", import.meta.url));
const engineSha = createHash("sha256").update(await readFile(engine)).digest("hex");
if (manifest.sources["web/game/engine.js"] !== engineSha)
    throw Error("Recorded game engine differs from the shot-trace implementation");
const states = (await readFile(join(run, "states.jsonl"), "utf8"))
    .split("\n")
    .filter(Boolean)
    .map(JSON.parse);
const rows = states.map((state) => {
    if (stateHash(state.raw_state) !== state.state_hash)
        throw Error(`State hash mismatch at decision ${state.decision_index}`);
    const { maze, tanks } = state.raw_state;
    const lines = tanks.map((tank, index) => {
        if (!tank.alive) return { points: [], bounces: 0, blocked: false };
        const trace = traceShot(maze, tank, tank.angle, maxDistance / 290, tanks[1 - index]);
        return {
            points: trace.points.map(({ x, y }) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]),
            bounces: trace.bounces,
            blocked: trace.muzzleBlocked,
        };
    });
    return { decision_index: state.decision_index, state_hash: state.state_hash, lines };
});
await writeFile(
    out,
    JSON.stringify({ source_run: run, engine_sha256: engineSha, max_distance_px: maxDistance, rows }) + "\n",
    { flag: "wx" },
);
console.log(JSON.stringify({ out, frames: rows.length, bounces: rows.reduce((n, row) => n + row.lines.filter((line) => line.bounces > 0).length, 0) }));
