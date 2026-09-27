import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { join } from "node:path";
import { PINS, SUPALOSA_DIR, SDK_DIR, GAME_DIR } from "./paths.mjs";
export { GAME_DIR };
const revision = execFileSync(
    "git",
    ["-C", SUPALOSA_DIR, "rev-parse", "HEAD"],
    { encoding: "utf8" },
).trim();
if (revision !== PINS.supalosa.revision)
    throw new Error("SUPALOSA_DIR must use the revision in pins.json");
export const CD = await import(
    pathToFileURL(
        join(SUPALOSA_DIR, "node_modules/@chronodivide/game-api/dist/index.js"),
    )
);
export const { SupalosaBot } = await import(
    pathToFileURL(
        join(SUPALOSA_DIR, "packages/chronodivide-bot/dist/bot/bot.js"),
    )
);
export const { Ra2Engine } = await import(
    pathToFileURL(join(SDK_DIR, "dist/index.js"))
);
