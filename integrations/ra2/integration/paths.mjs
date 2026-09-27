import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { readFileSync } from "node:fs";
export const ROOT = fileURLToPath(new URL("../", import.meta.url));
export const PINS = JSON.parse(
    readFileSync(new URL("../pins.json", import.meta.url)),
);
export const GAME_DIR = resolve(process.env.RA2_GAME_DIR ?? ROOT + "assets");
export const SDK_DIR = resolve(process.env.RA2_SDK_DIR ?? ROOT + "engine");
export const SUPALOSA_DIR = resolve(
    process.env.SUPALOSA_DIR ?? ROOT + "deps/supalosa",
);
