import { CD, GAME_DIR } from "./dependencies.mjs";
import { fileURLToPath } from "node:url";
export { CD };
export const ROOT = fileURLToPath(new URL("../", import.meta.url));
// A frozen reference game exposes only static map/rules metadata. It never drives
// the match; all dynamic observations and commands belong to ra2-game-env.
export async function loadMetadata(mapName) {
    class MetadataBot extends CD.Bot {
        onGameStart(api) {
            this.api = api;
        }
    }
    await CD.cdapi.init(GAME_DIR);
    const bot = new MetadataBot("metadata", "Russians");
    const instance = await CD.cdapi.createGame({
        online: false,
        agents: [bot, new MetadataBot("metadata2", "Americans")],
        mapName,
        gameMode: CD.cdapi.getAvailableGameModes(mapName)[0],
        gameSpeed: 4,
        credits: 10000,
        unitCount: 0,
        shortGame: true,
        mcvRepacks: true,
        cratesAppear: false,
        superWeapons: false,
        buildOffAlly: false,
    });
    while (!bot.api) await instance.update();
    const api = bot.api,
        map = api.mapApi,
        rules = api.rulesApi;
    const size = map.getRealMapSize();
    const tiles = map.getTilesInRect({
        x: 0,
        y: 0,
        width: size.width,
        height: size.height,
    });
    const bounds = {
        x: Math.min(...tiles.map((t) => t.rx)),
        y: Math.min(...tiles.map((t) => t.ry)),
        maxX: Math.max(...tiles.map((t) => t.rx)),
        maxY: Math.max(...tiles.map((t) => t.ry)),
    };
    bounds.width = bounds.maxX - bounds.x + 1;
    bounds.height = bounds.maxY - bounds.y + 1;
    const techno = new Map([
        ...rules.buildingRules,
        ...rules.infantryRules,
        ...rules.vehicleRules,
        ...rules.aircraftRules,
    ]);
    const placements = new Map(
        [...rules.buildingRules.keys()].map((name) => [
            name,
            api.getBuildingPlacementData(name),
        ]),
    );
    const countries = new Map(
        ["metadata", "metadata2"].map((name) => {
            const c = api.getPlayerData(name).country;
            return [c.name, c];
        }),
    );
    const resources = new Map(
        map.getAllTilesResourceData().map((r) => [r.tile.id, r]),
    );
    return {
        rules,
        techno,
        placements,
        countries,
        tiles,
        bounds,
        size,
        resources,
        starts: map.getStartingLocations(),
        general: api.getGeneralRules(),
        getTile: (x, y) => map.getTile(x, y),
        staticTick: instance.getCurrentTick(),
        dispose: () => instance.dispose(),
    };
}
