import { CD } from "./metadata.mjs";
import { SupalosaBot } from "./dependencies.mjs";
const QUEUES = [
    "structures",
    "armory",
    "infantry",
    "vehicles",
    "aircraft",
    "ships",
];
const NO_TARGET = new Map([
    [CD.OrderType.Deploy, 28],
    [CD.OrderType.DeploySelected, 28],
    [CD.OrderType.Stop, 27],
    [CD.OrderType.Scatter, 40],
    [CD.OrderType.Guard, 26],
]);
const UNIT_TARGET = new Map([
    [CD.OrderType.Attack, 23],
    [CD.OrderType.ForceAttack, 23],
    [CD.OrderType.Capture, 44],
    [CD.OrderType.Occupy, 45],
    [CD.OrderType.Dock, 47],
    [CD.OrderType.Gather, 46],
    [CD.OrderType.Repair, 43],
    [CD.OrderType.EnterTransport, 41],
    [CD.OrderType.PlaceBomb, 48],
]);
const WORLD_TARGET = new Map([
    [CD.OrderType.Move, 20],
    [CD.OrderType.ForceMove, 21],
    [CD.OrderType.AttackMove, 22],
    [CD.OrderType.Attack, 24],
    [CD.OrderType.ForceAttack, 24],
    [CD.OrderType.GuardArea, 25],
]);

/** A policy-facing facade. refresh accepts ONLY a normal seat observation.
 * No privileged state or reference-game units are accepted by this class.
 */
export class SupalosaAdapter {
    constructor({
        name,
        country,
        metadata,
        catalog,
        seats,
        seed = 1,
        log = false,
        strictNormal = false,
    }) {
        this.strictNormal = strictNormal;
        this.name = name;
        this.meta = metadata;
        this.catalog = catalog;
        this.seats = seats;
        this.seed = seed >>> 0;
        this.units = new Map();
        this.actions = [];
        this.tick = 0;
        this.metrics = {
            queued: 0,
            byFunction: {},
            unmapped: [],
            placementAttempts: 0,
            coordinateMisses: 0,
        };
        this.repairs = new Set();
        this.lastCooldown = new Map();
        this.badPlacements = new Map();
        this.typeIndex = new Map(catalog.unit_types.map((n, i) => [n, i]));
        this.fnByName = new Map(catalog.functions.map((f) => [f.name, f]));
        this.buildFns = new Map(
            catalog.functions
                .filter((f) => /^(Build|Train)_/.test(f.name))
                .map((f) => [f.name.replace(/^(Build|Train)_/, ""), f.id]),
        );
        this.game = this.makeGame();
        this.production = this.makeProduction();
        this.commands = this.makeActions();
        this.bot = new SupalosaBot(name, country, [], log);
        const logger = {
            info: (...a) => {
                if (log) console.log(`[${name} @ ${this.tick}]`, ...a);
            },
            debug: () => {},
            log: () => {},
            warn: console.warn,
            error: console.error,
        };
        this.bot.context = {
            game: this.game,
            player: {
                name,
                actions: this.commands,
                production: this.production,
                getPlayerData: () => this.playerData(name),
                isDefeated: () => false,
                isAlliedWith: (other) => other === name,
                canPlaceBuilding: (building, tile) =>
                    this.canPlace(building, tile),
                getVisibleUnits: (type, filter) =>
                    this.game.getVisibleUnits(name, type, filter),
            },
            logger,
        };
    }
    random() {
        this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0;
        return this.seed / 4294967296;
    }
    // Projection must be calibrated against engine world_size before the first tick.
    cell(rx, ry) {
        const b = this.meta.bounds;
        const x = Math.round(((rx - b.x) * (this.res - 1)) / (b.width - 1));
        const y = Math.round(((ry - b.y) * (this.res - 1)) / (b.height - 1));
        return x < 0 || y < 0 || x >= this.res || y >= this.res
            ? -1
            : y * this.res + x;
    }
    tileFromGrid(x, y) {
        const b = this.meta.bounds;
        return this.meta.getTile(
            Math.round(b.x + (x * (b.width - 1)) / (this.res - 1)),
            Math.round(b.y + (y * (b.height - 1)) / (this.res - 1)),
        );
    }
    layer(name, tile) {
        if (!tile) return 0;
        const idx = this.channels.get(name),
            cell = this.cell(tile.rx, tile.ry);
        return idx === undefined || cell < 0
            ? 0
            : this.obs.world.data.data[idx * this.res * this.res + cell];
    }
    refresh(obs, tick) {
        if (!obs || obs.version !== "ra2-raw-interface-v2")
            throw new Error("Expected raw v2 seat observation");
        this.obs = obs;
        this.tick = tick;
        this.res = obs.world.data.shape[1];
        if (
            obs.vectors.world_size[0] !== this.meta.bounds.width ||
            obs.vectors.world_size[1] !== this.meta.bounds.height
        )
            throw new Error(
                "Static metadata bounds do not match the authoritative engine world_size",
            );
        if (
            this.res < Math.max(this.meta.bounds.width, this.meta.bounds.height)
        )
            throw new Error(
                "Adapter requires a resolution large enough to preserve native tile coordinates",
            );
        this.channels = new Map(obs.world.channels.map((n, i) => [n, i]));
        this.units = new Map();
        const { features, data } = obs.units,
            cols = data.shape[1],
            fi = new Map(features.map((n, i) => [n, i]));
        for (let row = 0; row < data.shape[0]; row++) {
            const get = (n) => data.data[row * cols + fi.get(n)];
            if (
                this.strictNormal &&
                get("alliance") === 4 &&
                (get("display_type") !== 1 || get("is_blip"))
            )
                continue;
            const name = this.catalog.unit_types[get("unit_type")],
                rules = this.meta.techno.get(name);
            if (!rules) continue;
            const tile = this.tileFromGrid(get("x"), get("y"));
            if (!tile) {
                this.metrics.coordinateMisses++;
                continue;
            }
            const id = get("tag"),
                owner =
                    this.seats.find((s) => s.player_id === get("owner"))
                        ?.name ?? "Neutral";
            const cooldown = get("weapon_cooldown"),
                oldCooldown = this.lastCooldown.get(id) ?? cooldown;
            this.lastCooldown.set(id, cooldown);
            const weapon = (name, type) => {
                if (!name) return undefined;
                const w = this.meta.rules.getWeapon(name);
                return {
                    type,
                    rules: w,
                    projectileRules: this.meta.rules.getProjectile(
                        w.projectile,
                    ),
                    minRange: w.minimumRange ?? 0,
                    maxRange: w.range,
                    speed: w.speed,
                    cooldownTicks: cooldown,
                };
            };
            const placement = this.meta.placements.get(name);
            this.units.set(id, {
                id,
                name,
                rules,
                owner,
                type: rules.type,
                tile,
                worldPosition: new CD.Vector3(tile.rx * 256, tile.ry * 256, 0),
                tileElevation: 0,
                foundation: placement?.foundation ?? { width: 1, height: 1 },
                hitPoints: get("health"),
                maxHitPoints: rules.strength,
                sight: get("sight_range"),
                veteranLevel: get("veterancy_level"),
                purchaseValue: get("purchase_value"),
                primaryWeapon: weapon(rules.primary, 0),
                secondaryWeapon: weapon(rules.secondary, 1),
                attackState:
                    cooldown > oldCooldown
                        ? CD.AttackState.JustFired
                        : CD.AttackState.Idle,
                direction: (get("facing") * 360) / 256,
                turretFacing: (get("turret_facing") * 360) / 256,
                zone: get("zone"),
                buildStatus:
                    get("build_progress") === 100
                        ? CD.BuildStatus.Ready
                        : CD.BuildStatus.BuildUp,
                onBridge: false,
                isPoweredOn: !!get("is_powered"),
                hasWrenchRepair: this.repairs.has(id),
                isIdle: !!get("is_idle"),
                canMove: !!get("can_move"),
                velocity: new CD.Vector3(),
                stance: get("is_deployed")
                    ? CD.StanceType.Deployed
                    : CD.StanceType.Guard,
                harvestedOre: get("mineral_contents"),
                harvestedGems: get("vespene_contents"),
                passengerSlotCount: get("cargo_space_taken"),
                passengerSlotMax: get("cargo_space_max"),
                ammo: get("ammo"),
                isWarpedOut: false,
                guardMode: !!get("is_idle"),
                _alliance: get("alliance"),
                _display: get("display_type"),
            });
        }
        for (const [key, state] of this.badPlacements) {
            if (tick > state.tick && tick < state.tick + 60) {
                const queue = obs.vectors.production.find(
                    (q) =>
                        this.catalog.unit_types[q.item] === state.name &&
                        q.status === "Ready",
                );
                if (!queue) this.badPlacements.delete(key);
            } else if (tick >= state.tick + 60) this.badPlacements.delete(key);
        }
    }
    start() {
        this.bot.onGameStart(this.game);
    }
    decide() {
        this.actions = [];
        this.bot.onGameTick(this.game);
        return this.actions;
    }
    push(functionId, args = {}) {
        if (functionId === undefined) throw new Error("Unmapped raw action");
        this.actions.push({ function: functionId, ...args });
        this.metrics.queued++;
        this.metrics.byFunction[functionId] =
            (this.metrics.byFunction[functionId] ?? 0) + 1;
    }
    playerData(name) {
        const seat = this.seats.find((s) => s.name === name),
            own = name === this.name;
        const country = seat
            ? this.meta.countries.get(seat.country)
            : undefined;
        const cell = seat?.start_cell,
            tile =
                cell != null
                    ? this.tileFromGrid(
                          cell % this.res,
                          Math.floor(cell / this.res),
                      )
                    : undefined;
        return {
            name,
            country,
            startLocation: new CD.Vector2(tile?.rx ?? 0, tile?.ry ?? 0),
            isObserver: !seat,
            isAi: false,
            isCombatant: !!seat,
            credits: own ? this.obs.vectors.minerals : 0,
            power: {
                total: own ? this.obs.vectors.power_cap : 0,
                drain: own ? this.obs.vectors.power_used : 0,
                isLowPower: own ? !!this.obs.vectors.is_low_power : false,
            },
            radarDisabled: false,
        };
    }
    makeGame() {
        const self = this;
        const map = {
            getRealMapSize: () => self.meta.size,
            getStartingLocations: () => self.meta.starts,
            getTile: (x, y) => self.meta.getTile(x, y),
            getTilesInRect: (rect, size) => {
                const x = rect.x ?? rect.rx,
                    y = rect.y ?? rect.ry,
                    w = size?.width ?? rect.width,
                    h = size?.height ?? rect.height;
                const result = [];
                for (let ry = Math.ceil(y); ry < y + h; ry++)
                    for (let rx = Math.ceil(x); rx < x + w; rx++) {
                        const t = self.meta.getTile(rx, ry);
                        if (t) result.push(t);
                    }
                return result;
            },
            isVisibleTile: (tile) =>
                self.strictNormal
                    ? self.layer("visibility_map", tile) === 2
                    : self.layer("visibility_map", tile) > 0,
            isPassableTile: (tile, speed) =>
                !!self.layer(
                    speed === CD.SpeedType.Float ? "navigable" : "pathable",
                    tile,
                ),
            getTileResourceData: (tile) => {
                const value = self.layer("resource_map", tile),
                    original = self.meta.resources.get(tile.id);
                return value || original?.spawnsOre
                    ? {
                          tile,
                          ore: value,
                          gems: 0,
                          spawnsOre: !!original?.spawnsOre,
                      }
                    : undefined;
            },
            getAllTilesResourceData: () =>
                self.meta.tiles
                    .map((t) => map.getTileResourceData(t))
                    .filter(Boolean),
            getReachabilityMap: (speed) => self.reachability(speed),
        };
        const game = {
            map,
            mapApi: map,
            rules: self.meta.rules,
            rulesApi: self.meta.rules,
            getGeneralRules: () => self.meta.general,
            getTickRate: () => 15,
            getBaseTickRate: () => 15,
            getCurrentTick: () => self.tick,
            getCurrentTime: () => self.tick / 15,
            generateRandom: () => self.random(),
            generateRandomInt: (min, max) =>
                min + Math.floor(self.random() * (max - min + 1)),
            getPlayers: () => self.seats.map((s) => s.name),
            getPlayerData: (name) => self.playerData(name),
            areAlliedPlayers: (a, b) => a === b,
            isPlayerDefeated: () => false,
            getVisibleUnits: (player, type, filter) =>
                [...self.units.values()]
                    .filter((u) => {
                        const own = u.owner === player,
                            combatant = self.seats.some(
                                (s) => s.name === u.owner,
                            );
                        return (
                            (type === "self" || type === "allied"
                                ? own
                                : type === "enemy"
                                ? !own && combatant
                                : !own) &&
                            (!filter || filter(u.rules))
                        );
                    })
                    .map((u) => u.id),
            getGameObjectData: (id) => self.units.get(id),
            getUnitData: (id) => self.units.get(id),
            getUnitsInArea: (box) =>
                [...self.units.values()]
                    .filter(
                        (u) =>
                            u.tile.rx >= box.min.x &&
                            u.tile.rx <= box.max.x &&
                            u.tile.ry >= box.min.y &&
                            u.tile.ry <= box.max.y,
                    )
                    .map((u) => u.id),
            getBuildingPlacementData: (name) => self.meta.placements.get(name),
            canPlaceBuilding: (player, name, tile) => self.canPlace(name, tile),
        };
        return game;
    }
    canPlace(name, tile) {
        if (!tile || this.badPlacements.has(`${name}:${tile.rx}:${tile.ry}`))
            return false;
        const p = this.meta.placements.get(name);
        if (!p) return false;
        for (let y = 0; y < p.foundation.height; y++)
            for (let x = 0; x < p.foundation.width; x++) {
                if (
                    !this.layer(
                        "buildable",
                        this.meta.getTile(tile.rx + x, tile.ry + y),
                    )
                )
                    return false;
            }
        const adjacent = this.meta.techno.get(name).adjacent;
        return [...this.units.values()].some(
            (u) =>
                u.owner === this.name &&
                u.rules.baseNormal &&
                tile.rx <= u.tile.rx + u.foundation.width + adjacent &&
                tile.rx + p.foundation.width >= u.tile.rx - adjacent &&
                tile.ry <= u.tile.ry + u.foundation.height + adjacent &&
                tile.ry + p.foundation.height >= u.tile.ry - adjacent,
        );
    }
    reachability(speed) {
        // Raw observations have no native connectivity query. Construct a snapshot
        // connectivity graph; include occupied own tiles so units can leave them.
        if (
            this.regionCache?.tick === this.tick &&
            this.regionCache.speed === speed
        )
            return this.regionCache.api;
        const ownTiles = new Set(
            [...this.units.values()]
                .filter((u) => u.owner === this.name)
                .map((u) => u.tile.id),
        );
        const regions = new Map(),
            allowed = (t) =>
                this.game.map.isPassableTile(t, speed) || ownTiles.has(t.id);
        let region = 0;
        for (const tile of this.meta.tiles) {
            if (regions.has(tile.id) || !allowed(tile)) continue;
            const frontier = [tile];
            regions.set(tile.id, ++region);
            for (let i = 0; i < frontier.length; i++) {
                const t = frontier[i];
                for (const [dx, dy] of [
                    [1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                ]) {
                    const n = this.meta.getTile(t.rx + dx, t.ry + dy);
                    if (n && !regions.has(n.id) && allowed(n)) {
                        regions.set(n.id, region);
                        frontier.push(n);
                    }
                }
            }
        }
        const key = (n) => (n.tile ?? n).id;
        const api = {
            getRegionId: (n) => regions.get(key(n)),
            isReachable: (a, b) =>
                regions.has(key(a)) &&
                regions.get(key(a)) === regions.get(key(b)),
        };
        this.regionCache = { tick: this.tick, speed, api };
        return api;
    }
    makeProduction() {
        return {
            getAvailableObjects: (type) =>
                this.obs.vectors.production
                    .filter(
                        (q) => type === undefined || q.queue === QUEUES[type],
                    )
                    .flatMap((q) =>
                        q.available
                            .map((i) =>
                                this.meta.techno.get(
                                    this.catalog.unit_types[i],
                                ),
                            )
                            .filter(Boolean),
                    ),
            getQueueData: (type) => {
                const q = this.obs.vectors.production.find(
                    (q) => q.queue === QUEUES[type],
                );
                return {
                    type,
                    status: q ? CD.QueueStatus[q.status] : CD.QueueStatus.Idle,
                    size: q?.size ?? 0,
                    maxSize: 99,
                    items: q?.item
                        ? [
                              {
                                  rules: this.meta.techno.get(
                                      this.catalog.unit_types[q.item],
                                  ),
                                  quantity: q.size || 1,
                              },
                          ]
                        : [],
                };
            },
        };
    }
    makeActions() {
        return {
            queueForProduction: (queue, name, type, quantity) => {
                for (let i = 0; i < quantity; i++)
                    this.push(this.buildFns.get(name));
            },
            unqueueFromProduction: (queue, name) =>
                this.push(61, { extra: { item: this.typeIndex.get(name) } }),
            pauseProduction: (q) =>
                this.push(62, { extra: { queue: QUEUES[q] } }),
            resumeProduction: (q) =>
                this.push(63, { extra: { queue: QUEUES[q] } }),
            placeBuilding: (name, rx, ry) => {
                this.metrics.placementAttempts++;
                this.badPlacements.set(`${name}:${rx}:${ry}`, {
                    tick: this.tick,
                    name,
                });
                this.push(60, {
                    world: this.cell(rx, ry),
                    extra: { item: this.typeIndex.get(name) },
                });
            },
            orderUnits: (ids, order, ...target) => {
                let fn,
                    args = {};
                if (!target.length) fn = NO_TARGET.get(order);
                else if (target.length === 1) {
                    fn = UNIT_TARGET.get(order);
                    args.target_unit_tag = target[0];
                } else {
                    fn = WORLD_TARGET.get(order);
                    args.world = this.cell(target[0], target[1]);
                }
                if (fn === undefined) {
                    this.metrics.unmapped.push({
                        tick: this.tick,
                        order,
                        target,
                    });
                    throw new Error(`Unsupported order ${CD.OrderType[order]}`);
                }
                if (args.world === -1) return;
                for (let i = 0; i < ids.length; i += 64)
                    this.push(fn, { ...args, unit_tags: ids.slice(i, i + 64) });
            },
            toggleRepairWrench: (id) => {
                this.repairs.has(id)
                    ? this.repairs.delete(id)
                    : this.repairs.add(id);
                this.push(50, { unit_tags: [id] });
            },
            quitGame: () => this.push(81),
            toggleAlliance: () => {
                throw new Error(
                    "Alliance changes not supported in 1v1 adapter",
                );
            },
            setUnitDebugText: () => {},
            setGlobalDebugText: () => {},
            sayAll: () => {},
        };
    }
}
