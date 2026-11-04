import { boot } from "../shared/game-client.js";
import { adapter } from "../classic/adapter.js";

await boot(adapter, { humanChallenge: true });
