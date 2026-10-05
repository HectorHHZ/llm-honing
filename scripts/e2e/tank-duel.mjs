// Real Python DecisionServer -> Node bridge -> both game engines. The model is a CPU fixture, unless HONING_URL
// names a decision server that is already running (a real model on a GPU); TANK_SECONDS sets how long each
// engine plays (default 2, at most 120).
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { once } from "node:events";
import {
    createGameServer,
    configuration,
} from "../../integrations/tank-duel/server.mjs";
import { smokeModel } from "../../integrations/tank-duel/scripts/smoke-model.mjs";

const seconds = Number(process.env.TANK_SECONDS ?? 2);
const child = process.env.HONING_URL
    ? null
    : spawn(
          process.env.PYTHON || "python",
          ["scripts/e2e/fake_decision_server.py"],
          {
              stdio: ["ignore", "pipe", "inherit"],
          },
      );
let server;
const timeout = setTimeout(
    () => {
        child?.kill();
        process.exitCode = 1;
    },
    20000 + 3 * seconds * 1000,
);
try {
    let honingUrl = process.env.HONING_URL;
    if (child) {
        const lines = createInterface({ input: child.stdout }),
            ready = await Promise.race([
                once(lines, "line"),
                once(child, "error").then(([e]) => {
                    throw e;
                }),
                once(child, "exit").then(() => {
                    throw Error("Decision server exited before readiness");
                }),
            ]);
        honingUrl = JSON.parse(ready[0]).url;
    }
    server = createGameServer(configuration({ HONING_URL: honingUrl }));
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const url = `http://127.0.0.1:${server.address().port}`;
    for (const [edition, controllerMode] of [
        ["classic", "plans"],
        ["classic", "raw"],
        ["auto-aim", "raw"],
    ])
        console.log(
            JSON.stringify(
                await smokeModel({ url, edition, controllerMode, seconds }),
            ),
        );
} finally {
    clearTimeout(timeout);
    child?.kill();
    if (server)
        await new Promise((resolve) => {
            server.close(resolve);
            server.closeAllConnections();
        });
}
