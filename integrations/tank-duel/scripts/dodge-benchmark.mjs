import { Game } from "../web/classic/engine.js";
import { pathToFileURL } from "node:url";
// Optional previous engine module path: keep its relative imports alongside it.
const versions = [["current", Game]];
if (process.argv[2])
    versions.unshift([
        "previous",
        (await import(pathToFileURL(process.argv[2]).href)).Game,
    ]);
const results = [];
for (const [version, Engine] of versions)
    for (const level of ["light", "standard", "strong"]) {
        let survived = 0,
            frames = 0,
            assisted = 0;
        for (const gap of [70, 100, 140, 200])
            for (let j = 0; j < 8; j++) {
                const g = new Engine(1);
                g.maze.walls = [];
                g.setAssistEnabled(true);
                g.setAssistProfile(level);
                Object.assign(g.tanks[0], {
                    x: 450,
                    y: 315,
                    angle: (j * Math.PI) / 4,
                });
                Object.assign(g.tanks[1], { x: 780, y: 530 });
                g.bullets = [
                    {
                        id: "incoming",
                        owner: 1,
                        x: 450 - gap,
                        y: 315,
                        vx: 290,
                        vy: 0,
                        age: 1,
                        bounces: 0,
                        trail: [],
                    },
                ];
                g.start();
                for (let k = 0; k < 90 && g.phase === "running"; k++) {
                    g.step(1 / 60);
                    frames++;
                    if (g.tanks[0].assistActive) assisted++;
                }
                if (g.tanks[0].alive) survived++;
            }
        results.push({
            version,
            level,
            survived,
            total: 32,
            assist_fraction: assisted / frames,
        });
    }
console.log(
    JSON.stringify(
        {
            fixture:
                "Four incoming-shell gaps x eight hull headings, no walls, stationary controls; not match win rate.",
            results,
        },
        null,
        2,
    ),
);
