import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { ROOT, PINS } from "../integration/paths.mjs";
const run = (args, cwd = ROOT) =>
    execFileSync(args[0], args.slice(1), { cwd, stdio: "inherit" });
mkdirSync(join(ROOT, "deps"), { recursive: true });
for (const [name, pin] of [
    ["supalosa", PINS.supalosa],
    ["laya", PINS.layaSource],
]) {
    const dest = join(ROOT, "deps", name);
    if (!existsSync(dest)) {
        run(["git", "clone", pin.url, dest]);
        run(["git", "checkout", "--detach", pin.revision], dest);
    }
    const revision = execFileSync("git", ["rev-parse", "HEAD"], {
        cwd: dest,
        encoding: "utf8",
    }).trim();
    if (revision !== pin.revision)
        throw new Error(
            `${dest} is not at the pinned commit; refusing to change an existing checkout`,
        );
    if (name === "supalosa") {
        run(["npm", "ci"], dest);
        run(["npm", "run", "build"], dest);
    }
}
