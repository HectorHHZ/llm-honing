// Scan a recorded Tank War episode in decision order and retain every local
// clean comparison until the first fault-caused difference is found.
import { readFile, appendFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { cleanCounterfactual } from "./clean-counterfactual.mjs";

function argsFrom(argv) {
    const args = Object.fromEntries(
        argv.map((argument) => {
            const match =
                /^--(run|out|clean-decision-server|clean-harness-dir|limit)=(.+)$/.exec(
                    argument,
                );
            if (!match) throw Error(`Unknown argument ${argument}`);
            return [match[1], match[2]];
        }),
    );
    if (!args.run || !args.out) throw Error("--run and --out are required");
    return args;
}

const args = argsFrom(process.argv.slice(2));
const run = resolve(args.run);
const out = resolve(args.out);
const states = (await readFile(join(run, "states.jsonl"), "utf8"))
    .split("\n")
    .filter(Boolean)
    .map(JSON.parse);
const triggerFile = join(run, "private_fault_events.jsonl");
let candidates;
try {
    candidates = (await readFile(triggerFile, "utf8"))
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line).decision_index);
} catch (error) {
    if (error.code !== "ENOENT") throw error;
    candidates = states.map((state) => state.decision_index);
}
candidates = [...new Set(candidates)].sort((a, b) => a - b);
const limit = args.limit === undefined ? candidates.length : Number(args.limit);
if (!Number.isInteger(limit) || limit < 1) throw Error("--limit must be a positive integer");
await writeFile(out, "", { flag: "wx" });
let first = null;
let scanned = 0;
for (const decisionIndex of candidates.slice(0, limit)) {
    const result = await cleanCounterfactual({
        run,
        decisionIndex,
        cleanHarnessDir: args["clean-harness-dir"] &&
            resolve(args["clean-harness-dir"]),
        cleanDecisionServer: args["clean-decision-server"],
    });
    await appendFile(out, JSON.stringify(result) + "\n");
    scanned++;
    if (result.locally_verified) {
        first = decisionIndex;
        break;
    }
}
console.log(JSON.stringify({ run, scanned, first_verified_decision: first, out }));
