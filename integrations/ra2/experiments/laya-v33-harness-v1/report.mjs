// Read saved results only. Never load a model or an engine.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { RUNS_DIR } from "./runtime.mjs";
import { verdict } from "./run.mjs";

export function summarize(runsDir = RUNS_DIR) {
    const groups = new Map(),
        unfinished = [],
        invalid = [];
    if (!existsSync(runsDir)) return { groups: [], unfinished, invalid };
    for (const entry of readdirSync(runsDir, { withFileTypes: true }).sort(
        (a, b) => a.name.localeCompare(b.name),
    )) {
        if (!entry.isDirectory()) continue;
        const dir = join(runsDir, entry.name),
            resultPath = join(dir, "result.json");
        if (!existsSync(resultPath)) {
            unfinished.push({ path: dir, reason: "no_result_file" });
            continue;
        }
        let result;
        try {
            result = JSON.parse(readFileSync(resultPath, "utf8"));
        } catch (error) {
            invalid.push({
                path: dir,
                reason: `unreadable_result: ${error.message}`,
            });
            continue;
        }
        if (
            !result.valid ||
            !result.comparisonId ||
            !["red", "blue"].includes(result.candidateSeat)
        ) {
            invalid.push({
                path: dir,
                reason: result.error ?? "invalid_or_unidentified_run",
            });
            continue;
        }
        const outcome = verdict(result, result.candidateSeat);
        if (outcome === "unknown_terminal") {
            invalid.push({ path: dir, reason: "unknown_terminal_outcome" });
            continue;
        }
        let group = groups.get(result.comparisonId);
        if (!group) {
            group = {
                comparisonId: result.comparisonId,
                configuration: result.configuration,
                wins: 0,
                losses: 0,
                draws: 0,
                unresolved: 0,
                runs: [],
                seeds: new Map(),
                inferenceCalls: { candidate: 0, baseline: 0 },
                inferenceSeconds: { candidate: 0, baseline: 0 },
                skippedBaseSlots: 0,
            };
            groups.set(result.comparisonId, group);
        }
        group[
            {
                win: "wins",
                loss: "losses",
                draw: "draws",
                unresolved: "unresolved",
            }[outcome]
        ]++;
        const other = result.candidateSeat === "red" ? "blue" : "red";
        for (const [role, seat] of [
            ["candidate", result.candidateSeat],
            ["baseline", other],
        ]) {
            group.inferenceCalls[role] +=
                result.stats?.[seat]?.inferenceCalls ?? 0;
            group.inferenceSeconds[role] +=
                (result.stats?.[seat]?.inferenceMs ?? 0) / 1000;
        }
        group.skippedBaseSlots += result.scheduling?.skippedBaseSlots ?? 0;
        group.runs.push({
            path: dir,
            seed: result.seed,
            candidateSeat: result.candidateSeat,
            outcome,
            tick: result.tick,
        });
        if (!group.seeds.has(result.seed))
            group.seeds.set(result.seed, { red: [], blue: [] });
        group.seeds
            .get(result.seed)
            [result.candidateSeat].push({ path: dir, outcome });
    }
    return {
        groups: [...groups.values()].map(({ seeds, ...group }) => ({
            ...group,
            validRuns: group.runs.length,
            winsOverAllValidRuns: `${group.wins}/${group.runs.length}`,
            pairedSeeds: [...seeds]
                .filter(
                    ([, seats]) =>
                        seats.red.length === 1 && seats.blue.length === 1,
                )
                .map(([seed]) => seed),
            unpairedOrRepeatedSeeds: [...seeds]
                .filter(
                    ([, seats]) =>
                        seats.red.length !== 1 || seats.blue.length !== 1,
                )
                .map(([seed, seats]) => ({
                    seed,
                    redRuns: seats.red.length,
                    blueRuns: seats.blue.length,
                })),
        })),
        unfinished,
        invalid,
    };
}

if (
    process.argv[1] &&
    pathToFileURL(resolve(process.argv[1])).href === import.meta.url
) {
    if (process.argv.length > 3) {
        console.error("Usage: node report.mjs [runs-directory]");
        process.exitCode = 1;
    } else
        console.log(
            JSON.stringify(
                summarize(
                    process.argv[2] ? resolve(process.argv[2]) : RUNS_DIR,
                ),
                null,
                2,
            ),
        );
}
