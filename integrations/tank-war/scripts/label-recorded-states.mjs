// Give the planner's answer on the exact states the model saw. These are
// reference actions, never injection-cause labels.
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { adapter } from "../web/game/adapter.js";
import { plannerTeacher, TEACHER_MODEL } from "../web/game/teacher.js";
import { decide } from "../model-bridge.mjs";
import { restoreGame, stateHash } from "./state-replay.mjs";

const lines = (text) =>
    text
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line));

export async function labelRecordedStates(run) {
    const manifest = JSON.parse(
        await readFile(join(run, "manifest.json"), "utf8"),
    );
    const states = lines(await readFile(join(run, "states.jsonl"), "utf8"));
    if (manifest.agent_seat.controllerMode !== "plans")
        throw Error("Planner labels require planned steering");
    const labels = [];
    for (const saved of states) {
        if (stateHash(saved.raw_state) !== saved.state_hash)
            throw Error(
                `State hash mismatch at decision ${saved.decision_index}`,
            );
        const game = restoreGame(saved.raw_state);
        const observed = adapter.observe(game, saved.player, {
            ...manifest.agent_seat,
            interval: manifest.interval_ms,
            rtt: 0,
        });
        if (JSON.stringify(observed) !== JSON.stringify(saved.observation))
            throw Error(
                `Observation mismatch at decision ${saved.decision_index}`,
            );
        const response = await decide(
            {
                model: TEACHER_MODEL,
                agentIds: ["planner", "planner"],
                contextLimit: null,
                ask: plannerTeacher(observed),
            },
            saved.request,
        );
        labels.push({
            decision_index: saved.decision_index,
            state_hash: saved.state_hash,
            source: TEACHER_MODEL,
            source_game_revision: manifest.sources["web/game/planner.js"],
            reference_action: {
                motion: response.policy_motion_id,
                weapon: response.weapon_choice,
            },
            acceptable_actions: [
                {
                    motion: response.policy_motion_id,
                    weapon: response.weapon_choice,
                },
            ],
            review_status: "unreviewed",
        });
    }
    return labels;
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
    const args = Object.fromEntries(
        process.argv.slice(2).map((argument) => {
            const match = /^--(run|out)=(.+)$/.exec(argument);
            if (!match) throw Error(`Unknown argument ${argument}`);
            return [match[1], match[2]];
        }),
    );
    if (!args.run || !args.out)
        throw Error(
            "Usage: node scripts/label-recorded-states.mjs --run=DIR --out=FILE",
        );
    const labels = await labelRecordedStates(resolve(args.run));
    await writeFile(
        resolve(args.out),
        labels.map((label) => JSON.stringify(label) + "\n").join(""),
    );
    console.log(
        JSON.stringify({ labelled: labels.length, out: resolve(args.out) }),
    );
}
