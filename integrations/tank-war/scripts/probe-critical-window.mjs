// Repeat every supplied decision against the pinned Tank War reference path.
// Only neutral measurements enter the blinded diagnoser input.
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { cleanCounterfactual } from "./clean-counterfactual.mjs";

const args = Object.fromEntries(
    process.argv.slice(2).map((arg) => {
        const match =
            /^--(run|out|start|end-exclusive|reference-decision-server)=(.+)$/.exec(
                arg,
            );
        if (!match) throw Error(`Unknown argument ${arg}`);
        return [match[1], match[2]];
    }),
);
for (const key of [
    "run",
    "out",
    "start",
    "end-exclusive",
    "reference-decision-server",
])
    if (!args[key]) throw Error(`Missing --${key}`);
const start = Number(args.start),
    end = Number(args["end-exclusive"]);
if (
    !Number.isInteger(start) ||
    !Number.isInteger(end) ||
    start < 0 ||
    end <= start ||
    end - start > 11
)
    throw Error("Probe chunk must contain 1 to 11 decisions");
const run = resolve(args.run);
const calls = (await readFile(join(run, "model_calls.jsonl"), "utf8"))
    .split("\n")
    .filter(Boolean)
    .map(JSON.parse);
if (calls.length < end) throw Error("Probe range exceeds recorded calls");

function distributions(reply, prefix = "main") {
    const found = new Map();
    for (const [head, answer] of Object.entries(reply.answers ?? {}))
        found.set(`${prefix}/${head}`, answer);
    for (const [branch, nested] of Object.entries(reply.branches ?? {}))
        for (const [key, value] of distributions(nested, branch))
            found.set(key, value);
    return found;
}

function probabilities(observed, reference) {
    const before = distributions(observed),
        after = distributions(reference);
    if (before.size !== after.size)
        return {
            max_abs_probability_change: null,
            choice_changed: null,
            option_set_changed: true,
        };
    let max = 0,
        choiceChanged = false;
    for (const [key, answer] of before) {
        const other = after.get(key);
        if (
            !other ||
            JSON.stringify(Object.keys(answer.probabilities)) !==
                JSON.stringify(Object.keys(other.probabilities))
        )
            return {
                max_abs_probability_change: null,
                choice_changed: null,
                option_set_changed: true,
            };
        choiceChanged ||= answer.choice !== other.choice;
        for (const option of Object.keys(answer.probabilities))
            max = Math.max(
                max,
                Math.abs(
                    answer.probabilities[option] - other.probabilities[option],
                ),
            );
    }
    return {
        max_abs_probability_change: max,
        choice_changed: choiceChanged,
        option_set_changed: false,
    };
}

function questionTree(request) {
    return {
        questions: request.questions,
        branches: Object.fromEntries(
            Object.entries(request.branches ?? {}).map(([name, branch]) => [
                name,
                questionTree(branch),
            ]),
        ),
    };
}

const indices = [];
for (let decisionIndex = start; decisionIndex < end; decisionIndex++) {
    const comparison = await cleanCounterfactual({
        run,
        decisionIndex,
        cleanDecisionServer: args["reference-decision-server"],
    });
    if (comparison.reference_model_calls.length !== 1)
        throw Error(`Expected one recorded reference call at ${decisionIndex}`);
    const observed = calls[decisionIndex];
    const reference = comparison.reference_model_calls[0];
    indices.push({
        decision_index: decisionIndex,
        same_model_input:
            JSON.stringify(observed.request) ===
            JSON.stringify(reference.request),
        same_question_text:
            JSON.stringify(questionTree(observed.request)) ===
            JSON.stringify(questionTree(reference.request)),
        ...probabilities(observed.reply, reference.reply),
        selected_action_changed: comparison.local_decision_changed,
        command_changed: comparison.local_command_changed,
        execution_changed: comparison.local_execution_changed,
        repeat_selected_action: comparison.clean.selected_action,
        repeat_command: comparison.clean.acknowledged_command.controls,
        repeat_execution: comparison.clean.executed_action,
    });
}
await writeFile(
    resolve(args.out),
    JSON.stringify({
        probe_version: "tank183-critical-all-decisions-v1",
        identity_tolerance: 0.01,
        indices,
    }) + "\n",
    { flag: "wx" },
);
console.log(
    JSON.stringify({ indices: indices.length, out: resolve(args.out) }),
);
