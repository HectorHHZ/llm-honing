// Run the same three reference interventions at every index in a supplied window.
// The output contains evidence only; the private injection manifest is not copied.
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { adapter } from "../web/game/adapter.js";
import { healthFor, upstream, validateRequest } from "../model-bridge.mjs";
import { cleanCounterfactual } from "./clean-counterfactual.mjs";
import { loadHarness } from "./match.mjs";
import { restoreGame, stateHash } from "./state-replay.mjs";

const lines = (text) => text.split("\n").filter(Boolean).map(JSON.parse);

function options(argv) {
    const args = Object.fromEntries(
        argv.map((argument) => {
            const match =
                /^--(run|out|start|end-exclusive|reference-decision-server|observed-harness-dir)=(.+)$/.exec(
                    argument,
                );
            if (!match) throw Error(`Unknown argument ${argument}`);
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
    const start = Number(args.start);
    const end = Number(args["end-exclusive"]);
    if (
        !Number.isInteger(start) ||
        !Number.isInteger(end) ||
        start < 0 ||
        end <= start ||
        end - start > 11
    )
        throw Error("Probe window must contain 1 to 11 decision indices");
    return { ...args, start, end };
}

function distributions(reply) {
    const rows = [];
    function collect(branch, item) {
        for (const [head, answer] of Object.entries(item.answers ?? {})) {
            if (!answer.probabilities)
                throw Error(`Missing probabilities for ${branch}/${head}`);
            rows.push({
                branch,
                head,
                choice: answer.choice,
                probabilities: answer.probabilities,
            });
        }
    }
    collect("main", reply);
    for (const [branch, item] of Object.entries(reply.branches ?? {}))
        collect(branch, item);
    return rows;
}

export function probabilityProbe(observed, baseline) {
    const oldRows = distributions(observed);
    const newRows = distributions(baseline);
    const oldByKey = new Map(
        oldRows.map((row) => [`${row.branch}/${row.head}`, row]),
    );
    if (oldRows.length !== newRows.length)
        throw Error("Reference model changed question/branch count");
    const comparisons = [];
    let maxDelta = 0;
    for (const row of newRows) {
        const previous = oldByKey.get(`${row.branch}/${row.head}`);
        if (
            !previous ||
            JSON.stringify(Object.keys(previous.probabilities)) !==
                JSON.stringify(Object.keys(row.probabilities))
        )
            throw Error("Reference model changed option labels");
        const options = Object.fromEntries(
            Object.entries(row.probabilities).map(([option, probability]) => {
                const before = previous.probabilities[option];
                const delta = Math.abs(probability - before);
                if (!Number.isFinite(delta))
                    throw Error("Nonfinite probability difference");
                maxDelta = Math.max(maxDelta, delta);
                return [option, { observed: before, reference: probability }];
            }),
        );
        comparisons.push({
            branch: row.branch,
            head: row.head,
            observed_choice: previous.choice,
            reference_choice: row.choice,
            options,
        });
    }
    return {
        max_abs_probability_difference: maxDelta,
        exceeds_identity_tolerance_0p01: maxDelta > 0.01,
        distributions: comparisons,
    };
}

export function textDifference(observed, reference) {
    if (observed === reference) return null;
    const before = JSON.stringify(observed) ?? "undefined";
    const after = JSON.stringify(reference) ?? "undefined";
    if (before === after) return null;
    let first = 0;
    while (
        first < Math.min(before.length, after.length) &&
        before[first] === after[first]
    )
        first++;
    return {
        observed_excerpt: before.slice(Math.max(0, first - 100), first + 700),
        reference_excerpt: after.slice(Math.max(0, first - 100), first + 700),
    };
}

function rerender(state, manifest, harness) {
    if (stateHash(state.raw_state) !== state.state_hash)
        throw Error("Raw state hash mismatch");
    const game = restoreGame(state.raw_state);
    const player = state.player;
    Object.assign(game.playerSettings[player], {
        assistEnabled: harness.seat.assistEnabled,
        assistProfile: harness.seat.assistProfile,
        fireGuardEnabled: harness.seat.fireGuardEnabled,
    });
    const observed = adapter.observe(game, player, {
        ...harness.seat,
        interval: manifest.interval_ms,
        rtt: 0,
    });
    const body = {
        ...observed.request,
        id: state.request.id,
        player,
        round_id: game.roundSerial,
        map_revision: game.maze.revision,
    };
    validateRequest(body, "tank-war");
    const fields = new Set([
        ...Object.keys(state.request),
        ...Object.keys(body),
    ]);
    const changed = Object.fromEntries(
        [...fields]
            .sort()
            .map((field) => [
                field,
                textDifference(state.request[field], body[field]),
            ])
            .filter(([, difference]) => difference !== null),
    );
    return {
        same_rendered_request: Object.keys(changed).length === 0,
        changed_fields: changed,
    };
}

async function main() {
    const args = options(process.argv.slice(2));
    const run = resolve(args.run);
    const states = lines(await readFile(join(run, "states.jsonl"), "utf8"));
    const calls = lines(await readFile(join(run, "model_calls.jsonl"), "utf8"));
    const manifest = JSON.parse(
        await readFile(join(run, "manifest.json"), "utf8"),
    );
    if (states.length !== calls.length || args.end > states.length)
        throw Error("Probe window exceeds recorded decisions");
    const baselineHarness = await loadHarness();
    const backend = {
        url: args["reference-decision-server"].replace(/\/$/, ""),
        transport: "honing",
    };
    const health = await healthFor(
        backend,
        "tank-war",
        AbortSignal.timeout(60000),
    );
    if (
        !health.ready ||
        health.metadata.weights_sha256 !==
            "f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b" ||
        health.metadata.context_limit !== 8192
    )
        throw Error("Reference checkpoint is not the pinned clean release");
    const results = [];
    for (let index = args.start; index < args.end; index++) {
        const state = states[index];
        if (state.decision_index !== index)
            throw Error(`Decision index mismatch ${index}`);
        const referenceReply = await upstream(
            backend,
            "/v1/decide",
            calls[index].request,
            AbortSignal.timeout(300000),
        );
        const controller = await cleanCounterfactual({
            run,
            decisionIndex: index,
            cleanHarnessDir:
                args["observed-harness-dir"] &&
                resolve(args["observed-harness-dir"]),
        });
        results.push({
            decision_index: index,
            reference_render: rerender(state, manifest, baselineHarness),
            reference_checkpoint: probabilityProbe(
                calls[index].reply,
                referenceReply,
            ),
            reference_controller: {
                observed_selected_action: controller.faulty.selected_action,
                reference_selected_action: controller.clean.selected_action,
                observed_command: controller.faulty.acknowledged_command,
                reference_command: controller.clean.acknowledged_command,
                observed_execution: controller.faulty.executed_action,
                reference_execution: controller.clean.executed_action,
                selected_action_changed: controller.local_decision_changed,
                command_changed: controller.local_command_changed,
                execution_changed: controller.local_execution_changed,
            },
        });
    }
    await writeFile(
        resolve(args.out),
        JSON.stringify({
            probe_version: "tank183-window-v1",
            identity_tolerance: 0.01,
            indices: results,
        }) + "\n",
        { flag: "wx" },
    );
    console.log(
        JSON.stringify({ indices: results.length, out: resolve(args.out) }),
    );
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) await main();
