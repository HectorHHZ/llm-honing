// Evaluate one recorded fault state with the clean harness, checkpoint, and
// controller. This private evidence never enters the blinded diagnoser input.
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { adapter } from "../web/game/adapter.js";
import {
    decide,
    healthFor,
    upstream,
    validateRequest,
} from "../model-bridge.mjs";
import { loadHarness } from "./match.mjs";
import { restoreGame, stateHash } from "./state-replay.mjs";

const lines = (text) => text.split("\n").filter(Boolean).map(JSON.parse);

export async function cleanCounterfactual({
    run,
    decisionIndex,
    cleanHarnessDir,
    cleanDecisionServer,
}) {
    const manifest = JSON.parse(
        await readFile(join(run, "manifest.json"), "utf8"),
    );
    const states = lines(await readFile(join(run, "states.jsonl"), "utf8"));
    const executions = lines(
        await readFile(join(run, "execution.jsonl"), "utf8"),
    );
    const recordedCalls = lines(
        await readFile(join(run, "model_calls.jsonl"), "utf8"),
    );
    const state = states.find((row) => row.decision_index === decisionIndex);
    const faulty = executions.find(
        (row) => row.decision_index === decisionIndex,
    );
    if (!state || !faulty || !recordedCalls[decisionIndex])
        throw Error(`Missing complete decision ${decisionIndex}`);
    if (stateHash(state.raw_state) !== state.state_hash)
        throw Error(`Raw-state hash mismatch at decision ${decisionIndex}`);
    const harness = await loadHarness(cleanHarnessDir);
    if (harness.seat.controllerMode !== "plans")
        throw Error("Exp 0 requires planned steering");
    const game = restoreGame(state.raw_state);
    const player = state.player;
    game.setPlayerSettings(player, {
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
    const useRecordedReply = !cleanDecisionServer;
    if (
        useRecordedReply &&
        JSON.stringify(body) !== JSON.stringify(state.request)
    )
        throw Error("Recorded reply cannot score a changed clean request");
    let model, ask;
    if (useRecordedReply) {
        model = manifest.model;
        const recorded = recordedCalls[decisionIndex];
        ask = async (request) => {
            if (JSON.stringify(request) !== JSON.stringify(recorded.request))
                throw Error("Recorded model request differs");
            return recorded.reply;
        };
    } else {
        const backend = {
            url: cleanDecisionServer.replace(/\/$/, ""),
            transport: "honing",
        };
        model = (
            await healthFor(backend, "tank-war", AbortSignal.timeout(60000))
        ).metadata;
        ask = (request) =>
            upstream(
                backend,
                "/v1/decide",
                request,
                AbortSignal.timeout(300000),
            );
    }
    const response = await decide(
        {
            model: model.model,
            agentIds: ["agent", "agent"],
            contextLimit: null,
            branches: model.branches,
            ask,
        },
        body,
        { questions: harness.questions, notes: harness.notes },
    );
    if (!adapter.validControls(response, observed))
        throw Error("Clean response has invalid controls");
    const applied = adapter.apply(game, player, response, {
        ...harness.seat,
        interval: manifest.interval_ms,
    });
    const shotsBefore = game.shots[player];
    game.step(1 / 60);
    const tank = game.tanks[player];
    const clean = {
        selected_action: {
            motion: response.policy_motion_id,
            weapon: response.weapon_choice,
        },
        acknowledged_command: applied,
        executed_action: {
            drive: tank.executedDrive,
            turn: tank.executedTurn,
            shot_fired: game.shots[player] > shotsBefore,
        },
    };
    const localDecisionChanged =
        JSON.stringify(clean.selected_action) !==
        JSON.stringify(faulty.selected_action);
    const localCommandChanged =
        JSON.stringify(clean.acknowledged_command.controls) !==
        JSON.stringify(faulty.acknowledged_command.controls);
    const localExecutionChanged =
        JSON.stringify(clean.executed_action) !==
        JSON.stringify(faulty.executed_action);
    return {
        decision_index: decisionIndex,
        source_state_hash: state.state_hash,
        clean_harness_files: harness.files,
        clean_model: model,
        faulty: {
            selected_action: faulty.selected_action,
            acknowledged_command: faulty.acknowledged_command,
            executed_action: faulty.executed_action,
        },
        clean,
        local_decision_changed: localDecisionChanged,
        local_command_changed: localCommandChanged,
        local_execution_changed: localExecutionChanged,
        locally_verified:
            localDecisionChanged ||
            localCommandChanged ||
            localExecutionChanged,
        outcome_counterfactual: "not_run",
    };
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
    const args = Object.fromEntries(
        process.argv.slice(2).map((argument) => {
            const match =
                /^--(run|decision-index|clean-harness-dir|clean-decision-server|out)=(.+)$/.exec(
                    argument,
                );
            if (!match) throw Error(`Unknown argument ${argument}`);
            return [match[1], match[2]];
        }),
    );
    if (!args.run || !args["decision-index"] || !args.out)
        throw Error(
            "Usage: node clean-counterfactual.mjs --run=DIR --decision-index=N --out=FILE [--clean-harness-dir=DIR] [--clean-decision-server=URL]",
        );
    const result = await cleanCounterfactual({
        run: resolve(args.run),
        decisionIndex: Number(args["decision-index"]),
        cleanHarnessDir:
            args["clean-harness-dir"] && resolve(args["clean-harness-dir"]),
        cleanDecisionServer: args["clean-decision-server"],
    });
    await writeFile(resolve(args.out), JSON.stringify(result, null, 2), {
        flag: "wx",
    });
    console.log(
        JSON.stringify({
            decision_index: result.decision_index,
            locally_verified: result.locally_verified,
            out: resolve(args.out),
        }),
    );
}
