import { verifyEnsemble } from "./permutation-client.mjs";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { openLog } from "./jsonl.mjs";
import { resolve } from "node:path";
import { STAGED_PROTOCOL, verifyStagedTrace } from "./staged.mjs";
const dir = resolve(process.argv[2]);
const lines = (name) =>
    readFileSync(dir + "/" + name, "utf8")
        .trim()
        .split("\n")
        .filter(Boolean)
        .map(JSON.parse);
const result = JSON.parse(readFileSync(dir + "/result.json", "utf8")),
    actions = lines("actions.jsonl");
const policySeat = result.policySeat ?? result.layaSeat;
assert(["red", "blue"].includes(policySeat), "Missing policy seat");
if (result.policySeat && result.layaSeat)
    assert.equal(
        result.policySeat,
        result.layaSeat,
        "Contradictory policy seats",
    );
const selectedModelActions = [];
let decisionCount = 0,
    selectedCount = 0,
    droppedCount = 0,
    omittedCount = 0;
let calls = 0,
    questions = 0,
    waits = 0,
    peakTokens = 0,
    trimmed = 0,
    parameterCalls = 0,
    maxStages = 0;
let rootQuestions = 0,
    rootWaits = 0,
    parameterQuestions = 0,
    parameterCancels = 0;
const latencies = [],
    choiceCounts = {},
    functionCounts = {};
function auditCall(record, stageIndex = 1) {
    if (!Object.keys(record.questions).length) return;
    verifyEnsemble(record);
    calls += record.forwardCalls ?? 1;
    const forwards = record.ensemble
        ? [record.ensemble.original, record.ensemble.swapped]
        : [record];
    if (result.modelFamily && result.modelFamily !== "laya")
        throw new Error(
            `This audit covers the laya family only, not ${result.modelFamily}`,
        );
    latencies.push(...forwards.map((r) => r.latencyMs));
    for (const r of forwards)
        for (const a of Object.values(r.tokenAudit))
            peakTokens = Math.max(peakTokens, a.inputTokens);
    trimmed += record.trimmed.length * (record.forwardCalls ?? 1);
    // The actual model text must include exactly the retained recorded context.
    const retained = structuredClone(record.contextParts);
    for (const removed of record.trimmed) {
        const at =
            removed.section === "logs"
                ? 0
                : retained[removed.section].length - 1;
        assert.equal(retained[removed.section][at], removed.text);
        retained[removed.section].splice(at, 1);
    }
    const packed = ["essential", "details", "memory", "logs"]
        .flatMap((k) => retained[k])
        .join("\n");
    assert.equal(
        record.stateText,
        packed,
        "Model text differs from the audited context",
    );
    for (const [id, q] of Object.entries(record.questions)) {
        const answer = record.answers[id];
        assert(answer);
        assert(Object.hasOwn(q.criteria, answer.choice));
        questions++;
        const group = record.candidates.find((g) => g.id === id);
        assert(group);
        const choice = group.choices.find((c) => c.key === answer.choice);
        assert(choice);
        const waiting = choice.kind ? choice.kind === "cancel" : !choice.action;
        waits += Number(waiting);
        if (stageIndex === 1) {
            rootQuestions++;
            rootWaits += Number(waiting);
        } else {
            parameterQuestions++;
            parameterCancels += Number(waiting);
        }
        const key =
            result.decisionMode === "staged" ? `${stageIndex}:${id}` : id;
        choiceCounts[key] = (choiceCounts[key] ?? 0) + 1;
        const ps = Object.values(answer.probabilities);
        assert.deepEqual(
            Object.keys(answer.probabilities).sort(),
            Object.keys(q.criteria).sort(),
        );
        assert(ps.every((p) => Number.isFinite(p) && p >= 0 && p <= 1));
        assert(Math.abs(ps.reduce((a, b) => a + b, 0) - 1) < 0.005);
        // Probabilities are logged rounded; a 0.0001 tolerance allows a rounded tie.
        assert(answer.probabilities[answer.choice] + 0.0001 >= Math.max(...ps));
        const a = record.tokenAudit[id];
        assert(a);
        assert.equal(a.options, Object.keys(q.criteria).length);
        assert.equal(a.stateTruncated, false);
        assert(a.stateTokens <= a.stateRoom);
        peakTokens = Math.max(peakTokens, a.inputTokens);
    }
}
for await (const line of createInterface({
    input: openLog(dir + "/decisions.jsonl"),
    crlfDelay: Infinity,
})) {
    if (!line.trim()) continue;
    const d = JSON.parse(line),
        i = decisionCount++;
    selectedCount += d.selected.length;
    droppedCount += d.dropped.length;
    omittedCount += d.omitted.length;
    selectedModelActions.push(
        ...d.selected.map((c) => ({ tick: d.tick, action: c.action })),
    );
    assert.equal(d.seat, policySeat);
    assert.equal(d.tick, i * result.cadenceTicks);
    if (result.candidateMode === "intent") {
        assert(
            d.selected.length <= 1,
            "Intent policy submitted multiple commands",
        );
        if (d.tick % result.decisionInterval !== 0)
            assert.equal(d.stages.length, 0, "Off-schedule model call");
    }
    let answers;
    if (result.decisionMode === "staged") {
        assert.equal(d.decisionMode, "staged");
        answers = verifyStagedTrace(
            {
                tick: d.tick,
                parts: d.contextParts,
                groups: d.candidates,
                questions: d.leafQuestions,
                omitted: d.omitted,
            },
            d,
            assert,
        );
        assert.equal(
            d.inferenceCalls,
            d.stages.reduce((n, s) => n + (s.forwardCalls ?? 1), 0),
        );
        parameterCalls += d.stages
            .slice(1)
            .reduce((n, s) => n + (s.forwardCalls ?? 1), 0);
        maxStages = Math.max(maxStages, d.stages.length);
        for (const stage of d.stages) auditCall(stage, stage.stageIndex);
    } else {
        answers = d.answers;
        auditCall(d);
        maxStages = Math.max(
            maxStages,
            Object.keys(d.questions).length ? 1 : 0,
        );
    }
    for (const c of d.selected) {
        const group = d.candidates.find((g) => g.id === c.group);
        assert(group);
        assert.deepEqual(
            group.choices.find((x) => x.key === c.key).action,
            c.action,
        );
        assert.equal(answers[c.group].choice, c.key);
        for (const tag of c.action.unit_tags ?? [])
            assert(
                d.normalUnits.some((u) => u.tag === tag && u.alliance === 1),
                "foreign actor",
            );
        if (c.action.target_unit_tag !== undefined)
            assert(
                d.normalUnits.some(
                    (u) =>
                        u.tag === c.action.target_unit_tag &&
                        (u.alliance === 1 ||
                            u.alliance === 2 ||
                            (u.display_type === 1 && !u.is_blip)),
                ),
                "hidden target",
            );
        functionCounts[c.action.function] =
            (functionCounts[c.action.function] ?? 0) + 1;
    }
}
assert.equal(calls, result.inferenceCalls);
assert.equal(
    decisionCount,
    Math.ceil(result.tick / result.cadenceTicks),
    "Recorded decisions do not cover every simulation cadence",
);
const modelActions = actions
    .filter((a) => a.seat === policySeat)
    .map((a) => ({ tick: a.tick, action: a.action }));
assert.deepEqual(
    modelActions,
    selectedModelActions,
    "unselected model action inserted",
);
for (const seat of ["red", "blue"]) {
    const a = actions.filter((a) => a.seat === seat);
    assert.equal(a.length, result.stats[seat].submitted);
    assert.equal(
        a.filter((a) => a.result?.accepted).length,
        result.stats[seat].accepted,
    );
    const perTick = new Map();
    for (const x of a) perTick.set(x.tick, (perTick.get(x.tick) ?? 0) + 1);
    assert([...perTick.values()].every((n) => n <= result.maxActionsPerSeat));
}
latencies.sort((a, b) => a - b);
const pct = (p) =>
    latencies[
        Math.min(latencies.length - 1, Math.floor(latencies.length * p))
    ] ?? 0;
const audit = {
    passed: true,
    protocol: result.protocol,
    tick: result.tick,
    policySeat,
    ...(result.layaSeat ? { layaSeat: result.layaSeat } : {}),
    cadenceTicks: result.cadenceTicks,
    decisions: decisionCount,
    inferenceCalls: calls,
    parameterCalls,
    maxStagesObserved: maxStages,
    choiceQuestions: questions,
    latencyUnit: result.inferenceCallUnit ?? "native_model_forward",
    choiceCountUnit: "canonical_staged_choice",
    waitChoices: waits,
    rootQuestions,
    rootWaitChoices: rootWaits,
    parameterQuestions,
    cancelledParameterChoices: parameterCancels,
    selectedActions: selectedCount,
    droppedActions: droppedCount,
    omittedCandidateGroups: omittedCount,
    trimmedStateItems: trimmed,
    peakInputTokens: peakTokens,
    latencyMs: { p50: pct(0.5), p95: pct(0.95), max: pct(1) },
    choiceGroups: choiceCounts,
    selectedFunctions: functionCounts,
    checks: [
        ...(result.decisionMode === "staged"
            ? [
                  "every stage uses the same tick and carries exact prior selections",
                  "every parameter path resolves to an original grounded candidate",
              ]
            : []),
        "actual model text matches retained context",
        "binary target decisions equal mapped normalized means of two actual model calls",
        "submitted model actions exactly equal selected leaves",
        "all actors owned and targets currently observable",
        "options and essential context never silently truncated",
        `engine cadence ${result.cadenceTicks} ticks; model interval ${
            result.decisionInterval ?? result.cadenceTicks
        } ticks; per-seat raw cap ${result.maxActionsPerSeat} verified`,
        "recorded receipts agree with aggregate stats",
    ],
};
if (!process.argv.includes("--check-only"))
    writeFileSync(dir + "/audit.json", JSON.stringify(audit, null, 2));
console.log(JSON.stringify(audit));
