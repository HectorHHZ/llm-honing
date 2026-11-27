import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { playMatch } from "../scripts/match.mjs";
import { parseControllerFault } from "../scripts/controller-faults.mjs";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const lines = async (file) =>
    (await readFile(file, "utf8"))
        .split("\n")
        .filter(Boolean)
        .map((l) => JSON.parse(l));
// The file build-controller-faults.py writes for one kind.
const faultText = (kind) =>
    JSON.stringify(
        { layer: "controller", kind, family: kind, instance: `${kind}-v1` },
        null,
        2,
    ) + "\n";

async function teacherRound(kind) {
    const dir = await mkdtemp(join(tmpdir(), "tank-controller-fault-"));
    let controllerFaultFile;
    if (kind) {
        controllerFaultFile = join(dir, "fault.json");
        await writeFile(controllerFaultFile, faultText(kind));
    }
    const out = join(dir, "out");
    // On seed 2002 the teacher fires within six seconds.
    const result = await playMatch({
        seed: 2002,
        seat: "blue",
        out,
        teacher: true,
        seconds: 6,
        controllerFaultFile,
    });
    const events = join(out, "private_fault_events.jsonl");
    return {
        result,
        out,
        manifest: JSON.parse(await readFile(join(out, "manifest.json"))),
        decisions: await lines(join(out, "decisions.jsonl")),
        events: existsSync(events) ? await lines(events) : [],
    };
}

test("the switch is the controller-faults.mjs the Exp 0 v0 freeze pins", async () => {
    const freeze = JSON.parse(
        await readFile(join(ROOT, "exp0/v0/freeze.json"), "utf8"),
    );
    assert.equal(
        sha256(await readFile(join(ROOT, "scripts/controller-faults.mjs"))),
        freeze.game.sources_sha256["scripts/controller-faults.mjs"],
    );
});

test("a controller fault file must name a known kind and a plain instance", () => {
    for (const kind of [
        "swap_selected_turn",
        "drop_fire_after_selection",
        "wrong_legal_projection",
    ])
        assert.equal(parseControllerFault(faultText(kind)).kind, kind);
    for (const bad of [
        { layer: "harness", kind: "swap_selected_turn", instance: "a" },
        { layer: "controller", kind: "invert_selected_drive", instance: "a" },
        { layer: "controller", kind: "swap_selected_turn", instance: "A b" },
    ])
        assert.throws(
            () => parseControllerFault(JSON.stringify(bad)),
            /Invalid controller fault manifest/,
        );
});

test("a dropped fire keeps the selected fire in the log and fires no shell", async () => {
    const clean = await teacherRound(null);
    const fault = await teacherRound("drop_fire_after_selection");
    assert.ok(clean.result.shots.blue > 0);
    assert.equal(fault.result.shots.blue, 0);
    assert.equal(clean.manifest.controller_fault_sha256, null);
    assert.equal(
        fault.manifest.controller_fault_sha256,
        sha256(faultText("drop_fire_after_selection")),
    );
    assert.ok(
        "scripts/controller-faults.mjs" in fault.manifest.sources &&
            !existsSync(join(clean.out, "private_fault_events.jsonl")),
    );
    // Each event is a decision whose selected fire the controller dropped, and the log still shows the selection.
    assert.ok(fault.events.length > 0);
    const weapon = new Map(
        fault.decisions
            .filter((row) => "weapon" in row.chosen)
            .map((row) => [row.decision, row.chosen.weapon]),
    );
    for (const event of fault.events) {
        assert.deepEqual(
            [event.before.fire, event.after.fire, weapon.get(event.decision)],
            [true, false, "fire"],
        );
    }
});

test("a wrong legality projection executes another admissible motion than the one selected", async () => {
    const fault = await teacherRound("wrong_legal_projection");
    const motion = new Map(
        fault.decisions
            .filter((row) => "motion" in row.chosen)
            .map((row) => [row.decision, row]),
    );
    assert.ok(fault.events.length > 0);
    for (const event of fault.events) {
        const row = motion.get(event.decision);
        assert.equal(row.chosen.motion, event.before.motion);
        assert.notEqual(event.after.motion, event.before.motion);
    }
    // The game's own revalidation can still restore the selection, so most, not all, projections execute.
    assert.ok(
        fault.events.some(
            (event) =>
                motion.get(event.decision).applied.motion_id ===
                event.after.motion,
        ),
    );
});

test("the match command line takes --controller-fault-file", async () => {
    const dir = await mkdtemp(join(tmpdir(), "tank-controller-fault-cli-"));
    await writeFile(join(dir, "fault.json"), faultText("swap_selected_turn"));
    const run = spawnSync(
        process.execPath,
        [
            "scripts/match.mjs",
            "--seed=2002",
            "--seat=red",
            "--teacher",
            "--seconds=2",
            `--out=${join(dir, "out")}`,
            `--controller-fault-file=${join(dir, "fault.json")}`,
        ],
        { cwd: ROOT, encoding: "utf8" },
    );
    assert.equal(run.status, 0, run.stderr);
    const events = await lines(join(dir, "out/private_fault_events.jsonl"));
    assert.ok(events.length > 0);
    for (const { before, after } of events)
        assert.deepEqual([before.turn, after.turn].sort(), ["left", "right"]);
});
