import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { summarize } from "../report.mjs";

test("report separates configurations, unresolved/invalid runs, and unpaired seeds", () => {
    const root = mkdtempSync(join(tmpdir(), "harness-report-"));
    const add = (name, data) => {
        mkdirSync(join(root, name));
        if (data)
            writeFileSync(
                join(root, name, "result.json"),
                JSON.stringify(data),
            );
    };
    const result = {
        valid: true,
        comparisonId: "a",
        candidateSeat: "red",
        seed: 1,
        ended: true,
        truncated: false,
        outcome: { red: 1, blue: -1 },
        tick: 123,
    };
    try {
        add("red-win", result);
        add("blue-loss", { ...result, candidateSeat: "blue" });
        add("cutoff", {
            ...result,
            seed: 2,
            ended: false,
            truncated: true,
            outcome: { red: 0, blue: 0 },
        });
        add("other-profile", { ...result, comparisonId: "b" });
        add("error", { ...result, valid: false, error: "test error" });
        add("interrupted", null);
        const report = summarize(root),
            a = report.groups.find((g) => g.comparisonId === "a");
        assert.equal(a.wins, 1);
        assert.equal(a.losses, 1);
        assert.equal(a.unresolved, 1);
        assert.equal(a.winsOverAllValidRuns, "1/3");
        assert.deepEqual(a.pairedSeeds, [1]);
        assert.deepEqual(a.unpairedOrRepeatedSeeds, [
            { seed: 2, redRuns: 1, blueRuns: 0 },
        ]);
        assert.equal(report.groups.length, 2);
        assert.equal(report.invalid.length, 1);
        assert.equal(report.unfinished.length, 1);
        add("repeat-red", result);
        assert.deepEqual(
            summarize(root).groups.find((g) => g.comparisonId === "a")
                .pairedSeeds,
            [],
        );
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});
