import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
    DEFAULT_QUESTIONS_PATH,
    loadQuestions,
    t,
} from "../integration/laya/text.mjs";

const defaults = JSON.parse(readFileSync(DEFAULT_QUESTIONS_PATH, "utf8"));

test("placeholders are filled from the values given", () => {
    assert.equal(
        t("staged.frameCompact", { stage: 2, tick: 90 }),
        "Decision stage 2, tick 90. Simulation paused; this decision's actions have not executed yet.",
    );
});

test("a missing placeholder value is refused", () => {
    assert.throws(
        () => t("staged.frameCompact", { stage: 2 }),
        /needs a value for \{tick\}/,
    );
});

test("an unknown key is refused", () => {
    assert.throws(() => t("staged.nothing"), /Unknown or non-string/);
});

test("a replacement file changes the text the controller shows", () => {
    const dir = mkdtempSync(join(tmpdir(), "ra2-questions-"));
    const path = join(dir, "questions.json");
    writeFileSync(
        path,
        JSON.stringify({
            ...defaults,
            "intent.goal.firstPower": "Power first.",
        }),
    );
    loadQuestions(path);
    try {
        assert.equal(t("intent.goal.firstPower"), "Power first.");
    } finally {
        loadQuestions(DEFAULT_QUESTIONS_PATH);
    }
});

test("a replacement file missing a key is refused before use", () => {
    const dir = mkdtempSync(join(tmpdir(), "ra2-questions-"));
    const path = join(dir, "questions.json");
    const partial = { ...defaults };
    delete partial["staged.commitments"];
    writeFileSync(path, JSON.stringify(partial));
    assert.throws(
        () => loadQuestions(path),
        /missing keys: staged.commitments/,
    );
    assert.equal(t("staged.commitments"), defaults["staged.commitments"]);
});
