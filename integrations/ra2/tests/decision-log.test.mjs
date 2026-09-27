import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import { CompressedDecisionLog } from "../integration/laya/decision-log.mjs";

test("compressed decision log preserves every ordered row and decoded checksum", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ra2-log-"));
    try {
        const path = join(dir, "decisions.jsonl.gz");
        const log = new CompressedDecisionLog(path);
        const rows = Array.from({ length: 1000 }, (_, tick) => ({
            tick,
            text: "决策和参数".repeat(700),
            action: { function: 22, unit_tags: [tick + 1] },
        }));
        for (const row of rows) await log.append(row);
        const result = await log.close();
        assert.deepEqual(await log.close(), result);
        const decoded = gunzipSync(await readFile(path));
        assert.equal(
            decoded.toString(),
            rows.map((row) => JSON.stringify(row) + "\n").join(""),
        );
        assert.equal(result.records, rows.length);
        assert.equal(result.uncompressedBytes, decoded.length);
        assert.equal(
            result.contentSha256,
            createHash("sha256").update(decoded).digest("hex"),
        );
        await assert.rejects(log.append({ tick: 1001 }), /closed/);
    } finally {
        await rm(dir, { recursive: true, force: true });
    }
});

test("write errors reject completion and never overwrite an existing artifact", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ra2-log-"));
    try {
        const path = join(dir, "existing");
        await writeFile(path, "retained");
        const log = new CompressedDecisionLog(path);
        await assert.rejects(log.close(), { code: "EEXIST" });
        assert.equal(await readFile(path, "utf8"), "retained");
        const missing = new CompressedDecisionLog(
            join(dir, "missing", "log.gz"),
        );
        await assert.rejects(missing.close(), { code: "ENOENT" });
    } finally {
        await rm(dir, { recursive: true, force: true });
    }
});
