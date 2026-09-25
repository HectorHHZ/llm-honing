import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import zlib from "node:zlib";
import { openLog } from "../integration/laya/jsonl.mjs";

async function decoded(path) {
    const blocks = [];
    for await (const block of openLog(path)) blocks.push(block);
    return Buffer.concat(blocks);
}

test(
    "archive reader preserves ordered bytes and rejects corrupt zstd",
    { skip: typeof zlib.zstdCompressSync !== "function" },
    async () => {
        const directory = await mkdtemp(join(tmpdir(), "ra2-archive-"));
        try {
            const path = join(directory, "decisions.jsonl");
            const data = Buffer.from('{"text":"正常视角"}\n'.repeat(5000));
            await writeFile(path + ".zst", zlib.zstdCompressSync(data));
            assert.deepEqual(await decoded(path), data);
            await writeFile(path + ".zst", "corrupt");
            await assert.rejects(decoded(path));
            // Restored gzip takes precedence, so unchanged frozen readers also work.
            await writeFile(path + ".gz", zlib.gzipSync(data));
            assert.deepEqual(await decoded(path), data);
        } finally {
            await rm(directory, { recursive: true, force: true });
        }
    },
);
