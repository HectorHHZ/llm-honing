import { createWriteStream } from "node:fs";
import { createHash } from "node:crypto";
import { once } from "node:events";
import { PassThrough } from "node:stream";
import { pipeline } from "node:stream/promises";
import { createGzip } from "node:zlib";

// Backpressure bounds memory; successful close is required before a match result
// can be published. Readers already support losslessly compressed JSONL.
export class CompressedDecisionLog {
    constructor(path) {
        this.input = new PassThrough();
        this.hash = createHash("sha256");
        this.records = 0;
        this.bytes = 0;
        this.failure = null;
        this.closed = false;
        this.completion = pipeline(
            this.input,
            createGzip({ level: 3 }),
            createWriteStream(path, { flags: "wx" }),
        ).catch((error) => {
            // Retain failure immediately instead of an unhandled rejection.
            // append/close still reject it to the match runner.
            this.failure = error;
        });
    }
    async append(value) {
        if (this.closed) throw new Error("Decision log is closed");
        if (this.failure) throw this.failure;
        const line = JSON.stringify(value) + "\n";
        if (!this.input.write(line)) await once(this.input, "drain");
        if (this.failure) throw this.failure;
        this.hash.update(line);
        this.records++;
        this.bytes += Buffer.byteLength(line);
    }
    async close() {
        if (!this.closed) {
            this.closed = true;
            this.input.end();
        }
        await this.completion;
        if (this.failure) throw this.failure;
        this.summary ??= {
            format: "gzip-jsonl",
            records: this.records,
            uncompressedBytes: this.bytes,
            contentSha256: this.hash.digest("hex"),
        };
        return this.summary;
    }
}
