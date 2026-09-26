import { createReadStream, existsSync } from "node:fs";
import zlib from "node:zlib";

// Compression is storage only: consumers receive exactly the original bytes.
export function openLog(path) {
    if (existsSync(path)) return createReadStream(path);
    const gzip = existsSync(path + ".gz");
    if (!gzip && typeof zlib.createZstdDecompress !== "function")
        throw new Error(
            "Zstd archives require Node >=22.15; restore gzip for older readers",
        );
    const decoder = gzip ? zlib.createGunzip() : zlib.createZstdDecompress();
    const source = createReadStream(path + (gzip ? ".gz" : ".zst"));
    source.on("error", (error) => decoder.destroy(error));
    decoder.on("close", () => source.destroy());
    return source.pipe(decoder);
}
