import { createReadStream, existsSync } from "node:fs";
import { createGunzip } from "node:zlib";

// Compression is storage only: consumers receive exactly the original bytes.
export function openLog(path) {
    if (existsSync(path)) return createReadStream(path);
    const source = createReadStream(path + ".gz"),
        decoder = createGunzip();
    source.on("error", (error) => decoder.destroy(error));
    return source.pipe(decoder);
}
