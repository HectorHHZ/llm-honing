import test from "node:test";
import assert from "node:assert/strict";
import { LayaClient } from "../integration/laya/client.mjs";
test("worker startup failure rejects inference without a fallback policy", async () => {
    const client = new LayaClient({ python: "/nonexistent/ra2-python" });
    await assert.rejects(client.ready, /ENOENT/);
    await assert.rejects(client.predict({}), /ENOENT/);
    assert.equal(client.closed, true);
});
