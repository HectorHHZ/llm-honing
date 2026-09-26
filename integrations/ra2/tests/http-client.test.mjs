import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { HttpDecisionClient } from "../integration/laya/http-client.mjs";

function fakeServer(handler) {
    return new Promise((resolve) => {
        const server = createServer(async (req, res) => {
            let body = "";
            for await (const chunk of req) body += chunk;
            const [status, payload] = handler(
                req,
                body ? JSON.parse(body) : null,
            );
            res.writeHead(status, { "Content-Type": "application/json" });
            res.end(JSON.stringify(payload));
        });
        server.listen(0, "127.0.0.1", () =>
            resolve({
                server,
                url: `http://127.0.0.1:${server.address().port}`,
            }),
        );
    });
}

const metadata = { revision: "r1", dtype: "torch.bfloat16" };

test("waits for health, then sends parts and questions with an id", async () => {
    const seen = [];
    const { server, url } = await fakeServer((req, body) => {
        if (req.url === "/v1/health") return [200, { ready: true, metadata }];
        seen.push(body);
        return [200, { id: body.id, answers: { root: { choice: "power" } } }];
    });
    const client = new HttpDecisionClient({ url });
    assert.deepEqual(await client.ready, { ...metadata, decisionServer: url });
    const answer = await client.predict({
        parts: { essential: ["s"] },
        questions: { root: {} },
        extra: 1,
    });
    assert.equal(answer.answers.root.choice, "power");
    assert.deepEqual(seen, [
        { id: 1, parts: { essential: ["s"] }, questions: { root: {} } },
    ]);
    server.close();
});

test("a server error is raised, not swallowed", async () => {
    const { server, url } = await fakeServer((req) =>
        req.url === "/v1/health"
            ? [200, { ready: true, metadata }]
            : [500, { error: "RuntimeError: boom" }],
    );
    const client = new HttpDecisionClient({ url });
    await assert.rejects(
        client.predict({ parts: {}, questions: { q: {} } }),
        /boom/,
    );
    server.close();
});

test("an answer to a different request id is refused", async () => {
    const { server, url } = await fakeServer((req) =>
        req.url === "/v1/health"
            ? [200, { ready: true, metadata }]
            : [200, { id: 99, answers: {} }],
    );
    const client = new HttpDecisionClient({ url });
    await assert.rejects(
        client.predict({ parts: {}, questions: { q: {} } }),
        /different request/,
    );
    server.close();
});

test("a server that never becomes ready fails the startup", async () => {
    const { server, url } = await fakeServer(() => [503, { error: "loading" }]);
    const client = new HttpDecisionClient({ url, startupMs: 1500 });
    await assert.rejects(client.ready, /not ready: loading/);
    server.close();
});

test("a URL without a scheme is refused", () => {
    assert.throws(
        () => new HttpDecisionClient({ url: "localhost:8000" }),
        /must start with http/,
    );
});
