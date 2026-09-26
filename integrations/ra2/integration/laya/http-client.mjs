// A decision client that talks to a running Honing decision server
// (honing/backends/jev/server.py) instead of starting its own worker.
// Same surface as LayaClient: ready, metadata, predict, close.
export class HttpDecisionClient {
    constructor({
        url,
        startupMs = 180000,
        requestMs = 60000,
        fetchImpl = fetch,
    } = {}) {
        if (typeof url !== "string" || !/^https?:\/\//.test(url))
            throw new Error(
                "A decision server URL must start with http:// or https://",
            );
        this.url = url.replace(/\/$/, "");
        this.requestMs = requestMs;
        this.fetch = fetchImpl;
        this.seq = 0;
        this.closed = false;
        this.ready = this.waitReady(startupMs).then((metadata) => {
            this.metadata = { ...metadata, decisionServer: this.url };
            return this.metadata;
        });
    }
    async waitReady(startupMs) {
        const deadline = Date.now() + startupMs;
        let lastError = null;
        while (Date.now() < deadline) {
            try {
                const response = await this.fetch(this.url + "/v1/health", {
                    signal: AbortSignal.timeout(5000),
                });
                const body = await response.json();
                if (response.ok && body.ready) return body.metadata;
                lastError = new Error(
                    body.error ?? `health returned ${response.status}`,
                );
            } catch (e) {
                lastError = e;
            }
            await new Promise((r) => setTimeout(r, 1000));
        }
        throw new Error(
            `Decision server not ready: ${lastError?.message ?? "timeout"}`,
        );
    }
    async predict(packet) {
        await this.ready;
        if (this.closed) throw new Error("Decision client closed");
        const id = ++this.seq;
        const response = await this.fetch(this.url + "/v1/decide", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                id,
                parts: packet.parts,
                questions: packet.questions,
            }),
            signal: AbortSignal.timeout(this.requestMs),
        });
        const body = await response.json();
        if (!response.ok)
            throw new Error(body.error ?? `decide returned ${response.status}`);
        if (body.id !== id)
            throw new Error("Decision server answered a different request");
        return body;
    }
    close() {
        this.closed = true;
    }
}
