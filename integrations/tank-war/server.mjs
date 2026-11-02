import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
    HttpError,
    healthFor,
    decideWithHoning,
    validateRequest,
    upstream,
} from "./model-bridge.mjs";
const root = fileURLToPath(new URL("./web/", import.meta.url));
const types = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
};
export function configuration(env = process.env) {
    const editions = { "tank-war": [] };
    const add = (edition, id, url, label, transport, token, version) => {
        if (!url) return;
        if (
            !Object.hasOwn(editions, edition) ||
            !/^[a-zA-Z0-9_-]{1,64}$/.test(id) ||
            editions[edition].some((b) => b.id === id) ||
            typeof label !== "string" ||
            !label.trim() ||
            !["tank", "honing"].includes(transport) ||
            (version !== undefined &&
                (typeof version !== "string" || version.length > 200))
        )
            throw Error("Invalid or duplicate model registration");
        const parsed = new URL(url);
        if (
            !["http:", "https:"].includes(parsed.protocol) ||
            parsed.username ||
            parsed.password ||
            parsed.search ||
            parsed.hash
        )
            throw Error(
                "Model URL must be an HTTP(S) base URL without credentials, query or fragment",
            );
        editions[edition].push({
            id,
            url: url.replace(/\/$/, ""),
            label,
            transport,
            token,
            ...(version ? { version } : {}),
        });
    };
    for (const edition of Object.keys(editions))
        add(
            edition,
            "honing",
            env.HONING_URL,
            env.MODEL_LABEL || "Honing model",
            "honing",
            env.HONING_TOKEN,
        );
    add(
        "tank-war",
        "model",
        env.TANK_MODEL_URL,
        env.TANK_MODEL_LABEL || "Tank War model",
        "tank",
    );
    add("tank-war", "nanojev", env.NANOJEV_URL, "NanoJev", "tank");
    if (env.MODELS_JSON) {
        const models = JSON.parse(env.MODELS_JSON);
        if (!Array.isArray(models) || models.length > 32)
            throw Error(
                "MODELS_JSON must be an array of at most 32 model endpoints",
            );
        for (const m of models) {
            if (
                !m ||
                typeof m.url !== "string" ||
                !m.url ||
                (m.editions !== undefined &&
                    (!Array.isArray(m.editions) || !m.editions.length))
            )
                throw Error("Each model needs a URL");
            for (const edition of m.editions ?? ["tank-war"])
                add(
                    edition,
                    m.id,
                    m.url,
                    m.label,
                    m.transport || "honing",
                    m.token,
                    m.version,
                );
        }
    }
    return { editions, publicOrigin: env.PUBLIC_ORIGIN || "" };
}
export function createGameServer(config = configuration()) {
    const busy = new Set();
    return http.createServer(async (req, res) => {
        const send = (
            status,
            data,
            type = "application/json; charset=utf-8",
        ) => {
            res.writeHead(status, {
                "Content-Type": type,
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "same-origin",
            });
            res.end(
                type.startsWith("application/json") && !Buffer.isBuffer(data)
                    ? JSON.stringify(data)
                    : data,
            );
        };
        try {
            const url = new URL(req.url, "http://localhost");
            if (url.pathname.startsWith("/api/")) {
                const origin = req.headers.origin;
                if (
                    (origin &&
                        origin !== `http://${req.headers.host}` &&
                        origin !== config.publicOrigin) ||
                    req.headers["sec-fetch-site"] === "cross-site"
                )
                    throw new HttpError(403, "Invalid origin");
                if (url.pathname === "/api/config") {
                    if (req.method !== "GET")
                        throw new HttpError(405, "Method not allowed");
                    return send(200, {
                        editions: Object.fromEntries(
                            Object.entries(config.editions).map(
                                ([key, values]) => [
                                    key,
                                    values.map(({ id, label, version }) => ({
                                        id,
                                        label,
                                        ...(version ? { version } : {}),
                                    })),
                                ],
                            ),
                        ),
                    });
                }
                const match = /^\/api\/(tank-war)\/(health|decision)$/.exec(
                    url.pathname,
                );
                if (!match) throw new HttpError(404, "Not found");
                const [, edition, action] = match,
                    backend = config.editions[edition].find(
                        (b) =>
                            b.id ===
                            (url.searchParams.get("backend") ||
                                config.editions[edition][0]?.id),
                    );
                if (req.method !== (action === "health" ? "GET" : "POST"))
                    throw new HttpError(405, "Method not allowed");
                if (!backend)
                    throw new HttpError(503, "No matching model configured");
                const controller = new AbortController(),
                    timer = setTimeout(() => controller.abort(), 10000);
                timer.unref();
                const disconnected = () => {
                    if (!res.writableEnded) controller.abort();
                };
                res.once("close", disconnected);
                let lock;
                try {
                    if (action === "health")
                        return send(
                            200,
                            await healthFor(
                                backend,
                                edition,
                                controller.signal,
                            ),
                        );
                    if (
                        !req.headers["content-type"]?.startsWith(
                            "application/json",
                        )
                    )
                        throw new HttpError(415, "Expected application/json");
                    let size = 0;
                    const chunks = [];
                    for await (const chunk of req) {
                        size += chunk.length;
                        if (size > 65536)
                            throw new HttpError(413, "Request too large");
                        chunks.push(chunk);
                    }
                    let body;
                    try {
                        body = JSON.parse(
                            Buffer.concat(chunks).toString("utf8"),
                        );
                    } catch {
                        throw new HttpError(400, "Invalid JSON");
                    }
                    validateRequest(body, edition);
                    lock = `${edition}:${backend.id}:${body.player}`;
                    if (busy.has(lock)) {
                        lock = null;
                        throw new HttpError(
                            429,
                            "A decision for this seat is already in flight",
                        );
                    }
                    busy.add(lock);
                    const response =
                        backend.transport === "tank"
                            ? await upstream(
                                  backend,
                                  "/decision",
                                  body,
                                  controller.signal,
                              )
                            : await decideWithHoning(
                                  backend,
                                  edition,
                                  body,
                                  controller.signal,
                              );
                    return send(200, response);
                } finally {
                    clearTimeout(timer);
                    res.off("close", disconnected);
                    if (lock) busy.delete(lock);
                }
            }
            if (req.method !== "GET")
                throw new HttpError(405, "Method not allowed");
            if (["/", "/game"].includes(url.pathname)) {
                res.writeHead(302, { Location: "/game/" });
                return res.end();
            }
            let relative;
            try {
                relative = decodeURIComponent(url.pathname).slice(1);
            } catch {
                throw new HttpError(400, "Invalid path");
            }
            if (relative.includes("\0"))
                throw new HttpError(400, "Invalid path");
            if (!relative || relative.endsWith("/")) relative += "index.html";
            const file = path.resolve(root, relative);
            if (!file.startsWith(root)) throw new HttpError(403, "Forbidden");
            if (!types[path.extname(file)])
                throw new HttpError(404, "Not found");
            let content;
            try {
                content = await readFile(file);
            } catch {
                throw new HttpError(404, "Not found");
            }
            send(200, content, types[path.extname(file)]);
        } catch (error) {
            if (!res.headersSent && !res.destroyed)
                send(error.status || 500, {
                    error:
                        error instanceof HttpError
                            ? error.message
                            : "Internal server error",
                });
        }
    });
}
export function startServer(env = process.env) {
    const server = createGameServer(configuration(env)),
        port = Number(env.PORT || 4177),
        host = env.HOST || "127.0.0.1";
    server.listen(port, host, () =>
        console.log(`Tank War: http://${host}:${server.address().port}`),
    );
    return server;
}
if (
    process.argv[1] &&
    path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
    const server = startServer();
    for (const signal of ["SIGINT", "SIGTERM"])
        process.once(signal, () => {
            server.close(() => process.exit(0));
            setTimeout(() => process.exit(0), 2000).unref();
        });
}
