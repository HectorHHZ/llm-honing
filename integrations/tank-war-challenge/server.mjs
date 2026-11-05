import { RuntimeMetrics } from './runtime-metrics.mjs';
import http from "node:http";
import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MatchScheduler } from "./match-scheduler.mjs";
import { Leaderboard } from "./leaderboard.mjs";
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
async function readJson(req, limit = 65536) {
    if (!req.headers["content-type"]?.startsWith("application/json"))
        throw new HttpError(415, "Expected application/json");
    let size = 0;
    const chunks = [];
    for await (const chunk of req) {
        size += chunk.length;
        if (size > limit) throw new HttpError(413, "Request too large");
        chunks.push(chunk);
    }
    try {
        return JSON.parse(Buffer.concat(chunks).toString("utf8"));
    } catch {
        throw new HttpError(400, "Invalid JSON");
    }
}
export function configuration(env = process.env) {
    const editions = { classic: [], "auto-aim": [] };
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
        "classic",
        "model",
        env.CLASSIC_MODEL_URL,
        env.CLASSIC_MODEL_LABEL || "Classic model",
        "tank",
    );
    const nanojevUrls = (env.CLASSIC_NANOJEV_URLS || env.CLASSIC_NANOJEV_URL || "")
        .split(",").map(s => s.trim()).filter(Boolean);
    if (nanojevUrls.length > 8 || new Set(nanojevUrls).size !== nanojevUrls.length)
        throw Error("Expected at most eight distinct NanoJev workers");
    for (const url of nanojevUrls) {
        const parsed = new URL(url);
        if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password || parsed.search || parsed.hash)
            throw Error("Invalid NanoJev worker URL");
    }
    add("classic", "nanojev", nanojevUrls[0], "NanoJev", "tank");
    add(
        "auto-aim",
        "model",
        env.AUTOAIM_MODEL_URL,
        env.AUTOAIM_MODEL_LABEL || "Auto Aim model",
        "tank",
    );
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
                !Array.isArray(m.editions) ||
                !m.editions.length
            )
                throw Error("Each model needs a URL and editions");
            for (const edition of m.editions)
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
    const defaultEdition = env.DEFAULT_EDITION || "";
    if (defaultEdition && !Object.hasOwn(editions, defaultEdition))
        throw Error("Unknown DEFAULT_EDITION");
    const publicChallenge = env.PUBLIC_CHALLENGE === "1";
    const integer = (name, fallback, min, max) => {
        const value = env[name] === undefined ? fallback : Number(env[name]);
        if (!Number.isInteger(value) || value < min || value > max)
            throw Error(`Invalid ${name}: expected ${min}–${max}`);
        return value;
    };
    // Tree inference prepares requests on CPU while the GPU scores the previous
    // request. Bound those overlapping requests; keep legacy services serial.
    const requestsPerModel = integer("PUBLIC_REQUESTS_PER_MODEL", 1, 1, 8);
    const workerCount = publicChallenge
        ? Math.max(1, nanojevUrls.length) * requestsPerModel
        : 1;
    if (workerCount > 8) throw Error("At most eight model requests may run at once");
    if (publicChallenge) {
        const opponent = editions.classic.find(
            (model) => model.id === "nanojev",
        );
        if (!opponent)
            throw Error("Public play requires a configured NanoJev endpoint");
        editions.classic = [opponent];
        editions["auto-aim"] = [];
    }
    return {
        editions,
        defaultEdition,
        publicChallenge,
        nanojevUrls,
        admission: {
            capacity: integer("PUBLIC_MAX_MATCHES", 2, 1, 32),
            workers: workerCount,
            maxWaiting: integer("PUBLIC_MAX_WAITING", 50, 1, 500),
            leaseMs: integer("PUBLIC_MATCH_LEASE_MS", 15000, 3000, 60000),
            queueMs: integer("PUBLIC_DECISION_QUEUE_MS", 900, 100, 1200),
        },
        publicOrigin: env.PUBLIC_ORIGIN || "",
    };
}
export function createGameServer(config = configuration()) {
    const busy = new Set();
    const matches = config.publicChallenge
        ? new MatchScheduler(config.admission)
        : null;
    const leaderboard = matches
        ? new Leaderboard(path.join(path.dirname(fileURLToPath(import.meta.url)), "data", "leaderboard.json"))
        : null;
    if (matches) {
        matches.savedName = (owner) => leaderboard.nameFor(owner);
        matches.onRename = (s) => leaderboard.rename(s.owner, s.nickname);
        matches.onResult = (s, result) => leaderboard.record(s.owner, s.nickname, result);
    }
    const runtime = new RuntimeMetrics();
    const server = http.createServer(async (req, res) => {
        if (req.method === 'POST' && req.url?.split('?')[0] === '/api/classic/decision') {
            const started = performance.now(); runtime.inflight++;
            let done = false;
            const record = () => {
                if (done) return; done = true; runtime.inflight--;
                runtime.record({ok:res.writableFinished && res.statusCode === 200, status:res.writableFinished ? res.statusCode : 499, ms:performance.now()-started});
            };
            res.once('finish',record); res.once('close',record);
        }
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
                if (
                    config.publicChallenge &&
                    ![
                        "/api/config",
                        "/api/classic/health",
                        "/api/classic/decision",
                        "/api/play/join",
                        "/api/play/heartbeat",
                        "/api/play/leave",
                        "/api/play/begin",
                        "/api/play/finish",
                        "/api/play/rename",
                        "/api/leaderboard",
                    ].includes(url.pathname)
                )
                    throw new HttpError(404, "Not found");
                const origin = req.headers.origin;
                if (
                    (origin &&
                        origin !== `http://${req.headers.host}` &&
                        origin !== config.publicOrigin) ||
                    req.headers["sec-fetch-site"] === "cross-site"
                )
                    throw new HttpError(403, "Invalid origin");
                if (url.pathname === "/api/leaderboard") {
                    if (req.method !== "GET") throw new HttpError(405, "Method not allowed");
                    return send(200, leaderboard ? leaderboard.top() : { fair: [], modified: [] });
                }
                let owner = /(?:^|;\s*)tank_user=([0-9a-f-]{36})(?:;|$)/.exec(req.headers.cookie || "")?.[1];
                if (matches && url.pathname === "/api/play/join" && !owner) {
                    owner = randomUUID();
                    res.setHeader("Set-Cookie", `tank_user=${owner}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000${config.publicOrigin.startsWith("https:") ? "; Secure" : ""}`);
                }
                if (matches && url.pathname.startsWith("/api/play/")) {
                    if (req.method !== "POST")
                        throw new HttpError(405, "Method not allowed");
                    const body = await readJson(req, 1024);
                    const token = body?.token;
                    const action = url.pathname.split("/").at(-1);
                    if (!owner) throw new HttpError(401, "Start a match in this browser first");
                    matches.authorizeOwner(token, owner);
                    const result = action === "join" ? matches.join(token, owner)
                        : action === "begin" ? matches.beginRound(token, body.round_id)
                        : action === "finish" ? matches.finishRound(token, body.round_ticket, body.result)
                        : action === "rename" ? matches.rename(token, body.name)
                        : matches[action](token);
                    return send(200, result);
                }
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
                const match =
                    /^\/api\/(classic|auto-aim)\/(health|decision)$/.exec(
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
                    const body = await readJson(req);
                    validateRequest(body, edition);
                    if (
                        config.publicChallenge &&
                        (body.player !== 1 || body.protocol !== "plans-v1")
                    )
                        throw new HttpError(
                            400,
                            "This game accepts opponent decisions only",
                        );
                    const infer = (signal, worker = 0) => {
                        const target = config.publicChallenge && config.nanojevUrls?.length
                            ? {...backend, url: config.nanojevUrls[worker % config.nanojevUrls.length].replace(/\/$/, "")}
                            : backend;
                        return target.transport === "tank"
                            ? upstream(target, "/decision", body, signal)
                            : decideWithHoning(target, edition, body, signal);
                    };
                    if (matches) {
                        matches.validateToken(req.headers["x-match-token"]);
                        if (!owner) throw new HttpError(401, "Start a match in this browser first");
                        matches.authorizeOwner(req.headers["x-match-token"], owner);
                        matches.requireRound(req.headers["x-match-token"], body.round_id, body.round_ticket);
                        const response = await matches.enqueue(
                            req.headers["x-match-token"],
                            infer,
                            controller.signal,
                        );
                        return send(200, response);
                    }
                    lock = `${edition}:${backend.id}:${body.player}`;
                    if (busy.has(lock)) {
                        lock = null;
                        throw new HttpError(
                            429,
                            "A decision for this seat is already in flight",
                        );
                    }
                    busy.add(lock);
                    const response = await infer(controller.signal);
                    return send(200, response);
                } finally {
                    clearTimeout(timer);
                    res.off("close", disconnected);
                    if (lock) busy.delete(lock);
                }
            }
            if (req.method !== "GET")
                throw new HttpError(405, "Method not allowed");
            if (config.publicChallenge) {
                if (
                    [
                        "/",
                        "/index.html",
                        "/classic",
                        "/classic/",
                        "/classic/index.html",
                        "/auto-aim",
                        "/auto-aim/",
                        "/auto-aim/index.html",
                        "/play",
                        "/coop",
                        "/coop/",
                        "/coop/index.html",
                    ].includes(url.pathname)
                ) {
                    res.writeHead(302, {
                        Location: "/play/",
                        "Cache-Control": "no-store",
                    });
                    return res.end();
                }
                if (
                    !["/play/", "/play/index.html"].includes(url.pathname) &&
                    !/^\/(play|classic|shared)\/[^/]+\.(js|css)$/.test(
                        url.pathname,
                    )
                )
                    throw new HttpError(404, "Not found");
            }
            if (url.pathname === "/" && config.defaultEdition) {
                res.writeHead(302, { Location: `/${config.defaultEdition}/` });
                return res.end();
            }
            if (["/classic", "/auto-aim"].includes(url.pathname)) {
                res.writeHead(302, { Location: url.pathname + "/" });
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
                    ...(error.code ? { code: error.code } : {}),
                    error:
                        error instanceof HttpError
                            ? error.message
                            : "Internal server error",
                });
        }
    });
    // cloudflared keeps idle origin connections for 90s. Keep the origin alive
    // longer so the proxy retires idle sockets first, avoiding reuse/close races.
    server.keepAliveTimeout = 125000;
    server.keepAliveTimeoutBuffer = 5000;
    server.headersTimeout = 135000;
    server.runtimeStats = () => ({...runtime.snapshot(), scheduler:matches ? {
        capacity:matches.capacity, active:matches.activeCount, waiting:matches.waiting.length,
        pending:matches.pending.size, running:matches.runningJobs.size, ...matches.metrics,
    } : null});
    server.on("close", () => matches?.close());
    return server;
}
export function startServer(env = process.env) {
    const server = createGameServer(configuration(env)),
        port = Number(env.PORT || 4177),
        host = env.HOST || "127.0.0.1";
    server.listen(port, host, () =>
        console.log(`RICOCHET: http://${host}:${server.address().port}`),
    );
    if (env.TANK_METRICS_PORT) {
        const metricsPort=Number(env.TANK_METRICS_PORT);
        if (!Number.isInteger(metricsPort) || metricsPort < 1024 || metricsPort > 65535) throw Error('Invalid metrics port');
        const metrics=http.createServer((req,res)=>{
            res.setHeader('Content-Type','application/json');
            res.setHeader('Cache-Control','no-store');
            if(req.method!=='GET' || req.url!=='/metrics'){res.writeHead(404);return res.end('{}');}
            res.end(JSON.stringify(server.runtimeStats()));
        });
        metrics.on('error',e=>console.error('Metrics listener:',e.code));
        metrics.listen(metricsPort,'127.0.0.1');
        server.on('close',()=>metrics.close());
    }
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
