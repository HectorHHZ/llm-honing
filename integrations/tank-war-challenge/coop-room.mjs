import { Worker } from "node:worker_threads";
import { randomUUID } from "node:crypto";
import { HttpError, upstream } from "./model-bridge.mjs";

// One authoritative room. Each human holds a normal admission place, so solo
// plus co-op players remain within the same site capacity and FIFO queue.
export class CoopRoom {
    constructor(matches, config) {
        this.matches = matches;
        this.config = config;
        this.peers = new Map();
        this.members = [];
        this.worker = null;
        this.state = null;
        this.countdown = 0;
        this.serial = 0;
        this.roundTickets = new Map();
        this.busy = false;
        this.fault = null;
        this.timer = setInterval(() => this.tick(), 100);
        this.timer.unref();
    }
    hasOwner(owner) {
        return [...this.peers.values()].some((p) => p.owner === owner);
    }
    join(owner) {
        const existing = [...this.peers.values()].find(
            (p) => p.owner === owner,
        );
        if (existing) {
            this.matches.heartbeat(existing.token);
            return this.status(existing);
        }
        const token = randomUUID(),
            admission = this.matches.join(token, owner);
        const peer = {
            token,
            owner,
            nickname: admission.nickname,
            seq: -1,
            stream: null,
            ticket: null,
            id: null,
        };
        this.peers.set(token, peer);
        return this.status(peer);
    }
    authorize(token, owner) {
        const p = this.peers.get(token);
        if (!p || p.owner !== owner)
            throw new HttpError(
                403,
                "This player belongs to another browser or has left",
            );
        this.matches.authorizeOwner(token, owner);
        return p;
    }
    status(p) {
        const session = this.matches.sessions.get(p.token);
        const queued = this.matches.waiting.indexOf(p.token);
        return {
            token: p.token,
            nickname: p.nickname,
            id: p.id,
            state: !session
                ? "expired"
                : session.state === "queued"
                  ? "queued"
                  : this.members.includes(p)
                    ? "playing"
                    : "waiting",
            position: queued < 0 ? 0 : queued + 1,
            capacity: 8,
            round: this.state?.round ?? 0,
            countdown: this.countdown
                ? Math.max(0, Math.ceil((this.countdown - Date.now()) / 1000))
                : 0,
            players: this.members.map((m) => ({
                nickname: m.nickname,
                id: m.id,
                is_you: m === p,
            })),
            waiting: [...this.peers.values()]
                .filter((m) => !this.members.includes(m))
                .map((m) => ({ nickname: m.nickname, is_you: m === p })),
            fault: this.fault,
            game: this.state,
        };
    }
    input(token, owner, body) {
        const p = this.authorize(token, owner);
        const c = body.controls;
        if (
            !Number.isSafeInteger(body.seq) ||
            body.seq < 0 ||
            !c ||
            ![-1, 0, 1].includes(c.drive) ||
            ![-1, 0, 1].includes(c.turn) ||
            typeof c.fire !== "boolean"
        )
            throw new HttpError(400, "Invalid controls");
        this.matches.heartbeat(token);
        if (body.seq > p.seq) {
            p.seq = body.seq;
            if (this.members.includes(p) && body.round === this.state?.round)
                this.worker?.postMessage({
                    type: "input",
                    id: p.id,
                    controls: { drive: c.drive, turn: c.turn, fire: c.fire },
                });
        }
        return { ok: true };
    }
    heartbeat(token, owner) {
        const p = this.authorize(token, owner);
        this.matches.heartbeat(token);
        return { ok: true };
    }
    stream(owner, res) {
        const p = [...this.peers.values()].find((p) => p.owner === owner);
        if (!p) throw new HttpError(401, "Join the team first");
        p.stream?.end();
        p.stream = res;
        res.writeHead(200, {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            Connection: "keep-alive",
        });
        res.write("retry: 1000\n\n");
        res.on("close", () => {
            if (p.stream === res) p.stream = null;
        });
        this.send(p);
    }
    send(p) {
        if (p.stream && !p.stream.destroyed) {
            if (p.stream.writableLength > 262144) {
                p.stream.destroy();
                p.stream = null;
                return;
            }
            p.stream.write(`data: ${JSON.stringify(this.status(p))}\n\n`);
        }
    }
    leave(token, owner) {
        const p = this.authorize(token, owner);
        this.remove(p);
        return { ok: true };
    }
    remove(p) {
        p.stream?.end();
        this.peers.delete(p.token);
        this.members = this.members.filter((m) => m !== p);
        this.roundTickets.delete(p.token);
        this.worker?.postMessage({ type: "drop", id: p.id });
        this.matches.leave(p.token);
    }
    startRound() {
        this.roundTickets.clear();
        for (const p of [...this.members]) {
            try {
                const r = this.matches.beginRound(p.token, ++this.serial);
                this.roundTickets.set(p.token, {
                    serial: r.round_id,
                    ticket: r.round_ticket,
                });
            } catch {
                this.remove(p);
            }
        }
    }
    start() {
        this.members = [...this.peers.values()]
            .filter(
                (p) => this.matches.sessions.get(p.token)?.state === "active",
            )
            .slice(0, 8);
        if (!this.members.length) return;
        this.members.forEach((p, i) => {
            p.id = i === 0 ? 0 : i + 1;
            // Start the bounded three-round visit when the team actually takes the field.
            const session = this.matches.sessions.get(p.token);
            if (session && session.roundsUsed === 0)
                session.visitExpires =
                    this.matches.now() + this.matches.visitMs;
        });
        this.startRound();
        if (!this.members.length) return;
        this.fault = null;
        this.busy = false;
        const worker = new Worker(
            new URL("./coop-worker.mjs", import.meta.url),
            {
                workerData: {
                    players: this.members.map((p) => ({
                        id: p.id,
                        nickname: p.nickname,
                    })),
                    seed: Date.now() >>> 0,
                },
            },
        );
        this.worker = worker;
        worker.on("message", (m) => {
            if (this.worker !== worker) return;
            if (m.type === "state") this.state = m.state;
            if (m.type === "fault") {
                this.fault = m.message;
                console.error("Co-op:", m.message);
            }
            if (m.type === "infer") void this.infer(m, worker);
            if (m.type === "roundover") {
                for (const p of this.members) {
                    const r = this.roundTickets.get(p.token);
                    if (r)
                        try {
                            this.matches.finishRound(p.token, r.ticket);
                        } catch {}
                }
                this.roundTickets.clear();
                this.nextRoundAt = Date.now() + 3000;
                this.finished = m.round >= 3;
            }
        });
        worker.on("error", (e) => {
            console.error("Co-op worker:", e.message);
            if (this.worker === worker)
                this.abort("Room restarted. Join again.");
        });
        worker.on("exit", (code) => {
            if (this.worker === worker) this.abort("Room stopped. Join again.");
        });
    }
    async infer(m, worker) {
        if (this.busy) {
            worker.postMessage({ type: "decision", id: m.id });
            return;
        }
        const p = this.members.find((p) => this.roundTickets.has(p.token));
        if (!p) {
            worker.postMessage({ type: "decision", id: m.id });
            return;
        }
        const ticket = this.roundTickets.get(p.token),
            ctl = new AbortController();
        const timer = setTimeout(() => ctl.abort(), 3000);
        timer.unref();
        this.busy = true;
        try {
            const response = await this.matches.enqueue(
                p.token,
                async (signal, index) => {
                    const base = this.config.editions.classic.find(
                        (b) => b.id === "nanojev",
                    );
                    const urls = this.config.nanojevUrls;
                    return upstream(
                        {
                            ...base,
                            url: urls?.length
                                ? urls[index % urls.length]
                                : base.url,
                        },
                        "/decision",
                        {
                            ...m.body,
                            round_id: ticket.serial,
                            round_ticket: ticket.ticket,
                        },
                        signal,
                    );
                },
                ctl.signal,
            );
            if (this.worker === worker) {
                this.fault = null;
                worker.postMessage({ type: "decision", id: m.id, response });
            }
        } catch (e) {
            if (this.worker === worker) {
                this.fault = "Bot connection interrupted. Retrying…";
                worker.postMessage({ type: "decision", id: m.id });
            }
        } finally {
            clearTimeout(timer);
            this.busy = false;
        }
    }
    tick() {
        for (const p of [...this.peers.values()])
            if (!this.matches.sessions.has(p.token)) this.remove(p);
        if (this.worker && !this.members.length) {
            const w = this.worker;
            this.worker = null;
            void w.terminate();
            this.state = null;
            this.nextRoundAt = 0;
            this.fault = null;
        }
        if (this.worker && this.nextRoundAt && Date.now() >= this.nextRoundAt) {
            this.nextRoundAt = 0;
            if (this.finished) {
                const w = this.worker;
                this.worker = null;
                void w.terminate();
                this.members.forEach((p) => {
                    p.id = null;
                });
                this.members = [];
                this.state = null;
            } else {
                this.startRound();
                if (this.members.length)
                    this.worker.postMessage({ type: "next" });
            }
        }
        if (!this.worker) {
            const ready = [...this.peers.values()].some(
                (p) => this.matches.sessions.get(p.token)?.state === "active",
            );
            if (!ready) this.countdown = 0;
            else if (!this.countdown) this.countdown = Date.now() + 5000;
            else if (Date.now() >= this.countdown) {
                this.countdown = 0;
                this.start();
            }
        }
        for (const p of this.peers.values()) this.send(p);
    }
    abort(message) {
        this.fault = message;
        const w = this.worker;
        this.worker = null;
        void w?.terminate();
        for (const p of [...this.peers.values()]) {
            this.send(p);
            this.remove(p);
        }
        this.members = [];
        this.state = null;
        this.countdown = 0;
        this.nextRoundAt = 0;
    }
    close() {
        clearInterval(this.timer);
        this.abort("Server disconnected. Rejoin the team.");
    }
}
