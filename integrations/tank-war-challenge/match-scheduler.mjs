import { randomInt, randomUUID } from "node:crypto";
import { HttpError } from "./model-bridge.mjs";
import { cleanName } from "./leaderboard.mjs";

export class MatchError extends HttpError {
    constructor(status, code, message) {
        super(status, message);
        this.code = code;
    }
}

// Admission is FIFO; inference is round-robin across admitted matches. A model
// worker stays occupied until inference finishes, even if its HTTP client left:
// disconnecting fetch does not cancel computation in the legacy GPU service.
export class MatchScheduler {
    constructor({
        capacity = 2,
        workers = 1,
        maxRounds = 3,
        roundMs = 65000,
        visitMs = 210000,
        maxWaiting = 50,
        leaseMs = 15000,
        queueMs = 900,
        now = Date.now,
        sweepMs = 1000,
    } = {}) {
        if (!Number.isInteger(workers) || workers < 1 || workers > 8)
            throw Error("Invalid worker count");
        this.maxRounds = maxRounds;
        this.roundMs = roundMs;
        this.visitMs = visitMs;
        this.visitCounter = 0;
        this.workers = workers;
        this.runningJobs = new Map();
        this.capacity = capacity;
        this.maxWaiting = maxWaiting;
        this.leaseMs = leaseMs;
        this.queueMs = queueMs;
        this.now = now;
        this.sessions = new Map();
        this.waiting = [];
        this.pending = new Map();
        this.retired = new Map();
        this.turn = 0;
        this.closed = false;
        this.metrics = { completed: 0, failed: 0, superseded: 0, expired: 0 };
        this.sweeper = setInterval(() => this.sweep(), sweepMs);
        this.sweeper.unref();
    }
    get running() {
        return this.runningJobs.values().next().value || null;
    }
    validateToken(token) {
        if (
            typeof token !== "string" ||
            !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
                token,
            )
        )
            throw new MatchError(401, "invalid_session", "Start a match first");
    }
    nickname() {
        const adjectives = ["Swift", "Brave", "Lucky", "Calm", "Clever", "Bold", "Quick", "Sunny"];
        const animals = ["Fox", "Otter", "Panda", "Owl", "Lynx", "Finch", "Tiger", "Badger"];
        const used = new Set([...this.sessions.values()].map(s => s.nickname));
        const base = `${adjectives[randomInt(adjectives.length)]} ${animals[randomInt(animals.length)]}`;
        let number = randomInt(100, 1000);
        while (used.has(`${base} ${number}`)) number++;
        return `${base} ${number}`;
    }
    // A returning browser gets the name it chose last (nameFor, from the leaderboard), unless it is taken.
    nameFor(owner) {
        const saved = this.savedName?.(owner);
        const used = new Set([...this.sessions.values()].map((s) => s.nickname));
        return saved && !used.has(saved) ? saved : this.nickname();
    }
    rename(token, name) {
        this.heartbeat(token);
        const s = this.sessions.get(token), clean = cleanName(name);
        if (!clean) throw new MatchError(400, "invalid_name", "Names are 1 to 20 letters, digits or spaces");
        if ([...this.sessions.values()].some((other) => other !== s && other.nickname === clean))
            throw new MatchError(409, "name_taken", "Someone here already has that name");
        s.nickname = clean;
        this.onRename?.(s);
        return this.status(s);
    }
    join(token, owner = token) {
        this.validateToken(token);
        this.sweep();
        if (this.closed || this.retired.has(token))
            throw new MatchError(410, "session_expired", "Join again to play");
        const owned = [...this.sessions.values()].find(s => s.owner === owner && s.token !== token);
        if (owned) throw new MatchError(409, "session_in_use", "You already have a place in another tab. Close it or wait for it to expire.");
        if (this.sessions.has(token)) this.authorizeOwner(token, owner);
        if (!this.sessions.has(token)) {
            if (
                this.waiting.length >= this.maxWaiting &&
                this.activeCount >= this.capacity
            )
                throw new MatchError(
                    503,
                    "queue_full",
                    "The arena is full. Please try again shortly.",
                );
            this.sessions.set(token, {
                token,
                owner,
                nickname: this.nameFor(owner),
                roundsUsed: 0,
                round: null,
                completed: new Set(),
                state: "queued",
                touched: this.now(),
                served: this.turn,
            });
            this.waiting.push(token);
            this.promote();
        }
        return this.heartbeat(token);
    }
    get activeCount() {
        return [...this.sessions.values()].filter((s) => s.state === "active")
            .length;
    }
    heartbeat(token) {
        this.validateToken(token);
        this.sweep();
        const session = this.sessions.get(token);
        if (!session)
            throw new MatchError(
                410,
                "session_expired",
                "Your place expired. Join again to play.",
            );
        session.touched = this.now();
        return this.status(session);
    }
    status(session) {
        return {
            nickname: session.nickname,
            // This public roster never contains capability tokens, owners or
            // another player's round ticket. Identity is assigned by the server.
            lobby: {
                playing: [...this.sessions.values()].filter(s => s.state === "active")
                    .map(s => ({nickname: s.nickname, is_you: s === session})),
                waiting: this.waiting.map((token, index) => {
                    const s = this.sessions.get(token);
                    return {nickname: s.nickname, position: index + 1, is_you: s === session};
                }),
            },
            state: session.state,
            position:
                session.state === "queued"
                    ? this.waiting.indexOf(session.token) + 1
                    : 0,
            active: this.activeCount,
            capacity: this.capacity,
            heartbeat_ms: Math.min(3000, Math.floor(this.leaseMs / 3)),
            lease_ms: this.leaseMs,
            max_rounds: this.maxRounds,
            rounds_used: session.roundsUsed,
            visit: session.visit || 0,
            round_ticket: session.round?.ticket || null,
            round_id: session.round?.serial || null,
        };
    }
    promote() {
        while (
            !this.closed &&
            this.activeCount < this.capacity &&
            this.waiting.length
        ) {
            const session = this.sessions.get(this.waiting.shift());
            if (session) {
                session.state = "active";
                session.served = this.turn;
                session.roundsUsed = 0;
                session.round = null;
                session.visit = ++this.visitCounter;
                session.visitExpires = this.now() + this.visitMs;
                // Promotion is not a heartbeat: an abandoned queue tab must expire.
            }
        }
    }
    authorizeOwner(token, owner) {
        this.validateToken(token);
        const s = this.sessions.get(token);
        if (s && s.owner !== owner) throw new MatchError(403, "wrong_owner", "This place belongs to another browser");
    }
    beginRound(token, serial) {
        this.heartbeat(token);
        const s = this.sessions.get(token);
        if (s.state !== "active") throw new MatchError(409, "match_queued", "Wait for your place in the arena");
        if (!Number.isInteger(serial) || serial < 1) throw new MatchError(400, "invalid_round", "Invalid round");
        if (s.round) {
            if (s.round.serial !== serial) throw new MatchError(409, "round_in_progress", "Finish your current round first");
            return this.status(s);
        }
        if (s.roundsUsed >= this.maxRounds) throw new MatchError(409, "round_limit", "Join the queue for another turn");
        s.roundsUsed++;
        s.round = {ticket: randomUUID(), serial, expires: this.now() + this.roundMs};
        return this.status(s);
    }
    finishRound(token, ticket, result = null) {
        this.heartbeat(token);
        const s = this.sessions.get(token);
        if (s.completed.has(ticket)) return this.status(s);
        if (!s.round || s.round.ticket !== ticket) throw new MatchError(409, "invalid_round_ticket", "This round has expired");
        s.completed.add(ticket);
        // The round's result, as the player's browser reports it, goes to the leaderboard once per ticket.
        if (result && typeof result === "object")
            this.onResult?.(s, { winner: [0, 1].includes(result.winner) ? result.winner : null, modified: result.modified === true });
        if (s.completed.size > 12) s.completed.delete(s.completed.values().next().value);
        s.round = null;
        this.cancelWork(token);
        if (s.roundsUsed >= this.maxRounds) this.rotate(s);
        return this.status(s);
    }
    requireRound(token, serial, ticket) {
        this.sweep();
        const s = this.sessions.get(token);
        if (!s) throw new MatchError(410, "session_expired", "Start a match first");
        if (s.state !== "active") throw new MatchError(409, "match_queued", "Wait for your place in the arena");
        if (!s.round || s.round.ticket !== ticket || s.round.serial !== serial)
            throw new MatchError(409, "round_required", "Start an authorized round first. Reload if needed.");
    }
    cancelWork(token) {
        const pending = this.pending.get(token);
        if (pending) {this.pending.delete(token); pending.finish(new MatchError(409, "round_ended", "This round ended"));}
        for (const job of this.runningJobs.values()) if (job.token === token)
            job.finish(new MatchError(409, "round_ended", "This round ended"));
    }
    rotate(s) {
        this.cancelWork(s.token);
        s.round = null;
        s.state = "queued";
        // Promote existing waiters before adding this visitor at the tail.
        this.promote();
        this.waiting.push(s.token);
    }
    leave(token) {
        this.validateToken(token);
        this.remove(token);
        this.promote();
        return { state: "left" };
    }
    remove(token) {
        this.sessions.delete(token);
        this.waiting = this.waiting.filter((t) => t !== token);
        this.retired.set(token, this.now());
        // Bound tombstones used to reject a join that arrives after its leave.
        if (this.retired.size > 1000)
            this.retired.delete(this.retired.keys().next().value);
        const job = this.pending.get(token);
        if (job) {
            this.pending.delete(token);
            job.finish(
                new MatchError(410, "session_expired", "This match has ended"),
            );
        }
        for (const job of this.runningJobs.values())
            if (job.token === token) job.finish(
                new MatchError(410, "session_expired", "This match has ended"),
            );
    }
    sweep() {
        const now = this.now();
        for (const [token, s] of this.sessions) {
            if (now - s.touched >= this.leaseMs) this.remove(token);
            else if (s.state === "active" && now >= s.visitExpires) this.rotate(s);
            else if (s.round && now >= s.round.expires) {
                s.completed.add(s.round.ticket);
                s.round = null;
                this.cancelWork(token);
                if (s.roundsUsed >= this.maxRounds) this.rotate(s);
            }
        }
        for (const [token, at] of this.retired)
            if (now - at >= this.leaseMs * 2) this.retired.delete(token);
        this.promote();
    }
    enqueue(token, run, signal) {
        this.validateToken(token);
        this.sweep();
        const session = this.sessions.get(token);
        if (!session)
            throw new MatchError(410, "session_expired", "Start a match first");
        if (session.state !== "active")
            throw new MatchError(
                409,
                "match_queued",
                "Wait for your place in the arena",
            );
        if (signal?.aborted)
            throw new MatchError(499, "request_cancelled", "Request cancelled");
        // Only explicit heartbeats keep a match alive; request floods cannot do so.
        return new Promise((resolve, reject) => {
            const old = this.pending.get(token);
            if (old) {
                this.metrics.superseded++;
                old.finish(
                    new MatchError(
                        409,
                        "decision_superseded",
                        "A newer arena state replaced this request",
                    ),
                );
            }
            const job = { token, run, created: this.now(), done: false };
            const abort = () => {
                if (this.pending.get(token) === job) this.pending.delete(token);
                job.finish(
                    new MatchError(
                        499,
                        "request_cancelled",
                        "Request cancelled",
                    ),
                );
            };
            job.finish = (error, result) => {
                if (job.done) return;
                job.done = true;
                clearTimeout(job.timer);
                signal?.removeEventListener("abort", abort);
                if (error) reject(error);
                else resolve(result);
            };
            job.timer = setTimeout(() => {
                if (this.pending.get(token) !== job) return;
                this.pending.delete(token);
                this.metrics.expired++;
                job.finish(
                    new MatchError(
                        409,
                        "decision_expired",
                        "Read the arena again",
                    ),
                );
            }, this.queueMs);
            job.timer.unref();
            signal?.addEventListener("abort", abort, { once: true });
            this.pending.set(token, job);
            void this.pump();
        });
    }
    async pump() {
        if (this.runningJobs.size >= this.workers || this.closed) return;
        this.sweep();
        const runningTokens = new Set([...this.runningJobs.values()].map(job => job.token));
        const next = [...this.pending.values()].filter(job => !runningTokens.has(job.token)).sort(
            (a, b) =>
                this.sessions.get(a.token).served -
                this.sessions.get(b.token).served,
        )[0];
        if (!next) return;
        this.pending.delete(next.token);
        clearTimeout(next.timer);
        if (this.now() - next.created >= this.queueMs) {
            this.metrics.expired++;
            next.finish(
                new MatchError(409, "decision_expired", "Read the arena again"),
            );
            return this.pump();
        }
        const worker = Array.from({length: this.workers}, (_, i) => i).find(i => !this.runningJobs.has(i));
        this.runningJobs.set(worker, next);
        this.sessions.get(next.token).served = ++this.turn;
        void this.pump();
        try {
            const result = await next.run(AbortSignal.timeout(10000), worker);
            this.metrics.completed++;
            next.finish(null, result);
        } catch (error) {
            this.metrics.failed++;
            next.finish(error);
        } finally {
            this.runningJobs.delete(worker);
            void this.pump();
        }
    }
    close() {
        this.closed = true;
        clearInterval(this.sweeper);
        for (const token of this.sessions.keys()) this.remove(token);
    }
}
