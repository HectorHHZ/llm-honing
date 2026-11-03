// One capability token per visit to the queue, kept in memory (never in a URL).
// Leaving invalidates callbacks from previous requests as well as the server lease.
export class MatchAccess {
    constructor({
        fetcher = (...args) => fetch(...args),
        tokenFactory = () => crypto.randomUUID(),
        onChange = () => {},
        onActive = () => {},
        onLost = () => {},
        onQueued = () => {},
        now = () => performance.now(),
        beacon = (url, body) => navigator.sendBeacon?.(url, body),
    } = {}) {
        Object.assign(this, {
            fetcher,
            tokenFactory,
            onChange,
            onActive,
            onLost,
            onQueued,
            beacon,
            now,
        });
        this.state = "idle";
        this.position = 0;
        this.token = null;
        this.generation = 0;
        this.timer = null;
        this.roundTicket = null;
        this.roundId = null;
        this.roundsUsed = 0;
        this.maxRounds = 3;
        this.nickname = "";
        this.capacity = 16;
        this.lobby = { playing: [], waiting: [] };
        this.finishing = false;
        this.leaseDeadline = 0;
        this.heartbeatFailures = 0;
    }
    get active() {
        return this.state === "active";
    }
    get waiting() {
        return this.state === "joining" || this.state === "queued";
    }
    async call(action, token, keepalive = false, extra = {}) {
        const response = await this.fetcher(`/api/play/${action}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token, ...extra }),
            keepalive,
            signal: AbortSignal.timeout(action === 'heartbeat' && this.leaseDeadline
                ? Math.max(1, Math.floor(Math.min(4000, this.leaseDeadline - this.now() - 250)))
                : 4000),
        });
        let data;
        try { data = await response.json(); } catch { /* A proxy may return HTML. */ }
        if (!response.ok || !data) {
            const error = Error(data?.error || "Connection interrupted. Please try again.");
            error.status = response.status;
            error.retryable = response.status === 408 || response.status === 429 ||
                response.status >= 500 || (response.ok && !data);
            throw error;
        }
        return data;
    }
    async join() {
        if (this.waiting || this.active) return;
        this.token = this.tokenFactory();
        this.state = "joining";
        this.position = 0;
        const generation = ++this.generation;
        this.onChange();
        await this.refresh("join", generation, this.token);
    }
    async refresh(action, generation, token) {
        if (generation !== this.generation || token !== this.token) return;
        const sentAt = this.now();
        try {
            if (action === 'heartbeat' && sentAt >= this.leaseDeadline)
                throw Error("Your place expired. Please join again.");
            const status = await this.call(action, token);
            if (generation !== this.generation) return;
            if (
                !["active", "queued"].includes(status.state) ||
                !Number.isFinite(status.heartbeat_ms) ||
                status.heartbeat_ms < 250 ||
                status.heartbeat_ms > 5000 ||
                !Number.isInteger(status.position) ||
                status.position < 0
            )
                throw Error("Unable to confirm your place. Please join again.");
            this.accept(status, sentAt);
            if (generation !== this.generation) return;
            this.timer = setTimeout(
                () => void this.refresh("heartbeat", generation, token),
                status.heartbeat_ms,
            );
        } catch (error) {
            if (generation !== this.generation) return;
            const transient = error.retryable === true ||
                error instanceof TypeError || error.name === 'TimeoutError';
            const retryDelay = 250 * (this.heartbeatFailures + 1);
            if (action === 'heartbeat' && transient && this.heartbeatFailures < 2 &&
                this.now() + retryDelay + 250 < this.leaseDeadline) {
                this.heartbeatFailures++;
                this.timer = setTimeout(() => void this.refresh(action, generation, token), retryDelay);
                return;
            }
            this.leave();
            this.onLost(
                error instanceof TypeError || error.name === "TimeoutError"
                    ? "Connection interrupted. Please join again."
                    : error.message || "Connection lost. Join again to play.",
            );
        }
    }
    accept(status, sentAt = this.now()) {
        // Use request start, not response time: network delay cannot extend a lease.
        const lease = Number.isFinite(status.lease_ms) && status.lease_ms >= 3000 && status.lease_ms <= 60000
            ? status.lease_ms : 15000;
        if (sentAt + lease - 500 <= this.now())
            throw Error("Your place expired. Please join again.");
        this.leaseDeadline = sentAt + lease - 500;
        this.heartbeatFailures = 0;
        const wasActive = this.active;
        this.nickname = typeof status.nickname === "string" ? status.nickname : "";
        this.capacity = status.capacity || 16;
        this.lobby = status.lobby || { playing: [], waiting: [] };
        this.state = status.state;
        this.position = status.position;
        this.roundTicket = status.round_ticket || null;
        this.roundId = status.round_id || null;
        this.roundsUsed = status.rounds_used || 0;
        this.maxRounds = status.max_rounds || 3;
        this.onChange();
        if (wasActive && this.state === "queued") this.onQueued();
        if (!wasActive && this.active) this.onActive();
    }
    hasRound(serial) { return this.active && this.roundId === serial && !!this.roundTicket; }
    async roundCall(action, extra) {
        const generation = ++this.generation, token = this.token;
        clearTimeout(this.timer);
        const sentAt = this.now();
        try {
            const status = await this.call(action, token, false, extra);
            if (generation !== this.generation) return false;
            this.accept(status, sentAt);
            if (generation === this.generation) this.timer = setTimeout(
                () => void this.refresh("heartbeat", generation, token), status.heartbeat_ms);
            return true;
        } catch (error) {
            if (generation === this.generation) { this.leave(); this.onLost(error.message); }
            return false;
        }
    }
    async beginRound(serial) {
        if (this.hasRound(serial)) return true;
        if (!this.active) return false;
        return await this.roundCall("begin", {round_id: serial}) && this.hasRound(serial);
    }
    // result ({winner, modified}) goes to the leaderboard.
    async finishRound(result = null) {
        if (!this.roundTicket || this.finishing) return;
        this.finishing = true;
        try { await this.roundCall("finish", {round_ticket: this.roundTicket, ...(result ? {result} : {})}); }
        finally { this.finishing = false; }
    }
    async rename(name) {
        if (!this.token) return false;
        const sentAt = this.now(), status = await this.call("rename", this.token, false, {name});
        this.accept(status, sentAt);
        return true;
    }
    leave(unloading = false) {
        const token = this.token;
        ++this.generation;
        clearTimeout(this.timer);
        this.timer = null;
        this.token = null;
        this.leaseDeadline = 0;
        this.heartbeatFailures = 0;
        this.roundTicket = null;
        this.roundId = null;
        this.state = "idle";
        this.position = 0;
        this.nickname = "";
        this.lobby = { playing: [], waiting: [] };
        this.onChange();
        if (!token) return;
        if (unloading) {
            try {
                if (
                    this.beacon(
                        "/api/play/leave",
                        new Blob([JSON.stringify({ token })], {
                            type: "application/json",
                        }),
                    )
                )
                    return;
            } catch {
                /* Fall back to keepalive; lease expiry is the final fallback. */
            }
        }
        void this.call("leave", token, true).catch(() => {});
    }
}
