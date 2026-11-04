const percentile = (values, p) =>
    values.toSorted((a, b) => a - b)[Math.floor((values.length - 1) * p)];
// One in-flight request per seat. The floor follows measured fast-path RTT;
// forecasting uses the more conservative p75, so slower models get more lead.
export class AdaptiveCadence {
    constructor() {
        this.reset();
    }
    reset() {
        this.samples = [];
        this.backoff = 0;
    }
    observe(ms) {
        if (!Number.isFinite(ms) || ms <= 0) return;
        this.samples.push(ms);
        if (this.samples.length > 40) this.samples.shift();
        this.backoff *= 0.5;
    }
    busy() {
        this.backoff = Math.min(1500, Math.max(100, this.backoff * 1.5));
    }
    get interval() {
        return Math.round(
            Math.max(
                50,
                Math.min(
                    1500,
                    this.samples.length >= 4
                        ? percentile(this.samples, 0.2)
                        : 100,
                ),
            ) + this.backoff,
        );
    }
    get prediction() {
        return this.samples.length ? percentile(this.samples, 0.75) : 350;
    }
    get stats() {
        return {
            samples: this.samples.length,
            floor_ms: this.interval,
            p50_ms: this.samples.length
                ? Math.round(percentile(this.samples, 0.5))
                : null,
            p95_ms: this.samples.length
                ? Math.round(percentile(this.samples, 0.95))
                : null,
            serial_requests: true,
        };
    }
}
