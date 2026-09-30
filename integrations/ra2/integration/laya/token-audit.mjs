import assert from "node:assert/strict";

export function verifyTokenAudit(audit, options) {
    assert(audit);
    assert.equal(audit.options, options);
    assert.equal(audit.stateTruncated, false);
    assert(Number.isInteger(audit.inputTokens) && audit.inputTokens > 0);
    if (audit.encoding === "djev-remote-prompt") {
        // djev reports the whole prompt count, not a separate state budget.
        assert.equal(audit.accounting, "upstream-request-total");
        assert.equal(audit.stateTokens, null);
        assert.equal(audit.stateRoom, null);
    } else {
        assert(Number.isInteger(audit.stateTokens) && audit.stateTokens >= 0);
        assert(
            Number.isInteger(audit.stateRoom) &&
                audit.stateRoom >= audit.stateTokens,
        );
    }
}
