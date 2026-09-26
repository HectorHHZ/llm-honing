import assert from "node:assert/strict";

export const METHOD = "binary-target-permutation-mean-v1";
export function binaryTarget(request) {
    const ids = Object.keys(request.questions);
    if (ids.length !== 1 || request.dimensions?.[ids[0]] !== "target")
        return false;
    const q = request.questions[ids[0]],
        keys = Object.keys(q.criteria);
    return (
        q.type === "choice" &&
        keys.length === 3 &&
        keys[0] === "A" &&
        q.criteria.A.startsWith("Cancel ")
    );
}
export function permutation(questions) {
    const ids = Object.keys(questions);
    assert.equal(ids.length, 1);
    const id = ids[0],
        q = questions[id],
        keys = Object.keys(q.criteria);
    assert.equal(keys.length, 3);
    assert.equal(keys[0], "A");
    const sourceKeys = [keys[0], keys[2], keys[1]];
    const changed = structuredClone(questions);
    changed[id].criteria = Object.fromEntries(
        keys.map((key, i) => [key, q.criteria[sourceKeys[i]]]),
    );
    return { questions: changed, sourceKeys };
}
function probabilities(answer, keys) {
    assert(answer && typeof answer.probabilities === "object");
    assert.deepEqual(
        Object.keys(answer.probabilities).sort(),
        [...keys].sort(),
    );
    const values = keys.map((key) => answer.probabilities[key]);
    assert(
        values.every(
            (p) =>
                typeof p === "number" && Number.isFinite(p) && p >= 0 && p <= 1,
        ),
    );
    const sum = values.reduce((n, p) => n + p, 0);
    assert(Math.abs(sum - 1) < 0.005 && sum > 0);
    assert(keys.includes(answer.choice));
    assert(answer.probabilities[answer.choice] + 0.0001 >= Math.max(...values));
    return values.map((p) => p / sum); // Correct upstream four-decimal logging roundoff.
}
export function aggregate(questions, original, swapped) {
    const id = Object.keys(questions)[0],
        keys = Object.keys(questions[id].criteria);
    const { sourceKeys } = permutation(questions);
    assert.deepEqual(Object.keys(original.answers), [id]);
    assert.deepEqual(Object.keys(swapped.answers), [id]);
    const a = probabilities(original.answers[id], keys),
        b = probabilities(swapped.answers[id], keys);
    const scores = Object.fromEntries(keys.map((key, i) => [key, a[i] / 2]));
    sourceKeys.forEach((canonical, i) => {
        scores[canonical] += b[i] / 2;
    });
    const best = keys.reduce(
        (best, key) => (scores[key] > scores[best] ? key : best),
        keys[0],
    );
    return { [id]: { type: "choice", choice: best, probabilities: scores } };
}
export function verifyEnsemble(record) {
    if (!binaryTarget(record)) {
        assert.equal(record.forwardCalls ?? 1, 1);
        assert(!record.ensemble);
        return;
    }
    assert.equal(record.forwardCalls, 2);
    assert.equal(record.ensemble?.method, METHOD);
    const expected = permutation(record.questions);
    assert.deepEqual(record.ensemble.permutedQuestions, expected.questions);
    assert.deepEqual(record.ensemble.sourceKeys, expected.sourceKeys);
    const id = Object.keys(record.questions)[0];
    for (const response of [
        record.ensemble.original,
        record.ensemble.swapped,
    ]) {
        assert.equal(response.stateText, record.stateText);
        assert.deepEqual(response.trimmed, record.trimmed);
        assert.deepEqual(response.modelResult.answers, response.answers);
        const audit = response.tokenAudit[id];
        assert.equal(audit.options, 3);
        assert.equal(audit.stateTruncated, false);
        assert(audit.stateTokens <= audit.stateRoom);
        assert(Number.isInteger(audit.inputTokens) && audit.inputTokens > 0);
    }
    const mean = aggregate(
        record.questions,
        record.ensemble.original,
        record.ensemble.swapped,
    );
    assert.deepEqual(record.answers, mean);
    assert.deepEqual(record.modelResult.answers, mean);
    assert.deepEqual(record.tokenAudit, record.ensemble.original.tokenAudit);
}
export class PermutationClient {
    constructor(client) {
        this.client = client;
        this.ready = client.ready.then((metadata) => {
            this.metadata = {
                ...metadata,
                selectionMethod: METHOD,
                targetForwardCount: 2,
            };
            return this.metadata;
        });
    }
    async predict(packet) {
        if (!binaryTarget(packet)) return this.client.predict(packet);
        const { questions, sourceKeys } = permutation(packet.questions);
        const original = await this.client.predict(packet);
        const swapped = await this.client.predict({ ...packet, questions });
        assert.equal(
            original.stateText,
            swapped.stateText,
            "Permutations changed retained context",
        );
        assert.deepEqual(original.trimmed, swapped.trimmed);
        const answers = aggregate(packet.questions, original, swapped);
        const result = {
            id: original.id,
            answers,
            modelResult: { model: METHOD, answers: structuredClone(answers) },
            stateText: original.stateText,
            tokenAudit: original.tokenAudit,
            trimmed: original.trimmed,
            latencyMs: original.latencyMs + swapped.latencyMs,
            forwardCalls: 2,
            ensemble: {
                method: METHOD,
                sourceKeys,
                permutedQuestions: questions,
                original,
                swapped,
            },
        };
        verifyEnsemble({ ...packet, ...result });
        return result;
    }
    close() {
        this.client.close();
    }
}
