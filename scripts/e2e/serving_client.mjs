// The serving end-to-end check's client: the RA2 match's own client stack
// (decisionClientOptions -> createDecisionClient -> PermutationClient -> HttpDecisionClient)
// against a decision server process. Usage: node serving_client.mjs <url> <requests.json>
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
    createDecisionClient,
    decisionClientOptions,
} from "../../integrations/ra2/integration/models/factory.mjs";

const [url, requestsPath] = process.argv.slice(2);
const SECTIONS = ["essential", "details", "memory", "logs"];
const requests = JSON.parse(readFileSync(requestsPath, "utf8"));
// A binary target question: the permutation client asks twice, with options B and C swapped.
requests.push({
    parts: { essential: ["Tick 90. Enemy refinery and barracks in view."] },
    questions: {
        target: {
            type: "choice",
            instructions: "Which target should the selected units attack?",
            criteria: {
                A: "Cancel this pending action; submit no command for it.",
                B: "Attack the ore refinery",
                C: "Attack the barracks at the north gate",
            },
        },
    },
    dimensions: { target: "target" },
});

const options = decisionClientOptions({ "decision-server": url }, "intent");
const client = createDecisionClient(options);
const metadata = await client.ready;
assert.equal(metadata.model, "longest-option");
assert.equal(metadata.decisionServer, url);

let checked = 0;
for (const request of requests) {
    const body = await client.predict({
        parts: request.parts,
        questions: request.questions,
        ...(request.dimensions ? { dimensions: request.dimensions } : {}),
    });
    const state = SECTIONS.flatMap((k) => request.parts[k] ?? []).join("\n");
    assert.equal(
        body.stateText,
        state,
        "state text differs from the packed parts",
    );
    assert.deepEqual(
        Object.keys(body.answers).sort(),
        Object.keys(request.questions).sort(),
    );
    for (const [id, question] of Object.entries(request.questions)) {
        const answer = body.answers[id];
        const keys = Object.keys(question.criteria);
        assert.deepEqual(
            Object.keys(answer.probabilities).sort(),
            [...keys].sort(),
        );
        const sum = Object.values(answer.probabilities).reduce(
            (a, b) => a + b,
            0,
        );
        assert(Math.abs(sum - 1) < 1e-9, "probabilities do not sum to one");
        // The permutation client averages two forwards, which leaves a length-proportional
        // distribution unchanged, so the longest description wins either way.
        const expected = keys.reduce((best, key) =>
            question.criteria[key].length > question.criteria[best].length
                ? key
                : best,
        );
        assert.equal(answer.choice, expected, `question ${id}`);
        assert.equal(body.tokenAudit[id].options, keys.length);
    }
    if (request.dimensions) {
        assert.equal(body.forwardCalls, 2);
        assert.equal(body.ensemble.method, "binary-target-permutation-mean-v1");
    }
    checked++;
}
client.close();
console.log(
    JSON.stringify({
        served: metadata,
        requests: checked,
        permutationRequests: 1,
    }),
);
console.log("serving end to end: ok");
