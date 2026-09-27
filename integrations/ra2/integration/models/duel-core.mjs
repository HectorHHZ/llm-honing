import assert from "node:assert/strict";
import { predictStaged } from "../laya/staged.mjs";

export const DUEL_SEATS = ["red", "blue"];
export function normalSeats(players, seat) {
    assert(DUEL_SEATS.includes(seat));
    return players.map((p) =>
        p.name === seat ? { ...p } : { name: p.name, player_id: p.player_id },
    );
}

// There is deliberately no engine or privileged observer in this interface.
// Both packets exist before either client is invoked; no opponent decision is
// supplied to a policy, even when the two workers finish at different times.
export async function resolveDuelStep({
    tick,
    observations,
    view,
    policies,
    clients,
    predict = predictStaged,
}) {
    assert.equal(view, "normal");
    assert(Number.isInteger(tick) && tick >= 0);
    const packets = {};
    for (const seat of DUEL_SEATS) {
        assert(observations[seat], `Missing normal observation for ${seat}`);
        policies[seat].refresh(observations[seat], tick);
        packets[seat] = policies[seat].packet();
        assert.equal(packets[seat].tick, tick);
    }
    const predictions = await Promise.all(
        DUEL_SEATS.map((seat) => predict(packets[seat], clients[seat])),
    );
    return Object.fromEntries(
        DUEL_SEATS.map((seat, index) => {
            const policy = policies[seat],
                packet = packets[seat],
                prediction = predictions[index];
            const selection = policy.select(prediction.resolvedAnswers, 1);
            assert(
                selection.actions.length <= 1 && selection.selected.length <= 1,
                "Duel seat exceeds equal one-command budget",
            );
            assert.deepEqual(
                selection.actions,
                selection.selected.map((choice) => choice.action),
            );
            return [
                seat,
                {
                    actions: selection.actions,
                    record: {
                        tick,
                        seat,
                        decisionMode: "staged",
                        candidateMode: "intent",
                        availability: packet.availability,
                        actorExclusions: packet.actorExclusions,
                        normalUnits: policy.frame.rows,
                        policyMemory: {
                            kind: "normal-public-search-v1",
                            searchVisits: [
                                ...(policy.searchVisits ?? new Map()),
                            ],
                        },
                        ownVectors: {
                            minerals: policy.obs.vectors.minerals,
                            power_cap: policy.obs.vectors.power_cap,
                            power_used: policy.obs.vectors.power_used,
                            production: policy.obs.vectors.production,
                            units_truncated: policy.obs.vectors.units_truncated,
                        },
                        contextParts: packet.parts,
                        leafQuestions: packet.questions,
                        candidates: packet.groups,
                        omitted: packet.omitted,
                        ...prediction,
                        selected: selection.selected,
                        dropped: selection.dropped,
                    },
                },
            ];
        }),
    );
}
