export async function modelConfig(edition) {
    try {
        const r = await fetch("/api/config", {
            signal: AbortSignal.timeout(3000),
        });
        if (!r.ok) return [];
        const c = await r.json();
        return c.editions?.[edition] || [];
    } catch {
        return [];
    }
}
export function validHealth(h, protocols) {
    const m = h?.metadata;
    return (
        h?.ready === true &&
        typeof m?.model === "string" &&
        m.model.length > 0 &&
        Array.isArray(m.instances) &&
        m.instances.length === 2 &&
        new Set(m.instances).size === 2 &&
        m.instances.every((id) => typeof id === "string" && id.length > 0) &&
        protocols.every((p) =>
            (m.supported_protocols || [m.protocol]).includes(p),
        )
    );
}
export function validDecision(r, request, health) {
    return (
        r?.id === request.id &&
        r.round_id === request.round_id &&
        r.protocol === request.protocol &&
        r.model === health.metadata.model &&
        r.agent_id === health.metadata.instances[request.player] &&
        r.state_truncated === false
    );
}
export function validProbabilities(probabilities, keys) {
    return (
        probabilities &&
        Object.keys(probabilities).length === keys.length &&
        keys.every(
            (k) =>
                typeof probabilities[k] === "number" &&
                Number.isFinite(probabilities[k]) &&
                probabilities[k] >= 0 &&
                probabilities[k] <= 1,
        ) &&
        Math.abs(keys.reduce((sum, k) => sum + probabilities[k], 0) - 1) < 1e-5
    );
}
export function tokenLabel(data) {
    if (data.input_tokens == null) return "Token count unavailable";
    return data.context_limit == null
        ? `${data.total_input_tokens ?? data.input_tokens} tokens · ${data.structured_requests ?? 1} request(s)`
        : `${data.input_tokens} / ${data.context_limit} tokens`;
}
