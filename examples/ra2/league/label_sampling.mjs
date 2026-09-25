// Sampling changes training data only; complete match traces remain archived.
export function sampleCorrection(
    tick,
    actual,
    desired,
    previousSignature,
    mode,
) {
    const signature = JSON.stringify(desired);
    if (mode === "actions-periodic")
        return tick < 600 || actual.length > 0 || tick % 30 === 0;
    if (mode === "transition-periodic")
        return tick < 600 || tick % 30 === 0 || signature !== previousSignature;
    throw new Error(`Unknown correction sampling mode: ${mode}`);
}
