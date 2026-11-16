const originalRender = render;
render = function(value) {
  originalRender(value);
  const d = value.demo;
  if (!d) return;
  document.getElementById('recordingSummary').textContent =
    `Model: ${d.repo} | revision: ${d.revision.slice(0, 12)} | harness: ${d.harness} | map: ${value.scenario.layout} | final score: ${d.final_score} | budget: 400 native ticks`;
  document.getElementById('actionCaption').textContent = value.frame.step === 0
    ? 'Next recorded action (not applied yet)'
    : `Applied joint action: tick ${value.frame.step - 1} → ${value.frame.step}`;
  document.getElementById('recordedActions').textContent =
    `Chef 0 / Laya: ${d.decision.chosen}\nChef 1 / greedy: ${d.actions[1]}\nReward for this joint tick: ${d.reward}`;
  document.getElementById('recordedProbabilities').textContent =
    Object.entries(d.decision.response.answers.action.probabilities)
      .map(([action, probability]) => `${action.padEnd(9)} ${Number(probability).toFixed(4)}`).join('\n');
  document.getElementById('recordedRequest').textContent = JSON.stringify(d.decision.request, null, 2);
  document.getElementById('controlHint').textContent =
    'Replay only: chef 0 / Laya; chef 1 / autonomous greedy (solo). Seeking and playback make no model calls.';
};
