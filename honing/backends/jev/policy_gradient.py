"""Policy gradient over a decision model's option distributions, with a group baseline.

Each decision was made by the policy being trained (on-policy rollouts) and
earned a reward, for example its match's result. Decisions in the same group
(repeats of one scenario) share a baseline: a decision's advantage is its
reward minus the group's mean reward, divided by the group's standard
deviation. A group whose rewards are all equal teaches nothing and is skipped.

The update maximizes the clipped surrogate ``min(r * A, clip(r, 1 - eps, 1 + eps) * A)``
per question, where ``r`` is the chosen option's probability under the current
weights divided by its probability under the weights the update started from.
Those starting probabilities are recomputed from the model before the first
step, not read from the rollout: the rollout's probabilities come from inference,
which may apply temperature calibration that training does not, and would bias
``r``. The rollout's probabilities are reported for comparison.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence

from honing.core import ContractError, DecisionSample


def group_advantages(decisions: Sequence[DecisionSample], eps: float = 1e-6) -> list[float]:
    """Each decision's reward minus its group's mean, over the group's standard deviation; 0 when all agree."""
    rewards: dict[str, list[float]] = defaultdict(list)
    for decision in decisions:
        if decision.group is None or decision.reward is None:
            raise ContractError("policy gradient needs a group and a reward on every decision")
        rewards[decision.group].append(decision.reward)
    stats = {}
    for group, values in rewards.items():
        mean = statistics.fmean(values)
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        stats[group] = (mean, spread)
    advantages = []
    for decision in decisions:
        mean, spread = stats[decision.group]  # type: ignore[index]
        advantages.append(0.0 if spread <= eps else (decision.reward - mean) / spread)  # type: ignore[operator]
    if not all(math.isfinite(a) for a in advantages):
        raise ContractError("nonfinite advantage")
    return advantages
