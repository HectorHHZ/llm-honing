"""A tank run's decision log as typed decisions.

``decisions.jsonl`` has one row per question set the agent was asked: the state
exactly as the model read it (``contextParts``), the questions as sent, the
model's answers with a probability for every option, and the policy choice
(``chosen``). Planned steering asks twice per decision, the motion and
then the weapon for that motion, and each ask is its own row.

The policy choice is the training label: for a model's round it is what the
policy did (drawn from its probabilities in a rollout), and for a teacher's
round it is the teacher's answer. Safety can project a sampled motion before
execution; the log keeps that change in ``constraints`` and ``applied``. The
sampled choice and native behavior probabilities stay aligned.
"""

from __future__ import annotations

import json
from pathlib import Path

from honing.backends.tank.traces import decision_from_row
from honing.core import ContractError, DecisionSample


def decisions_from_log(
    run: Path,
    *,
    reward: float | None,
    group: str | None,
    release_id: str | None,
    harness_id: str | None,
    behavior: bool,
) -> list[DecisionSample]:
    """Every row of one run's decision log. ``behavior`` keeps the answers' probabilities as behavior probabilities."""
    log = run / "decisions.jsonl"
    if not log.exists():
        raise ContractError(f"{run} has no decision log")
    decisions = []
    with log.open(encoding="utf-8") as rows:
        for line in rows:
            if not line.strip():
                continue
            decisions.append(
                decision_from_row(
                    json.loads(line),
                    reward=reward,
                    group=group,
                    release_id=release_id,
                    harness_id=harness_id,
                    behavior=behavior,
                )
            )
    return decisions
