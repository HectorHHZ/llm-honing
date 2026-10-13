"""A tank run's decision log as typed decisions.

``decisions.jsonl`` has one row per question set the agent was asked: the state
exactly as the model read it (``contextParts``), the questions as sent, the
model's answers with a probability for every option, and the option the match
executed (``chosen``). Planned steering asks twice per decision, the motion and
then the weapon for that motion, and each ask is its own row.

The executed option is the training label: for a model's round it is what the
policy did (drawn from its probabilities in a rollout), and for a teacher's
round it is the teacher's answer.
"""

from __future__ import annotations

import json
from pathlib import Path

from honing.core import ContractError, DecisionQuestion, DecisionSample


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
            row = json.loads(line)
            questions = row["questions"]
            decisions.append(
                DecisionSample(
                    parts=tuple((name, tuple(lines)) for name, lines in row["contextParts"].items()),
                    questions=tuple(
                        (
                            key,
                            DecisionQuestion(
                                kind=q["type"], instructions=q["instructions"], options=tuple(q["criteria"].items())
                            ),
                        )
                        for key, q in questions.items()
                    ),
                    chosen=tuple((key, row["chosen"][key]) for key in questions),
                    behavior_probs=(
                        tuple(
                            (
                                key,
                                tuple(
                                    (option, float(row["answers"][key]["probabilities"][option]))
                                    for option in q["criteria"]
                                ),
                            )
                            for key, q in questions.items()
                        )
                        if behavior
                        else None
                    ),
                    reward=reward,
                    group=group,
                    source_release=release_id,
                    source_harness=harness_id,
                )
            )
    return decisions
