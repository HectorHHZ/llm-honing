"""Translate recorded choices into the existing typed_decisions training contract."""

from __future__ import annotations

from typing import Any

from honing.backends.griddly.model import validate_reply
from honing.core import DecisionQuestion, DecisionSample


def decision_samples(
    rows: list[dict[str, Any]],
    *,
    group: str,
    reward: float | None = None,
    release_id: str | None = None,
    harness_id: str | None = None,
    behavior: bool = False,
) -> list[DecisionSample]:
    result = []
    for row in rows:
        if row["forced_action"]:
            continue
        request = row["request"]
        question = request["questions"]["action"]
        result.append(
            DecisionSample(
                parts=tuple((name, tuple(lines)) for name, lines in request["parts"].items()),
                questions=(
                    (
                        "action",
                        DecisionQuestion("choice", question["instructions"], tuple(question["criteria"].items())),
                    ),
                ),
                chosen=(("action", row["chosen"]["action"]),),
                behavior_probs=(
                    (("action", tuple(validate_reply(request, row["response"]).items())),) if behavior else None
                ),
                reward=reward,
                group=group,
                source_release=release_id,
                source_harness=harness_id,
            )
        )
    return result
