"""Guard for a Laya request that carries several choice questions (v9_keys: `move` and `attack` in one request).
New file, 2026-11-25; core/laya.py is unchanged and its client (LayaClient: GET /v1/health, POST /v1/decide, raw
bytes saved before parsing) is used as it is. The service at 127.0.0.1:8809 is never started, stopped or changed.

The checks are those of core/laya.py `checked_choice`, applied to every question: id match, `trimmed == []`, the
full `stateText` echo, a `tokenAudit` for every question (not truncated, the right option count, consistent token
counts), probabilities over exactly the offered options, finite, within [0, 1], summing to one within rounding,
and the returned choice is the argmax. One failed check rejects the whole response: nothing may be executed.
"""

from __future__ import annotations

import math
from typing import Any

from .laya import MAX_OPTIONS, GuardRejection


def build_request(identity: int, essential: list[str], questions: dict[str, dict]) -> dict[str, Any]:
    if not questions:
        raise ValueError("no question")
    out = {}
    for key, q in questions.items():
        if not 1 <= len(q["criteria"]) <= MAX_OPTIONS:
            raise ValueError("Laya accepts 1..16 options per question")
        out[key] = {"type": "choice", "instructions": q["instructions"], "criteria": dict(q["criteria"])}
    return {"id": int(identity), "parts": {"essential": list(essential)}, "questions": out}


def checked_answers(
    response: dict[str, Any], request: dict[str, Any]
) -> dict[str, tuple[str, float, dict[str, float]]]:
    """{question: (choice, probability, probabilities)} for every question, or raise GuardRejection."""
    try:
        if type(response.get("id")) is not int or response["id"] != request["id"]:
            raise GuardRejection("Decision response id does not match the request")
        if response.get("trimmed") != []:
            raise GuardRejection("Decision response trimmed context or omitted its trimmed audit")
        state = "\n".join(
            text
            for section in ("essential", "details", "memory", "logs")
            for text in request["parts"].get(section, [])
        )
        if response.get("stateText") != state:
            raise GuardRejection("Decision response stateText differs from the complete submitted context")
        audits = response.get("tokenAudit")
        if not isinstance(audits, dict) or set(audits) != set(request["questions"]):
            raise GuardRejection("Decision response tokenAudit does not cover the submitted questions")
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(request["questions"]):
            raise GuardRejection("Decision response does not answer exactly the submitted questions")
        result = {}
        for key, question in request["questions"].items():
            audit = audits[key]
            if not isinstance(audit, dict) or audit.get("stateTruncated") is not False:
                raise GuardRejection(f"tokenAudit reports truncated or unaudited context for {key}")
            if type(audit.get("options")) is not int or audit["options"] != len(question["criteria"]):
                raise GuardRejection(f"tokenAudit has the wrong option count for {key}")
            for field in ("stateTokens", "stateRoom", "inputTokens", "maxInputTokens"):
                value = audit.get(field)
                if value is not None and (type(value) is not int or value < 0):
                    raise GuardRejection(f"tokenAudit has an invalid {field} for {key}")
            size, room, total, limit = (
                audit.get(f) for f in ("stateTokens", "stateRoom", "inputTokens", "maxInputTokens")
            )
            if (
                (size is not None and room is not None and size > room)
                or (size is not None and total is not None and size > total)
                or (total is not None and limit is not None and total > limit)
                or (state and any(value == 0 for value in (size, room, total, limit)))
            ):
                raise GuardRejection(f"tokenAudit contradicts complete context for {key}")
            offered = question["criteria"]
            try:
                answer = answers[key]
                choice, probabilities = answer["choice"], answer["probabilities"]
            except (KeyError, TypeError) as error:
                raise GuardRejection(f"Decision response has no distribution for {key}") from error
            if (
                not isinstance(probabilities, dict)
                or set(probabilities) != set(offered)
                or not isinstance(choice, str)
                or choice not in offered
            ):
                raise GuardRejection(f"Decision response does not cover exactly the offered options of {key}")
            values = list(probabilities.values())
            if any(
                isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
                for p in values
            ):
                raise GuardRejection(f"Invalid model probabilities for {key}")
            if abs(math.fsum(values) - 1) > len(offered) * 0.00005 + 1e-8:
                raise GuardRejection(f"Model probabilities for {key} do not sum to one within rounding precision")
            if probabilities[choice] != max(values):
                raise GuardRejection(f"Chosen option of {key} is not the model's argmax")
            result[key] = (choice, float(probabilities[choice]), {k: float(v) for k, v in probabilities.items()})
        return result
    except GuardRejection:
        raise
    except Exception as error:
        raise GuardRejection(f"Malformed decision response: {type(error).__name__}: {error}") from error
