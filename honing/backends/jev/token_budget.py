"""Encode complete questions and prune optional state before model inference.

The state has four sections. ``essential`` is never dropped; when the encoded
state does not fit, whole entries are removed from ``logs`` (oldest first), then
``memory`` and ``details`` (newest first), and each removal is reported. Every
option of every question must survive encoding intact, or the request fails.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

SECTIONS = ("essential", "details", "memory", "logs")


def prepare_request(
    agent: Any,
    parts: Mapping[str, Any],
    questions: Mapping[str, Any],
    max_len: int,
    head_max_len: int,
    build_sequence: Callable[..., Any],
    render_options: Callable[[Any], list[str]],
) -> tuple[str, dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Return the state text, a per-question token audit, and what was trimmed; fail rather than lose content."""
    sections = SECTIONS
    retained = {key: list(parts.get(key, [])) for key in sections}
    trimmed: list[dict[str, str]] = []
    internal = {key: agent._to_internal(question) for key, question in questions.items()}
    for key, question in internal.items():
        options = render_options(question)
        tokens = [
            agent.tok(" " + option.replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
            for option in options
        ]
        instructions = agent.tok(question["t"] + " question: " + question["ins"], add_special_tokens=False)[
            "input_ids"
        ]
        longest_option = max((len(token) for token in tokens), default=0)
        question_tokens = len(instructions) + sum(1 + len(token) for token in tokens)
        if longest_option > 48 or question_tokens > head_max_len:
            raise ValueError(
                f"Question {key} exceeds lossless option/instruction budget "
                f"(longest option {longest_option}/48 tokens; question {question_tokens}/{head_max_len})"
            )
    while True:
        state = "\n".join(text for key in sections for text in retained[key])
        state_ids = agent.tok(state, add_special_tokens=False)["input_ids"]
        audit: dict[str, dict[str, Any]] = {}
        fits = True
        for key, question in internal.items():
            empty, _ = build_sequence(agent.tok, "", question, max_len, head_max_len, state_ids=[])
            room = max_len - len(empty)
            ids, markers = build_sequence(agent.tok, state, question, max_len, head_max_len, state_ids=state_ids)
            audit[key] = {
                "stateTokens": len(state_ids),
                "stateRoom": room,
                "inputTokens": len(ids),
                "options": len(markers),
                "stateTruncated": len(state_ids) > room,
            }
            if len(markers) != len(render_options(question)):
                raise ValueError("Option marker loss")
            fits &= len(state_ids) <= room
        if fits:
            return state, audit, trimmed
        for section in ("logs", "memory", "details"):
            if retained[section]:
                text = retained[section].pop(0 if section == "logs" else -1)
                trimmed.append({"section": section, "text": text})
                break
        else:
            raise ValueError("Essential state exceeds token budget; refusing silent truncation")
