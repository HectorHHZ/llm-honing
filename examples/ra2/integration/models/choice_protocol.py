"""Shared choice-only boundary for experimental native decision-model adapters."""

import copy
import hashlib
import json
import math

SECTIONS = ("essential", "details", "memory", "logs")


def validate_questions(questions):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Expected at least one choice question")
    for key, question in questions.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Question IDs must be nonempty strings")
        if not isinstance(question, dict) or set(question) != {"type", "instructions", "criteria"}:
            raise ValueError("Unexpected question fields")
        if question["type"] != "choice" or not isinstance(question["instructions"], str):
            raise ValueError("Only explicit choice questions are supported")
        criteria = question["criteria"]
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError("Expected 2-255 ordered choices")
        if not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in criteria.items()):
            raise ValueError("Every choice needs an ID and a description")


def checked_answer(criteria, probabilities, choice=None):
    """Preserve offered key order; reject partial/invalid distributions and choices."""
    keys = list(criteria)
    if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
        raise ValueError("Model probabilities must cover exactly the offered choices")
    values = [probabilities[key] for key in keys]
    if not all(
        isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1 for p in values
    ):
        raise ValueError("Invalid model probability")
    if abs(math.fsum(values) - 1) > 1e-5:
        raise ValueError("Model probabilities do not sum to one")
    best = keys[max(range(len(keys)), key=values.__getitem__)]
    if choice is not None and choice != best:
        raise ValueError("Model choice differs from deterministic argmax in offered order")
    return {"choice": best, "probabilities": dict(zip(keys, values, strict=True))}


def canonical_ids(value):
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one encoded prompt")
        value = value[0]
    if not isinstance(value, list) or not value or any(type(x) is not int or x < 0 for x in value):
        raise ValueError("Invalid encoded prompt IDs")
    return value


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def pack_request(parts, questions, measure, max_length):
    """Prune optional whole records only; keep essential state and all choices."""
    validate_questions(questions)
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("Invalid input length budget")
    if not isinstance(parts, dict) or set(parts) - set(SECTIONS):
        raise ValueError("Unknown context sections")
    retained = {}
    for section in SECTIONS:
        rows = parts.get(section, [])
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise ValueError("Context sections must contain complete text records")
        retained[section] = list(rows)
    if not retained["essential"]:
        raise ValueError("Essential normal context is required")
    trimmed = []
    while True:
        state = "\n".join(row for section in SECTIONS for row in retained[section])
        measurements = measure(state, questions)
        if set(measurements) != set(questions):
            raise ValueError("Native encoding omitted a question")
        audit = {}
        for key, count in measurements.items():
            total, state_tokens = count["inputTokens"], count["stateTokens"]
            if type(total) is not int or type(state_tokens) is not int or not 0 <= state_tokens <= total:
                raise ValueError("Invalid native token accounting")
            audit[key] = {
                **count,
                "stateRoom": max_length - (total - state_tokens),
                "options": len(questions[key]["criteria"]),
                "stateTruncated": False,
                "maxInputTokens": max_length,
            }
        if all(a["inputTokens"] <= max_length for a in audit.values()):
            return state, audit, trimmed
        for section in ("logs", "memory", "details"):
            if retained[section]:
                row = retained[section].pop(0 if section == "logs" else -1)
                trimmed.append({"section": section, "text": row})
                break
        else:
            raise ValueError("Essential context and complete choices exceed native input budget")


def nano_payload(state, questions):
    return {"states": [{"id": "decision", "state": state, "questions": copy.deepcopy(questions)}]}


def omni_messages(state, question):
    """The pinned upstream loader's numbered choice template, including its suffix."""
    options = list(question["criteria"].values())
    choices = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(options))
    prompt = (
        f"{state}\n\n---\n\nQUESTION: {question['instructions']}\n\nOPTIONS:\n{choices}\n\n"
        f"Reply with only the number of the correct option (1-{len(options)}).\n"
        "Output a single number and nothing else."
    )
    return [{"role": "user", "content": prompt}]
