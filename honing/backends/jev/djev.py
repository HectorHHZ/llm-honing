"""Bridge a djev structured DiffusionGemma server to Honing's decision protocol.

The remote server owns tokenization, diffusion sampling and GPU resources. This
adapter sends complete state, preserves its distributions and reports its token
usage as request-level accounting, never as a local per-question budget.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from honing.backends.jev.token_budget import SECTIONS

FAMILY = "djev"
# The API this adapter was checked against, not a claim about a remote deployment.
CONTRACT_REVISION = "e5841cf41e9211608e698492658685c36e24e77a"


@dataclass(frozen=True)
class DjevSettings:
    url: str
    model: str
    timeout: float = 60.0
    samples: int = 1
    steps: int = 1

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("djev URL must be an HTTP(S) base URL without query or fragment")
        if parsed.username or parsed.password:
            raise ValueError("djev URL must not contain credentials")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("djev needs the upstream served model name")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("djev timeout must be finite and positive")
        for name, limit in (("samples", 32), ("steps", 8)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"djev {name} must be an integer from 1 to {limit}")


def pack_state(parts: Mapping[str, object]) -> str:
    if set(parts) - set(SECTIONS):
        raise ValueError("Unknown context sections")
    lines = []
    for section in SECTIONS:
        rows = parts.get(section, [])
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise ValueError("Context sections must contain complete text records")
        lines.extend(rows)
    if not parts.get("essential"):
        raise ValueError("Essential normal context is required")
    return "\n".join(lines)


def validate_questions(questions: Mapping[str, Any]) -> None:
    if not 1 <= len(questions) <= 64:
        raise ValueError("djev accepts 1-64 questions")
    for key, question in questions.items():
        if not isinstance(key, str) or not key.strip() or key != key.strip() or ":" in key or "\n" in key:
            raise ValueError("djev question IDs must be nonempty, unpadded, and contain no colon or newline")
        if not isinstance(question, dict) or set(question) - {"type", "instructions", "criteria"}:
            raise ValueError("Unexpected question fields (conditional and span questions are not supported)")
        if not isinstance(question.get("instructions"), str):
            raise ValueError("Question instructions must be text")
        kind = question.get("type")
        criteria: Any = question.get("criteria")
        if kind == "choice":
            valid = isinstance(criteria, dict) and 2 <= len(criteria) <= 26
            valid = valid and all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in criteria.items())
        elif kind == "noul":
            valid = criteria is None or (
                isinstance(criteria, dict)
                and set(criteria) <= {"true", "false"}
                and all(isinstance(v, str) for v in criteria.values())
            )
        elif kind == "score":
            valid = isinstance(criteria, list) and 2 <= len(criteria) <= 26
            valid = valid and all(isinstance(v, str) and v for v in criteria) and len(set(criteria)) == len(criteria)
        else:
            raise ValueError("djev adapter supports choice, noul and score questions only")
        if not valid:
            raise ValueError(f"Invalid {kind} criteria; choice/score support 2-26 alternatives")


def _probability(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def checked_answers(questions: Mapping[str, Any], raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != set(questions):
        raise ValueError("djev must answer exactly the requested questions")
    for key, question in questions.items():
        answer, kind = raw[key], question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError("djev answer has the wrong question type")
        if kind == "noul":
            if not _probability(answer.get("noul")):
                raise ValueError("Invalid djev noul probability")
            continue
        keys = list(question["criteria"]) if kind == "choice" else [str(i) for i in range(len(question["criteria"]))]
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(keys) or not all(_probability(p) for p in probs.values()):
            raise ValueError("djev probabilities must cover exactly the offered alternatives")
        if abs(math.fsum(probs.values()) - 1) > 1e-5:
            raise ValueError("djev probabilities do not sum to one")
        if not _probability(answer.get("confidence")) or abs(answer["confidence"] - max(probs.values())) > 1e-5:
            raise ValueError("djev confidence differs from the highest probability")
        if kind == "choice":
            choice = answer.get("choice")
            if choice not in keys or probs[choice] < max(probs.values()):
                raise ValueError("djev choice must be an offered probability maximum")
        else:
            expected = sum(i * probs[str(i)] for i in range(len(keys)))
            if answer.get("legend") != dict(zip(keys, question["criteria"], strict=True)):
                raise ValueError("djev score legend differs from the offered levels")
            score: Any = answer.get("score")
            if type(score) not in (int, float) or not math.isfinite(score) or abs(score - expected) > 1e-5:
                raise ValueError("djev score differs from its probability expectation")
    return raw


class DjevDecisionModel:
    """A remote structured read endpoint, with no local GPU or generated-choice fallback."""

    def __init__(self, settings: DjevSettings) -> None:
        self.settings = settings
        base = settings.url.rstrip("/")
        self._api = base if base.endswith("/v1") else base + "/v1"

    @property
    def metadata(self) -> Mapping[str, object]:
        return {
            "family": FAMILY,
            "model": self.settings.model,
            "upstream": self._api,
            "transport": "djev-systemone",
            "adapterContractRevision": CONTRACT_REVISION,
            "remoteWeightsVerified": False,
            "samples": self.settings.samples,
            "steps": self.settings.steps,
            "tokenAccounting": "upstream-request-total",
            "contextPolicy": "complete-state-no-client-trimming",
        }

    @classmethod
    @contextlib.contextmanager
    def load(cls, settings: DjevSettings) -> Iterator[DjevDecisionModel]:
        model = cls(settings)
        listing = model._request("/models")
        models = listing.get("data", [])
        if not isinstance(models, list) or not any(
            isinstance(m, dict) and m.get("id") == settings.model for m in models
        ):
            raise ValueError("djev /v1/models does not advertise the requested served model")
        yield model

    def _request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            self._api + path,
            data=None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            # Keep the upstream diagnostic; never retry as an invented game action.
            detail = error.read(4096).decode(errors="replace")
            raise RuntimeError(f"djev HTTP {error.code}: {detail}") from error
        if not isinstance(result, dict) or "error" in result:
            raise ValueError("djev returned an error or non-object response")
        return result

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, Any]) -> Mapping[str, object]:
        validate_questions(questions)
        state = pack_state(parts)
        started = time.perf_counter()
        raw = self._request(
            "/systemone",
            {
                "model": self.settings.model,
                "state": state,
                "questions": dict(questions),
                "samples": self.settings.samples,
                "steps": self.settings.steps,
            },
        )
        if raw.get("model") != self.settings.model:
            raise ValueError("djev answered with a different served model")
        answers = checked_answers(questions, raw.get("answers"))
        usage = raw.get("usage")
        tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("djev must report a positive request input token count")
        audit = {
            key: {
                "encoding": "djev-remote-prompt",
                "accounting": "upstream-request-total",
                "inputTokens": tokens,
                "stateTokens": None,
                "stateRoom": None,
                "options": 2 if q["type"] == "noul" else len(q["criteria"]),
                "stateTruncated": False,
            }
            for key, q in questions.items()
        }
        return {
            "answers": answers,
            "modelResult": raw,
            "stateText": state,
            "trimmed": [],
            "tokenAudit": audit,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }
