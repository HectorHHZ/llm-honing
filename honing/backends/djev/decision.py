"""DiffusionGemma decisions through a djev deployment, behind Honing's DecisionModel interface.

:class:`DjevDecisionModel` answers Honing's decision requests (the four state
sections and choice questions) with a djev deployment's ``POST /v1/request``.
It starts the deployment itself (:func:`~honing.backends.djev.runtime.djev_runtime`)
or connects to one already running. The HTTP decision server and the RA2
integration's stdio worker serve it like any other family, so a runner reaches
it through ``PolicyEndpoints`` as a ``jev_decision`` endpoint.

The state packs the sections as for the other families and is trimmed by whole
entries (oldest logs first, then the newest memory and details) until djev
accepts it: first to djev's state limit in characters, then, while djev
reports that the prompt and answer canvas exceed the model's context, one entry
at a time. Any other refusal is an error. An overrun uses the wording
``RA2Grader`` scores as 0. Every request is one joint read with seed 0 and one
sample, so an answer does not depend on the requests before it.

The token audit counts the state with DiffusionGemma's tokenizer; the input
is the prompt length djev reports; the room is the context limit less the
answer canvas and the rest of the prompt. djev reads DiffusionGemma without
training it, so this family has no trainer.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.backends.djev.runtime import DjevRuntimeSettings, djev_runtime
from honing.backends.jev.kev import QUESTION_OVERRUN, STATE_OVERRUN
from honing.backends.jev.token_budget import SECTIONS

FAMILY = "djev"
CONTEXT_REFUSAL = "exceed the model context limit"
OPTIONS = {"seed": 0, "samples": 1, "isolation": "joint", "diagnostics": True}


@dataclass(frozen=True)
class DjevSettings:
    """A djev deployment to start (``runtime``) or to reach (``url``), and the pins it must match.

    With ``url`` set, the deployment is not started or verified here; its
    ``/config`` and the tokenizer at ``tokenizer`` (the pinned model snapshot)
    are still read.
    """

    pins_path: Path
    runtime: DjevRuntimeSettings | None = None
    url: str | None = None
    tokenizer: Path | None = None
    model: str = "diffusiongemma-26b-a4b-it"
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if (self.runtime is None) == (self.url is None):
            raise ValueError("give either a djev runtime to start or the url of a running one")


class DjevRefusedError(ValueError):
    """djev refused a request (HTTP 422); the message is djev's."""


class DjevClient:
    """The djev API's ``/ready``, ``/config``, and ``/v1/request``, over HTTP."""

    def __init__(self, url: str, timeout_seconds: float = 300.0) -> None:
        self.url = url.rstrip("/")
        self._timeout = timeout_seconds

    def _call(self, path: str, body: Mapping[str, object] | None = None) -> Any:
        data = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        request = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            message = json.loads(error.read() or b"{}").get("error", {})
            text = message.get("message", "") if isinstance(message, dict) else str(message)
            if error.code == 422:
                raise DjevRefusedError(text) from None
            raise RuntimeError(f"djev {path} returned {error.code}: {text}") from None

    def config(self) -> dict[str, Any]:
        value: dict[str, Any] = self._call("/config")
        return value

    def ready(self) -> bool:
        try:
            self._call("/ready")
        except (RuntimeError, OSError):
            return False
        return True

    def request(self, state: str, questions: Mapping[str, object]) -> dict[str, Any]:
        value: dict[str, Any] = self._call(
            "/v1/request", {"state": state, "questions": dict(questions), "options": OPTIONS}
        )
        return value


def validate_questions(questions: Mapping[str, Any], limits: Mapping[str, int]) -> None:
    """Choice questions only, each within djev's character limits; a question over a limit is the harness's overrun."""
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("Expected at least one choice question")
    for key, question in questions.items():
        if not isinstance(question, Mapping) or set(question) != {"type", "instructions", "criteria"}:
            raise ValueError("Unexpected question fields")
        criteria = question["criteria"]
        if question["type"] != "choice" or not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= 255:
            raise ValueError("Only choice questions with 2-255 options are supported")
        longest = max(max(len(str(k)), len(str(v))) for k, v in criteria.items())
        if (
            len(question["instructions"]) > limits["instructions_characters"]
            or longest > limits["criterion_characters"]
        ):
            detail = (
                f"instructions {len(question['instructions'])}/{limits['instructions_characters']} characters; "
                f"longest option {longest}/{limits['criterion_characters']}"
            )
            raise ValueError(QUESTION_OVERRUN.format(key=key, detail=detail))


def checked_answer(criteria: Mapping[str, Any], answer: Any) -> dict[str, Any]:
    """djev's choice answer, checked: a distribution over exactly the offered options whose argmax is the choice."""
    keys = list(criteria)
    probabilities = answer.get("probabilities") if isinstance(answer, Mapping) else None
    if answer.get("type") != "choice" or not isinstance(probabilities, Mapping) or set(probabilities) != set(keys):
        raise ValueError("djev's probabilities must cover exactly the offered choices")
    values = [probabilities[key] for key in keys]
    if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in values):
        raise ValueError("djev returned an invalid probability")
    if abs(math.fsum(values) - 1) > 1e-4:
        raise ValueError("djev's probabilities do not sum to one")
    if probabilities[answer.get("choice")] + 1e-9 < max(values):
        raise ValueError("djev's choice is not the most probable option")
    return {"choice": answer["choice"], "probabilities": {key: probabilities[key] for key in keys}}


class DjevDecisionModel:
    """A djev deployment as a :class:`~honing.backends.jev.model.DecisionModel`; create it with :meth:`load`."""

    def __init__(self, client: DjevClient, tokenizer: Any, limits: Mapping[str, int], metadata: Mapping[str, object]):
        self._client = client
        self._tokenizer = tokenizer
        self._limits = dict(limits)
        self._metadata = dict(metadata)

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    @classmethod
    @contextlib.contextmanager
    def load(cls, settings: DjevSettings) -> Iterator[DjevDecisionModel]:
        pins = json.loads(settings.pins_path.read_text())
        pin = pins["djevModels"][settings.model]
        with contextlib.ExitStack() as stack:
            if settings.runtime is not None:
                deployment = stack.enter_context(djev_runtime(settings.runtime))
                url, environment = deployment["url"], deployment["environment"]
                if (environment["model"], environment["modelRevision"]) != (pin["repo"], pin["revision"]):
                    raise ValueError("the djev deployment serves another model than the one selected")
                max_model_len = settings.runtime.max_model_len
                extra = {"gpu": settings.runtime.gpu, "gpuUuid": deployment["gpuUuid"], "runtime": environment}
            else:
                url, extra, max_model_len = str(settings.url), {"deployment": "external"}, 32768
            client = DjevClient(url, settings.timeout_seconds)
            config = client.config()
            with contextlib.redirect_stdout(sys.stderr):
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    str(settings.tokenizer) if settings.tokenizer else pin["repo"],
                    revision=None if settings.tokenizer else pin["revision"],
                    local_files_only=True,
                    trust_remote_code=False,
                )
            metadata = {
                "family": FAMILY,
                "model": settings.model,
                "repo": pin["repo"],
                "revision": pin["revision"],
                "sourceRevision": pins["djevSource"]["revision"],
                "url": url,
                "djevModel": config.get("model"),
                "limits": config.get("limits"),
                "maxModelLen": max_model_len,
                "options": OPTIONS,
                **extra,
            }
            yield cls(client, tokenizer, config["limits"], metadata)

    def _tokens(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        started = time.perf_counter()
        validate_questions(questions, self._limits)
        retained = {key: list(parts.get(key, [])) for key in SECTIONS}  # type: ignore[call-overload]
        trimmed: list[dict[str, str]] = []

        def trim() -> None:
            for section in ("logs", "memory", "details"):
                if retained[section]:
                    text = retained[section].pop(0 if section == "logs" else -1)
                    trimmed.append({"section": section, "text": text})
                    return
            raise ValueError(STATE_OVERRUN)

        while True:
            state = "\n".join(text for key in SECTIONS for text in retained[key])
            if len(state) > self._limits["state_characters"]:
                trim()
                continue
            try:
                response = self._client.request(state, questions)
            except DjevRefusedError as refused:
                if CONTEXT_REFUSAL not in str(refused):
                    raise ValueError(f"djev refused the request: {refused}") from None
                trim()
                continue
            break
        answers = {key: checked_answer(q["criteria"], response["answers"].get(key)) for key, q in questions.items()}  # type: ignore[index]
        prompt_tokens = int(response["usage"]["input_tokens"])
        canvas = int(response.get("diagnostics", {}).get("canvas_tokens", 0))
        state_tokens = self._tokens(state)
        room = int(self._metadata["maxModelLen"]) - canvas - (prompt_tokens - state_tokens)  # type: ignore[call-overload]
        audit = {
            key: {
                "stateTokens": state_tokens,
                "stateRoom": room,
                "inputTokens": prompt_tokens,
                "canvasTokens": canvas,
                "stateCharacters": len(state),
                "options": len(q["criteria"]),  # type: ignore[index]
                "stateTruncated": False,
            }
            for key, q in questions.items()
        }
        return {
            "answers": answers,
            "stateText": state,
            "tokenAudit": audit,
            "trimmed": trimmed,
            "modelResult": response,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }
