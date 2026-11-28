"""Counterfactual probes: ask a decision model again on logged decisions, with one thing changed.

#138's second claim is that System 2 can test a diagnosis instead of only
stating it. A probe does the test. It takes an episode's decision traces,
rebuilds the ``/v1/decide`` request each decision sent (the rendered parts and
the questions, exactly as logged), changes one thing, asks again, and reports
where the probability of a target option went. The one thing is one of:

- the text, a :class:`TextVariant`: edits to the rendered state, or to the
  question and option wording, such as a field removed or restored;
- the rendering, a :class:`SettingsVariant`: each decision rendered again from
  its raw state under other harness settings, through a
  :class:`ContextRenderer` (a source with raw states, such as #169's tank
  states, supplies one);
- the weights, a :class:`WeightsVariant`: the logged requests answered by a
  server holding other weights, such as a ``direction_edit`` checkpoint.

The target is an option per question: given as a mapping, chosen per decision
by a callable (for example a teacher's label), or by default the option the
model gave the most probability when the decision was logged.

Every probe checks itself before it measures anything:

- **Weights.** The server answering for the traces' own weights must report
  their checksum (``weightsSha256`` in its ``/v1/health`` metadata), the one
  the traces carry (``DecisionTrace.policy_checksum``) or the caller names.
  A weights variant's server must report the variant's checksum, and it must
  be another checksum than the traces'.
- **Identity.** The logged requests asked again unchanged must reproduce the
  logged probabilities within ``tolerance``, with no change of the most
  probable option except where the logged top two were within ``tie``. A
  probe whose identity fails would measure the server instead of the variant,
  so it is refused (:class:`IdentityError`).

Requests go to the server in batches: one ``/v1/decide`` call carries a
request and up to the server's ``maxBranches`` more as branches, each answered
as if it had been sent alone.

This module imports only the standard library and :mod:`honing.core`, as the
trace loader and renderer do.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from honing.core import ContractError, DecisionTrace, require

DEFAULT_TOLERANCE = 0.01
DEFAULT_TIE = 0.01
# The metadata key a decision server uses for the checksum of the weights it serves.
CHECKSUM_KEY = "weightsSha256"

Request = dict[str, Any]
# Per question, the probability of every option.
Answer = dict[str, dict[str, float]]
Target = Mapping[str, str] | Callable[[DecisionTrace, str], str | None] | None


class IdentityError(ContractError):
    """The server did not reproduce the logged probabilities, so a probe against it would not measure the variant."""

    def __init__(self, report: IdentityReport) -> None:
        worst = (
            f" The first change of the most probable option beyond a tie: {report.flips_beyond_ties[0]}."
            if report.flips_beyond_ties
            else ""
        )
        super().__init__(
            f"the identity probe over {report.decisions} decisions differs from the logged probabilities by up to"
            f" {report.max_difference:.6f} (tolerance {report.tolerance:g}), and the most probable option changed"
            f" on {len(report.flips_beyond_ties)} questions whose logged top two were {report.tie:g} or more apart."
            f"{worst}"
        )
        self.report = report


def decide_request(trace: DecisionTrace) -> Request:
    """The ``/v1/decide`` request the decision sent: its rendered parts and its questions, as logged."""
    sample = trace.decision
    return {
        "parts": {name: list(lines) for name, lines in sample.parts},
        "questions": {
            key: {"type": question.kind, "instructions": question.instructions, "criteria": dict(question.options)}
            for key, question in sample.questions
        },
    }


def logged_answer(trace: DecisionTrace) -> Answer:
    """The probabilities the model gave when the decision was logged."""
    probs = trace.decision.behavior_probs or ()
    return {key: dict(options) for key, options in probs}


class DecideClient:
    """A decision server's ``/v1/health`` and batched ``/v1/decide``, over HTTP.

    ``batch`` caps the requests per call. The default is one request plus as
    many branches as the server's health says it takes (``maxBranches``).
    ``calls`` counts the ``/v1/decide`` calls made.
    """

    def __init__(self, url: str, *, timeout_seconds: float = 300.0, batch: int | None = None) -> None:
        require(timeout_seconds > 0, "timeout_seconds must be positive")
        require(batch is None or batch >= 1, f"batch must be at least 1, got {batch}")
        self.url = url.rstrip("/")
        self._timeout = timeout_seconds
        self._batch = batch
        self._health: Mapping[str, Any] | None = None
        self.calls = 0

    def _http(self, path: str, body: Mapping[str, Any] | None = None) -> Any:
        data = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise ContractError(f"{self.url}{path} answered {error.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise ContractError(f"cannot reach the decision server at {self.url}: {error}") from None

    def health(self) -> Mapping[str, Any]:
        """The server's health, asked once. A server that is not ready is refused."""
        if self._health is None:
            health = self._http("/v1/health")
            require(
                isinstance(health, dict) and health.get("ready") is True,
                f"the decision server at {self.url} is not ready: {health}",
            )
            self._health = health
        return self._health

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self.health().get("metadata") or {})

    @property
    def checksum(self) -> str | None:
        """The checksum of the weights the server holds, or ``None`` when its metadata names none."""
        value = self.metadata.get(CHECKSUM_KEY)
        return str(value) if value else None

    @property
    def batch(self) -> int:
        branches = self.health().get("maxBranches")
        most = 1 + (branches if isinstance(branches, int) and branches >= 0 else 0)
        return most if self._batch is None else min(self._batch, most)

    def answer(self, requests: Sequence[Mapping[str, Any]]) -> list[Answer]:
        """Every request's probabilities, in order, with as many requests per call as the server takes."""
        answers: list[Answer] = []
        size = self.batch
        for start in range(0, len(requests), size):
            chunk = list(requests[start : start + size])
            body: dict[str, Any] = {**chunk[0], "id": start}
            if len(chunk) > 1:
                body["branches"] = {
                    f"r{start + offset}": {**request, "id": start + offset}
                    for offset, request in enumerate(chunk[1:], 1)
                }
            response = self._http("/v1/decide", body)
            self.calls += 1
            replies = [response] + [
                (response.get("branches") or {}).get(f"r{start + offset}") for offset in range(1, len(chunk))
            ]
            for offset, (request, reply) in enumerate(zip(chunk, replies, strict=True)):
                answers.append(_probabilities(reply, request, start + offset))
        return answers


def _probabilities(reply: Any, request: Mapping[str, Any], index: int) -> Answer:
    """One reply's probabilities, checked to cover every option of every question asked."""
    require(isinstance(reply, dict) and isinstance(reply.get("answers"), dict), f"request {index} got no answers")
    answers: Answer = {}
    for key, question in request["questions"].items():
        found = (reply["answers"].get(key) or {}).get("probabilities")
        if not isinstance(found, dict):
            raise ContractError(f"request {index} has no probabilities for question {key!r}")
        probabilities: dict[str, Any] = found
        options = list(question["criteria"])
        require(
            sorted(probabilities) == sorted(options),
            f"request {index}, question {key!r}: probabilities for {sorted(probabilities)}, options {sorted(options)}",
        )
        values = {option: float(probabilities[option]) for option in options}
        require(
            all(map(math.isfinite, values.values())), f"request {index}, question {key!r}: a probability is not finite"
        )
        answers[key] = values
    return answers


@dataclass(frozen=True)
class TextEdit:
    """Replace ``old`` with ``new`` wherever it occurs. An empty ``new`` removes the text, such as a field."""

    old: str
    new: str = ""

    def __post_init__(self) -> None:
        require(bool(self.old), "a text edit needs the text it replaces")
        require(self.old != self.new, f"a text edit that replaces {self.old!r} with itself changes nothing")


@dataclass(frozen=True)
class TextVariant:
    """The logged requests with their text edited, answered by the traces' own weights.

    The edits apply in order to every line of the rendered state, and with
    ``questions`` also to every question's instructions and option
    descriptions. A variant with no edits is the identity probe.
    """

    name: str
    edits: tuple[TextEdit, ...] = ()
    questions: bool = False

    def requests(self, traces: Sequence[DecisionTrace]) -> list[Request]:
        return [self._edit(decide_request(trace)) for trace in traces]

    def _edit(self, request: Request) -> Request:
        def apply(text: str) -> str:
            for edit in self.edits:
                text = text.replace(edit.old, edit.new)
            return text

        parts = {name: [apply(line) for line in lines] for name, lines in request["parts"].items()}
        questions = request["questions"]
        if self.questions:
            questions = {
                key: {
                    **question,
                    "instructions": apply(question["instructions"]),
                    "criteria": {option: apply(text) for option, text in question["criteria"].items()},
                }
                for key, question in questions.items()
            }
        return {"parts": parts, "questions": questions}


IDENTITY = TextVariant("identity")


class ContextRenderer(Protocol):
    """Renders decisions again from the raw state behind them, under settings the renderer holds.

    It returns one ``/v1/decide`` request (``parts`` and ``questions``) per
    trace, in order, asking the same questions with the same options.
    """

    def render(self, traces: Sequence[DecisionTrace]) -> Sequence[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class SettingsVariant:
    """Each decision rendered again under other settings by ``renderer``, answered by the traces' own weights."""

    name: str
    renderer: ContextRenderer

    def requests(self, traces: Sequence[DecisionTrace]) -> list[Request]:
        rendered = list(self.renderer.render(traces))
        require(
            len(rendered) == len(traces),
            f"settings variant {self.name} rendered {len(rendered)} requests for {len(traces)} decisions",
        )
        requests = []
        for index, (trace, request) in enumerate(zip(traces, rendered, strict=True)):
            require(
                isinstance(request.get("parts"), dict) and isinstance(request.get("questions"), dict),
                f"settings variant {self.name}: request {index} needs parts and questions",
            )
            logged = decide_request(trace)["questions"]
            asked = {key: sorted(question["criteria"]) for key, question in request["questions"].items()}
            require(
                asked == {key: sorted(question["criteria"]) for key, question in logged.items()},
                f"settings variant {self.name}: decision {trace.step} of {trace.episode_id} asks other questions"
                f" or options than it logged",
            )
            requests.append({"parts": dict(request["parts"]), "questions": dict(request["questions"])})
        return requests


@dataclass(frozen=True, eq=False)
class WeightsVariant:
    """The logged requests answered by a server holding other weights, whose checksum is ``checksum``."""

    name: str
    client: DecideClient
    checksum: str

    def __post_init__(self) -> None:
        require(bool(self.checksum), "a weights variant names the checksum of its weights")


Variant = TextVariant | SettingsVariant | WeightsVariant


@dataclass(frozen=True)
class IdentityReport:
    """How closely the logged requests, asked again unchanged, reproduced the logged probabilities."""

    decisions: int
    questions: int
    max_difference: float
    flips: int
    flips_beyond_ties: tuple[str, ...]
    tolerance: float
    tie: float
    seconds: float

    @property
    def ok(self) -> bool:
        return self.max_difference <= self.tolerance and not self.flips_beyond_ties

    def to_json(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "questions": self.questions,
            "max_difference": self.max_difference,
            "flips": self.flips,
            "flips_beyond_ties": list(self.flips_beyond_ties),
            "tolerance": self.tolerance,
            "tie": self.tie,
            "ok": self.ok,
            "decisions_per_second": _rate(self.decisions, self.seconds),
        }


@dataclass(frozen=True)
class ProbeRow:
    """One question of one decision: the target option's probability as logged and under the variant."""

    episode_id: str
    step: int
    question: str
    target: str
    before: float
    after: float
    top_before: str
    top_after: str
    max_difference: float
    changed: bool

    @property
    def shift(self) -> float:
        """How far the variant moved the target's probability: positive is toward the target."""
        return self.after - self.before

    @property
    def flipped(self) -> bool:
        """Whether the most probable option changed."""
        return self.top_before != self.top_after

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "question": self.question,
            "target": self.target,
            "before": self.before,
            "after": self.after,
            "shift": self.shift,
            "top_before": self.top_before,
            "top_after": self.top_after,
            "flipped": self.flipped,
            "max_difference": self.max_difference,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class ProbeResult:
    """A variant's effect on every probed question, with the identity check that preceded it."""

    variant: str
    kind: Literal["text", "settings", "weights"]
    policy_checksum: str
    decisions: int
    changed: int
    seconds: float
    rows: tuple[ProbeRow, ...] = field(repr=False)
    identity: IdentityReport | None = None

    @property
    def decisions_per_second(self) -> float | None:
        return _rate(self.decisions, self.seconds)

    def summary(self, *, epsilon: float = 1e-9) -> dict[str, Any]:
        """Counts over the rows: how many moved toward the target, away from it, and changed their top option."""
        shifts = [row.shift for row in self.rows]
        return {
            "rows": len(self.rows),
            "changed_decisions": self.changed,
            "toward_target": sum(shift > epsilon for shift in shifts),
            "away_from_target": sum(shift < -epsilon for shift in shifts),
            "flipped": sum(row.flipped for row in self.rows),
            "flipped_to_target": sum(row.flipped and row.top_after == row.target for row in self.rows),
            "mean_shift": sum(shifts) / len(shifts) if shifts else 0.0,
            "max_difference": max((row.max_difference for row in self.rows), default=0.0),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "kind": self.kind,
            "policy_checksum": self.policy_checksum,
            "decisions": self.decisions,
            "seconds": self.seconds,
            "decisions_per_second": self.decisions_per_second,
            "summary": self.summary(),
            "identity": None if self.identity is None else self.identity.to_json(),
        }


def _rate(count: int, seconds: float) -> float | None:
    return count / seconds if seconds > 0 else None


def _top(probabilities: Mapping[str, float]) -> str:
    return max(probabilities, key=lambda option: probabilities[option])


def _gap(probabilities: Mapping[str, float]) -> float:
    ordered = sorted(probabilities.values(), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else math.inf


def _policy_checksum(traces: Sequence[DecisionTrace], given: str | None) -> str:
    carried = {trace.policy_checksum for trace in traces if trace.policy_checksum is not None}
    require(len(carried) <= 1, f"the traces come from {len(carried)} policies: {sorted(carried)}; probe one at a time")
    if given is not None:
        require(
            not carried or carried == {given},
            f"the traces were made by the weights {next(iter(carried), '')}, not {given}",
        )
        return given
    require(
        bool(carried),
        "the traces name no policy checksum, so the server's weights cannot be checked against them;"
        " name the checksum of the weights that made them",
    )
    return next(iter(carried))


def _require_serves(client: DecideClient, checksum: str, role: str) -> None:
    served = client.checksum
    require(
        served is not None,
        f"{role} at {client.url} reports no {CHECKSUM_KEY} in its metadata, so its weights cannot be checked",
    )
    require(
        served == checksum,
        f"{role} at {client.url} serves the weights {served}, but the probe needs {checksum}",
    )


def identity_report(
    traces: Sequence[DecisionTrace],
    answers: Sequence[Answer],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    tie: float = DEFAULT_TIE,
    seconds: float = 0.0,
) -> IdentityReport:
    """Compare answers to the logged requests with the logged probabilities."""
    worst = 0.0
    flips = 0
    beyond: list[str] = []
    questions = 0
    for trace, answer in zip(traces, answers, strict=True):
        for key, logged in logged_answer(trace).items():
            questions += 1
            now = answer[key]
            worst = max(worst, max(abs(now[option] - logged[option]) for option in logged))
            if _top(now) != _top(logged):
                flips += 1
                if _gap(logged) >= tie:
                    beyond.append(
                        f"{trace.episode_id} decision {trace.step} {key}: {_top(logged)} -> {_top(now)},"
                        f" logged gap {_gap(logged):.4f}"
                    )
    return IdentityReport(len(traces), questions, worst, flips, tuple(beyond), tolerance, tie, seconds)


def _target(target: Target, trace: DecisionTrace, key: str, logged: Mapping[str, float]) -> str | None:
    if target is None:
        return _top(logged)
    if callable(target):
        return target(trace, key)
    return target.get(key)


def probe(
    traces: Sequence[DecisionTrace],
    variant: Variant,
    *,
    server: DecideClient,
    target: Target = None,
    policy_checksum: str | None = None,
    check_identity: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    tie: float = DEFAULT_TIE,
) -> ProbeResult:
    """Ask ``variant`` on ``traces`` and report the target option's probability, logged and under the variant.

    ``server`` serves the weights that made the traces. It answers the identity
    check, and the text and settings variants. ``policy_checksum`` names those
    weights when the traces do not.
    """
    require(len(traces) > 0, "a probe needs at least one decision")
    require(tolerance >= 0 and tie >= 0, "tolerance and tie must be >= 0")
    expected = _policy_checksum(traces, policy_checksum)
    _require_serves(server, expected, "the server for the traces' weights")
    if isinstance(variant, WeightsVariant):
        require(
            variant.checksum != expected,
            f"weights variant {variant.name} names the traces' own weights {expected}; that is the identity probe",
        )
        _require_serves(variant.client, variant.checksum, f"the server for weights variant {variant.name}")

    logged = [decide_request(trace) for trace in traces]
    identity = None
    identity_answers: list[Answer] | None = None
    if check_identity:
        start = time.perf_counter()
        identity_answers = server.answer(logged)
        identity = identity_report(
            traces, identity_answers, tolerance=tolerance, tie=tie, seconds=time.perf_counter() - start
        )
        if not identity.ok:
            raise IdentityError(identity)

    kind: Literal["text", "settings", "weights"]
    if isinstance(variant, WeightsVariant):
        kind, requests, client = "weights", logged, variant.client
    elif isinstance(variant, SettingsVariant):
        kind, requests, client = "settings", variant.requests(traces), server
    else:
        kind, requests, client = "text", variant.requests(traces), server

    start = time.perf_counter()
    if kind == "text" and requests == logged and identity_answers is not None:
        answers, seconds = identity_answers, identity.seconds if identity is not None else 0.0
    else:
        answers = client.answer(requests)
        seconds = time.perf_counter() - start

    rows = []
    changed = 0
    for trace, request, original, answer in zip(traces, requests, logged, answers, strict=True):
        differs = kind == "weights" or request != original
        changed += differs
        for key, before in logged_answer(trace).items():
            option = _target(target, trace, key, before)
            if option is None:
                continue
            require(option in before, f"{trace.episode_id} decision {trace.step} {key} has no option {option!r}")
            after = answer[key]
            rows.append(
                ProbeRow(
                    episode_id=trace.episode_id,
                    step=trace.step,
                    question=key,
                    target=option,
                    before=before[option],
                    after=after[option],
                    top_before=_top(before),
                    top_after=_top(after),
                    max_difference=max(abs(after[o] - before[o]) for o in before),
                    changed=differs,
                )
            )
    return ProbeResult(variant.name, kind, expected, len(traces), changed, seconds, tuple(rows), identity)


__all__ = [
    "CHECKSUM_KEY",
    "DEFAULT_TIE",
    "DEFAULT_TOLERANCE",
    "IDENTITY",
    "ContextRenderer",
    "DecideClient",
    "IdentityError",
    "IdentityReport",
    "ProbeResult",
    "ProbeRow",
    "SettingsVariant",
    "TextEdit",
    "TextVariant",
    "WeightsVariant",
    "decide_request",
    "identity_report",
    "logged_answer",
    "probe",
]
