"""Shared constants, file helpers and the decision-response guard of the Exp 0 v0 episode contract.

Standard library only. See CONTRACT.md for the meaning of every field. Game collectors import
``guard_response`` (validate a decision server reply before any native action) and the file helpers;
the common tools import everything else.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:  # the common tools are flat modules in one directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema_lite import load_schema

FORMAT = "honing-exp0-episode-v0"
CONTRACT_VERSION = "0.2.0"
CONDITIONS = ("no_injection", "harness", "weights", "controller")
COMPONENTS = ("harness", "weights", "controller")
SHAPES = ("native", "option")
ROW_KINDS = ("model", "forced", "auto")
OUTCOME_STATUSES = ("win", "loss", "timeout", "other_terminal")
EVENT_KINDS = ("damage", "death", "heal", "kill", "goal", "checkpoint", "room", "pickup", "terminal", "other")
# The pinned Laya server (`honing.backends.jev.token_budget.prepare_request`) reads only these sections,
# in this order, and joins their lines with "\n" into the state text it encodes. Any other part name is
# silently ignored by the model, so the guard refuses it.
LAYA_SECTIONS = ("essential", "details", "memory", "logs")
# Pinned Laya rounds each option probability to four decimal places: at most half a unit per option.
ROUNDING_PER_OPTION = 0.5e-4

HERE = Path(__file__).resolve().parent
SCHEMA_DIR = HERE / "schemas"


# ----------------------------------------------------------------------------------------------- files
def canonical(value: Any) -> str:
    """One-line JSON, keys in insertion order (never sorted), no NaN, UTF-8 kept: every JSONL line's form.

    Key order is data here: the order of a request's options (``criteria``) is part of what Laya encodes, and
    the same options in another order get other probabilities (measured on a logged HK request: max |dp|
    0.0358 after a sorted-key rewrite). So rows are never written with ``sort_keys``.
    """
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def same_request(left: Any, right: Any) -> bool:
    """Order-sensitive equality of two requests (dict ``==`` ignores the order of options, Laya does not)."""
    return left is not None and right is not None and canonical(left) == canonical(right)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path: Path, rows: Iterable[Any], mode: str = "x") -> None:
    with Path(path).open(mode, encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_files(paths: Iterable[Path], root: Path) -> str:
    """SHA-256 over (relative path, file SHA-256) pairs: one pin for a frozen code or harness snapshot."""
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def fingerprint_tree(
    root: Path, suffixes: tuple[str, ...] = (".py", ".json", ".txt", ".md", ".cs", ".js", ".yaml")
) -> str:
    root = Path(root)
    return fingerprint_files(
        [p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts], root
    )


# ------------------------------------------------------------------------------------------- schemas
def schema(name: str) -> dict[str, Any]:
    return load_schema(SCHEMA_DIR / f"{name}.schema.json")


# --------------------------------------------------------------------------------------------- guard
class ResponseRejected(ValueError):  # noqa: N818
    """A decision server reply failed the guard. Nothing may be executed for this decision."""


def request_options(request: dict[str, Any], question_key: str) -> list[str]:
    questions = request.get("questions")
    if not isinstance(questions, dict) or set(questions) != {question_key}:
        raise ResponseRejected(f"the request must ask exactly one question named {question_key!r}")
    criteria = questions[question_key].get("criteria")
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise ResponseRejected("the question must offer at least two options")
    return list(criteria)


def laya_state_text(request: dict[str, Any]) -> str:
    parts = request["parts"]
    return "\n".join(text for key in LAYA_SECTIONS for text in parts.get(key, []))


# ------------------------------------------------------------------------- several questions (0.2.0)
# A decision may ask several choice questions in one request (Tank War v0: motion and weapon; full HK with a
# raw-key harness: move and attack). The plan names them in request order (``question_keys``); a plan with one
# question keeps ``question_key`` and every row field keeps its 0.1.x form (a choice is a string, the
# probabilities one option -> p mapping). With several questions the same fields are keyed by question:
# ``probabilities`` {question: {option: p}}, ``server_choice`` / ``model_choice`` / ``executed.option``
# {question: option}. The helpers below give every tool one view of both forms.
def plan_question_keys(plan: dict[str, Any] | None, default: str = "action") -> list[str]:
    """The questions every model request of this plan asks, in request order."""
    if not plan:
        return [default]
    if plan.get("question_keys"):
        return list(plan["question_keys"])
    return [plan["question_key"]]


def request_question_keys(request: dict[str, Any]) -> list[str]:
    questions = request.get("questions") if isinstance(request, dict) else None
    if not isinstance(questions, dict) or not questions:
        raise ResponseRejected("the request asks no question")
    return list(questions)


def multi(keys: list[str] | tuple[str, ...]) -> bool:
    return len(keys) > 1


def keyed(value: Any, keys: list[str] | tuple[str, ...]) -> Any:
    """A row's per-question value ({question: x}) from either form: the 0.1.x single-question form is wrapped."""
    if value is None:
        return None
    if multi(keys):
        return value
    return {keys[0]: value}


def unkeyed(value: dict[str, Any], keys: list[str] | tuple[str, ...]) -> Any:
    """Inverse of :func:`keyed`: the row form of a per-question value."""
    return value if multi(keys) else value[keys[0]]


def format_choice(value: Any) -> str:
    """A choice as text: "right" for one question, "move right, attack forward" for several."""
    if isinstance(value, dict):
        return ", ".join(f"{key} {option}" for key, option in value.items())
    return str(value)


def question_options(request: dict[str, Any], question_keys: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    """The offered options of each asked question; the request must ask exactly these questions, in this order."""
    questions = request.get("questions")
    if not isinstance(questions, dict) or list(questions) != list(question_keys):
        raise ResponseRejected(f"the request must ask exactly the questions {list(question_keys)} in this order")
    out = {}
    for key in question_keys:
        criteria = questions[key].get("criteria") if isinstance(questions[key], dict) else None
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ResponseRejected(f"question {key!r} must offer at least two options")
        out[key] = list(criteria)
    return out


def guard_decision(
    request: dict[str, Any],
    raw: bytes | str | dict[str, Any],
    question_keys: list[str] | tuple[str, ...],
    *,
    server: str = "laya",
) -> dict[str, Any]:
    """Validate one ``/v1/decide`` reply to a request that asks one or several questions.

    The checks of :func:`guard_response`, applied to every asked question (one failed check rejects the whole
    reply: nothing may be executed). Returns ``{"questions": {key: {"probabilities", "normalized",
    "server_choice"}}, "reply": <parsed>}``.
    """
    if isinstance(raw, (bytes, str)):
        try:
            reply = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise ResponseRejected(f"the reply is not JSON: {error}") from None
    else:
        reply = raw
    if not isinstance(reply, dict):
        raise ResponseRejected("the reply is not a JSON object")
    options = question_options(request, question_keys)
    if "id" in request and reply.get("id") != request["id"]:
        raise ResponseRejected(f"the reply id {reply.get('id')!r} does not echo the request id {request['id']!r}")
    if server == "laya":
        unknown = set(request.get("parts", {})) - set(LAYA_SECTIONS)
        if unknown:
            raise ResponseRejected(f"the model ignores request parts {sorted(unknown)}; use {LAYA_SECTIONS}")
        if reply.get("stateText") != laya_state_text(request):
            raise ResponseRejected("the reply's stateText differs from the request's state: incomplete context")
        if reply.get("trimmed") != []:
            raise ResponseRejected(f"the server trimmed state entries: {reply.get('trimmed')!r}")
        audit = reply.get("tokenAudit")
        if not isinstance(audit, dict) or set(audit) != set(question_keys):
            raise ResponseRejected("the reply has no token audit for exactly the asked questions")
        for key in question_keys:
            entry = audit[key]
            if not isinstance(entry, dict) or entry.get("stateTruncated") is not False:
                raise ResponseRejected(f"the token audit does not confirm an untruncated state for {key!r}")
            if entry.get("options") != len(options[key]):
                raise ResponseRejected(f"the token audit does not cover every offered option of {key!r}")
    answers = reply.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(question_keys):
        raise ResponseRejected("the reply does not answer exactly the asked questions")
    out: dict[str, dict[str, Any]] = {}
    for key in question_keys:
        answer = answers[key]
        if not isinstance(answer, dict):
            raise ResponseRejected(f"the reply does not answer question {key!r}")
        probabilities, choice = answer.get("probabilities"), answer.get("choice")
        if not isinstance(probabilities, dict) or set(probabilities) != set(options[key]):
            raise ResponseRejected(f"the probabilities of {key!r} do not cover exactly the offered options")
        for value in probabilities.values():
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ResponseRejected(f"invalid probability {value!r} for {key!r}")
        total = math.fsum(probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=len(options[key]) * ROUNDING_PER_OPTION + 1e-12):
            raise ResponseRejected(f"the probabilities of {key!r} sum to {total}, not one")
        if choice not in options[key] or probabilities[choice] < max(probabilities.values()):
            raise ResponseRejected(
                f"the server choice {choice!r} of {key!r} is not an offered option of maximal probability"
            )
        out[key] = {
            "probabilities": dict(probabilities),
            "normalized": {option: probabilities[option] / total for option in options[key]},
            "server_choice": choice,
        }
    return {"questions": out, "reply": reply}


def plan_faults(plan: dict[str, Any], component: str) -> list[dict[str, Any]]:
    """The plan's validated fault definitions of one component (0.2.0: one object or a list of them)."""
    value = (plan.get("faults") or {}).get(component)
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def plan_fault_instances(plan: dict[str, Any]) -> list[str]:
    return [fault["instance"] for component in COMPONENTS for fault in plan_faults(plan, component)]


def max_question_difference(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]) -> float:
    """The largest option probability difference over all questions ({question: {option: p}} on both sides)."""
    if set(left) != set(right):
        raise ValueError("probability vectors cover different questions")
    return max(max_abs_difference(left[key], right[key]) for key in left)


def guard_response(
    request: dict[str, Any], raw: bytes | str | dict[str, Any], question_key: str = "action", *, server: str = "laya"
) -> dict[str, Any]:
    """Validate one ``/v1/decide`` reply against the request it answers; return the parsed decision.

    Raises :class:`ResponseRejected` (with the reason) when the reply must not be executed. Returns
    ``{"probabilities": <as returned>, "normalized": <sum 1>, "server_choice": str, "reply": <parsed>}``.
    Checks: the reply parses to an object; it echoes the request ``id``; it answers exactly the asked
    question; the probabilities cover exactly the offered options, are finite in [0, 1] and sum to one
    within Laya's 4-decimal rounding; the server's choice is an offered option with maximal probability;
    and (server ``laya``) the model saw the complete state: only Laya's four sections are used, the
    reply's ``stateText`` equals their joined lines, nothing was ``trimmed`` and no question's state was
    truncated.
    """
    if isinstance(raw, (bytes, str)):
        try:
            reply = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise ResponseRejected(f"the reply is not JSON: {error}") from None
    else:
        reply = raw
    if not isinstance(reply, dict):
        raise ResponseRejected("the reply is not a JSON object")
    options = request_options(request, question_key)
    if "id" in request and reply.get("id") != request["id"]:
        raise ResponseRejected(f"the reply id {reply.get('id')!r} does not echo the request id {request['id']!r}")
    if server == "laya":
        unknown = set(request.get("parts", {})) - set(LAYA_SECTIONS)
        if unknown:
            raise ResponseRejected(f"the model ignores request parts {sorted(unknown)}; use {LAYA_SECTIONS}")
        if reply.get("stateText") != laya_state_text(request):
            raise ResponseRejected("the reply's stateText differs from the request's state: incomplete context")
        if reply.get("trimmed") != []:
            raise ResponseRejected(f"the server trimmed state entries: {reply.get('trimmed')!r}")
        audit = reply.get("tokenAudit")
        if not isinstance(audit, dict) or set(audit) != {question_key}:
            raise ResponseRejected("the reply has no token audit for the asked question")
        entry = audit[question_key]
        if not isinstance(entry, dict) or entry.get("stateTruncated") is not False:
            raise ResponseRejected("the token audit does not confirm an untruncated state")
        if entry.get("options") != len(options):
            raise ResponseRejected("the token audit does not cover every offered option")
    answers = reply.get("answers")
    if not isinstance(answers, dict) or set(answers) != {question_key} or not isinstance(answers[question_key], dict):
        raise ResponseRejected("the reply does not answer exactly the asked question")
    answer = answers[question_key]
    probabilities, choice = answer.get("probabilities"), answer.get("choice")
    if not isinstance(probabilities, dict) or set(probabilities) != set(options):
        raise ResponseRejected("the probabilities do not cover exactly the offered options")
    for value in probabilities.values():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ResponseRejected(f"invalid probability {value!r}")
    total = math.fsum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=len(options) * ROUNDING_PER_OPTION + 1e-12):
        raise ResponseRejected(f"the probabilities sum to {total}, not one")
    if choice not in options or probabilities[choice] < max(probabilities.values()):
        raise ResponseRejected(f"the server choice {choice!r} is not an offered option of maximal probability")
    return {
        "probabilities": dict(probabilities),
        "normalized": {option: probabilities[option] / total for option in options},
        "server_choice": choice,
        "reply": reply,
    }


class Sampler:
    """The one sampling rule of the contract (``selection.rule == "sample"``).

    One ``random.Random(seed)`` per episode; for every executed model decision, in order, one draw
    ``choices(options, weights)`` over the request's options in their request order, weighted by the
    normalized probabilities. The validator repeats the draws to check every recorded model choice.
    """

    def __init__(self, seed: int) -> None:
        import random

        self._rng = random.Random(seed)

    def choose(self, request: dict[str, Any], probabilities: dict[str, float], question_key: str = "action") -> str:
        options = request_options(request, question_key)
        weights = normalized({option: probabilities[option] for option in options})
        return self._rng.choices(options, weights=[weights[option] for option in options])[0]

    def choose_all(
        self,
        request: dict[str, Any],
        probabilities: dict[str, dict[str, float]],
        question_keys: list[str] | tuple[str, ...],
    ) -> dict[str, str]:
        """Several questions (0.2.0): one draw per question, in the request's question order."""
        options = question_options(request, question_keys)
        out = {}
        for key in question_keys:
            weights = normalized({option: probabilities[key][option] for option in options[key]})
            out[key] = self._rng.choices(options[key], weights=[weights[option] for option in options[key]])[0]
        return out


def normalized(probabilities: dict[str, float]) -> dict[str, float]:
    total = math.fsum(probabilities.values())
    return {key: value / total for key, value in probabilities.items()}


def max_abs_difference(left: dict[str, float], right: dict[str, float]) -> float:
    if set(left) != set(right):
        raise ValueError("probability vectors cover different options")
    return max(abs(left[key] - right[key]) for key in left)


def argmax(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=lambda key: probabilities[key])


# ------------------------------------------------------------------------------------------- service
class ServiceClient:
    """``GET /v1/health`` and ``POST /v1/decide`` of a Laya-compatible decision service, keeping raw bytes."""

    def __init__(self, url: str, timeout_seconds: float = 300.0) -> None:
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.query:
            raise ValueError("a decision service is an http(s) URL without credentials or query")
        self.url = url.rstrip("/")
        self.timeout = timeout_seconds
        self.calls = 0

    def _open(self, path: str, body: bytes | None) -> bytes:
        import urllib.request

        request = urllib.request.Request(
            self.url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read(16 * 1024 * 1024)

    def health(self) -> dict[str, Any]:
        value = json.loads(self._open("/v1/health", None))
        if (
            not isinstance(value, dict)
            or value.get("ready") is not True
            or not isinstance(value.get("metadata"), dict)
        ):
            raise ValueError(f"the decision service at {self.url} is not ready")
        return value

    def checkpoint(self) -> dict[str, Any]:
        """The health metadata, which names the served weights (``weightsSha256``) and ``role``."""
        return dict(self.health()["metadata"])

    def decide_raw(self, request: dict[str, Any]) -> tuple[bytes, float]:
        """POST one request exactly as given; return the raw reply bytes and the wall seconds."""
        import time

        started = time.perf_counter()
        raw = self._open("/v1/decide", json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        self.calls += 1
        return raw, time.perf_counter() - started


# ------------------------------------------------------------------------------------------- history
def public_history_entry(row: dict[str, Any]) -> dict[str, Any]:
    """What the clean harness may use of an earlier row when it re-renders a later decision (public only)."""
    executed = row.get("executed") or {}
    return {
        "index": row["index"],
        "kind": row["kind"],
        "model_choice": row.get("model_choice"),
        "executed_option": executed.get("option"),
        "native_actions": [window["native_action"] for window in executed.get("windows", [])],
        "native_before": row.get("native_before"),
        "native_after": row.get("native_after"),
        "events": [event for event in row.get("events", []) if event.get("visible")],
    }


# -------------------------------------------------------------------------------------------- runs
def load_run(run: Path) -> dict[str, Any]:
    """All contract files of one attempt directory (missing optional files are ``None``)."""
    run = Path(run)
    loaded: dict[str, Any] = {"dir": run, "name": run.name}
    for name in ("manifest", "result", "error"):
        path = run / f"{name}.json"
        loaded[name] = read_json(path) if path.is_file() else None
    loaded["rows"] = read_rows(run / "decisions.jsonl") if (run / "decisions.jsonl").is_file() else []
    states = run / "public_states.jsonl"
    loaded["public_states"] = read_rows(states) if states.is_file() else None
    return loaded


def model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["kind"] == "model"]
