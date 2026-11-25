"""The records Honing writes to Reef: one per answered decision request, and one report per graded episode.

Reef keeps every answered request as an ``AgentRecord`` and every piece of
feedback as a report that references the inferences it scores (#137). A
decision request has no chat messages, so Honing defines its own payload:

- **A decision record** is one request the decision server answered: its
  ``id``, the state's parts, the questions and the model's answers (a choice
  and a probability for every option). It is an inference record. The server
  writes it, so it holds what the model said, not what the game did.
- **An episode report** is the grade of one episode: the score, the feedback,
  the task, and the episode's decisions in play order. Each decision is the
  record id it references and the policy's choice for it. The choice is the
  training label, and only the game knows it: a rollout draws it from the
  probabilities, and safety can change the motion after the draw. So it
  travels in the report, not in the record. A request whose answer the game
  did not use is recorded but not referenced, and never trains.

Record ids are computed from the episode and the request's ``id``. A resend is
then a retry, which Reef stores once, and a report can name its records
without asking Reef for receipts. The ``episode`` tag must be unique to one
attempt of one episode: Reef refuses the same id with other content.

Tags travel in ``payload.metadata.tags``, where Reef stores the
``x-reef-tag-*`` headers of an inference, so one import batch can carry
records of several episodes.

Follows Reef's ``AgentRecord`` (reef/core/records_types.py), ``ReportPayload``
and ``with_request_tags`` (reef/api/wire.py) and ``import_records``
(reef/api/request_handling.py), Reef bc8c150. Differs: the payload is a
decision, not a chat exchange, and ids are derived, not drawn.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from honing.backends.tank.decisions import decision_from_row
from honing.core import ContractError, DecisionSample

DECISION_FORMAT = "honing-decision"
EPISODE_FORMAT = "honing-episode"
FORMAT_VERSION = 1

#: The tags Honing sets. A campaign's episodes carry the first five. A game
#: outside a campaign tags ``game``, and its scenario names the rest.
TAG_NAMES = ("episode", "task", "release", "harness", "policy", "game")
TAG_HEADER_PREFIX = "x-reef-tag-"

#: Reef's import route takes 1 to 1000 records and at most 10 MiB a request
#: (reef/api/routes/records.py). A tank decision's parts are tens of KB, so the
#: byte limit binds first. The margin covers the client's encoding.
MAX_BATCH_RECORDS = 1000
MAX_BATCH_BYTES = 9 * 1024 * 1024


def _digest(*fields: str) -> str:
    # 32 hex digits, the length of Reef's own uuid4 receipts.
    return hashlib.sha256("\x00".join(fields).encode("utf-8")).hexdigest()[:32]


def decision_record_id(episode: str, request_id: str) -> str:
    """The id of the record of request ``request_id`` in ``episode``."""
    _require_text(episode, "the episode tag")
    _require_text(request_id, "a decision request's id")
    return _digest(DECISION_FORMAT, str(FORMAT_VERSION), episode, request_id)


def episode_report_id(episode: str) -> str:
    """The id of ``episode``'s report, so a resent report is a retry."""
    _require_text(episode, "the episode tag")
    return _digest(EPISODE_FORMAT, str(FORMAT_VERSION), episode)


def tag_headers(tags: Mapping[str, str]) -> dict[str, str]:
    """``tags`` as the ``x-reef-tag-*`` headers of a request that Reef serves and records itself."""
    return {TAG_HEADER_PREFIX + name: value for name, value in _checked_tags(tags).items()}


@dataclass(frozen=True)
class RecordedDecision:
    """One decision request the server answered, as its Reef record holds it.

    ``parts`` maps each state section to its lines, ``questions`` and
    ``answers`` are the request's and the reply's objects as sent, and
    ``branch`` names the branch the request was in, or is ``None`` for the
    request itself.
    """

    request_id: str
    parts: Mapping[str, Sequence[str]]
    questions: Mapping[str, Mapping[str, Any]]
    answers: Mapping[str, Mapping[str, Any]]
    tags: Mapping[str, str]
    branch: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.request_id, "a decision request's id")
        _require_text(self.tags.get("episode"), "a decision record's episode tag")
        object.__setattr__(self, "tags", MappingProxyType(_checked_tags(self.tags)))
        if not isinstance(self.parts, Mapping) or not all(
            isinstance(lines, Sequence) and not isinstance(lines, str) and all(isinstance(x, str) for x in lines)
            for lines in self.parts.values()
        ):
            raise ContractError(f"decision {self.request_id}: parts must map each section to a list of lines")
        if not isinstance(self.questions, Mapping) or not self.questions:
            raise ContractError(f"decision {self.request_id}: questions must be a non-empty object")
        for key, question in self.questions.items():
            criteria = question.get("criteria") if isinstance(question, Mapping) else None
            if not isinstance(criteria, Mapping) or not criteria:
                raise ContractError(f"decision {self.request_id}: question {key} has no criteria")
            answer = self.answers.get(key) if isinstance(self.answers, Mapping) else None
            probabilities = answer.get("probabilities") if isinstance(answer, Mapping) else None
            if not isinstance(probabilities, Mapping) or set(probabilities) != set(criteria):
                raise ContractError(f"decision {self.request_id}: the answer to {key} needs a probability per option")
            if not all(_probability(p) for p in probabilities.values()):
                raise ContractError(f"decision {self.request_id}: the answer to {key} has a probability out of [0, 1]")
        if self.branch is not None:
            _require_text(self.branch, "a branch name")

    @property
    def episode(self) -> str:
        return self.tags["episode"]

    @property
    def record_id(self) -> str:
        return decision_record_id(self.episode, self.request_id)

    def payload(self) -> dict[str, Any]:
        """The record's payload, JSON-native."""
        return {
            "honing": {
                "format": DECISION_FORMAT,
                "version": FORMAT_VERSION,
                "request_id": self.request_id,
                "branch": self.branch,
                "parts": {name: list(lines) for name, lines in self.parts.items()},
                "questions": _plain(self.questions),
                "answers": _plain(self.answers),
            },
            "metadata": {"tags": dict(self.tags)},
        }

    def to_record(self) -> dict[str, Any]:
        """The record as one entry of ``POST /reef/records/batch``."""
        return {"agent_record_id": self.record_id, "request_type": "inference", "payload": self.payload()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RecordedDecision:
        """Read back a payload :meth:`payload` wrote, as Reef returns it. Refuses another format or version."""
        body = _versioned(payload.get("honing"), DECISION_FORMAT)
        keys = {"format", "version", "request_id", "branch", "parts", "questions", "answers"}
        if set(body) != keys:
            raise ContractError(f"a {DECISION_FORMAT} record has keys {sorted(keys)}, got {sorted(body)}")
        metadata = payload.get("metadata")
        tags = metadata.get("tags") if isinstance(metadata, Mapping) else None
        if not isinstance(tags, Mapping):
            raise ContractError(f"a {DECISION_FORMAT} record carries its tags in metadata.tags")
        return cls(
            request_id=body["request_id"],
            parts=body["parts"],
            questions=body["questions"],
            answers=body["answers"],
            tags=tags,
            branch=body["branch"],
        )

    @classmethod
    def from_row(cls, row: Mapping[str, Any], tags: Mapping[str, str]) -> RecordedDecision:
        """The record of the request behind one ``decisions.jsonl`` row: what the server was asked and answered."""
        return cls(
            request_id=row["request_id"],
            parts=row["contextParts"],
            questions=row["questions"],
            answers=row["answers"],
            tags=tags,
        )

    def sample(
        self,
        chosen: Mapping[str, str],
        *,
        reward: float | None,
        group: str | None,
        release_id: str | None,
        harness_id: str | None,
        behavior: bool,
    ) -> DecisionSample:
        """The decision as a training sample, with ``chosen`` from the episode's report.

        Gives what :func:`~honing.backends.tank.decisions.decision_from_row` gives
        for the ``decisions.jsonl`` row this record was made from.
        """
        if set(chosen) != set(self.questions):
            raise ContractError(
                f"decision {self.request_id}: the report chose {sorted(chosen)}, "
                f"the questions are {sorted(self.questions)}"
            )
        for key, option in chosen.items():
            if option not in self.questions[key]["criteria"]:
                raise ContractError(f"decision {self.request_id}: {option!r} is not an option of {key}")
        row = {"contextParts": self.parts, "questions": self.questions, "answers": self.answers, "chosen": chosen}
        return decision_from_row(
            row, reward=reward, group=group, release_id=release_id, harness_id=harness_id, behavior=behavior
        )


@dataclass(frozen=True)
class EpisodeReport:
    """One graded episode, as its Reef report holds it.

    ``decisions`` are the episode's decisions in play order: each is the id of
    the record it references and the policy's choice for every question.
    """

    episode: str
    score: float
    feedback: str | None = None
    task: str | None = None
    decisions: tuple[tuple[str, Mapping[str, str]], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_text(self.episode, "the episode tag")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ContractError(f"episode {self.episode}: the score must be a finite number, got {self.score!r}")
        if self.feedback is not None and not isinstance(self.feedback, str):
            raise ContractError(f"episode {self.episode}: feedback must be text or None")
        if self.task is not None:
            _require_text(self.task, "the task id")
        decisions = tuple((record_id, MappingProxyType(dict(chosen))) for record_id, chosen in self.decisions)
        ids = [record_id for record_id, _ in decisions]
        if len(set(ids)) != len(ids):
            raise ContractError(f"episode {self.episode}: a report references each record once")
        for record_id, chosen in decisions:
            _require_text(record_id, "a referenced record id")
            if not chosen or not all(isinstance(k, str) and isinstance(v, str) for k, v in chosen.items()):
                raise ContractError(f"episode {self.episode}: record {record_id} needs a choice per question")
        object.__setattr__(self, "decisions", decisions)

    @property
    def record_id(self) -> str:
        return episode_report_id(self.episode)

    @property
    def references(self) -> tuple[str, ...]:
        return tuple(record_id for record_id, _ in self.decisions)

    def payload(self) -> dict[str, Any]:
        """The report's payload, as ``POST /reef/report`` takes it."""
        return {
            "score": float(self.score),
            "feedback": self.feedback,
            "references": list(self.references),
            "metadata": {
                "task": self.task,
                "honing": {
                    "format": EPISODE_FORMAT,
                    "version": FORMAT_VERSION,
                    "episode": self.episode,
                    "chosen": [dict(chosen) for _, chosen in self.decisions],
                },
            },
        }

    def to_record(self) -> dict[str, Any]:
        """The report as one entry of ``POST /reef/records/batch``, after the records it references."""
        return {"agent_record_id": self.record_id, "request_type": "report", "payload": self.payload()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EpisodeReport:
        """Read back a payload :meth:`payload` wrote, as Reef returns it."""
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ContractError(f"an {EPISODE_FORMAT} report needs metadata")
        body = _versioned(metadata.get("honing"), EPISODE_FORMAT)
        keys = {"format", "version", "episode", "chosen"}
        if set(body) != keys:
            raise ContractError(f"an {EPISODE_FORMAT} report has keys {sorted(keys)}, got {sorted(body)}")
        references, chosen = payload.get("references"), body["chosen"]
        if not isinstance(references, list) or not isinstance(chosen, list) or len(references) != len(chosen):
            raise ContractError(f"an {EPISODE_FORMAT} report has one choice per reference")
        return cls(
            episode=body["episode"],
            score=payload.get("score"),  # type: ignore[arg-type]
            feedback=payload.get("feedback"),
            task=metadata.get("task"),
            decisions=tuple(zip(references, chosen, strict=True)),
        )

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        episode: str,
        score: float,
        feedback: str | None = None,
        task: str | None = None,
    ) -> EpisodeReport:
        """The report of an episode whose ``decisions.jsonl`` holds ``rows``, in order."""
        return cls(
            episode=episode,
            score=score,
            feedback=feedback,
            task=task,
            decisions=tuple(
                (decision_record_id(episode, row["request_id"]), {key: row["chosen"][key] for key in row["questions"]})
                for row in rows
            ),
        )


def batches(
    records: Iterable[Mapping[str, Any]],
    *,
    max_records: int = MAX_BATCH_RECORDS,
    max_bytes: int = MAX_BATCH_BYTES,
) -> Iterator[list[Mapping[str, Any]]]:
    """``records`` in order, split into bodies that Reef's import route accepts.

    Refuses a single record larger than ``max_bytes``, which no batch can carry.
    """
    batch: list[Mapping[str, Any]] = []
    size = len(b'{"records": []}')
    for record in records:
        encoded = len(json.dumps(record).encode("utf-8")) + len(b", ")
        if encoded + len(b'{"records": []}') > max_bytes:
            raise ContractError(f"record {record.get('agent_record_id')} is {encoded} bytes, over {max_bytes}")
        if batch and (len(batch) == max_records or size + encoded > max_bytes):
            yield batch
            batch, size = [], len(b'{"records": []}')
        batch.append(record)
        size += encoded
    if batch:
        yield batch


def _checked_tags(tags: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(tags, Mapping):
        raise ContractError(f"tags must be an object, got {tags!r}")
    unknown = set(tags) - set(TAG_NAMES)
    if unknown:
        raise ContractError(f"unknown tags {sorted(unknown)}; Honing's tags are {list(TAG_NAMES)}")
    for name, value in tags.items():
        _require_text(value, f"the {name} tag")
    return {name: tags[name] for name in TAG_NAMES if name in tags}


def _versioned(body: object, kind: str) -> Mapping[str, Any]:
    if not isinstance(body, Mapping) or body.get("format") != kind:
        raise ContractError(f"not a {kind} payload")
    if body.get("version") != FORMAT_VERSION:
        raise ContractError(f"{kind} version {body.get('version')!r}; this Honing reads version {FORMAT_VERSION}")
    return body


def _require_text(value: object, what: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"{what} must be a non-empty string without surrounding spaces, got {value!r}")


def _probability(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "DECISION_FORMAT",
    "EPISODE_FORMAT",
    "FORMAT_VERSION",
    "MAX_BATCH_BYTES",
    "MAX_BATCH_RECORDS",
    "TAG_HEADER_PREFIX",
    "TAG_NAMES",
    "EpisodeReport",
    "RecordedDecision",
    "batches",
    "decision_record_id",
    "episode_report_id",
    "tag_headers",
]
