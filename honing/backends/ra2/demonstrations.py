"""RA2 demonstrations as typed-decision training data.

Reads the match directories the RA2 teacher and DAgger labelling write (#49's
format): ``manifest.json``, ``result.json``, ``samples.jsonl``, and ``source/``.
Every check the RA2 trainer made on them is kept, so nothing that is not labelled
training data reaches a trainer: the match must be a train-split scenario from
the protocol, explicitly marked as a teacher demonstration or DAgger labels and
not a model evaluation, with its source files matching their recorded hashes and
every row naming the same scenario as its match.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.core import (
    ContractError,
    DataContract,
    DecisionQuestion,
    DecisionSample,
    TrainingBatch,
)

LABELLED_KINDS = frozenset({"teacher-demonstration", "dagger-labels"})


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def decision_from_row(row: Mapping[str, object]) -> DecisionSample:
    """One labelled decision row as a DecisionSample; its group is ``map:seed:seat``."""
    questions = row["questions"]
    answers = row["answers"]
    if not isinstance(questions, Mapping) or not isinstance(answers, Mapping):
        raise ContractError("a decision row needs questions and answers objects")
    parts = row["parts"]
    if not isinstance(parts, Mapping):
        raise ContractError("a decision row needs a parts object")
    return DecisionSample(
        parts=tuple((name, tuple(lines)) for name, lines in parts.items()),
        questions=tuple(
            (
                key,
                DecisionQuestion(kind=q["type"], instructions=q["instructions"], options=tuple(q["criteria"].items())),
            )
            for key, q in questions.items()
        ),
        chosen=tuple((key, answers[key]["choice"]) for key in questions),
        group=f"{row['map']}:{row['seed']}:{row['seat']}",
    )


def read_match(directory: Path, protocol: Mapping[str, object]) -> list[DecisionSample]:
    """The decisions of one labelled training match, after every check."""
    manifest = json.loads((directory / "manifest.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    train = protocol["train"]
    if not isinstance(train, Mapping):
        raise ContractError("the protocol has no train section")
    if manifest["split"] != "train" or manifest["seed"] not in train["seeds"] or manifest["map"] not in train["maps"]:
        raise ContractError(f"{directory.name} is not a training scenario")
    if result.get("notModelEvaluation") is not True or result["kind"] not in LABELLED_KINDS:
        raise ContractError(f"{directory.name} is not labelled training data (teacher demonstration or DAgger)")
    for relative, expected in manifest["sourceSha256"].items():
        if _sha256(directory / "source" / relative) != expected:
            raise ContractError(f"{directory.name}: source {relative} differs from its recorded hash")
    decisions = []
    with (directory / "samples.jsonl").open() as rows:
        for line in rows:
            row = json.loads(line)
            if (row["split"], row["seed"], row["seat"], row["map"]) != (
                "train",
                manifest["seed"],
                manifest["seat"],
                manifest["map"],
            ):
                raise ContractError(f"{directory.name}: a row names another scenario than its match")
            decisions.append(decision_from_row(row))
    return decisions


class RA2DemonstrationProvider:
    """Satisfies :class:`~honing.core.DataProvider`: labelled RA2 matches as one typed-decisions batch.

    Returns the batch once, then ``None``; a contract other than
    ``typed_decisions`` gets ``None``.
    """

    def __init__(self, matches: Sequence[Path], protocol: Mapping[str, object], batch_id: str = "ra2-demonstrations"):
        self._matches = list(matches)
        self._protocol = protocol
        self._batch_id = batch_id
        self._served = False

    def next_batch(self, contract: DataContract) -> TrainingBatch | None:
        if contract.kind != "typed_decisions" or self._served:
            return None
        decisions = [d for directory in self._matches for d in read_match(directory, self._protocol)]
        if not decisions:
            return None
        self._served = True
        return TrainingBatch(self._batch_id, contract, decisions=tuple(decisions))
