"""Corrected decisions: the decisions of tank rounds System 2 blamed on the weights, relabelled for training.

System 2 diagnoses a whole round (:class:`~honing.harness.diagnosis.Diagnosis`).
When it names the weights, a supervised move can train them on that round's
decisions with better answers. The answers are the planner teacher's by
default: ``node scripts/raw-states.mjs label RUN`` writes them to the run's
``labels.jsonl``, one line per row of ``decisions.jsonl`` (#204). The other
source is System 2 itself: the option its diagnosis names for a decision it
points at.

A corrected decision is a row's state and questions as the model saw them,
with the label as the choice. It records where the label came from
(``label_source``) and the release and harness the round was diagnosed under,
and its group is ``tank:<seed>:<seat>``, as a demonstration's is.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, get_args

from honing.backends.tank.decisions import decision_from_row
from honing.backends.tank.raw_states import read_labels
from honing.backends.tank.traces import verify_outputs
from honing.core import ContractError, DataContract, DecisionSample, TrainingBatch, require_choice
from honing.harness.diagnosis import Diagnosis

LabelSource = Literal["teacher", "system2"]
#: The ``label_source`` of a decision System 2 labelled itself.
SYSTEM2 = "system2"


class CorrectedDecisionProvider:
    """Satisfies :class:`~honing.core.DataProvider`: the corrected decisions of weights diagnoses, as one batch.

    ``runs`` maps an episode id to its run directory. Only diagnoses with
    cause ``weights`` produce decisions, and each needs its run. With
    ``labels="teacher"`` every row of the round becomes a decision that
    chooses the teacher's answer, including the rows where the model chose the
    same: these are the states the faulty weights led to, and the teacher's
    answers there are what training should teach. With ``labels="system2"``
    only the decisions the diagnosis names an option for do.

    The weights diagnoses must all name the same release and harness, or the
    provider is refused when it is built: a correction made for one release's
    weights says nothing about another's (design principle P4). A decision
    whose state, questions and choice equal one of ``demonstrations``, or an
    earlier corrected decision, is dropped, so nothing is trained on twice.

    The first ``typed_decisions`` request gets every decision as the batch
    ``batch_id``. Later requests get ``None``, and so does every request when
    no decision is left to train on.
    """

    def __init__(
        self,
        diagnoses: Iterable[Diagnosis],
        runs: Mapping[str, Path],
        *,
        labels: LabelSource = "teacher",
        demonstrations: Iterable[DecisionSample] = (),
        batch_id: str = "tank-corrected-decisions",
    ) -> None:
        require_choice(labels, get_args(LabelSource), "labels")
        weights = [diagnosis for diagnosis in diagnoses if diagnosis.cause == "weights"]
        for diagnosis in weights:
            if diagnosis.release_id is None or diagnosis.harness_id is None:
                raise ContractError(
                    f"the weights diagnosis of {diagnosis.episode_id} names no release or no harness, and a"
                    " correction is only good for the weights and harness it was made under"
                )
            if diagnosis.episode_id not in runs:
                raise ContractError(f"no run directory for {diagnosis.episode_id}, which is diagnosed as weights")
        releases = sorted({str(diagnosis.release_id) for diagnosis in weights})
        harnesses = sorted({str(diagnosis.harness_id) for diagnosis in weights})
        if len(releases) > 1 or len(harnesses) > 1:
            raise ContractError(
                f"the weights diagnoses are of releases {releases} under harnesses {harnesses}, and one batch"
                " corrects one release under one harness"
            )
        seen = {_key(sample) for sample in demonstrations}
        decisions = []
        for diagnosis in weights:
            for sample in _corrected(diagnosis, Path(runs[diagnosis.episode_id]), labels):
                if _key(sample) not in seen:
                    seen.add(_key(sample))
                    decisions.append(sample)
        self._decisions = tuple(decisions)
        self._batch_id = batch_id
        self._served = False

    def next_batch(self, contract: DataContract) -> TrainingBatch | None:
        if contract.kind != "typed_decisions" or self._served or not self._decisions:
            return None
        self._served = True
        return TrainingBatch(self._batch_id, contract, decisions=self._decisions)


def _key(sample: DecisionSample) -> tuple[object, ...]:
    """What a decision teaches: the state, the questions and the choice."""
    return (sample.parts, sample.questions, sample.chosen)


def _corrected(diagnosis: Diagnosis, run: Path, labels: LabelSource) -> list[DecisionSample]:
    """The round's decisions that ``labels`` labels, each choosing its label."""
    manifest = verify_outputs(run, ("decisions.jsonl",))
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    answers: dict[int, dict[str, str]] = {}
    if labels == "teacher":
        header, lines = read_labels(run)
        if [line["request_id"] for line in lines] != [row["request_id"] for row in rows]:
            raise ContractError(f"{run}/labels.jsonl does not label the rows of its decisions.jsonl. Label it again")
        source = str(header["source"])
        answers = {step: line["labels"] for step, line in enumerate(lines)}
    else:
        source = SYSTEM2
        for decision in diagnosis.decisions:
            if decision.option is not None:
                answers.setdefault(decision.step, {})[decision.question] = decision.option
    group = f"tank:{manifest['seed']}:{manifest['seat']}"
    samples = []
    for step, chosen in sorted(answers.items()):
        if step >= len(rows):
            raise ContractError(f"{diagnosis.episode_id} has {len(rows)} decisions, and none is step {step}")
        questions = sorted(rows[step]["questions"])
        if sorted(chosen) != questions:
            raise ContractError(
                f"step {step} of {diagnosis.episode_id} asks {questions}, and its {labels} labels answer"
                f" {sorted(chosen)}"
            )
        sample = decision_from_row(
            {**rows[step], "chosen": chosen},
            reward=None,
            group=group,
            release_id=diagnosis.release_id,
            harness_id=diagnosis.harness_id,
            behavior=False,
        )
        samples.append(replace(sample, label_source=source))
    return samples


__all__ = ["SYSTEM2", "CorrectedDecisionProvider", "LabelSource"]
