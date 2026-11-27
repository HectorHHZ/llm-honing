"""Reef's job data as Honing's training batches (#227).

Honing's recipe in ``integrations/reef/`` has Reef spool each training job's
data to ``<spool>/<job id>.json``, and the job runs until Honing writes
``<job id>.consumed`` next to it. :class:`ReefDataProvider` is that reader: at
a train move it turns the oldest job data no move has consumed into one
``typed_decisions`` batch, then marks it consumed. Reef consumed the job's
records when it submitted the job, so data drawn for a move the gate rejects
never trains again, even after a restart. One campaign reads a spool.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from honing.backends.reef.records import EPISODE_FORMAT, FORMAT_VERSION, RecordedDecision
from honing.core import ContractError, DataContract, DecisionSample, TrainingBatch

#: What ``integrations/reef/backend.py`` writes for each job.
JOB_FORMAT = "honing-job-data"
JOB_VERSION = 1
#: The decision record's fields that each step of a spooled trajectory carries under ``extra.honing``.
DECISION_KEYS = ("format", "version", "request_id", "branch", "parts", "questions", "answers")


def decisions_from_job(job: Mapping[str, Any], *, behavior: bool) -> tuple[DecisionSample, ...]:
    """The decisions of one spooled job, episode by episode in the job's order, each in play order.

    A decision is the sample ``decision_from_row`` builds from its row: the
    episode's score is its reward, its task the group, and its ``release``,
    ``harness`` and ``policy`` tags its sources. A runner knows the checksum of
    the weights it served but not their release, so ``policy`` is what keeps a
    batch to the head's weights. ``behavior`` keeps the answers' probabilities
    as behavior probabilities.
    """
    if job.get("format") != JOB_FORMAT or job.get("version") != JOB_VERSION:
        raise ContractError(f"job data must be {JOB_FORMAT} version {JOB_VERSION}")
    decisions: list[DecisionSample] = []
    for sample in job["samples"]:
        trajectory = sample["trajectory"]
        episode = trajectory["extra"]["honing"]
        if episode.get("format") != EPISODE_FORMAT or episode.get("version") != FORMAT_VERSION:
            raise ContractError(f"job {job['job_id']} holds a trajectory that is not a {EPISODE_FORMAT} episode")
        tags = episode["tags"]
        for step in trajectory["steps"]:
            honing = step["extra"]["honing"]
            record = RecordedDecision.from_payload(
                {"honing": {key: honing[key] for key in DECISION_KEYS}, "metadata": {"tags": tags}}
            )
            decisions.append(
                record.sample(
                    honing["chosen"],
                    reward=trajectory["extra"]["reef"]["reward"],
                    group=sample["group_id"],
                    release_id=tags.get("release"),
                    harness_id=tags.get("harness"),
                    behavior=behavior,
                    policy_checksum=tags.get("policy"),
                )
            )
    return tuple(decisions)


class ReefDataProvider:
    """Satisfies :class:`~honing.core.DataProvider`: the next job data Reef spooled, as one typed-decisions batch.

    The batch id names the Reef job (``reef:<scenario>/<job id>``), and the
    campaign writes it into the commit record of the move that trained on it.
    A decision keeps its behavior probabilities when the contract asks for
    them. With nothing spooled it waits up to ``wait_s`` seconds, then returns
    ``None``, as it does for any contract other than ``typed_decisions``.
    """

    def __init__(self, spool_dir: Path, *, wait_s: float = 0.0, poll_s: float = 1.0) -> None:
        if wait_s < 0 or poll_s <= 0:
            raise ContractError("wait_s must be >= 0 and poll_s > 0")
        self.spool_dir = Path(spool_dir)
        self._wait_s = wait_s
        self._poll_s = poll_s

    def pending(self) -> list[Path]:
        """The spooled jobs no move consumed and Reef did not cancel, oldest first."""
        jobs = [
            path
            for path in self.spool_dir.glob("*.json")
            if not path.with_suffix(".consumed").exists() and not path.with_suffix(".cancelled").exists()
        ]
        return sorted(jobs, key=lambda path: (json.loads(path.read_text())["submitted_at"], path.name))

    def next_batch(self, contract: DataContract) -> TrainingBatch | None:
        if contract.kind != "typed_decisions":
            return None
        deadline = time.monotonic() + self._wait_s
        while not (pending := self.pending()):
            if time.monotonic() >= deadline:
                return None
            time.sleep(self._poll_s)
        path = pending[0]
        job = json.loads(path.read_text())
        batch = TrainingBatch(
            f"reef:{job['scenario']}/{job['job_id']}",
            contract,
            decisions=decisions_from_job(job, behavior=contract.requires_logprobs),
        )
        _mark(path.with_suffix(".consumed"))
        return batch


def _mark(path: Path) -> None:
    """Create ``path`` and make it durable: a consumed job must stay consumed after a crash."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
