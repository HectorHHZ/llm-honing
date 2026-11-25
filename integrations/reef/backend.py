"""A Reef training backend that trains nothing: it spools each job's data for Honing and saves no checkpoint.

Honing's campaign decides when to train and its gate decides what ships
(#137, P2), so Reef must publish nothing. ``submit`` writes the job's data to
``<spool_dir>/<job id>.json`` durably and returns. The job runs until Honing's
data provider draws it and writes ``<job id>.consumed``. Then ``status``
reports it finished with no checkpoints, which Reef reads as a job that
trained nothing: it ends the job and publishes no snapshot. Reef consumed the
data's records when it submitted the job, so they never train again, even
after a restart.

Select it with ``training.backend: integrations.reef.backend:Deployment`` and
``training.options.spool-dir``.

Follows the in-process backend of Reef's ``tests/support/training_deployment.py``
and the job contract of ``TrainingComponent`` (reef/train/component.py), Reef
bc8c150. Differs: a job finishes when a reader outside Reef says so, and saves
no checkpoint.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact import Artifact, DefaultStart
from reef.core.config import config_option
from reef.core.weight_layout import ArtifactFormat
from reef.storage.data.job_data import JobDataKind
from reef.train.component import (
    JobStatus,
    TrainingBackendFacts,
    TrainingComponent,
    TrainingComponentFactory,
    TrainingJob,
)
from reef.train.deployment import InProcessTrainingDeployment

NAME = "honing-spool"
SPOOL_FORMAT = "honing-job-data"
SPOOL_VERSION = 1
START = "honing:none"
_NO_START = "Honing's release chain is the only one, so Reef trains from no artifact"


@dataclass(frozen=True)
class Options:
    #: The directory Honing's data provider reads. It must be on a disk both processes see.
    spool_dir: str = config_option("")
    inference_timeout_s: float = config_option(300.0)
    train_timeout_s: float | None = config_option(None)

    def __post_init__(self) -> None:
        if not self.spool_dir:
            raise ValueError("training.options.spool-dir names the directory Honing reads job data from")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_durably(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class SpoolComponent(TrainingComponent):
    """Spools each job's data for Honing; a job finishes once Honing marks it consumed."""

    def __init__(self, spool_dir: Path) -> None:
        self.spool_dir = spool_dir
        self.spool_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return NAME

    @property
    def artifact_format(self) -> ArtifactFormat | None:
        return None

    @property
    def start_files(self) -> tuple[str, ...] | None:
        return ()

    def root_start(self, root: Any) -> str:
        return START

    def import_start(self, artifact_dir: Path) -> str:
        raise ValueError(_NO_START)

    def snapshot_start(self, artifact: Artifact) -> str:
        raise ValueError(_NO_START)

    def check_job(self, job: TrainingJob) -> None:
        if job.data.kind not in (None, "trajectory"):
            raise ValueError(f"{NAME} spools trajectory jobs only, not {job.data.kind} jobs")

    def path(self, job_id: str) -> Path:
        return self.spool_dir / f"{job_id}.json"

    def submit(self, job: TrainingJob) -> None:
        self.check_job(job)
        path = self.path(job.job_id)
        if path.exists():
            return
        data = job.data.read()
        record = {
            "format": SPOOL_FORMAT,
            "version": SPOOL_VERSION,
            "job_id": job.job_id,
            "scenario": job.scenario,
            "data_id": data.data_id,
            "submitted_at": time.time(),
            "parameters": _plain(job.parameters),
            "samples": [
                {
                    "trajectory": _plain(sample.trajectory),
                    "group_id": sample.group_id,
                    "source_agent_record_ids": list(sample.source_agent_record_ids),
                }
                for sample in data.samples
            ],
        }
        _write_durably(path, json.dumps(record))

    def status(self, job_id: str) -> JobStatus:
        if self.path(job_id).with_suffix(".cancelled").exists():
            return JobStatus(job_id, "failed", error="cancelled")
        if self.path(job_id).with_suffix(".consumed").exists():
            return JobStatus(job_id, "finished")
        if self.path(job_id).exists():
            return JobStatus(job_id, "running")
        return JobStatus(job_id, "failed", error=f"{NAME} has no job {job_id}")

    def cancel(self, job_id: str) -> None:
        if self.path(job_id).exists() and not self.path(job_id).with_suffix(".consumed").exists():
            _write_durably(self.path(job_id).with_suffix(".cancelled"), "")


@dataclass(frozen=True)
class SpoolFacts(TrainingBackendFacts):
    """The facts of :class:`SpoolComponent`, from its settings alone."""

    @property
    def name(self) -> str:
        return NAME

    @property
    def artifact_format(self) -> ArtifactFormat | None:
        return None

    @property
    def shares_scenarios(self) -> bool:
        return False

    @property
    def concurrent_jobs(self) -> bool:
        return False

    @property
    def default_start_inputs(self) -> Mapping[str, str]:
        return {"model": NAME}

    def default_start(self) -> DefaultStart:
        return DefaultStart(inputs=self.default_start_inputs)

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any] | None:
        return None

    @property
    def start_files(self) -> tuple[str, ...] | None:
        return ()

    def check_root_start(self, root: Any) -> None:
        return

    def check_import_start(self, artifact_dir: Path) -> None:
        raise ValueError(_NO_START)


class SpoolFactory(TrainingComponentFactory):
    kind = "integrations.reef.backend:component_factory"

    def config_type(self) -> type[Options]:
        return Options

    def facts(self, config: Mapping[str, Any], model_path: str, environ: Mapping[str, str]) -> SpoolFacts:
        return SpoolFacts()

    def __call__(self, config: Mapping[str, Any], model_path: str, environ: Mapping[str, str]) -> SpoolComponent:
        return SpoolComponent(Path(config["spool_dir"]))


component_factory = SpoolFactory()


class Deployment(InProcessTrainingDeployment):
    """The ``training.backend`` that spools job data for Honing. It loads no model."""

    component_type = SpoolFactory.kind
    requires_local_model = False
