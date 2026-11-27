"""Honing's record-to-sample and sample-to-job stages: one trajectory sample per reported tank episode.

Reef runs these in its samples service (#137, stages 3 and 4). They read the
records ``honing/backends/reef/records.py`` defines, as plain JSON: this
package imports Reef, never Honing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.records_types import AgentRecord
from reef.samples import Record2Sample, ReportContext, ReportSamples, SkippedReport
from reef.samples.context import PipelineContext
from reef.train.sample2job import Sample2Job, Sample2JobSettings
from reef.train.types import JobData, Sample, TrajectorySample

DECISION_FORMAT = "honing-decision"
EPISODE_FORMAT = "honing-episode"
FORMAT_VERSION = 1
#: The ATIF version of Reef's own trajectories at bc8c150 (reef/core/trajectories.py).
ATIF_VERSION = "ATIF-v1.7"


def _honing(payload: Mapping[str, Any], kind: str) -> Mapping[str, Any] | None:
    body = payload.get("honing")
    if isinstance(body, Mapping) and body.get("format") == kind and body.get("version") == FORMAT_VERSION:
        return body
    return None


class DecisionEpisodeSamples(ReportSamples):
    """One sample per reported episode: its decisions in play order as one ATIF trajectory.

    Each step is one decision record's ``honing`` payload, with the policy's
    choice from the report and the record's id, under ``extra.honing``. The
    report's score is the reward in ``extra.reef``, and the task groups the
    sample. A report of another format, or with no score, trains nothing.

    Follows Reef's ``ReportSamples.make_sample`` (reef/samples/reports.py, Reef
    bc8c150): one sample per report, its score as the reward. Differs: builds
    the trajectory from decision records, not chat payloads, since
    ``make_trajectory`` reads chat messages.
    """

    #: Reef's staleness rule compares a sample's Reef artifact with the one served. Decision records name no Reef
    #: artifact, and Honing keeps a batch to one release itself (``batch_from_one_release``, by the release tag).
    goes_stale = False

    def make_sample(self, context: ReportContext) -> Sample | SkippedReport:
        report = context.report.payload
        metadata = report.get("metadata")
        episode = _honing(metadata, EPISODE_FORMAT) if isinstance(metadata, Mapping) else None
        if episode is None:
            return SkippedReport(f"report {context.report.agent_record_id} is not a {EPISODE_FORMAT} report")
        if context.score is None:
            return SkippedReport(f"report {context.report.agent_record_id} has no score")
        chosen = episode.get("chosen")
        # Decision records hold no chat messages, so each is its own trajectory and the inferences are the
        # report's references, in order.
        if not isinstance(chosen, list) or len(chosen) != len(context.inferences) or not chosen:
            raise ValueError(
                f"report {context.report.agent_record_id} has {len(chosen) if isinstance(chosen, list) else 'no'} "
                f"choices for {len(context.inferences)} decisions"
            )
        steps = [
            self.step(index, record, choice)
            for index, (record, choice) in enumerate(zip(context.inferences, chosen, strict=True))
        ]
        feedback = report.get("feedback")
        tags = context.inferences[0].payload.get("metadata", {}).get("tags", {})
        task = metadata.get("task") if isinstance(metadata.get("task"), str) else None
        trajectory = {
            "schema_version": ATIF_VERSION,
            "session_id": episode.get("episode"),
            "agent": {"name": "honing", "version": str(FORMAT_VERSION)},
            "steps": steps,
            "extra": {
                "reef": {
                    "reward": context.score,
                    "feedback": feedback if isinstance(feedback, (str, Mapping)) else None,
                },
                "honing": {
                    "format": EPISODE_FORMAT,
                    "version": FORMAT_VERSION,
                    "episode": episode.get("episode"),
                    "task": task,
                    "tags": dict(tags),
                },
            },
        }
        return TrajectorySample(trajectory, group_id=task)

    @staticmethod
    def step(index: int, record: AgentRecord, choice: object) -> dict[str, Any]:
        decision = _honing(record.payload, DECISION_FORMAT)
        if decision is None:
            raise ValueError(f"record {record.agent_record_id} is not a {DECISION_FORMAT} record")
        # Ordered fields are [name, value] pairs, since Reef stores payloads with sorted keys.
        questions = decision.get("questions")
        keys = {pair[0] for pair in questions} if isinstance(questions, list) else None
        if not isinstance(choice, Mapping) or keys is None or set(choice) != keys:
            raise ValueError(f"record {record.agent_record_id} needs a choice for each of its questions")
        return {
            "step_id": index + 1,
            "source": "agent",
            "message": "",
            "extra": {"honing": {**decision, "record_id": record.agent_record_id, "chosen": dict(choice)}},
        }


class DecisionJobData(Sample2Job):
    """Put the episode samples into one job's data unchanged.

    Follows OpenClaw-RL's ``OpenClawRLJobData.make_job_data``
    (recipes/openclawrl/processor.py, Reef bc8c150), which keeps the samples as
    they are. Differs: no record file and no source rewrite.

    Config: ``buffer_size`` and ``job_samples`` (``Sample2JobSettings``).
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        settings = Sample2JobSettings.parse(config, plugin_class=cls)
        return cls(context, settings.buffer_size, settings.job_samples)

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        if not all(isinstance(sample, TrajectorySample) for sample in samples):
            raise TypeError("Honing's job data holds episode trajectories only")
        return JobData(data_id, samples)
