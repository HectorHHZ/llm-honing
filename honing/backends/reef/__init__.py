"""Honing's side of Reef (#137): the records it writes, the route its chat calls take, and the data provider.

Reef runs as its own service in its own Python 3.12 environment. Honing
reaches it only from this package, through ``reef-client`` (the ``reef``
extra) or Reef's OpenAI-compatible routes, and never imports ``reef`` itself.
See ``README.md`` here for the record format and the Reef code each module
follows.

``recorder`` (the decision server's recorder, #223) and ``reporter`` (the
grader's reports, #225) import ``reef_client``, so they are not imported here:
import ``honing.backends.reef.recorder`` or ``.reporter`` itself.
"""

from honing.backends.reef.chat import ReefChatRoute, chat_model
from honing.backends.reef.provider import JOB_FORMAT, JOB_VERSION, ReefDataProvider, decisions_from_job
from honing.backends.reef.records import (
    DECISION_FORMAT,
    EPISODE_FORMAT,
    FORMAT_VERSION,
    MAX_BATCH_BYTES,
    MAX_BATCH_RECORDS,
    TAG_HEADER_PREFIX,
    TAG_NAMES,
    EpisodeReport,
    RecordedDecision,
    batches,
    decision_record_id,
    episode_report_id,
    tag_headers,
)

__all__ = [
    "DECISION_FORMAT",
    "EPISODE_FORMAT",
    "FORMAT_VERSION",
    "JOB_FORMAT",
    "JOB_VERSION",
    "MAX_BATCH_BYTES",
    "MAX_BATCH_RECORDS",
    "TAG_HEADER_PREFIX",
    "TAG_NAMES",
    "EpisodeReport",
    "RecordedDecision",
    "ReefChatRoute",
    "ReefDataProvider",
    "batches",
    "chat_model",
    "decision_record_id",
    "decisions_from_job",
    "episode_report_id",
    "tag_headers",
]
