"""Honing's side of Reef (#137): the records it writes, and later its client and data provider.

Reef runs as its own service in its own Python 3.12 environment. Honing
reaches it only from this package, through ``reef-client`` (the ``reef``
extra), and never imports ``reef`` itself. See ``README.md`` here for the
record format and the Reef code each module follows.
"""

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
