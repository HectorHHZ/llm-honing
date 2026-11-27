"""The report Honing's grader posts for every graded episode (#225), as a Reef report type.

Reef parses a deployment's report type at admission and refuses a report that
does not parse, naming the broken field. A recipe declares the type in its
``report_type``; this module declares it once, for every recipe that reads
Honing's episode reports.

Follows Reef's ``ScoredRolloutReport`` (reef/core/reports/scored_rollout.py,
Reef bc8c150): a finite ``score`` on top. Differs: ``metadata.task`` is
required, so a report that names no task is refused at admission. A report
type declares only numbers, strings, booleans and maps of strings to strings,
and only ``score`` sits on top: the rest of what Honing sends, ``feedback`` and
``metadata.honing`` (each reference's choice), passes through unchecked here,
and its reader checks it with ``EpisodeReport.from_payload``
(``honing/backends/reef/records.py``).

Reef loads this module in its own environment, so it imports Reef and never Honing.
"""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.reports import ScoredRolloutReport


@dataclass(frozen=True)
class HoningEpisodeReport(ScoredRolloutReport):
    """One graded episode: its score, and the task it played under ``metadata.task``."""

    task: str


__all__ = ["HoningEpisodeReport"]
