"""Post a Reef report for every graded episode (#225), on a background thread.

:class:`ReefReportingGrader` wraps a :class:`~honing.core.TaskGrader`. After
the grader returns, it hands the task, the outcome and the grade to
:meth:`ReefReporter.submit` and returns the grade unchanged. The reporter's
thread reads the run directory and posts one report to ``POST /reef/report``:

- **The report** is #229's :class:`~honing.backends.reef.records.EpisodeReport`:
  the grade's score and feedback, ``metadata.task``, and one reference per
  ``decisions.jsonl`` row, in play order, with the row's ``chosen`` (the
  policy's draw, before safety changed anything). Its id is
  :func:`~honing.backends.reef.records.episode_report_id`, so a resend is a retry.
- **The episode** is the ``episode`` tag in ``manifest.json``'s ``reef_tags``,
  which the match process keeps (#223). A run without one made no records to
  reference: it is skipped and counted (``skipped_no_tags``), and its grade is
  unaffected. So is a run with no decision rows (``skipped_no_decisions``) and
  one whose files #229's checks refuse (``invalid``).
- **Records first.** Reef refuses a report whose reference it has not stored.
  The decision server records on its own thread, so a report can arrive before
  its records. A report refused for an unknown reference is sent again, whole,
  for up to ``reference_wait_seconds``; then it is logged with its episode and
  counted as ``unknown_reference``, and never sent without the missing references.
- **Reef down.** As the recorder: connection failures and HTTP 408, 429 and 5xx
  are retried with a backoff of 0.5 s doubling to 30 s and no limit; 401, 403
  and 404 are retried the same way but logged as errors and counted apart
  (``rejected_token``, ``no_route``). Any other refusal (400, or 409 for a
  known id with other content) is logged and counted as ``refused``.
- **Memory.** Past ``max_pending_reports`` waiting reports, a new episode is not
  queued: it is counted as ``dropped`` and logged. :meth:`ReefReporter.close`
  keeps sending for ``flush_seconds``; what is left is counted as ``unsent``.

Follows reef-client's ``ReefClient.report`` (reef_client/client.py,
reef-client 91af039) and Reef's ``POST /reef/report`` (reef/api/routes/records.py,
Reef bc8c150): a client-supplied ``agent_record_id`` makes a resend safe.
Differs: it waits without a retry limit, and waits for missing references
before it counts them.
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from reef_client import ReefClient, ReefClientError

from honing.backends.reef.recorder import _UNAVAILABLE, backoff_seconds
from honing.backends.reef.records import EpisodeReport
from honing.core import ContractError, EpisodeOutcome, Grade, Task, TaskGrader

logger = logging.getLogger(__name__)

DEFAULT_MAX_PENDING_REPORTS = 10_000
DEFAULT_REFERENCE_WAIT_SECONDS = 300.0
#: How Reef words a report that references a record it has not stored (``validate_record``, reef/control/ingest.py,
#: Reef bc8c150). Reef answers 400 for it as for any malformed report, so only the message tells them apart.
UNKNOWN_REFERENCE = "must identify an existing inference"
_MISCONFIGURED = {
    401: ("rejected_token", "Reef rejected the token (REEF_TOKEN)"),
    403: ("rejected_token", "Reef rejected the token (REEF_TOKEN)"),
    404: ("no_route", "no such route, check reef_url (or a scenario Reef will not create, check reef_scenario)"),
}
_LOG_SECONDS = 30.0


@dataclass(frozen=True)
class Skipped:
    """A graded episode that makes no report: ``reason`` is ``no_tags``, ``no_decisions`` or ``invalid``."""

    reason: str
    detail: str


def episode_report(task: Task, outcome: EpisodeOutcome, grade: Grade) -> EpisodeReport | Skipped:
    """The report of one graded episode, read from its run directory, or why it makes none."""
    if outcome.workspace_location is None:
        return Skipped("no_tags", f"the outcome of {task.task_id} names no run directory")
    run = Path(outcome.workspace_location)
    try:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Skipped("no_tags", f"{run} has no manifest.json")
    except (OSError, ValueError) as error:
        return Skipped("invalid", f"{run}/manifest.json: {error}")
    tags = manifest.get("reef_tags") if isinstance(manifest, dict) else None
    if not isinstance(tags, dict) or not tags.get("episode"):
        return Skipped("no_tags", f"{run}/manifest.json has no reef_tags episode")
    episode = tags["episode"]
    if tags.get("task", task.task_id) != task.task_id:
        return Skipped("invalid", f"{run} was tagged task {tags['task']!r}, but graded as {task.task_id!r}")
    try:
        text = (run / "decisions.jsonl").read_text(encoding="utf-8")
    except FileNotFoundError:
        return Skipped("no_decisions", f"{run} has no decisions.jsonl")
    except OSError as error:
        return Skipped("invalid", f"{run}/decisions.jsonl: {error}")
    try:
        rows = [json.loads(line) for line in text.split("\n") if line.strip()]
        if not rows:
            return Skipped("no_decisions", f"{run}/decisions.jsonl has no rows")
        return EpisodeReport.from_rows(
            rows, episode=episode, score=grade.score, feedback=grade.feedback, task=task.task_id
        )
    except (ContractError, KeyError, TypeError, AttributeError, ValueError) as error:
        return Skipped("invalid", f"{run}: {type(error).__name__}: {error}")


@dataclass
class _Pending:
    """A report waiting on Reef, and when it was first refused for an unknown reference."""

    report: EpisodeReport
    unknown_since: float | None = None
    unknown_attempts: int = 0
    not_before: float = 0.0


@dataclass
class _Counts:
    episodes: int = 0
    sent: int = 0
    refused: int = 0
    unknown_reference: int = 0
    skipped_no_tags: int = 0
    skipped_no_decisions: int = 0
    invalid: int = 0
    dropped: int = 0
    rejected_token: int = 0
    no_route: int = 0
    unsent: int = 0


class ReefReporter:
    """Posts one report per episode handed to :meth:`submit` to one Reef scenario.

    ``url``, ``scenario`` and ``token`` are the Reef the decision server records
    in (its ``--reef-url``, ``--reef-scenario`` and ``REEF_TOKEN``): a report
    can only reference records of its own scenario.
    """

    def __init__(
        self,
        url: str,
        scenario: str,
        *,
        token: str | None = None,
        max_pending_reports: int = DEFAULT_MAX_PENDING_REPORTS,
        reference_wait_seconds: float = DEFAULT_REFERENCE_WAIT_SECONDS,
        flush_seconds: float = 10.0,
        timeout_seconds: float = 30.0,
        backoff_seconds: tuple[float, float] = (0.5, 30.0),
    ) -> None:
        if not scenario.strip():
            raise ContractError("a Reef reporter needs a scenario")
        if max_pending_reports <= 0 or timeout_seconds <= 0 or flush_seconds < 0 or reference_wait_seconds < 0:
            raise ContractError(
                "max_pending_reports and timeout_seconds must be positive, "
                "flush_seconds and reference_wait_seconds not negative"
            )
        self.scenario = scenario
        self._client = ReefClient(url, token=token, timeout_s=timeout_seconds)
        self._max_pending_reports = max_pending_reports
        self._reference_wait_seconds = reference_wait_seconds
        self._flush_seconds = flush_seconds
        self._timeout_seconds = timeout_seconds
        self._backoff_seconds = backoff_seconds
        self._lock = threading.Condition()
        self._queued: deque[tuple[Task, EpisodeOutcome, Grade]] = deque()
        #: Reports made and not yet stored or refused. Only the reporter's thread changes it.
        self._pending: deque[_Pending] = deque()
        #: Episodes accepted and not yet skipped or settled: queued, being made into a report, or pending. Changed
        #: only under the lock, since a report being made is in neither deque.
        self._waiting = 0
        self._counts = _Counts()
        self._closing = False
        self._deadline = float("inf")
        self._last_drop_log = -float("inf")
        self._failure = ""
        self._misconfigured = False
        self._mistake_logged: dict[str, float] = {}
        self._thread = threading.Thread(target=self._run, name="reef-reporter", daemon=True)
        self._thread.start()

    def submit(self, task: Task, outcome: EpisodeOutcome, grade: Grade) -> None:
        """Queue one graded episode. Never blocks on Reef or raises."""
        with self._lock:
            self._counts.episodes += 1
            if self._closing or self._waiting >= self._max_pending_reports:
                self._counts.dropped += 1
                now = time.monotonic()
                if now - self._last_drop_log >= _LOG_SECONDS:
                    self._last_drop_log = now
                    logger.warning(
                        "Reef reporter: %d episodes not reported so far (%s); the latest is %s",
                        self._counts.dropped,
                        (
                            "the reporter is closing"
                            if self._closing
                            else f"{self._max_pending_reports} reports are waiting for Reef"
                        ),
                        task.task_id,
                    )
                return
            self._waiting += 1
            self._queued.append((task, outcome, grade))
            self._lock.notify()

    def counts(self) -> dict[str, int]:
        """What the reporter has done so far."""
        with self._lock:
            counts = dict(vars(self._counts))
            counts["pending"] = self._waiting
            return counts

    def close(self) -> None:
        """Send what is queued for up to ``flush_seconds``, then stop; what is left is logged as unsent."""
        with self._lock:
            self._closing = True
            self._deadline = time.monotonic() + self._flush_seconds
            self._lock.notify()
        self._thread.join(self._flush_seconds + self._timeout_seconds + 1.0)
        with self._lock:
            self._counts.unsent = unsent = self._waiting
        if unsent:
            logger.warning("Reef reporter: closed with %d episode reports not sent to Reef", unsent)

    def _run(self) -> None:
        failures = 0
        while True:
            with self._lock:
                while not self._queued and not self._pending and not self._closing:
                    self._lock.wait()
                if self._closing and (time.monotonic() >= self._deadline or not (self._queued or self._pending)):
                    return
                taken = list(self._queued)
                self._queued.clear()
            for item in taken:
                self._make_report(*item)
            now = time.monotonic()
            ready = next((pending for pending in self._pending if pending.not_before <= now), None)
            if ready is None:
                if self._pending:
                    # Only reports waiting for their records: a new episode can go first.
                    self._wait(min(pending.not_before for pending in self._pending), wake_on_submit=True)
                continue
            try:
                outcome = self._deliver(ready)
            except Exception as error:  # a fault here must not stop reporting: wait as for an outage
                self._failure = f"{type(error).__name__}: {error}"
                logger.warning("Reef reporter: sending a report failed: %s", self._failure)
                outcome = "down"
            if outcome == "settled":
                self._pending.remove(ready)
                if failures:
                    logger.warning("Reef reporter: Reef took reports again after %d failed attempts", failures)
                failures = 0
                continue
            if outcome == "later":
                # Waits for its records at the back of the queue, so the reports behind it go on.
                self._pending.remove(ready)
                self._pending.append(ready)
                continue
            failures += 1
            if failures == 1 and not self._misconfigured:
                logger.warning(
                    "Reef reporter: Reef at %s is not taking reports (%s); retrying",
                    self._client.service_url,
                    self._failure,
                )
            # A new episode does not end this wait, so a busy campaign does not retry at its grading rate.
            self._wait(time.monotonic() + backoff_seconds(failures, *self._backoff_seconds), wake_on_submit=False)

    def _wait(self, until: float, *, wake_on_submit: bool) -> None:
        """Wait until ``until`` or the end of the flush, or, with ``wake_on_submit``, until an episode is queued."""
        with self._lock:
            while (remaining := min(until, self._deadline) - time.monotonic()) > 0:
                if wake_on_submit and self._queued:
                    return
                self._lock.wait(remaining)

    def _make_report(self, task: Task, outcome: EpisodeOutcome, grade: Grade) -> None:
        try:
            made = episode_report(task, outcome, grade)
        except Exception as error:  # the grade stands, so this is the reporter's fault: say so
            made = Skipped("invalid", f"{type(error).__name__}: {error}")
        if isinstance(made, Skipped):
            with self._lock:
                name = "invalid" if made.reason == "invalid" else f"skipped_{made.reason}"
                setattr(self._counts, name, getattr(self._counts, name) + 1)
                self._waiting -= 1
            log = logger.warning if made.reason == "invalid" else logger.info
            log("Reef reporter: no report for %s: %s", task.task_id, made.detail)
            return
        self._pending.append(_Pending(made))

    def _deliver(self, pending: _Pending) -> str:
        """Send one report: ``settled`` once Reef stored or refused it, ``later`` to await its records, or ``down``."""
        report = pending.report
        try:
            receipt = self._client.report(self.scenario, {**report.payload(), "agent_record_id": report.record_id})
        except ReefClientError as error:
            self._misconfigured = error.status in _MISCONFIGURED
            if self._misconfigured:
                self._mistake(error)
                return "down"
            if error.status in _UNAVAILABLE:
                self._failure = f"HTTP {error.status}: {error.body[:200]}"
                return "down"
            self._failure = ""
            if error.status == 400 and UNKNOWN_REFERENCE in error.body:
                now = time.monotonic()
                if pending.unknown_since is None:
                    pending.unknown_since = now
                if now - pending.unknown_since < self._reference_wait_seconds:
                    pending.unknown_attempts += 1
                    pending.not_before = now + backoff_seconds(pending.unknown_attempts, *self._backoff_seconds)
                    return "later"
                self._settle(pending, "unknown_reference", f"Reef answered {error.status}: {error.body[:400]}")
                return "settled"
            self._settle(pending, "refused", f"Reef answered {error.status}: {error.body[:400]}")
            return "settled"
        except (OSError, http.client.HTTPException, ValueError) as error:
            self._failure, self._misconfigured = f"{type(error).__name__}: {error}", False
            return "down"
        self._failure = ""
        got = receipt.get("agent_record_id") if isinstance(receipt, dict) else None
        if got != report.record_id:
            self._settle(pending, "refused", f"Reef acknowledged {got!r}, not the report sent")
        else:
            self._settle(pending, "sent", "")
        return "settled"

    def _mistake(self, error: ReefClientError) -> None:
        """Count a deployment mistake, and log it the first time and then every 30 s while it lasts."""
        name, message = _MISCONFIGURED[error.status]
        self._failure = f"HTTP {error.status}: {error.body[:200]}"
        with self._lock:
            count = getattr(self._counts, name) + 1
            setattr(self._counts, name, count)
        now, last = time.monotonic(), self._mistake_logged.get(name)
        if last is None:
            logger.error("Reef reporter: %s (%s); reports wait and are retried", message, self._failure)
        elif now - last < _LOG_SECONDS:
            return
        else:
            logger.error("Reef reporter: %s: %d attempts so far", message, count)
        self._mistake_logged[name] = now

    def _settle(self, pending: _Pending, outcome: str, detail: str) -> None:
        with self._lock:
            setattr(self._counts, outcome, getattr(self._counts, outcome) + 1)
            self._waiting -= 1
        if outcome != "sent":
            report = pending.report
            logger.error(
                "Reef reporter: the report of episode %s (task %s, %d references) was not stored: %s",
                report.episode,
                report.task,
                len(report.references),
                detail,
            )


class ReefReportingGrader:
    """Satisfies :class:`~honing.core.TaskGrader`: grades with ``grader``, then reports the episode to Reef.

    The grade is returned as ``grader`` gave it, and its errors propagate as
    they did. Reporting never fails or delays a grade: ``reporter`` only queues.
    """

    def __init__(self, grader: TaskGrader, reporter: ReefReporter) -> None:
        self._grader = grader
        self.reporter = reporter

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        grade = await self._grader.grade(task, outcome)
        self.reporter.submit(task, outcome, grade)
        return grade

    def close(self) -> None:
        """Send the reports still waiting, for up to the reporter's ``flush_seconds``."""
        self.reporter.close()


__all__ = [
    "DEFAULT_MAX_PENDING_REPORTS",
    "DEFAULT_REFERENCE_WAIT_SECONDS",
    "UNKNOWN_REFERENCE",
    "ReefReporter",
    "ReefReportingGrader",
    "Skipped",
    "episode_report",
]
