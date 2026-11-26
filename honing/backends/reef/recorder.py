"""Record the decision server's answered requests in Reef (#223), on a background thread.

The decision server hands every answered ``/v1/decide`` request to
:meth:`ReefRecorder.submit` after it has sent the reply. The recorder's thread
turns the request and its reply into decision records (:func:`decision_records`)
and posts them to ``POST /reef/records/batch`` in the bodies :func:`batches`
makes. A recording failure never reaches the decision: ``submit`` only queues,
and everything else runs on the recorder's thread.

- **Reef down.** A batch that fails to arrive is sent again, with the same ids,
  after a backoff of 0.5 s doubling to 30 s. A connection failure and HTTP 408,
  429 and 5xx count as down: Reef cannot take records now.
- **A deployment mistake.** HTTP 401 or 403 (Reef rejected the token) and 404
  (no such route or scenario) are retried the same way, but logged as errors
  and counted apart (``rejected_token``, ``no_route``), so a wrong token or URL
  does not look like an outage.
- **Records Reef refuses.** Reef stores a batch whole or not at all, and answers
  400 (a malformed record) or 409 (a known id with other content). The recorder
  then splits the batch in half and sends each half again, until each refused
  record is alone. Those are logged with Reef's message and counted as
  ``refused``; the rest of the batch is stored.
- **Memory.** Queued requests are charged their request and reply bytes until
  every record they made is stored or refused. Past ``max_pending_bytes`` a new
  request is not queued: it is counted as ``dropped_requests`` and logged. A
  restart while Reef is down loses what is queued.

Follows reef-client's ``upload_batch`` (reef_client/record_import.py,
reef-client 91af039): unchanged ids on a retry, the same retried statuses, and a
receipt for every record. Differs: it waits without a retry limit, tells a
refusal from an outage, and isolates the refused records instead of stopping.
"""

from __future__ import annotations

import http.client
import itertools
import json
import logging
import random
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef_client import ReefClient, ReefClientError

from honing.backends.reef.records import TAG_HEADER_PREFIX, TAG_NAMES, RecordedDecision, batches
from honing.core import ContractError

logger = logging.getLogger(__name__)

BATCH_PATH = "/reef/records/batch"
DEFAULT_MAX_PENDING_BYTES = 256 * 1024 * 1024
#: Statuses that mean Reef cannot take records now, so the same batch is sent again later, as reef-client's importer
#: retries them.
_UNAVAILABLE = frozenset({408, 429, 500, 502, 503, 504})
#: Statuses of a deployment mistake, each with its count and message. Refusing record by record would only multiply
#: the mistake, so they wait as an outage does, and the memory cap bounds the wait. 404 is also Reef's answer for a
#: scenario it will not create (``ScenarioNotFoundError``).
_MISCONFIGURED = {
    401: ("rejected_token", "Reef rejected the token (REEF_TOKEN)"),
    403: ("rejected_token", "Reef rejected the token (REEF_TOKEN)"),
    404: ("no_route", "no such route, check --reef-url (or a scenario Reef will not create, check --reef-scenario)"),
}
_DROP_LOG_SECONDS = 30.0


@dataclass(frozen=True)
class Skipped:
    """A request or branch that made no record: ``reason`` is ``no_episode``, ``no_id`` or ``invalid``."""

    reason: str
    detail: str


def decision_records(
    request: Mapping[str, Any], response: Mapping[str, Any], tags: Mapping[str, str]
) -> tuple[list[RecordedDecision], list[Skipped]]:
    """The records of one answered decide request, and what was skipped.

    ``request`` and ``response`` are the decide body and the reply as sent.
    The request makes one record, and each of its ``branches`` one more with
    ``branch`` set to its name, since each branch is a request with its own
    ``id``. A request without an ``episode`` tag makes none: its record could not
    have a stable id. An integer id is recorded as its decimal string. Any other
    id that is not a string, or a request #229's checks refuse, is skipped.

    This is the one place that maps a request to records, so recording one
    record per question would change only this function.
    """
    branches = request.get("branches") or {}
    replies = response.get("branches") or {}
    calls: list[tuple[str | None, Mapping[str, Any], Mapping[str, Any]]] = [(None, request, response)]
    calls += [(name, branch, replies.get(name) or {}) for name, branch in branches.items()]
    if not tags.get("episode"):
        return [], [Skipped("no_episode", f"request {request.get('id')!r} has no episode tag")] * len(calls)
    records: list[RecordedDecision] = []
    skipped: list[Skipped] = []
    for name, sent, reply in calls:
        where = "" if name is None else f" (branch {name})"
        request_id = sent.get("id")
        if isinstance(request_id, int) and not isinstance(request_id, bool):
            request_id = str(request_id)
        if not isinstance(request_id, str):
            skipped.append(Skipped("no_id", f"a request{where} has id {request_id!r}, not a string or an integer"))
            continue
        try:
            records.append(
                RecordedDecision(
                    request_id=request_id,
                    parts=sent.get("parts"),  # type: ignore[arg-type]
                    questions=sent.get("questions"),  # type: ignore[arg-type]
                    answers=reply.get("answers"),  # type: ignore[arg-type]
                    tags=tags,
                    branch=name,
                )
            )
        except (ContractError, AttributeError, TypeError) as error:
            skipped.append(Skipped("invalid", f"request {request_id!r}{where}: {error}"))
    return records, skipped


def backoff_seconds(failures: int, low: float, high: float) -> float:
    """The wait after ``failures`` failed attempts in a row: ``low`` doubling to ``high``, times a jitter in [0.5, 1].

    The exponent is capped, since an outage of hours is thousands of attempts and ``2.0 ** 1024`` overflows.
    """
    return min(high, low * 2.0 ** min(failures - 1, 32)) * random.uniform(0.5, 1.0)


def request_tags(headers: Sequence[tuple[str, str]]) -> tuple[dict[str, str], list[str]]:
    """Honing's tags among a request's ``x-reef-tag-*`` headers, and the names of the others, which are ignored.

    Follows Reef's ``parse_request_headers`` (reef/api/wire.py, Reef bc8c150): names are lowercased and an empty
    value is no tag. Differs: only ``TAG_NAMES`` are kept, so an unknown tag cannot make a record invalid.
    """
    tags: dict[str, str] = {}
    ignored: list[str] = []
    for key, value in headers:
        name = key.lower()
        if not name.startswith(TAG_HEADER_PREFIX) or not value.strip():
            continue
        name = name[len(TAG_HEADER_PREFIX) :]
        if name in TAG_NAMES:
            tags[name] = value.strip()
        else:
            ignored.append(name)
    return tags, ignored


@dataclass
class _Request:
    """One answered request waiting on the recorder: its body, reply and headers, and what it is charged."""

    body: bytes
    reply: bytes
    headers: list[tuple[str, str]]
    size: int
    #: Records made from it that Reef has neither stored nor refused yet.
    open_records: int = 0


@dataclass
class _Counts:
    requests: int = 0
    dropped_requests: int = 0
    sent: int = 0
    refused: int = 0
    invalid: int = 0
    skipped_no_episode: int = 0
    skipped_no_id: int = 0
    ignored_tags: int = 0
    rejected_token: int = 0
    no_route: int = 0
    unsent: int = 0


class ReefRecorder:
    """Sends the decision records of the requests handed to :meth:`submit` to one Reef scenario.

    ``url`` is Reef's service URL, ``scenario`` the ``x-reef-scenario`` every
    batch names (one workload in Reef, created by the first import if the
    deployment allows it) and ``token`` its Bearer token. ``flush_seconds`` is
    how long :meth:`close` keeps sending.
    """

    def __init__(
        self,
        url: str,
        scenario: str,
        *,
        token: str | None = None,
        max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES,
        flush_seconds: float = 10.0,
        timeout_seconds: float = 30.0,
        backoff_seconds: tuple[float, float] = (0.5, 30.0),
    ) -> None:
        if not scenario.strip():
            raise ContractError("a Reef recorder needs a scenario")
        if max_pending_bytes <= 0 or flush_seconds < 0 or timeout_seconds <= 0:
            raise ContractError("max_pending_bytes and timeout_seconds must be positive, flush_seconds not negative")
        self.scenario = scenario
        self._client = ReefClient(url, token=token, timeout_s=timeout_seconds)
        self._max_pending_bytes = max_pending_bytes
        self._flush_seconds = flush_seconds
        self._timeout_seconds = timeout_seconds
        self._backoff_seconds = backoff_seconds
        self._lock = threading.Condition()
        self._queued: deque[_Request] = deque()
        #: Records made and not yet stored or refused, in the order they were answered.
        self._outbox: deque[tuple[dict[str, Any], _Request]] = deque()
        self._pending_bytes = 0
        self._counts = _Counts()
        self._closing = False
        self._deadline = float("inf")
        self._last_drop_log = -float("inf")
        #: Why the last batch did not arrive, for the log, and whether it was a deployment mistake, logged apart.
        self._failure = ""
        self._misconfigured = False
        #: When each deployment mistake was last logged.
        self._mistake_logged: dict[str, float] = {}
        self._thread = threading.Thread(target=self._run, name="reef-recorder", daemon=True)
        self._thread.start()

    def submit(self, body: bytes, reply: bytes, headers: Sequence[tuple[str, str]]) -> None:
        """Queue one answered request: its decide body, its reply and its headers. Never blocks on Reef or raises."""
        size = len(body) + len(reply)
        with self._lock:
            self._counts.requests += 1
            if self._closing or self._pending_bytes + size > self._max_pending_bytes:
                self._counts.dropped_requests += 1
                now = time.monotonic()
                if now - self._last_drop_log >= _DROP_LOG_SECONDS:
                    self._last_drop_log = now
                    logger.warning(
                        "Reef recorder: %d requests not recorded so far: %s",
                        self._counts.dropped_requests,
                        (
                            "the recorder is closing"
                            if self._closing
                            else f"{self._pending_bytes} bytes are waiting for Reef, the limit is "
                            f"{self._max_pending_bytes}"
                        ),
                    )
                return
            self._pending_bytes += size
            self._queued.append(_Request(body, reply, list(headers), size))
            self._lock.notify()

    def counts(self) -> dict[str, int]:
        """What the recorder has done so far, for ``/v1/health``."""
        with self._lock:
            counts = dict(vars(self._counts))
            counts["pending_requests"] = len(self._queued)
            counts["pending_records"] = len(self._outbox)
            counts["pending_bytes"] = self._pending_bytes
            return counts

    def close(self) -> None:
        """Send what is queued for up to ``flush_seconds``, then stop; what is left is logged as unsent."""
        with self._lock:
            self._closing = True
            self._deadline = time.monotonic() + self._flush_seconds
            self._lock.notify()
        # A request already on the wire may take its whole timeout.
        self._thread.join(self._flush_seconds + self._timeout_seconds + 1.0)
        with self._lock:
            unsent = len(self._outbox) + len(self._queued)
            self._counts.unsent = unsent
        if unsent:
            logger.warning("Reef recorder: closed with %d records or requests not sent to Reef", unsent)

    def _run(self) -> None:
        failures = 0
        while True:
            with self._lock:
                while not self._queued and not self._outbox and not self._closing:
                    self._lock.wait()
                if self._closing and (time.monotonic() >= self._deadline or not (self._queued or self._outbox)):
                    return
                taken = list(self._queued)
                self._queued.clear()
            for request in taken:
                self._make_records(request)
            if not self._outbox:
                continue
            # batches() encodes records only until its first batch is full, however long the outbox is.
            entries = list(itertools.islice(self._outbox, len(next(batches(record for record, _ in self._outbox)))))
            try:
                settled = self._deliver(entries)
            except Exception as error:  # a fault here must not stop recording: wait as for an outage
                self._failure = f"{type(error).__name__}: {error}"
                logger.warning("Reef recorder: sending a batch failed: %s", self._failure)
                settled = 0
            for _ in range(settled):
                self._outbox.popleft()
            if settled == len(entries):
                if failures:
                    logger.warning("Reef recorder: Reef took records again after %d failed attempts", failures)
                failures = 0
                continue
            failures += 1
            if failures == 1 and not self._misconfigured:
                logger.warning(
                    "Reef recorder: Reef at %s is not taking records (%s); retrying",
                    self._client.service_url,
                    self._failure,
                )
            until = time.monotonic() + backoff_seconds(failures, *self._backoff_seconds)
            with self._lock:
                # A submit notifies too; only the backoff, or the end of the flush, ends the wait, so a busy
                # server does not retry at its request rate while Reef is down.
                while (remaining := min(until, self._deadline) - time.monotonic()) > 0:
                    self._lock.wait(remaining)

    def _make_records(self, request: _Request) -> None:
        try:
            tags, ignored = request_tags(request.headers)
            records, skipped = decision_records(json.loads(request.body), json.loads(request.reply), tags)
        except Exception as error:  # the server answered it, so this is the recorder's fault: say so
            logger.warning("Reef recorder: could not read an answered request: %s: %s", type(error).__name__, error)
            records, skipped, ignored = [], [Skipped("invalid", str(error))], []
        entries = []
        for record in records:
            entry = record.to_record()
            try:
                next(batches([entry]))
            except ContractError as error:  # larger than any batch Reef takes
                skipped.append(Skipped("invalid", str(error)))
                continue
            entries.append(entry)
        with self._lock:
            self._counts.ignored_tags += len(ignored)
            for skip in skipped:
                setattr(self._counts, _COUNT_OF[skip.reason], getattr(self._counts, _COUNT_OF[skip.reason]) + 1)
                if skip.reason == "invalid":
                    logger.warning("Reef recorder: not recorded: %s", skip.detail)
            request.open_records = len(entries)
            if not entries:
                self._pending_bytes -= request.size
        for entry in entries:
            self._outbox.append((entry, request))

    def _deliver(self, entries: list[tuple[dict[str, Any], _Request]]) -> int:
        """Send ``entries`` as one batch; return how many of the first ones Reef stored or refused."""
        try:
            body, _ = self._client.post(BATCH_PATH, self.scenario, {"records": [record for record, _ in entries]})
        except ReefClientError as error:
            self._misconfigured = error.status in _MISCONFIGURED
            if self._misconfigured:
                self._mistake(error)
                return 0
            if error.status in _UNAVAILABLE:
                self._failure = f"HTTP {error.status}: {error.body[:200]}"
                return 0
            if len(entries) == 1:
                self._settle_entries(entries, "refused", f"Reef answered {error.status}: {error.body[:400]}")
                return 1
            # Reef stored none of the batch: find the records it refuses, and store the others.
            half = len(entries) // 2
            settled = self._deliver(entries[:half])
            return settled if settled < half else half + self._deliver(entries[half:])
        except (OSError, http.client.HTTPException, ValueError) as error:
            self._failure, self._misconfigured = f"{type(error).__name__}: {error}", False
            return 0
        expected = [record["agent_record_id"] for record, _ in entries]
        receipts = body.get("records") if isinstance(body, dict) else None
        got = [receipt.get("agent_record_id") for receipt in receipts] if isinstance(receipts, list) else None
        if got != expected:
            self._settle_entries(entries, "refused", f"Reef acknowledged {got}, not the records sent")
        else:
            self._settle_entries(entries, "sent", "")
        return len(entries)

    def _mistake(self, error: ReefClientError) -> None:
        """Count a deployment mistake, and log it the first time and then every 30 s while it lasts."""
        name, message = _MISCONFIGURED[error.status]
        self._failure = f"HTTP {error.status}: {error.body[:200]}"
        with self._lock:
            count = getattr(self._counts, name) + 1
            setattr(self._counts, name, count)
        now, last = time.monotonic(), self._mistake_logged.get(name)
        if last is None:
            logger.error("Reef recorder: %s (%s); records wait and are retried", message, self._failure)
        elif now - last < _DROP_LOG_SECONDS:
            return
        else:
            logger.error("Reef recorder: %s: %d attempts so far", message, count)
        self._mistake_logged[name] = now

    def _settle_entries(self, entries: Sequence[tuple[dict[str, Any], _Request]], outcome: str, detail: str) -> None:
        with self._lock:
            for record, request in entries:
                setattr(self._counts, outcome, getattr(self._counts, outcome) + 1)
                if outcome != "sent":
                    logger.warning("Reef recorder: record %s not stored: %s", record["agent_record_id"], detail)
                request.open_records -= 1
                if request.open_records == 0:
                    self._pending_bytes -= request.size


_COUNT_OF = {"no_episode": "skipped_no_episode", "no_id": "skipped_no_id", "invalid": "invalid"}

__all__ = [
    "BATCH_PATH",
    "DEFAULT_MAX_PENDING_BYTES",
    "ReefRecorder",
    "Skipped",
    "backoff_seconds",
    "decision_records",
    "request_tags",
]
