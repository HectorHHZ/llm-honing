"""Stands in for Reef's record import route, ``POST /reef/records/batch``, and its report route, ``POST /reef/report``.

Follows Reef's ``import_records`` (reef/api/request_handling.py), ``accept_records``
(reef/control/dispatcher.py) and ``RecordStore`` (reef/storage/scenario/records.py),
Reef bc8c150:

- the ``x-reef-scenario`` header names the scenario, or the request is a 400;
- the body is exactly ``{"records": [...]}`` with 1 to 1000 records of exactly
  ``agent_record_id``, ``request_type`` and ``payload``, at most 10 MiB, or a 400 (413);
- a batch is stored whole or not at all;
- an identical resend returns the stored receipt, and the same id with other content,
  stored or earlier in the batch, is a 409;
- ``x-reef-tag-*`` headers are merged into an inference's ``payload.metadata.tags``;
- a report is checked as a record-only recipe checks it (no report type): a client
  ``agent_record_id`` or a drawn one, a finite number or no ``score``, a string or
  object ``feedback``, an object ``metadata``, and unique ``references`` that each
  name a stored inference of the scenario, or a 400 with Reef's message. Its payload
  is stored as Reef normalizes it (``ReportPayload.to_dict``), so an identical
  resend returns the stored receipt and a changed one is a 409.

``down`` makes every request a 503, as an overloaded Reef answers, and ``answer`` makes it
that status instead (401 for a wrong token, 404 for a wrong URL). Payloads are compared
and returned as stored, with no key sorting.

The report route follows Reef's ``accept_typed`` (reef/api/routes/records.py),
``ReportPayload`` (reef/api/wire.py) and ``validate_record`` (reef/control/ingest.py),
Reef bc8c150. Differs: no report type, and no trajectory rule, since decision
records link into no trajectory.
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MAX_BYTES = 10 * 1024 * 1024


class FakeReef(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int = 0) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.down = False
        self.answer: HTTPStatus | None = None
        self.lock = threading.Lock()
        #: (scenario, record id) -> (request type, payload), in arrival order.
        self.records: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self.posts = 0
        self.headers_seen: list[dict[str, str]] = []
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def payloads(self, scenario: str) -> dict[str, dict[str, Any]]:
        with self.lock:
            return {record_id: payload for (name, record_id), (_, payload) in self.records.items() if name == scenario}

    def reports(self, scenario: str) -> dict[str, dict[str, Any]]:
        """The scenario's stored reports by id, as Reef normalized them."""
        with self.lock:
            return {
                record_id: payload
                for (name, record_id), (kind, payload) in self.records.items()
                if name == scenario and kind == "report"
            }

    def close(self) -> None:
        self.shutdown()
        self._thread.join()
        self.server_close()


class _RefusedError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class _Handler(BaseHTTPRequestHandler):
    server: FakeReef

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        with self.server.lock:
            self.server.posts += 1
            self.server.headers_seen.append({key.lower(): value for key, value in self.headers.items()})
            if self.server.answer is not None:
                self._send(self.server.answer, {"error": f"answering {int(self.server.answer)}"})
                return
            if self.server.down:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Reef is down"})
                return
            try:
                if self.path == "/reef/report":
                    self._send(HTTPStatus.OK, self._report(body))
                    return
                receipts = self._import(body)
            except _RefusedError as refused:
                self._send(refused.status, {"error": str(refused)})
                return
        self._send(HTTPStatus.OK, {"records": receipts})

    def _report(self, body: bytes) -> dict[str, str]:
        scenario = (self.headers.get("x-reef-scenario") or "").strip()
        if not scenario:
            raise _RefusedError(HTTPStatus.BAD_REQUEST, "missing or empty x-reef-scenario")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise _RefusedError(HTTPStatus.BAD_REQUEST, "a report is a JSON object")
        record_id = payload.pop("agent_record_id", None) or uuid.uuid4().hex
        if not isinstance(record_id, str) or not record_id.strip():
            raise _RefusedError(HTTPStatus.BAD_REQUEST, "agent_record_id must be a non-empty string")
        score, feedback = payload.get("score"), payload.get("feedback")
        references, metadata = payload.get("references", []), payload.get("metadata", {})
        if score is not None and (
            isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
        ):
            raise _RefusedError(HTTPStatus.BAD_REQUEST, "score must be a finite number")
        if feedback is not None and not isinstance(feedback, (str, dict)):
            raise _RefusedError(HTTPStatus.BAD_REQUEST, "feedback must be a string or an object")
        if not isinstance(metadata, dict):
            raise _RefusedError(HTTPStatus.BAD_REQUEST, "metadata must be an object")
        if not isinstance(references, list) or not all(isinstance(reference, str) for reference in references):
            raise _RefusedError(HTTPStatus.BAD_REQUEST, "references must be a list of strings")
        normalized = {
            key: value
            for key, value in (
                ("score", None if score is None else float(score)),
                ("feedback", feedback),
                ("references", references),
                ("metadata", metadata),
            )
            if value not in (None, [], {})
        }
        content = ("report", normalized)
        stored = self.server.records.get((scenario, record_id))
        if stored == content:
            return {"agent_record_id": record_id, "scenario": scenario, "request_type": "report"}
        if len(set(references)) != len(references):
            raise _RefusedError(HTTPStatus.BAD_REQUEST, "report references must be unique")
        for reference in references:
            if self.server.records.get((scenario, reference), ("",))[0] != "inference":
                raise _RefusedError(
                    HTTPStatus.BAD_REQUEST,
                    f"report reference {reference!r} must identify an existing inference in scenario {scenario!r}",
                )
        if stored is not None:
            raise _RefusedError(HTTPStatus.CONFLICT, f"record {record_id!r} is stored with other content")
        self.server.records[(scenario, record_id)] = content
        return {"agent_record_id": record_id, "scenario": scenario, "request_type": "report"}

    def _import(self, body: bytes) -> list[dict[str, str]]:
        if self.path != "/reef/records/batch":
            raise _RefusedError(HTTPStatus.NOT_FOUND, f"no route {self.path}")
        scenario = (self.headers.get("x-reef-scenario") or "").strip()
        if not scenario:
            raise _RefusedError(HTTPStatus.BAD_REQUEST, "missing or empty x-reef-scenario")
        if len(body) > MAX_BYTES:
            raise _RefusedError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
        request = json.loads(body)
        records = request.get("records") if isinstance(request, dict) else None
        if set(request or ()) != {"records"} or not isinstance(records, list) or not 1 <= len(records) <= 1000:
            raise _RefusedError(HTTPStatus.BAD_REQUEST, "body must contain only 'records', 1 to 1000 records")
        tags = {
            key.lower()[len("x-reef-tag-") :]: value.strip()
            for key, value in self.headers.items()
            if key.lower().startswith("x-reef-tag-") and value.strip()
        }
        batch: dict[str, tuple[str, dict[str, Any]]] = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != {"agent_record_id", "request_type", "payload"}:
                raise _RefusedError(HTTPStatus.BAD_REQUEST, "a record is agent_record_id, request_type and payload")
            record_id, kind, payload = record["agent_record_id"], record["request_type"], record["payload"]
            if not isinstance(record_id, str) or not record_id.strip() or kind not in ("inference", "report"):
                raise _RefusedError(HTTPStatus.BAD_REQUEST, "a record needs an id and an inference or report type")
            if kind == "inference" and tags:
                metadata = dict(payload.get("metadata") or {})
                metadata["tags"] = {**(metadata.get("tags") or {}), **tags}
                payload = {**payload, "metadata": metadata}
            content = (kind, payload)
            if record_id in batch and batch[record_id] != content:
                raise _RefusedError(HTTPStatus.CONFLICT, f"record {record_id!r} is given twice with other content")
            stored = self.server.records.get((scenario, record_id))
            if stored is not None and stored != content:
                raise _RefusedError(HTTPStatus.CONFLICT, f"record {record_id!r} is stored with other content")
            batch[record_id] = content
        for record_id, content in batch.items():
            self.server.records.setdefault((scenario, record_id), content)
        return [
            {
                "agent_record_id": record["agent_record_id"],
                "scenario": scenario,
                "request_type": record["request_type"],
            }
            for record in records
        ]


__all__ = ["FakeReef"]
