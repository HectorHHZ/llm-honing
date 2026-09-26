"""Serve one decision model over HTTP.

Routes:

- ``GET /v1/health`` returns ``{"ready": true, "metadata": ...}``.
- ``POST /v1/decide`` takes ``{"parts": {...}, "questions": {...}}``, with an
  optional ``id`` echoed back, and returns the model's answers with the same
  fields the stdio worker emits.

One model loaded on one GPU answers one request at a time; concurrent requests
wait their turn, so many games can share one loaded model. A malformed request
is a 400 and a model failure is a 500, each with ``{"error": ...}``.

Run: ``python -m honing.backends.jev.server --pins <pins.json> --model typed-decisions --gpu 0``
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from honing.backends.jev.model import DecisionModel

MAX_BODY_BYTES = 16 * 1024 * 1024


class RequestError(ValueError):
    """A request the server refuses without calling the model."""


def parse_decide(body: bytes) -> tuple[object, Mapping[str, object], Mapping[str, object]]:
    """Validate a decide request and return its id, parts, and questions."""
    try:
        request = json.loads(body)
    except json.JSONDecodeError as error:
        raise RequestError(f"body is not JSON: {error}") from None
    if not isinstance(request, dict):
        raise RequestError("body must be a JSON object")
    unknown = set(request) - {"id", "parts", "questions"}
    if unknown:
        raise RequestError(f"unknown fields: {', '.join(sorted(unknown))}")
    parts, questions = request.get("parts"), request.get("questions")
    if not isinstance(parts, dict):
        raise RequestError("parts must be an object of state sections")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a non-empty object")
    return request.get("id"), parts, questions


class DecisionServer(ThreadingHTTPServer):
    """An HTTP server around one :class:`DecisionModel`, answering one request at a time."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], model: DecisionModel) -> None:
        super().__init__(address, _Handler)
        self.model = model
        self.lock = threading.Lock()

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}"


class _Handler(BaseHTTPRequestHandler):
    server: DecisionServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/v1/health":
            self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
            return
        self._send(HTTPStatus.OK, {"ready": True, "metadata": dict(self.server.model.metadata)})

    def do_POST(self) -> None:
        if self.path != "/v1/decide":
            self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(HTTPStatus.BAD_REQUEST, {"error": f"body must be 1 to {MAX_BODY_BYTES} bytes"})
            return
        try:
            request_id, parts, questions = parse_decide(self.rfile.read(length))
        except RequestError as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        try:
            with self.server.lock:
                response = dict(self.server.model.predict(parts, questions))
        except Exception as error:  # the model's own failure, reported rather than hidden
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})
            return
        if request_id is not None:
            response["id"] = request_id
        self._send(HTTPStatus.OK, response)


def main(argv: Sequence[str] | None = None) -> None:
    from honing.backends.jev.laya import LayaDecisionModel, LayaSettings

    parser = argparse.ArgumentParser(description="Serve one Laya checkpoint over HTTP.")
    parser.add_argument(
        "--pins", type=Path, required=True, help="pins.json naming the Laya source and model revisions"
    )
    parser.add_argument("--model", required=True, help="model name in pins.json, e.g. typed-decisions")
    parser.add_argument("--model-config-dir", type=Path, required=True, help="directory of prepared model records")
    parser.add_argument("--laya-source", type=Path, required=True, help="Laya checkout at the pinned revision")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint-manifest", type=Path, help="a trained checkpoint to serve instead")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    args = parser.parse_args(argv)
    settings = LayaSettings(
        pins_path=args.pins,
        model_type=args.model,
        model_config_dir=args.model_config_dir,
        source_dir=args.laya_source,
        gpu=args.gpu,
        checkpoint_manifest=args.checkpoint_manifest,
        max_len=args.max_len,
    )
    with LayaDecisionModel.load(settings) as model:
        server = DecisionServer((args.host, args.port), model)
        # One line on stdout tells a launcher where the server is and what it loaded.
        print(json.dumps({"ready": True, "url": server.url, "metadata": dict(model.metadata)}), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
