"""Serve one decision model over HTTP.

Routes:

- ``GET /v1/health`` returns ``{"ready": true, "metadata": ...}``.
- ``POST /v1/decide`` takes ``{"parts": {...}, "questions": {...}}``, with an
  optional ``id`` echoed back, and returns the model's answers with the same
  fields the stdio worker emits.

One model loaded on one GPU answers one request at a time; concurrent requests
wait their turn, so many games can share one loaded model. Every model call runs
on the thread that runs :meth:`DecisionServer.serve`, the one that loaded the
model, never on a request's thread: a CUDA kernel can be chosen per thread, and
NanoJev answered a few requests differently (by up to 0.0015) on a new thread
than on the loading thread. The server therefore answers exactly as the stdio
worker does. A malformed request is a 400 and a model failure is a 500, each
with ``{"error": ...}``.

Run ``python -m honing.backends.jev.server --pins <pins.json> --gpu 0`` with the family's settings:

- Laya: ``--model typed-decisions --model-config-dir <records> --laya-source <checkout>``
- Kev: ``--family kev --kev-source <checkout>``
- NanoJev: ``--family nanojev --nanojev-source <checkout>``
- djev (DiffusionGemma): ``--family djev --djev-source <checkout> --djev-cache <local dir>`` starts a deployment on
  the GPU, or ``--family djev --djev-url <url>`` uses a running one; run it with the djev environment's Python
"""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from honing.backends.jev.model import DecisionModel

MAX_BODY_BYTES = 16 * 1024 * 1024
DJEV_MODEL = "diffusiongemma-26b-a4b-it"


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


@dataclass
class _Decision:
    """One model call waiting for the model thread."""

    parts: Mapping[str, object]
    questions: Mapping[str, object]
    done: threading.Event = field(default_factory=threading.Event)
    response: Mapping[str, object] | None = None
    error: Exception | None = None


class DecisionServer(ThreadingHTTPServer):
    """An HTTP server around one :class:`DecisionModel`, answering one request at a time on the model's thread.

    :meth:`serve` accepts requests on a background thread and runs every model
    call on the calling thread until :meth:`stop`.
    """

    daemon_threads = True

    def __init__(self, address: tuple[str, int], model: DecisionModel) -> None:
        super().__init__(address, _Handler)
        self.model = model
        self._decisions: queue.Queue[_Decision | None] = queue.Queue()

    def decide(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> dict[str, object]:
        """Hand one call to the model thread and wait for its answer; a model failure is raised here."""
        decision = _Decision(parts, questions)
        self._decisions.put(decision)
        decision.done.wait()
        if decision.error is not None:
            raise decision.error
        return dict(decision.response or {})

    def serve(self) -> None:
        """Accept requests on a background thread and answer them on this one until :meth:`stop`."""
        accepting = threading.Thread(target=self.serve_forever, daemon=True)
        accepting.start()
        try:
            while (decision := self._decisions.get()) is not None:
                self._answer(decision)
        finally:
            self.shutdown()
            accepting.join()

    def _answer(self, decision: _Decision) -> None:
        try:
            decision.response = self.model.predict(decision.parts, decision.questions)
        except Exception as error:  # reported to the waiting request as a 500
            decision.error = error
        finally:
            decision.done.set()

    def stop(self) -> None:
        """Make :meth:`serve` return once the calls already handed over are answered."""
        self._decisions.put(None)

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
            response = self.server.decide(parts, questions)
        except Exception as error:  # the model's own failure, reported rather than hidden
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})
            return
        if request_id is not None:
            response["id"] = request_id
        self._send(HTTPStatus.OK, response)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve one decision model (Laya or Kev) over HTTP.")
    parser.add_argument("--family", choices=("laya", "kev", "nanojev", "djev"), default="laya")
    parser.add_argument(
        "--pins", type=Path, required=True, help="pins.json naming the model source and model revisions"
    )
    parser.add_argument(
        "--model", help="model name in pins.json: typed-decisions (laya), kev-0.8b or kev-4b (kev), nanojev (nanojev)"
    )
    parser.add_argument("--model-config-dir", type=Path, help="laya: directory of prepared model records")
    parser.add_argument("--laya-source", type=Path, help="laya: Laya checkout at the pinned revision")
    parser.add_argument("--kev-source", type=Path, help="kev: Kev checkout at the pinned revision")
    parser.add_argument("--nanojev-source", type=Path, help="nanojev: NanoJev checkout at the pinned revision")
    parser.add_argument("--djev-url", help="djev: a running djev deployment's API to use")
    parser.add_argument("--djev-source", type=Path, help="djev: djev checkout at the pinned revision, to start one")
    parser.add_argument("--djev-cache", type=Path, help="djev: local directory for the deployment's compile caches")
    parser.add_argument("--djev-ports", default="8100,8101", help="djev: API and vLLM ports of a started deployment")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="kev: serving precision")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint-manifest", type=Path, help="a trained checkpoint to serve instead")
    parser.add_argument("--max-len", type=int, default=2048, help="laya, nanojev: input token limit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    args = parser.parse_args(argv)
    # SIGTERM ends the server through its context managers, so a model holding processes or a GPU is released.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    with _load(parser, args) as model:
        server = DecisionServer((args.host, args.port), model)
        # One line on stdout tells a launcher where the server is and what it loaded.
        print(json.dumps({"ready": True, "url": server.url, "metadata": dict(model.metadata)}), flush=True)
        try:
            server.serve()
        finally:
            server.server_close()


def _load(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Any:
    """The model's loading context for the chosen family; a missing family setting is a usage error."""
    if args.family == "djev":
        from honing.backends.djev import DjevDecisionModel, DjevRuntimeSettings, DjevSettings

        if (args.djev_url is None) == (args.djev_source is None):
            parser.error("--family djev needs --djev-url (a running deployment) or --djev-source (to start one)")
        runtime = None
        if args.djev_source is not None:
            api_port, model_port = (int(p) for p in args.djev_ports.split(","))
            runtime = DjevRuntimeSettings(
                pins_path=args.pins,
                source_dir=args.djev_source,
                python=Path(sys.executable),
                gpu=args.gpu,
                api_port=api_port,
                model_port=model_port,
                cache_dir=args.djev_cache,
            )
        return DjevDecisionModel.load(
            DjevSettings(pins_path=args.pins, runtime=runtime, url=args.djev_url, model=args.model or DJEV_MODEL)
        )
    if args.family == "nanojev":
        from honing.backends.jev.nanojev import NanoJevDecisionModel, NanoJevSettings

        if args.nanojev_source is None:
            parser.error("--family nanojev needs --nanojev-source")
        return NanoJevDecisionModel.load(
            NanoJevSettings(
                pins_path=args.pins,
                source_dir=args.nanojev_source,
                model=args.model or "nanojev",
                gpu=args.gpu,
                checkpoint_manifest=args.checkpoint_manifest,
                max_length=args.max_len,
            )
        )
    if args.family == "kev":
        from honing.backends.jev.kev import KevDecisionModel, KevSettings

        if args.kev_source is None:
            parser.error("--family kev needs --kev-source")
        return KevDecisionModel.load(
            KevSettings(
                pins_path=args.pins,
                source_dir=args.kev_source,
                model=args.model or "kev-0.8b",
                gpu=args.gpu,
                checkpoint_manifest=args.checkpoint_manifest,
                dtype=args.dtype,
            )
        )
    from honing.backends.jev.laya import LayaDecisionModel, LayaSettings

    if args.model is None or args.model_config_dir is None or args.laya_source is None:
        parser.error("--family laya needs --model, --model-config-dir, and --laya-source")
    return LayaDecisionModel.load(
        LayaSettings(
            pins_path=args.pins,
            model_type=args.model,
            model_config_dir=args.model_config_dir,
            source_dir=args.laya_source,
            gpu=args.gpu,
            checkpoint_manifest=args.checkpoint_manifest,
            max_len=args.max_len,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
