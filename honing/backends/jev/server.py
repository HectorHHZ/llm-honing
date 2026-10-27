"""Serve one decision model over HTTP.

Routes:

- ``GET /v1/health`` returns ``{"ready": true, "metadata": ..., "maxBranches": 16}``.
- ``POST /v1/decide`` takes ``{"parts": {...}, "questions": {...}}``, with an
  optional ``id`` echoed back, and returns the model's answers with the same
  fields the stdio worker emits. An optional ``branches`` object names up to
  ``maxBranches`` more requests of the same shape (``{"id", "parts",
  "questions"}``, no further branches), answered in the same model call; the
  response then carries ``branches`` with each one's answer under its name. Each
  branch is answered as if it had been sent alone. A game that asks a follow-up
  question whose state depends on the first answer, such as Tank War's weapon
  question after its motion question, sends the follow-up for every answer it
  may choose as a branch and keeps the one it chooses.

One model loaded on one GPU answers one call at a time; concurrent requests
wait their turn, so many games can share one loaded model. Every model call runs
on the thread that runs :meth:`DecisionServer.serve`, the one that loaded the
model, never on a request's thread: a CUDA kernel can be chosen per thread, and
NanoJev answered a few requests differently (by up to 0.0015) on a new thread
than on the loading thread. The server therefore answers exactly as the stdio
worker does. A model that offers ``prepare`` and ``run`` (NanoJev's tree modes)
has its CPU-only ``prepare`` (encoding the request) run on the request's thread,
so one call is encoded while the GPU runs another, and ``run`` on the model
thread. A malformed request is a 400 and a model failure is a 500, each with
``{"error": ...}``.

Run ``python -m honing.backends.jev.server --pins <pins.json> --gpu 0`` with the family's settings:

- Laya: ``--model typed-decisions --model-config-dir <records> --laya-source <checkout>``
- Kev: ``--family kev --kev-source <checkout>``
- NanoJev: ``--family nanojev --nanojev-source <checkout>``, optionally ``--nanojev-inference tree-compiled``
- StartLux-Decision: ``--family startlux --startlux-source <checkout> --model startlux-4b``
- Nimble-v4-max: ``--family nimble --nimble-source <source snapshot> --nimble-kit <checkout>``
- djev: ``--family djev --djev-url http://127.0.0.1:8011 --model dgemma`` (no local pins/GPU)
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from honing.backends.jev.model import DecisionModel
from honing.backends.jev.nanojev_tree import INFERENCE_MODES

MAX_BODY_BYTES = 16 * 1024 * 1024
# Requests a decide body may add under ``branches``; Tank War sends one per admissible motion, at most nine.
MAX_BRANCHES = 16


class RequestError(ValueError):
    """A request the server refuses without calling the model."""


@dataclass(frozen=True)
class DecideRequest:
    """One request to answer: its optional id, echoed back, the state's parts and the questions."""

    id: object
    parts: Mapping[str, object]
    questions: Mapping[str, object]


def _request(request: object, fields: set[str], branch: str | None = None) -> DecideRequest:
    """One request's fields, checked; ``branch`` names the branch it is in the error messages."""
    where = "" if branch is None else f" in branch {branch}"
    if not isinstance(request, dict):
        raise RequestError(f"{'body' if branch is None else 'branch ' + branch} must be a JSON object")
    unknown = set(request) - fields
    if unknown:
        raise RequestError(f"unknown fields{where}: {', '.join(sorted(unknown))}")
    parts, questions = request.get("parts"), request.get("questions")
    if not isinstance(parts, dict):
        raise RequestError(f"parts{where} must be an object of state sections")
    if not isinstance(questions, dict) or not questions:
        raise RequestError(f"questions{where} must be a non-empty object")
    return DecideRequest(request.get("id"), parts, questions)


def parse_decide(body: bytes) -> tuple[DecideRequest, dict[str, DecideRequest]]:
    """Validate a decide request and return it with its named branches (empty without ``branches``)."""
    try:
        request = json.loads(body)
    except json.JSONDecodeError as error:
        raise RequestError(f"body is not JSON: {error}") from None
    first = _request(request, {"id", "parts", "questions", "branches"})
    branches = request.get("branches", {})
    if not isinstance(branches, dict) or len(branches) > MAX_BRANCHES:
        raise RequestError(f"branches must be an object of at most {MAX_BRANCHES} requests")
    fields = {"id", "parts", "questions"}
    return first, {name: _request(branch, fields, name) for name, branch in branches.items()}


@dataclass
class _Decision:
    """One model call, of one or more requests, waiting for the model thread; ``prepared`` is the model's own
    encoding of the requests when it offers ``prepare``."""

    requests: list[tuple[Mapping[str, object], Mapping[str, object]]]
    prepared: object = None
    done: threading.Event = field(default_factory=threading.Event)
    responses: list[Mapping[str, object]] = field(default_factory=list)
    error: Exception | None = None


class DecisionServer(ThreadingHTTPServer):
    """An HTTP server around one :class:`DecisionModel`, answering one call at a time on the model's thread.

    :meth:`serve` accepts requests on a background thread and runs every model
    call on the calling thread until :meth:`stop`. A call answers one request and
    its branches, with the model's ``run`` (after ``prepare`` on the request's
    thread), its ``predict_many``, or ``predict`` once per request, whichever it offers.
    """

    daemon_threads = True

    def __init__(self, address: tuple[str, int], model: DecisionModel) -> None:
        super().__init__(address, _Handler)
        self.model = model
        self._decisions: queue.Queue[_Decision | None] = queue.Queue()

    def decide(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> dict[str, object]:
        """Hand one call to the model thread and wait for its answer; a model failure is raised here."""
        return self.decide_many([(parts, questions)])[0]

    def decide_many(
        self, requests: Sequence[tuple[Mapping[str, object], Mapping[str, object]]]
    ) -> list[dict[str, object]]:
        """Answer several requests in one model call, each as if alone, in order; a model failure is raised here.

        A model with ``prepare`` encodes them here, on the request's thread; the model thread then runs them.
        """
        prepare = getattr(self.model, "prepare", None)
        decision = _Decision(list(requests))
        if callable(prepare):
            decision.prepared = prepare(decision.requests)
        self._decisions.put(decision)
        decision.done.wait()
        if decision.error is not None:
            raise decision.error
        return [dict(response) for response in decision.responses]

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
        model: Any = self.model
        try:
            if decision.prepared is not None:
                decision.responses = list(model.run(decision.prepared))
            elif len(decision.requests) > 1 and callable(getattr(model, "predict_many", None)):
                decision.responses = list(model.predict_many(decision.requests))
            else:
                decision.responses = [model.predict(parts, questions) for parts, questions in decision.requests]
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
        self._send(
            HTTPStatus.OK, {"ready": True, "metadata": dict(self.server.model.metadata), "maxBranches": MAX_BRANCHES}
        )

    def do_POST(self) -> None:
        if self.path != "/v1/decide":
            self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(HTTPStatus.BAD_REQUEST, {"error": f"body must be 1 to {MAX_BODY_BYTES} bytes"})
            return
        try:
            first, branches = parse_decide(self.rfile.read(length))
        except RequestError as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        requests = [first, *branches.values()]
        try:
            responses = self.server.decide_many([(r.parts, r.questions) for r in requests])
        except Exception as error:  # the model's own failure, reported rather than hidden
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})
            return
        for request, response in zip(requests, responses, strict=True):
            if request.id is not None:
                response["id"] = request.id
        response = responses[0]
        if branches:
            response["branches"] = dict(zip(branches, responses[1:], strict=True))
        self._send(HTTPStatus.OK, response)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve a local decision model or bridge a remote djev server.")
    parser.add_argument("--family", choices=("laya", "kev", "nanojev", "startlux", "nimble", "djev"), default="laya")
    parser.add_argument("--pins", type=Path, help="local families: pins.json naming source and model revisions")
    parser.add_argument(
        "--model",
        help="local model name in pins.json, or the upstream served name for djev",
    )
    parser.add_argument("--model-config-dir", type=Path, help="laya: directory of prepared model records")
    parser.add_argument("--laya-source", type=Path, help="laya: Laya checkout at the pinned revision")
    parser.add_argument("--kev-source", type=Path, help="kev: Kev checkout at the pinned revision")
    parser.add_argument("--nanojev-source", type=Path, help="nanojev: NanoJev checkout at the pinned revision")
    parser.add_argument(
        "--nanojev-inference",
        choices=INFERENCE_MODES,
        default="reference",
        help="nanojev: score each candidate path on its own (reference) or a request's paths as one token tree",
    )
    parser.add_argument(
        "--startlux-source", type=Path, help="startlux: StartLux-Decision checkout at the pinned revision"
    )
    parser.add_argument(
        "--nimble-source", type=Path, help="nimble: snapshot of the pinned source dataset, holding code/nimble"
    )
    parser.add_argument("--nimble-kit", type=Path, help="nimble: Decision Index kit checkout at the pinned revision")
    parser.add_argument("--djev-url", help="djev: upstream base URL, optionally ending in /v1")
    parser.add_argument("--djev-timeout", type=float, default=60.0)
    parser.add_argument("--djev-samples", type=int, default=1)
    parser.add_argument("--djev-steps", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="kev: serving precision")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint-manifest", type=Path, help="a trained checkpoint to serve instead")
    parser.add_argument(
        "--max-len",
        type=int,
        help="laya, nanojev (default 2048), startlux, nimble (default 8192): input token limit",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    args = parser.parse_args(argv)
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
        from honing.backends.jev.djev import DjevDecisionModel, DjevSettings

        if not args.djev_url or not args.model:
            parser.error("--family djev needs --djev-url and --model (the upstream served name)")
        if args.checkpoint_manifest is not None or args.pins is not None:
            parser.error("djev is remote: local pins/checkpoint manifests cannot verify or select its weights")
        return DjevDecisionModel.load(
            DjevSettings(
                url=args.djev_url,
                model=args.model,
                timeout=args.djev_timeout,
                samples=args.djev_samples,
                steps=args.djev_steps,
            )
        )
    if args.pins is None:
        parser.error("local model families need --pins")
    if args.family == "startlux":
        from honing.backends.jev.startlux import StartLuxDecisionModel, StartLuxSettings

        if args.startlux_source is None:
            parser.error("--family startlux needs --startlux-source")
        if args.checkpoint_manifest is not None:
            parser.error("startlux serves the pinned releases only; it has no trained checkpoints yet")
        return StartLuxDecisionModel.load(
            StartLuxSettings(
                pins_path=args.pins,
                source_dir=args.startlux_source,
                model=args.model or "startlux-4b",
                gpu=args.gpu,
                max_length=args.max_len or 8192,
            )
        )
    if args.family == "nimble":
        from honing.backends.jev.nimble import NimbleDecisionModel, NimbleSettings

        if args.nimble_source is None or args.nimble_kit is None:
            parser.error("--family nimble needs --nimble-source and --nimble-kit")
        if args.checkpoint_manifest is not None:
            parser.error("nimble serves the pinned release only; it has no trained checkpoints yet")
        return NimbleDecisionModel.load(
            NimbleSettings(
                pins_path=args.pins,
                source_dir=args.nimble_source,
                kit_dir=args.nimble_kit,
                model=args.model or "nimble-v4-max",
                gpu=args.gpu,
                max_length=args.max_len or 8192,
            )
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
                max_length=args.max_len or 2048,
                inference=args.nanojev_inference,
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
            max_len=args.max_len or 2048,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
