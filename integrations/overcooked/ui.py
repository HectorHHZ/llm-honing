"""Loopback GUI showing native PNGs and the exact public map/state/rules."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from integrations.overcooked.decisions import DecisionClient, make_request
from integrations.overcooked.engine import ACTIONS, LAYOUTS, ROOT, Engine, Scenario, canonical
from integrations.overcooked.policy import joint_baseline
from integrations.overcooked.recording import Recorder, new_run, replay


class Application:
    def __init__(self, scenario: Scenario, decision_server: str | None = None, replay_dir: Path | None = None) -> None:
        self.lock = threading.RLock()
        self.busy = False
        self.client = DecisionClient(decision_server) if decision_server else None
        self.replay_data = replay(replay_dir) if replay_dir else None
        if self.replay_data:
            scenario = Scenario(**self.replay_data["manifest"]["scenario"])
        self.engine = Engine(scenario)
        self.history: list[dict[str, Any]] = []
        self.index = 0
        self.recorder: Recorder | None = None
        if not self.replay_data:
            self.recorder = Recorder(self.engine, new_run("live"), {"controller": "human", "kind": "interactive"})
        self.frame = self.replay_data["frames"][0] if self.replay_data else self.engine.observe()
        self.picture = self.engine.png(self.frame)
        self.pictures = {self.frame["frame_hash"]: self.picture}

    def refresh_picture(self) -> None:
        self.picture = self.engine.png(self.frame)
        self.pictures[self.frame["frame_hash"]] = self.picture
        while len(self.pictures) > 64:
            del self.pictures[next(iter(self.pictures))]

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "frame": self.frame,
                "busy": self.busy,
                "layouts": LAYOUTS,
                "scenario": self.engine.scenario.record(),
                "model_available": self.client is not None,
                "replay": self.replay_data is not None,
                "replay_length": len(self.replay_data["frames"]) if self.replay_data else 0,
                "replay_index": self.index,
            }

    def advance(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.busy:
                raise ValueError("a decision is already pending")
            if self.replay_data:
                index = payload.get("index", self.index + 1)
                if type(index) is not int or not 0 <= index < len(self.replay_data["frames"]):
                    raise ValueError("replay index is outside the recording")
                self.index = index
                self.frame = self.replay_data["frames"][index]
                self.refresh_picture()
                return self.state()
            if self.frame["status"] != "running":
                raise ValueError("episode has finished; reset to start a new kitchen")
            expected = payload.get("frame_hash")
            if expected != self.frame["frame_hash"]:
                raise ValueError("stale observation; refresh before choosing an action")
            mode = payload.get("policy", "human")
            if mode not in {"human", "baseline", "model"}:
                raise ValueError("unknown controller")
            if mode == "model" and self.client is None:
                raise ValueError("start the server with --decision-server to enable model play")
            action = payload.get("action", "STAY")
            if mode == "human" and action not in ACTIONS:
                raise ValueError("unknown native action")
            self.busy = True
            before = self.frame
            scenario = self.engine.scenario
            actions = joint_baseline(before, scenario.controlled_player, scenario.partner)
        try:
            # The HTTP wait occurs outside the state lock; observers can still read the frozen frame.
            request, response = None, None
            if mode == "model":
                request = make_request(before, scenario.controlled_player, history=self.history[-4:])
                response = self.client.decide(request)
                action = response["answers"]["action"]["choice"]
            elif mode == "baseline":
                action = actions[scenario.controlled_player]
            actions[scenario.controlled_player] = action
            with self.lock:
                self.frame = self.engine.step(actions)
                self.recorder.append(
                    before,
                    actions,
                    self.frame,
                    {
                        # UI recordings are interactive demonstrations, not graded model episodes.
                        "request": None,
                        "response": None,
                        "controller": mode,
                        "model_request": request,
                        "model_response": response,
                        "chosen": action,
                    },
                )
                self.history.append({"actions": actions, "reward": self.frame["reward"]})
                self.refresh_picture()
                if self.frame["status"] != "running":
                    self.recorder.finish()
        finally:
            with self.lock:
                self.busy = False
        return self.state()

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.busy:
                raise ValueError("wait for the pending decision before resetting")
            if self.replay_data:
                return self.advance({"index": 0})
            allowed = {"layout", "old_dynamics", "controlled_player", "partner", "max_steps", "target_deliveries"}
            if set(payload) - allowed:
                raise ValueError("unknown scenario field")
            scenario = replace(self.engine.scenario, **payload)
            self.recorder.finish()
            self.engine.close()
            self.engine = Engine(scenario)
            self.recorder = Recorder(self.engine, new_run("live"), {"controller": "human", "kind": "interactive"})
            self.history.clear()
            self.frame = self.engine.observe()
            self.refresh_picture()
            return self.state()

    def close(self) -> None:
        with self.lock:
            if self.recorder:
                self.recorder.finish()
            self.engine.close()


def make_server(app: Application, port: int = 4185) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            pass

        def send(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def json(self, value: object, status: int = 200) -> None:
            self.send(canonical(value).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                self.send((ROOT / "web/index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/frame.png":
                with app.lock:
                    wanted = parse_qs(urlsplit(self.path).query).get("hash", [app.frame["frame_hash"]])[0]
                    picture = app.pictures.get(wanted)
                if picture is None:
                    self.json({"error": "frame expired; refresh the observation"}, 409)
                else:
                    self.send(picture, "image/png")
            elif path in {"/api/state", "/api/map", "/api/rules"}:
                state = app.state()
                self.json(state if path == "/api/state" else state["frame"][path.rsplit("/", 1)[1]])
            else:
                self.json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            try:
                # Reject cross-origin browser commands; service is bound to loopback only.
                origin = self.headers.get("Origin")
                if origin and origin not in {
                    f"http://127.0.0.1:{self.server.server_port}",
                    f"http://localhost:{self.server.server_port}",
                }:
                    self.json({"error": "cross-origin command refused"}, 403)
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("request must contain a JSON body no larger than 64 KiB")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("request body must be an object")
                if self.path == "/api/step":
                    self.json(app.advance(data))
                elif self.path == "/api/reset":
                    self.json(app.reset(data))
                elif self.path == "/api/export":
                    with app.lock:
                        if app.busy or app.recorder is None:
                            raise ValueError("cannot export while busy or replaying")
                        self.json({"out": str(app.recorder.snapshot())})
                else:
                    self.json({"error": "not found"}, 404)
            except (ValueError, TypeError, KeyError) as error:
                self.json({"error": str(error)}, 400)
            except Exception as error:
                self.json({"error": str(error)}, 502)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4185)
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--old-dynamics", action="store_true")
    parser.add_argument("--decision-server")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    app = Application(
        Scenario(layout=args.layout, max_steps=args.max_steps, old_dynamics=args.old_dynamics),
        args.decision_server,
        args.replay,
    )
    server = make_server(app, args.port)
    print(f"Overcooked GUI: http://127.0.0.1:{server.server_port}/", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{server.server_port}/")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
