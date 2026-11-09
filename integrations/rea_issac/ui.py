#!/usr/bin/env python3
"""Local UI and a synchronous, explicit-step HTTP interface."""

import argparse
import json
import threading
import uuid
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO

from integrations.rea_issac.engine import VERSION, Action, Engine, replay
from integrations.rea_issac.policy import Baseline

ROOT = Path(__file__).resolve().parent


class Session:
    def __init__(self, seed: int) -> None:
        self.lock = threading.RLock()
        self.log: TextIO | None = None
        self.engine: Engine
        self.reset(seed)

    def reset(self, seed: int, max_ticks: int = 18000) -> dict[str, Any]:
        new_engine = Engine(seed, max_ticks)
        if self.log:
            self.save("reset")
            self.log.close()
        self.engine = new_engine
        self.baseline = Baseline()
        self.actions: list[dict[str, Any]] = []
        self.bot = False
        self.playback: list[dict[str, Any]] | None = None
        self.playback_index = 0
        self.saved_terminal = False
        self.run = (
            ROOT
            / "runs"
            / (
                "s"
                + str(seed)
                + "-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + uuid.uuid4().hex[:6]
            )
        )
        self.run.mkdir(parents=True, exist_ok=False)
        self.log = (self.run / "actions.jsonl").open("w", buffering=1)
        (self.run / "manifest.json").write_text(
            json.dumps(
                {
                    "engine_version": VERSION,
                    "seed": seed,
                    "max_ticks": max_ticks,
                    "initial_hash": new_engine.digest(),
                    "observation": "current room and discovered map only",
                },
                indent=2,
            )
            + "\n"
        )
        return self.state()

    def export(self) -> dict[str, Any]:
        return {
            "engine_version": VERSION,
            "seed": self.engine.seed,
            "max_ticks": self.engine.max_ticks,
            "actions": self.actions,
            "final_hash": self.engine.digest(),
            "result": self.engine.result(),
        }

    def save(self, reason: str = "checkpoint") -> None:
        (self.run / "replay.json").write_text(json.dumps(self.export(), separators=(",", ":")))
        (self.run / "result.json").write_text(
            json.dumps({"recording_end": reason, **self.engine.result()}, indent=2) + "\n"
        )

    def state(self) -> dict[str, Any]:
        return {
            "observation": self.engine.observe(),
            "controller": "replay" if self.playback is not None else "baseline" if self.bot else "human",
            "run": self.run.name,
            "replay_index": self.playback_index,
            "replay_length": len(self.playback) if self.playback is not None else 0,
            "playback_complete": self.playback is not None and self.playback_index >= len(self.playback),
        }

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.playback is not None:
            if self.playback_index >= len(self.playback):
                return self.state()
            entry = self.playback[self.playback_index]
            self.playback_index += 1
        else:
            action = self.baseline.act(self.engine.observe()) if self.bot else payload.get("action", {})
            action = Action.parse(action)
            entry = {
                "action": dict(action.__dict__),
                "frames": payload.get("frames", 1),
            }
        before = self.engine.tick
        self.engine.step(entry["action"], entry["frames"])
        if self.engine.tick != before:
            self.actions.append(entry)
            assert self.log is not None
            self.log.write(json.dumps({"tick_before": before, "tick_after": self.engine.tick, **entry}) + "\n")
            if self.engine.tick // 30 != before // 30:
                with (self.run / "states.jsonl").open("a") as f:
                    f.write(json.dumps(self.engine.observe(), separators=(",", ":")) + "\n")
        if self.engine.status != "running" and not self.saved_terminal:
            self.save("terminal")
            self.saved_terminal = True
        return self.state()

    def load_replay(self, record: dict[str, Any]) -> dict[str, Any]:
        verified = replay(record)
        self.reset(record["seed"], record["max_ticks"])
        self.playback = record["actions"]
        self.verified_hash = verified.digest()
        return self.state()


def handler(session: Session) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

        def log_message(self, *args: Any) -> None:
            pass

        def json_response(self, value: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            with session.lock:
                if self.path == "/api/state":
                    self.json_response(session.state())
                    return
                if self.path == "/api/export":
                    session.save("export")
                    self.json_response(session.export())
                    return
                if self.path == "/api/health":
                    self.json_response({"status": "ready", "engine_version": VERSION})
                    return
            if self.path == "/":
                self.path = "/index.html"
            super().do_GET()

        def do_POST(self) -> None:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8 * 1024 * 1024:
                    raise ValueError("request body must be 1 byte to 8 MiB")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                with session.lock:
                    if self.path == "/api/reset":
                        result = session.reset(body.get("seed", 7), body.get("max_ticks", 18000))
                    elif self.path == "/api/step":
                        result = session.step(body)
                    elif self.path == "/api/controller":
                        if body.get("controller") not in ("human", "baseline") or session.playback is not None:
                            raise ValueError("controller must be human or baseline in a live run")
                        session.bot = body["controller"] == "baseline"
                        result = session.state()
                    elif self.path == "/api/replay":
                        result = session.load_replay(body)
                    else:
                        self.json_response({"error": "unknown route"}, 404)
                        return
                    self.json_response(result)
            except (ValueError, KeyError, TypeError) as error:
                self.json_response({"error": str(error)}, 400)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4183)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    session = Session(args.seed)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(session))
    print(f"REA-Issac: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Run artifacts: {session.run}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with session.lock:
            session.save("server_shutdown")
            assert session.log is not None
            session.log.close()


if __name__ == "__main__":
    main()
