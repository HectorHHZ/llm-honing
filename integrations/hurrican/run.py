"""Run the pinned Hurrican native game with an existing Honing decision server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

# Resolve the sibling stdlib-only receiver without exposing integrations as importable runtimes.
_shared_import_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _shared_import_path)
try:
    from _shared.decision import DecisionClient as DecisionClient
    from _shared.decision import checked_action as checked_decision_action
    from _shared.decision import finite_json_float as finite_json_float
    from _shared.decision import reject_json_constant as reject_json_constant
    from _shared.decision import strict_json_loads as strict_json_loads
    from _shared.decision import unique_object as unique_object
finally:
    # Remove only our inserted object, preserving preexisting equal path entries.
    for _shared_index, _shared_entry in enumerate(sys.path):
        if _shared_entry is _shared_import_path:
            del sys.path[_shared_index]
            break

ACTIONS = {
    "wait": "Wait without shooting.",
    "left": "Move left.",
    "right": "Move right.",
    "shoot": "Shoot in the current facing direction.",
    "left_shoot": "Move left and shoot.",
    "right_shoot": "Move right and shoot.",
    "jump": "Hold jump without shooting.",
    "jump_left_shoot": "Hold left, jump and shoot.",
    "jump_right_shoot": "Hold right, jump and shoot.",
    "crouch_shoot": "Crouch and shoot in the current facing direction.",
}

PROMPT = (
    "Play Hurrican, a run-and-gun platformer. Progress right through the jungle, "
    "shoot enemies and survive. Coordinates are world pixels; positive x is right "
    "and positive y is DOWN. Enemy positions are nearby on-screen destroyable objects. "
    "Each action lasts {frames} fixed60Hz native updates. If jump was held, "
    "a jump action releases jump on its first update while keeping movement/shoot. The game pauses "
    "while you decide. Shooting and jumping follow the original game's rules."
)


def request_for(observation: dict[str, Any], frames: int) -> dict[str, Any]:
    """Submit required instructions and full observed state without optional history."""
    return {
        "id": observation["id"],
        "parts": {
            "essential": [
                PROMPT.format(frames=frames),
                json.dumps(observation, separators=(",", ":"), allow_nan=False),
            ]
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Choose the next input to progress and survive.",
                "criteria": dict(ACTIONS),
            }
        },
    }


def checked_action(response: dict[str, Any], request: dict[str, Any]) -> tuple[str, float]:
    """Validate the shared serving contract before the game-specific native input."""
    return checked_decision_action(response, request, ACTIONS)


def write_json(path: Path, value: Any) -> None:
    """Retain finite JSON with readable newlines."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class NativeClient:
    """Own one native process and receive observations only at paused boundaries."""

    def __init__(self, game: Path, data: Path, session: Path, *, capture: bool = False, timeout: float = 90) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0 < timeout <= 3570
        ):
            raise ValueError("timeout must be finite and greater than 0, at most 3570 seconds")
        self.timeout = timeout
        self.session = session
        self.messages: queue.Queue[dict[str, Any] | Exception] = queue.Queue()
        self.log = (session / "native.log").open("w", encoding="utf-8")
        environment = {
            **os.environ,
            "HONING_HURRICAN_SESSION": str(session.resolve()),
            "HONING_HURRICAN_IDLE_SECONDS": str(math.ceil(timeout) + 30),
        }
        if capture:
            environment["HONING_HURRICAN_CAPTURE"] = "1"
        else:
            environment.pop("HONING_HURRICAN_CAPTURE", None)
        self.reader: threading.Thread | None = None
        try:
            self.process = subprocess.Popen(
                [
                    str(game.resolve()),
                    "--windowmode",
                    "--novsync",
                    "--pathdata",
                    str(data.resolve()),
                    "--pathsave",
                    str(session.resolve()),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
                start_new_session=True,
            )
            write_json(session / "launch.json", {"pid": self.process.pid, "command": self.process.args})
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
        except BaseException:
            # The caller has not acquired this object yet, so its finally cannot
            # clean up receipt/thread initialization failures. Write no receipt
            # here: the session is caller-owned, and its storage may have failed.
            try:
                self._reap_owned_process()
            finally:
                self._close_reader_log()
            raise

    def _read(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.log.write(line)
                self.log.flush()
                if line.startswith("HONING "):
                    try:
                        value = json.loads(
                            line[7:], parse_constant=reject_json_constant, parse_float=finite_json_float
                        )
                        self.messages.put(value)
                    except ValueError as error:
                        self.messages.put(error)
        finally:
            self.messages.put(RuntimeError("Native game exited; see native.log"))

    def receive(self) -> dict[str, Any]:
        try:
            value = self.messages.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError("No paused native observation") from error
        if isinstance(value, Exception):
            raise value
        if not isinstance(value, dict) or type(value.get("id")) is not int or type(value.get("tick")) is not int:
            raise ValueError("Invalid native observation")
        return value

    def command(self, text: str) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        return self.receive()

    def step(self, observation: dict[str, Any], action: str, frames: int) -> dict[str, Any]:
        if action not in ACTIONS or type(frames) is not int or not 1 <= frames <= 120:
            raise ValueError("Invalid native action or frame duration")
        result = self.command(f"step {observation['id']} {frames} {list(ACTIONS).index(action)}")
        if (
            result.get("error")
            or result["id"] != observation["id"] + 1
            or result["tick"] != observation["tick"] + frames
        ):
            raise ValueError(f"Native step did not advance exactly as requested: {result}")
        return result

    def _reap_owned_process(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def _close_reader_log(self) -> None:
        try:
            if self.reader is not None and self.reader.ident is not None:
                self.reader.join(timeout=2)
        finally:
            process = getattr(self, "process", None)
            if process is not None and process.stdout is not None:
                process.stdout.close()
            self.log.close()

    def close(self) -> None:
        """Close stdin, then reap or terminate only this launcher-created process group."""
        try:
            self._reap_owned_process()
            write_json(
                self.session / "cleanup.json",
                {"pid": self.process.pid, "returncode": self.process.returncode, "alive": self.process.poll() is None},
            )
        finally:
            self._close_reader_log()


def decide_step(
    native: NativeClient, client: DecisionClient, observation: dict[str, Any], frames: int, session: Path
) -> dict[str, Any]:
    """Preserve raw response bytes before validation; rejection cannot call step."""
    request = request_for(observation, frames)
    stem = f"decision{observation['id']:04d}_{time.time_ns()}"
    write_json(session / f"{stem}.request.json", request)
    raw_path = session / f"{stem}.response.raw"
    started = time.monotonic()
    try:
        response = client.decide(request, raw_path=raw_path)
        action, probability = checked_action(response, request)
    except Exception as error:
        write_json(
            session / f"{stem}.rejection.json",
            {"request": request, "reason": f"{type(error).__name__}: {error}", "rawResponse": raw_path.name},
        )
        raise
    after = native.step(observation, action, frames)
    record = {
        "before": observation,
        "action": action,
        "probability": probability,
        "frames": frames,
        "wallSeconds": time.monotonic() - started,
        "after": after,
    }
    with (session / "trace.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
    return after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, required=True, help="External Hurrican directory containing data/ and lang/"
    )
    parser.add_argument("--output", type=Path, required=True, help="New local directory; private raw trace and frames")
    parser.add_argument("--decision-url", default="http://127.0.0.1:8809")
    parser.add_argument("--decisions", type=int, default=12)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.decisions <= 10000 or not 1 <= args.frames <= 120:
        parser.error("decisions must be1..10000 and frames1..120")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3570:
        parser.error("timeout must be finite and greater than 0, at most 3570 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    client = DecisionClient(args.decision_url, args.timeout)
    write_json(args.output / "model.json", client.health())
    write_json(
        args.output / "binary.json",
        {"path": str(args.game.resolve()), "sha256": hashlib.sha256(args.game.read_bytes()).hexdigest()},
    )
    native = NativeClient(args.game, args.data, args.output, capture=args.capture, timeout=args.timeout)
    try:
        observation = native.receive()
        write_json(args.output / "initial.json", observation)
        for _ in range(args.decisions):
            observation = decide_step(native, client, observation, args.frames, args.output)
        write_json(args.output / "final.json", observation)
    finally:
        native.close()
    print(args.output)


if __name__ == "__main__":
    main()
