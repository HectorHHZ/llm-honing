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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

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
    "Each input is held for {frames} fixed60Hz native updates. The game pauses "
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
    """Require the complete submitted state and audited choices before sending input to Hurrican."""
    if not isinstance(response, dict):
        raise ValueError("Decision response must be an object")
    if not isinstance(response.get("answers"), dict) or set(response["answers"]) != set(request["questions"]):
        raise ValueError("Decision answers do not cover exactly the submitted questions")
    if type(response.get("id")) is not int or response["id"] != request["id"]:
        raise ValueError("Decision response id does not match the observation")
    # This adapter submits only required state; there is no optional history to prune.
    if response.get("trimmed") != []:
        raise ValueError("Decision response trimmed context or omitted its trimmed audit")
    state = "\n".join(
        text for section in ("essential", "details", "memory", "logs") for text in request["parts"].get(section, [])
    )
    if response.get("stateText") != state:
        raise ValueError("Decision response stateText differs from the complete submitted context")
    audits = response.get("tokenAudit")
    if not isinstance(audits, dict) or set(audits) != set(request["questions"]):
        raise ValueError("Decision response tokenAudit does not cover the submitted questions")
    for key, question in request["questions"].items():
        audit = audits[key]
        if (
            not isinstance(audit, dict)
            or not {"stateTokens", "stateRoom", "inputTokens", "options", "stateTruncated"} <= set(audit)
            or audit.get("stateTruncated") is not False
        ):
            raise ValueError(f"Decision tokenAudit reports truncated or unaudited context for {key}")
        if type(audit.get("options")) is not int or audit["options"] != len(question["criteria"]):
            raise ValueError(f"Decision tokenAudit has the wrong option count for {key}")
        # DJEV reports null token counts. Validate all available counts without inventing them.
        for field in ("stateTokens", "stateRoom", "inputTokens", "maxInputTokens"):
            value = audit.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Decision tokenAudit has an invalid {field} for {key}")
        size, room, total, limit = (
            audit.get(field) for field in ("stateTokens", "stateRoom", "inputTokens", "maxInputTokens")
        )
        if (
            (size is not None and room is not None and size > room)
            or (size is not None and total is not None and size > total)
            or (total is not None and limit is not None and total > limit)
            or (state and any(value == 0 for value in (size, room, total, limit)))
        ):
            raise ValueError(f"Decision tokenAudit contradicts complete context for {key}")
    try:
        answer = response["answers"]["action"]
        if answer.get("type") != "choice":
            raise ValueError("Decision answer is not a choice")
        action, probabilities = answer["choice"], answer["probabilities"]
    except (KeyError, TypeError) as error:
        raise ValueError("Decision response has no action distribution") from error
    if (
        not isinstance(probabilities, dict)
        or set(probabilities) != set(ACTIONS)
        or not isinstance(action, str)
        or action not in ACTIONS
    ):
        raise ValueError("Decision response does not cover the offered actions")
    values = list(probabilities.values())
    if any(
        isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
        for p in values
    ):
        raise ValueError("Invalid model probabilities")
    # Laya returns four decimal places, allowing at most 0.00005 rounding error per option.
    if abs(math.fsum(values) - 1) > len(ACTIONS) * 0.00005 + 1e-8:
        raise ValueError("Model probabilities do not sum to one within rounding precision")
    if probabilities[action] != max(values):
        raise ValueError("Chosen input is not the model's argmax")
    return action, float(probabilities[action])


def reject_json_constant(value: str) -> Any:
    """Keep NaN/Infinity out of the JSON trace as well as the game input."""
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def finite_json_float(value: str) -> float:
    """Reject exponent overflow before a decoded response can enter the JSON trace."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return result


class DecisionClient:
    """Use Honing's existing health and decide routes without loading a model in the game client."""

    def __init__(self, url: str, timeout_seconds: float = 180) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("decision-url must be an HTTP(S) server base URL")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout must be positive and finite")
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _call(
        self, route: str, payload: dict[str, Any] | None = None, *, raw_path: Path | None = None
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.url + route, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            if raw_path is not None:
                raw_path.write_bytes(error.read())
            raise
        if raw_path is not None:
            raw_path.write_bytes(raw)
        result = json.loads(raw, parse_constant=reject_json_constant, parse_float=finite_json_float)
        if not isinstance(result, dict):
            raise ValueError("Decision server response must be an object")
        return result

    def health(self) -> dict[str, Any]:
        result = self._call("/v1/health")
        if result.get("ready") is not True:
            raise ValueError("Decision server is not ready")
        return result

    def decide(self, request: dict[str, Any], *, raw_path: Path | None = None) -> dict[str, Any]:
        return self._call("/v1/decide", request, raw_path=raw_path)


def write_json(path: Path, value: Any) -> None:
    """Retain finite JSON with readable newlines."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class NativeClient:
    """Own one native process and receive observations only at paused boundaries."""

    def __init__(self, game: Path, data: Path, session: Path, *, capture: bool = False, timeout: float = 90) -> None:
        self.timeout = timeout
        self.session = session
        self.messages: queue.Queue[dict[str, Any] | Exception] = queue.Queue()
        self.log = (session / "native.log").open("w", encoding="utf-8")
        environment = {**os.environ, "HONING_HURRICAN_SESSION": str(session.resolve())}
        if capture:
            environment["HONING_HURRICAN_CAPTURE"] = "1"
        else:
            environment.pop("HONING_HURRICAN_CAPTURE", None)
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

    def close(self) -> None:
        """Close stdin, then reap or terminate only this launcher-created process group."""
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        write_json(
            self.session / "cleanup.json",
            {"pid": self.process.pid, "returncode": self.process.returncode, "alive": self.process.poll() is None},
        )
        self.reader.join(timeout=2)
        self.log.close()


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
    args.output.mkdir(parents=True, exist_ok=False)
    client = DecisionClient(args.decision_url, args.timeout)
    write_json(args.output / "model.json", client.health())
    write_json(
        args.output / "binary.json",
        {"path": str(args.game.resolve()), "sha256": hashlib.sha256(args.game.read_bytes()).hexdigest()},
    )
    native = NativeClient(args.game, args.data, args.output, capture=args.capture)
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
