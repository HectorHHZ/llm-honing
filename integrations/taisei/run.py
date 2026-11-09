"""Run a paused native Taisei segment with an existing Honing decision service."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ACTIONS = {
    "stay": "Stay still, shoot and focus.",
    "left": "Move left, shoot and focus.",
    "right": "Move right, shoot and focus.",
    "up": "Move up, shoot and focus.",
    "down": "Move down, shoot and focus.",
    "up_left": "Move diagonally up-left, shoot and focus.",
    "up_right": "Move diagonally up-right, shoot and focus.",
    "down_left": "Move diagonally down-left, shoot and focus.",
    "down_right": "Move diagonally down-right, shoot and focus.",
}


def request_for(observation: dict[str, Any], frames: int) -> dict[str, Any]:
    """Send a bounded current-state sensor in the existing Honing decision contract."""
    return {
        "id": observation["id"],
        "parts": {
            "essential": [
                "Play Taisei, a vertical bullet-hell shooter. Survive and shoot enemies. "
                "Coordinates are viewport pixels, x right, y down. nearestBullets entries "
                "are [x,y,vx,vy], native movement velocity parameters in pixels/tick, "
                "up to eight nearest on-screen enemy "
                "bullets; other bullets, lasers and enemies are not in this partial sensor. "
                f"Each action holds movement, shot and focus for {frames} ticks at 60Hz. "
                "Diagonal speed is normalized by the game. The game pauses during inference.",
                json.dumps(observation, separators=(",", ":"), allow_nan=False),
            ]
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Choose the next movement to survive.",
                "criteria": dict(ACTIONS),
            }
        },
    }


def checked_action(response: dict[str, Any], request: dict[str, Any]) -> tuple[str, float]:
    """Require the complete submitted state and audited choices before sending input to Taisei."""
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
        if not isinstance(audit, dict) or audit.get("stateTruncated") is not False:
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
    if not isinstance(response.get("answers"), dict) or set(response["answers"]) != {"action"}:
        raise ValueError("Decision answers differ from offered questions")
    try:
        answer = response["answers"]["action"]
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


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Publish a complete command with a same-directory rename."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


class NativeClient:
    """Consume one command per paused observation from an isolated stage process."""

    def __init__(self, session: Path, process: subprocess.Popen) -> None:
        self.session = session
        self.process = process
        self.previous = 0

    def observe(self, timeout: float = 90) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Taisei exited: {self.process.returncode}; see game.log")
            try:
                value = json.loads((self.session / "observation.json").read_text())
            except (OSError, ValueError):
                time.sleep(0.01)
                continue
            if type(value.get("id")) is int and value["id"] > self.previous:
                self.previous = value["id"]
                return value
            time.sleep(0.01)
        raise TimeoutError("Native observation timed out")

    def step(self, observation_id: int, action: str, frames: int) -> None:
        if observation_id != self.previous or action not in {*ACTIONS, "stop"}:
            raise ValueError("Invalid native action or stale observation")
        if type(frames) is not int or not 1 <= frames <= 120:
            raise ValueError("frames must be an integer from 1 through 120")
        atomic_json(self.session / "action.json", {"id": observation_id, "action": action, "frames": frames})


def decision_step(client: DecisionClient, native: NativeClient, observation: dict[str, Any], frames: int) -> str:
    """Preserve rejected bytes before validating; only accepted responses reach native.step."""
    request = request_for(observation, frames)
    raw = native.session / f"response-{observation['id']:06d}.raw"
    atomic_json(native.session / f"request-{observation['id']:06d}.json", request)
    record: dict[str, Any] = {"request": request, "responseRawFile": raw.name}
    try:
        response = client.decide(request, raw_path=raw)
        record["response"] = response
        action, probability = checked_action(response, request)
        record["acceptedAction"] = action
        record["probability"] = probability
    except Exception as error:
        record["rejected"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (native.session / "decisions.jsonl").open("a") as trace:
            trace.write(json.dumps(record, allow_nan=False) + "\n")
    native.step(observation["id"], action, frames)
    return action


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    session = args.output.resolve()
    client = DecisionClient(args.decision_url, args.timeout)
    atomic_json(session / "model.json", client.health())
    environment = {
        **os.environ,
        "HONING_SESSION_DIR": str(session),
        "TAISEI_STORAGE_PATH": str(session / "storage"),
        "TAISEI_CACHE_PATH": str(session / "cache"),
        "TAISEI_RES_PATH": str(args.resources.resolve()),
        "TAISEI_TASKMGR_NUM_THREADS": "4",
        "LP_NUM_THREADS": "4",
        "TAISEI_AUDIO_BACKEND": "null",
        "DISPLAY": args.display,
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "TAISEI_NOASYNC": "1",
    }
    if args.capture:
        (session / "frames").mkdir()
        environment["TAISEI_FRAMEDUMP"] = str(session / "frames/frame-")
    command = [
        str(args.game.resolve()),
        "--play",
        "--sid",
        "1",
        "--diff",
        "Easy",
        "--shotmode",
        "youmuA",
        "--width",
        "800",
        "--height",
        "600",
    ]
    xvfb = None
    started = time.monotonic()
    completed = 0
    initial = final = None
    with (session / "game.log").open("wb") as game_log, (session / "display.log").open("wb") as display_log:
        process = None
        try:
            # Refuse to share a display implicitly; never stop a pre-existing server.
            display_number = args.display.removeprefix(":").split(".")[0]
            if Path(f"/tmp/.X11-unix/X{display_number}").exists():
                raise RuntimeError("Display occupied; choose an unused --display")
            xvfb = subprocess.Popen(
                ["Xvfb", args.display, "-screen", "0", "800x600x24", "-nolisten", "tcp"],
                stdout=display_log,
                stderr=subprocess.STDOUT,
            )
            time.sleep(0.5)
            if xvfb.poll() is not None:
                raise RuntimeError("Owned Xvfb failed")
            process = subprocess.Popen(command, env=environment, stdout=game_log, stderr=subprocess.STDOUT)
            atomic_json(session / "launch.json", {"gamePid": process.pid, "displayPid": xvfb.pid, "command": command})
            native = NativeClient(session, process)
            initial = native.observe()
            observation = initial
            for _index in range(args.decisions):
                action = decision_step(client, native, observation, args.frames)
                after = native.observe()
                with (session / "actions.jsonl").open("a") as trace:
                    trace.write(
                        json.dumps({"before": observation, "action": action, "frames": args.frames, "after": after})
                        + "\n"
                    )
                completed += 1
                print(f"{completed}: frame {observation['frame']}->{after['frame']} {action}", flush=True)
                observation = after
            final = observation
            native.step(observation["id"], "stop", 1)
            process.wait(timeout=20)
        finally:
            for owned in (process, xvfb):
                if owned is not None and owned.poll() is None:
                    owned.terminate()
                    try:
                        owned.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        owned.kill()
                        owned.wait()
            atomic_json(
                session / "result.json",
                {
                    "completedDecisions": completed,
                    "requestedFramesPerAction": args.frames,
                    "initial": initial,
                    "final": final,
                    "wallSeconds": time.monotonic() - started,
                    "simulationSeconds": (
                        None if initial is None or final is None else (final["frame"] - initial["frame"]) / 60
                    ),
                    "nativeExitCode": None if process is None else process.returncode,
                    "ownedDisplayExitCode": None if xvfb is None else xvfb.returncode,
                    "trainingPerformed": False,
                    "heuristicFallback": False,
                    "controllerAssistance": "shot and focus are held by every offered movement action",
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True, help="Pinned checkout resources directory")
    parser.add_argument("--decision-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display", default=":191")
    parser.add_argument("--decisions", type=int, default=12)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if args.decisions < 1 or not 1 <= args.frames <= 120:
        parser.error("decisions must be positive, frames 1..120")
    run(args)


if __name__ == "__main__":
    main()
