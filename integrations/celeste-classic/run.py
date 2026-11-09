"""Run a paused Celeste Classic C port with Honing decisions and local frame capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import select
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ACTIONS = {
    "wait": "Wait without pressing buttons.",
    "left": "Move left.",
    "right": "Move right.",
    "jump": "Hold jump.",
    "jump_left": "Hold jump and left.",
    "jump_right": "Hold jump and right.",
    "dash_left": "Hold dash and left.",
    "dash_right": "Hold dash and right.",
    "dash_up": "Hold dash and up.",
    "dash_down": "Hold dash and down.",
    "dash_up_left": "Hold dash, up and left.",
    "dash_up_right": "Hold dash, up and right.",
    "dash_down_left": "Hold dash, down and left.",
    "dash_down_right": "Hold dash, down and right.",
}
MASKS = dict(zip(ACTIONS, (0, 1, 2, 16, 17, 18, 33, 34, 36, 40, 37, 38, 41, 42), strict=True))


def request_for(observation: dict[str, Any], frames: int) -> dict[str, Any]:
    return {
        "id": observation["id"],
        "parts": {
            "essential": [
                "Play Celeste Classic. Climb to the top of the room and avoid spikes. "
                "Coordinates are room pixels: x right, y down; velocity pixels/tick. "
                "Tiles are 8px: # solid, ^ v < > spikes, . other. Entities cover the whole current room. "
                "Dashes refill on ground. Jump/dash trigger on button edges; "
                "repeating a held button does not retrigger. "
                f"Hold one action for {frames} game ticks at 30Hz. Game update and draw pause while you decide.",
                json.dumps(observation, separators=(",", ":"), allow_nan=False),
            ]
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Choose the next input to climb and survive.",
                "criteria": dict(ACTIONS),
            }
        },
    }


def checked_action(response: dict[str, Any], request: dict[str, Any]) -> tuple[str, float]:
    """Require the complete submitted state and audited choices before sending input to the Classic engine."""
    if type(response.get("id")) is not int or response["id"] != request["id"]:
        raise ValueError("Decision response id does not match the observation")
    retained = {key: list(request["parts"].get(key, [])) for key in ("essential", "details", "memory", "logs")}
    trimmed = response.get("trimmed")
    if not isinstance(trimmed, list):
        raise ValueError("Missing context removal audit")
    for removal in trimmed:
        if (
            not isinstance(removal, dict)
            or set(removal) != {"section", "text"}
            or removal["section"] != "logs"
            or not retained["logs"]
            or removal["text"] != retained["logs"][0]
        ):
            raise ValueError("Required or unsubmitted context was removed")
        retained["logs"].pop(0)
    state = "\n".join(text for section in retained for text in retained[section])
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
    if not isinstance(response.get("answers"), dict) or set(response["answers"]) != set(request["questions"]):
        raise ValueError("Answers do not cover the exact questions")
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


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
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
        result = json.loads(
            raw, parse_constant=reject_json_constant, parse_float=finite_json_float, object_pairs_hook=unique_object
        )
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def process_receipt(process: subprocess.Popen) -> dict[str, Any]:
    stat = Path(f"/proc/{process.pid}/stat")
    return {
        "pid": process.pid,
        "argv": process.args,
        "startTicks": stat.read_text().split()[21] if stat.exists() else None,
    }


class Native:
    """Own one player and one Xvfb; only accepted commands advance update/draw."""

    def __init__(
        self,
        game: Path,
        output: Path,
        *,
        capture: bool = False,
        watchdog: int = 240,
        seed: int = 7,
        library_path: str = "",
    ) -> None:
        self.game = game.resolve()
        self.output = output.resolve()
        self.capture, self.watchdog, self.seed, self.library_path = capture, watchdog, seed, library_path
        self.player: subprocess.Popen | None = None
        self.display: subprocess.Popen | None = None
        self.receipts: dict[str, Any] = {}
        self.owns_output = False

    def start(self) -> dict[str, Any]:
        if self.output.exists():
            raise ValueError("output must be a fresh directory")
        if not self.game.is_file():
            raise ValueError("game executable does not exist")
        self.output.mkdir(parents=True)
        self.owns_output = True
        (self.output / "frames").mkdir()
        read_fd, write_fd = os.pipe()
        try:
            with (self.output / "xvfb.log").open("wb") as log:
                self.display = subprocess.Popen(
                    ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "768x768x24", "-nolisten", "tcp", "-ac"],
                    pass_fds=(write_fd,),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            self.receipts["display"] = process_receipt(self.display)
            write_json(self.output / "processes.json", self.receipts)
            os.close(write_fd)
            write_fd = -1
            if not select.select([read_fd], [], [], 15)[0]:
                raise TimeoutError("owned Xvfb did not allocate a new display")
            display_number = os.read(read_fd, 80).decode().strip()
            if not display_number.isdigit() or self.display.poll() is not None:
                raise RuntimeError("owned Xvfb failed to allocate display")
        finally:
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
        env = {
            **os.environ,
            "DISPLAY": ":" + display_number,
            "SDL_AUDIODRIVER": "dummy",
            "HONING_SESSION_DIR": str(self.output),
            "HONING_CAPTURE": "1" if self.capture else "0",
            "HONING_SEED": str(self.seed),
            "HONING_WATCHDOG_SECONDS": str(self.watchdog),
        }
        if self.library_path:
            env["LD_LIBRARY_PATH"] = self.library_path + ":" + env.get("LD_LIBRARY_PATH", "")
        with (self.output / "player.log").open("wb") as log:
            self.player = subprocess.Popen(
                [str(self.game)], cwd=self.game.parent, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        self.receipts["player"] = process_receipt(self.player)
        self.receipts["allocatedDisplay"] = ":" + display_number
        write_json(self.output / "processes.json", self.receipts)
        return self.wait(0)

    def state(self) -> dict[str, Any]:
        return json.loads((self.output / "observation.json").read_text())

    def wait(self, previous: int, timeout: float = 30) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = self.state()
                if type(state.get("id")) is int and state["id"] > previous:
                    if state.get("status") != "paused":
                        raise RuntimeError(f"Native bridge is {state.get('status')}")
                    return state
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if self.player is not None and self.player.poll() is not None:
                raise RuntimeError("Native game exited; see player.log")
            time.sleep(0.01)
        raise TimeoutError("Native observation timed out")

    def send_raw(self, command: bytes) -> None:
        temporary = self.output / "command.tmp"
        temporary.write_bytes(command)
        temporary.replace(self.output / "command.txt")

    def step(self, before: dict[str, Any], action: str, frames: int) -> dict[str, Any]:
        if action not in MASKS or type(frames) is not int or not 1 <= frames <= 30:
            raise ValueError("invalid action or frame duration")
        self.send_raw(f"step {before['id']} {MASKS[action]} {frames}\n".encode())
        after = self.wait(before["id"])
        check_transition(before, after, MASKS[action], frames)
        return after

    def close(self) -> None:
        cleanup: dict[str, Any] = {}
        for name, process in (("player", self.player), ("display", self.display)):
            if process is None:
                continue
            if name == "player" and process.poll() is None:
                try:
                    state = self.state()
                    self.send_raw(f"stop {state['id']} 0 0\n".encode())
                    process.wait(timeout=3)
                except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            cleanup[name] = {
                "pid": process.pid,
                "returncode": process.returncode,
                "reaped": process.poll() is not None,
            }
        if self.owns_output and self.output.is_dir():
            write_json(self.output / "cleanup.json", cleanup)


def check_transition(before: dict[str, Any], after: dict[str, Any], mask: int, frames: int) -> None:
    if after["id"] != before["id"] + 1 or after["lastMask"] != mask or after["lastFrames"] != frames:
        raise ValueError("native action echo mismatch")
    for counter in ("ticks", "episodeTicks", "nativeUpdates", "nativeDraws"):
        if after[counter] - before[counter] != frames:
            raise ValueError(f"native {counter} did not advance by the requested ticks")
    if after["clockFrame"] != (before["clockFrame"] + frames) % 30:
        raise ValueError("original engine clock did not advance")


def decide_and_step(native: Native, client: DecisionClient, before: dict[str, Any], frames: int) -> dict[str, Any]:
    request = request_for(before, frames)
    raw_path = native.output / f"response-{before['id']:06d}.raw"
    record: dict[str, Any] = {"before": before, "request": request, "rawResponse": raw_path.name}
    try:
        response = client.decide(request, raw_path=raw_path)
        record["response"] = response
        action, probability = checked_action(response, request)
        record.update({"action": action, "probability": probability})
        after = native.step(before, action, frames)
        record["after"] = after
        return after
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            line = json.dumps(record, allow_nan=False)
        except ValueError:
            record.pop("response", None)  # The complete rejected bytes remain in rawResponse.
            line = json.dumps(record, allow_nan=False)
        with (native.output / "decisions.jsonl").open("a") as log:
            log.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-path", default="")
    parser.add_argument("--decision-url", default="http://127.0.0.1:8809")
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames <= 30 or not 1 <= args.decisions <= 1000 or not 0 <= args.seed <= 1000000000:
        parser.error("frames 1..30, decisions 1..1000, seed 0..1000000000 required")
    client = DecisionClient(args.decision_url, args.timeout)
    health = client.health()
    native = Native(
        args.game,
        args.output,
        capture=args.capture,
        watchdog=min(3600, math.ceil(args.timeout) + 30),
        seed=args.seed,
        library_path=args.library_path,
    )
    started = time.monotonic()
    try:
        observation = native.start()
        first = observation
        write_json(native.output / "model.json", health)
        source_dir = Path(__file__).resolve().parent
        write_json(
            native.output / "adapter.json",
            {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_dir.glob("*.py"))},
        )
        write_json(native.output / "build.json", json.loads((native.game.parent / "build.json").read_text()))
        for _ in range(args.decisions):
            observation = decide_and_step(native, client, observation, args.frames)
            print(f"id={observation['id']} ticks={observation['ticks']} room={observation['room']}", flush=True)
        write_json(
            native.output / "result.json",
            {
                "decisions": args.decisions,
                "frames": observation["ticks"],
                "simulationSeconds": observation["ticks"] / 30,
                "wallSeconds": time.monotonic() - started,
                "initial": first,
                "final": observation,
                "capture": args.capture,
            },
        )
    finally:
        native.close()


if __name__ == "__main__":
    main()
