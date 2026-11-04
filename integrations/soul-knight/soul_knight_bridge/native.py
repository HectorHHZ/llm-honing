"""One synchronous native request at a time, with matching response IDs."""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

JSON = dict[str, Any]  # Unity snapshots are a dynamic JSON boundary.
BUTTONS = ("attack", "interact", "skill", "switch_weapon")
MAX_AIM_COORDINATE = 1_000_000


def normalize_action(action: JSON) -> JSON:
    """Validate before writing IPC; buttons omitted by the caller are false."""
    if not isinstance(action, dict) or set(action) - {"move", "aim", *BUTTONS}:
        raise ValueError("action must contain only move, aim and the four buttons")
    result = {"move": [0, 0], "aim": [1, 0], **dict.fromkeys(BUTTONS, False), **action}
    for field in ("move", "aim"):
        pair = result[field]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{field} must contain two finite numbers")
        for value in pair:
            try:
                finite = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and abs(value) <= 3.4028234663852886e38
                )
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{field} must contain two finite numbers")
        result[field] = list(pair)
    if any(value not in (-1, 0, 1) for value in result["move"]):
        raise ValueError("move components must be -1, 0, or 1")
    if any(abs(value) > MAX_AIM_COORDINATE for value in result["aim"]):
        raise ValueError("aim components must be within[-1000000,1000000] world units")
    if any(not isinstance(result[key], bool) for key in BUTTONS):
        raise ValueError("buttons must be booleans")
    return result


class NativeClient:
    """Serialize IPC calls and retain their exact envelopes for action receipts.

    A timeout stops waiting. It does not cancel Unity execution, so callers must
    not automatically retry a timed-out step or reset.
    """

    def __init__(self, ipc_dir: str | Path, timeout_seconds: float = 30) -> None:
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.ipc_dir = Path(ipc_dir).resolve()
        self.timeout_seconds = timeout_seconds
        self.last_request: JSON | None = None
        self.last_response: JSON | None = None

    def _request(self, operation: str, **fields: Any) -> JSON:
        if not self.ipc_dir.is_dir():
            raise FileNotFoundError("IPC directory must come from an explicitly launched native player")
        with (self.ipc_dir / "client.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Never replace an operation Unity has not consumed, including after a timeout.
            request_path = self.ipc_dir / "request.json"
            if request_path.exists():
                raise RuntimeError("A native request is still pending; do not retry blindly")
            request = {"id": uuid.uuid4().hex, "op": operation, **fields}
            self.last_request, self.last_response = request, None
            temporary = self.ipc_dir / "request.tmp"
            temporary.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
            os.replace(temporary, request_path)
            deadline = time.monotonic() + self.timeout_seconds
            while time.monotonic() < deadline:
                try:
                    response = json.loads((self.ipc_dir / "response.json").read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    response = None
                if isinstance(response, dict) and response.get("id") == request["id"]:
                    self.last_response = response
                    if response.get("ok") is not True:
                        raise RuntimeError(response.get("error", "Native request failed"))
                    if not isinstance(response.get("state"), dict):
                        raise RuntimeError("Native response is missing its snapshot")
                    return response["state"]
                time.sleep(0.01)
        raise TimeoutError(f"Native {operation} timed out; completion is unknown and no retry was sent")

    def state(self) -> JSON:
        """Read a paused snapshot without advancing a rendered frame."""
        return self._request("state")

    def reset(self, seed: int = 7) -> JSON:
        """Reload the author's gameplay scene with Unity.Random seeded."""
        if not isinstance(seed, int) or isinstance(seed, bool) or not -(2**31) <= seed < 2**31:
            raise ValueError("seed must be a signed int32")
        return self._request("reset", seed=seed)

    def step(self, action: JSON, frames: int = 50) -> JSON:
        """Hold movement for a rendered-frame batch and edge-trigger the buttons."""
        if not isinstance(frames, int) or isinstance(frames, bool) or not 1 <= frames <= 250:
            raise ValueError("frames must be an integer in 1..250")
        return self._request("step", action=normalize_action(action), frames=frames)
