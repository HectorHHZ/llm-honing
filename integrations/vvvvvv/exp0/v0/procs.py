"""Process ownership helpers for the exp0-games collectors.

Every process is started here in its own session and stopped only through the handle or the recorded
pid + /proc start time; nothing is ever stopped by name. Xvfb displays are taken from :350..:399 only
(the range the collectors of one machine share).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

DISPLAY_RANGE = range(350, 400)


def proc_start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def receipt(process: subprocess.Popen) -> dict:
    return {
        "pid": process.pid,
        "procStartTicks": proc_start_ticks(process.pid),
        "command": [str(a) for a in process.args],
        "started_unix": time.time(),
    }


def display_free(number: int) -> bool:
    return not Path(f"/tmp/.X11-unix/X{number}").exists() and not Path(f"/tmp/.X{number}-lock").exists()


class OwnedXvfb:
    """One Xvfb on a free display in ``displays`` (default DISPLAY_RANGE), owned by this object.

    ``displays`` (added 2026-11-26 for the VVVVVV v0 release collection, which owns :420..:439 only) restricts the
    candidates; the default stays DISPLAY_RANGE, so the hallowdale collector is unchanged."""

    def __init__(
        self,
        log_path: Path,
        size: str = "1024x768x24",
        prefer: int | None = None,
        displays: range | list[int] | None = None,
    ):
        self.log_path = Path(log_path)
        self.size = size
        self.prefer = prefer
        self.displays = list(displays) if displays is not None else list(DISPLAY_RANGE)
        if prefer is not None and prefer not in self.displays:
            raise ValueError(f"preferred display :{prefer} is outside the allowed displays")
        self.process: subprocess.Popen | None = None
        self.number: int | None = None
        self.info: dict = {}

    def start(self) -> int:
        candidates = [self.prefer] if self.prefer is not None else []
        candidates += [n for n in self.displays if n != self.prefer]
        log = self.log_path.open("w")
        for number in candidates:
            if number is None or not display_free(number):
                continue
            process = subprocess.Popen(
                ["Xvfb", f":{number}", "-screen", "0", self.size, "-nolisten", "tcp"],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and process.poll() is None and display_free(number):
                time.sleep(0.05)
            if process.poll() is None and not display_free(number):
                self.process, self.number = process, number
                self.info = {**receipt(process), "display": f":{number}"}
                return number
        raise RuntimeError(f"no free display in :{self.displays[0]}..:{self.displays[-1]}")

    def stop(self) -> dict:
        process = self.process
        if process is None:
            return {}
        if process.poll() is None:  # noqa: SIM102
            # identity check: the pid must still be the process we started
            if proc_start_ticks(process.pid) == self.info.get("procStartTicks"):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        self.info["returncode"] = process.returncode
        self.process = None
        return dict(self.info)


def stop_owned(process: subprocess.Popen, expected_ticks: str | None, grace: float = 5.0) -> int | None:
    """Terminate a child we started, after checking pid identity; returns its exit code."""
    if process.poll() is None and proc_start_ticks(process.pid) == expected_ticks:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=grace)
    return process.returncode


def write_json(path: Path, value) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)
