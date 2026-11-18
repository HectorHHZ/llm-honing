"""Read the public Brogue display at native keyboard input boundaries."""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

VERSION = "1.15.1"
WIDTH, HEIGHT = 100, 34
OUTCOMES = ("quit", "death", "victory", "mastery", "recording_ended")

# WebBrogue CE 1.15.1 display layout, also used by the native console.
SIDEBAR_WIDTH = 20
MAP_LEFT = SIDEBAR_WIDTH + 1
MAP_TOP = 3
MAP_WIDTH = WIDTH - MAP_LEFT
MAP_HEIGHT = HEIGHT - MAP_TOP - 2
DEPTH_LABEL = re.compile(r"\s*-- Depth: (\d+) --")


@dataclass(frozen=True)
class Observation:
    """The text a player can see, with native depth and gold counters."""

    rows: tuple[str, ...]
    deepest_level: int
    gold: int
    outcome: str | None = None
    message: str | None = None


class _Display:
    def __init__(self) -> None:
        self.rows = [[" "] * WIDTH for _ in range(HEIGHT)]
        # The player health and nutrition fills are drawn in the background.
        self.bar_colors = [[b"\0\0\0"] * SIDEBAR_WIDTH for _ in range(2)]
        self.status = [0] * 4
        self.last: Observation | None = None

    def accept(self, packet: bytes) -> Observation | None:
        """Apply one native display datagram, returning a snapshot at a boundary."""
        for offset in range(0, len(packet), 10):
            record = packet[offset : offset + 10]
            x, y, code = record[:3]
            if (x, y) == (254, 254):
                record = packet[offset:]
                if len(record) != 100 or code >= len(OUTCOMES) or self.last is None:
                    raise RuntimeError("Invalid WebBrogue game-over event")
                message = record[19:69].split(b"\0", 1)[0].decode("utf-8", errors="replace")
                return Observation(
                    self.last.rows,
                    max(self.status[0], int.from_bytes(record[7:9], "big")),
                    int.from_bytes(record[11:15], "big"),
                    OUTCOMES[code],
                    message,
                )
            if (x, y) == (255, 255):
                if code >= len(self.status):
                    raise RuntimeError("Invalid WebBrogue status code")
                self.status[code] = int.from_bytes(record[3:7], "big")
                if code == 3:
                    self.last = Observation(
                        tuple("".join(row).rstrip() for row in self.rows),
                        self.status[0],
                        self.status[1],
                    )
                    return self.last
            elif x < WIDTH and y < HEIGHT:
                self.rows[y][x] = chr(int.from_bytes(record[2:4], "big"))
                if x < SIDEBAR_WIDTH and y in (1, 2):
                    self.bar_colors[y - 1][x] = record[7:10]
            else:
                raise RuntimeError("Invalid WebBrogue display coordinate")
        return None

    def observe_for_ai(self, frame: Observation, seed: int, keys: int) -> dict[str, object]:
        rows = frame.rows
        depth = DEPTH_LABEL.match(rows[-1])
        footer = rows[-1][MAP_LEFT:].split()
        done = frame.outcome is not None
        show_map = not done and footer == ["Explore", "Rest", "(z)", "Search", "(s)", "Menu", "Inventory"]
        result: dict[str, object] = {
            "schema_version": 1,
            "game": "brogue-ce",
            "seed": str(seed),
            "status": "ended" if done else "playing" if show_map else "prompt",
            "done": done,
            "clock": {"unit": "key", "value": keys},
            "depth": int(depth[1]) if depth else None,
            "deepest_level": frame.deepest_level,
            "gold": frame.gold,
            "outcome": frame.outcome,
            "outcome_message": frame.message,
            "map": None,
            "player": None,
            "action_space": None if done else {"kind": "key"},
        }
        if not show_map:
            result["display"] = list(rows)
            return result

        glyphs = [row[MAP_LEFT:].ljust(MAP_WIDTH) for row in rows[MAP_TOP : MAP_TOP + MAP_HEIGHT]]
        positions = [{"x": x, "y": y} for y, row in enumerate(glyphs) for x, glyph in enumerate(row) if glyph == "@"]
        sidebar = [row[:SIDEBAR_WIDTH].strip() for row in rows[:-1]]
        bars = {
            name: {"label": label, "background": [color.hex() for color in colors]}
            for name, label, colors in zip(("health", "nutrition"), sidebar[1:3], self.bar_colors, strict=True)
            if label.startswith(name.capitalize())
        }
        result.update(
            map={"kind": "glyph_grid", "width": MAP_WIDTH, "height": MAP_HEIGHT, "rows": glyphs},
            player={"position": positions[0] if len(positions) == 1 else None, "label": sidebar[0], "bars": bars},
            sidebar=[line for line in sidebar if line],
            messages=[row[MAP_LEFT:] for row in rows[:MAP_TOP] if row[MAP_LEFT:].strip()],
            location_hint=rows[-2][MAP_LEFT:].strip(),
            action_space={
                "kind": "key",
                "move": dict(zip(("n", "ne", "e", "se", "s", "sw", "w", "nw"), "kulnjbhy", strict=True)),
                "explore": "x",
                "inventory": "i",
                "rest": "z",
                "search": "s",
                "ascend": "<",
                "descend": ">",
            },
        )
        return result


def check_engine(engine: Path) -> None:
    version = subprocess.check_output([engine, "--version"], text=True, timeout=10).splitlines()[0]
    help_text = subprocess.check_output([engine, "--help"], text=True, timeout=10)
    if version != f"Brogue version: CE {VERSION}" or "--server-mode" not in help_text:
        raise ValueError(f"Use the WebBrogue CE {VERSION} build from integrations/brogue_ce/setup.sh")


class BrogueSession:
    """One seeded game, isolated from other games and blocked between inputs."""

    def __init__(self, engine: Path, seed: int, *, timeout_seconds: float = 20) -> None:
        if type(seed) is not int or not 1 <= seed < 2**64:
            raise ValueError("seed must be in 1..2^64-1")
        engine = engine.resolve()
        check_engine(engine)
        self._timeout = timeout_seconds
        self._seed = seed
        self._keys = 0
        self._display = _Display()
        # Keep the socket paths within the Unix-domain address length limit.
        self._temp = tempfile.TemporaryDirectory(prefix="honing-br-", dir="/tmp")
        self._directory = Path(self._temp.name)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._process: subprocess.Popen[bytes] | None = None
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
            self._socket.settimeout(0.2)
            self._socket.bind(str(self._directory / "client-socket"))
            with (self._directory / "engine.log").open("wb") as log:
                self._process = subprocess.Popen(
                    [engine, "--server-mode", "-s", str(seed), "--no-effects", "--data-dir", engine.parent],
                    cwd=self._directory,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            self._observation = self._read()
            if self._display.status[2] != seed & 0xFFFFFFFF:
                raise RuntimeError("WebBrogue started with an unexpected seed")
        except BaseException:
            self.close()
            raise

    def _read(self) -> Observation:
        assert self._process is not None
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            try:
                packet = self._socket.recv(1000)
            except TimeoutError:
                if self._process.poll() is not None:
                    log = (self._directory / "engine.log").read_text(errors="replace").strip()
                    raise RuntimeError(
                        f"Brogue exited before an input boundary ({self._process.returncode}): {log}"
                    ) from None
                continue
            if not packet or len(packet) % 10:
                raise RuntimeError("Invalid WebBrogue datagram length")
            frame = self._display.accept(packet)
            if frame is not None:
                return frame
        raise TimeoutError("Brogue did not reach an input boundary")

    def observe(self) -> Observation:
        return self._observation

    def observe_for_ai(self) -> dict[str, object]:
        """Return the current public map, sidebar and input mode as JSON data."""
        return self._display.observe_for_ai(self._observation, self._seed, self._keys)

    def step(self, key: str) -> Observation:
        """Send one printable ASCII key, Escape, Return or Tab."""
        if self._observation.outcome is not None:
            raise RuntimeError("Brogue has already ended")
        if len(key) != 1 or not (32 <= ord(key) <= 126 or key in "\x1b\r\t"):
            raise ValueError("expected one Brogue keyboard input")
        self._socket.sendto(bytes((0, 0, ord(key), 0, 0)), str(self._directory / "server-socket"))
        self._observation = self._read()
        self._keys += 1
        return self._observation

    def save_recording(self, target: Path) -> None:
        """Copy a native recording after the game has finished."""
        if self._observation.outcome is None:
            raise RuntimeError("Finish the Brogue game before saving its recording")
        assert self._process is not None
        if self._process.wait(timeout=self._timeout):
            raise RuntimeError("Brogue failed to finish its recording")
        (source,) = self._directory.glob("*.broguerec")
        with source.open("rb") as recording, target.open("xb") as destination:
            shutil.copyfileobj(recording, destination)

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._socket.close()
        self._temp.cleanup()

    def __enter__(self) -> BrogueSession:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
