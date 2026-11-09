"""Read the public Brogue display at native keyboard input boundaries."""

from __future__ import annotations

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
            else:
                raise RuntimeError("Invalid WebBrogue display coordinate")
        return None


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

    def step(self, key: str) -> Observation:
        """Send one printable ASCII key, Escape, Return or Tab."""
        if self._observation.outcome is not None:
            raise RuntimeError("Brogue has already ended")
        if len(key) != 1 or not (32 <= ord(key) <= 126 or key in "\x1b\r\t"):
            raise ValueError("expected one Brogue keyboard input")
        self._socket.sendto(bytes((0, 0, ord(key), 0, 0)), str(self._directory / "server-socket"))
        self._observation = self._read()
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
