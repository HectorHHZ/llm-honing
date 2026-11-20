"""Check the counterfactual probe CLI end to end, on the recorded tank rounds, against stand-in decision servers.

Run from the repository root: python scripts/e2e/check_probe.py

It unpacks the four rounds kept under ``tests/fixtures/decision_traces`` (played
by the dry run's stand-in model at quality 0.7), serves that model through
Honing's real ``DecisionServer`` under a made-up weights checksum, and a weaker
one (quality 0.5, which holds its fire) under another. Then it runs
``scripts/system2/probe.py`` as a separate process five times:

1. the identity probe: every logged probability reproduced to within floating-point rounding (the
   rounds were recorded on macOS, and ``math.exp`` can differ in the last bit between platforms);
2. a text variant that strips ``Fire permitted=true.`` from the state: every
   weapon decision the model fired on turns to hold;
3. a weights variant answered by the weaker model: the same decisions turn;
4. a wrong ``--policy-checksum``: refused, exit 1;
5. the weaker model posing as the traces' weights: the identity probe fails, exit 1.

It prints one line per check and exits 1 when any fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # the stand-in model lives under tests/

from honing.backends.jev.server import DecisionServer  # noqa: E402
from tests.fakes.tank import FIRING_QUALITY, START_QUALITY, ForecastModel  # noqa: E402

FIXTURES = ROOT / "tests/fixtures/decision_traces"
RUNS = ("s3114-blue-destroyed", "s3138-red-opponent-self-hit", "s3104-blue-opponent-destroyed", "s3102-red-timeout")
RECORDED, WEAKER = "a" * 64, "b" * 64
PERMITTED = "Fire permitted=true."
# How far a recomputed probability may differ from the logged one on another platform.
ROUNDING = 1e-12


def _runs(scratch: Path) -> list[Path]:
    for packed in sorted(FIXTURES.glob("*.tar.gz")):
        with tarfile.open(packed) as archive:
            archive.extractall(scratch, filter="data")
    return [FIXTURES / name if (FIXTURES / name).is_dir() else scratch / name for name in RUNS]


def _serve(quality: float, checksum: str) -> DecisionServer:
    server = DecisionServer(("127.0.0.1", 0), ForecastModel(quality, checksum))
    threading.Thread(target=server.serve, daemon=True).start()
    return server


def _cli(arguments: Sequence[str]) -> tuple[int, dict[str, Any] | None, str]:
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/system2/probe.py"), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    try:
        report = json.loads(process.stdout) if process.stdout.strip() else None
    except json.JSONDecodeError:
        report = None
    return process.returncode, report, process.stderr.strip()


def checks(runs: Sequence[Path], recorded: str, weaker: str) -> list[tuple[str, Callable[[], str | None]]]:
    """Each check's name, and a function that returns a failure message or None."""
    paths = [str(run) for run in runs]
    base = [*paths, "--server", recorded, "--policy-checksum", RECORDED]
    weapon = ["--question", "weapon", "--target", "weapon=fire"]

    def identity() -> str | None:
        code, report, err = _cli(base)
        if code != 0 or report is None:
            return f"exit {code}: {err}"
        found = report["identity"]
        exact = found["max_difference"] <= ROUNDING and found["flips"] == 0
        if not (found["ok"] and exact and report["decisions"] > 400):
            return f"identity not reproduced: {found}"
        return None

    def stripped(arguments: Sequence[str]) -> str | None:
        code, report, err = _cli([*base, *weapon, *arguments])
        if code != 0 or report is None:
            return f"exit {code}: {err}"
        summary = report["summary"]
        turned = summary["flipped"]
        if not (turned > 0 and turned == summary["away_from_target"] and summary["toward_target"] == 0):
            return f"expected only fire-to-hold turns, got {summary}"
        return None

    def refused(arguments: Sequence[str], words: str) -> str | None:
        code, _, err = _cli(arguments)
        return None if code == 1 and words in err else f"expected exit 1 mentioning {words!r}, got exit {code}: {err}"

    return [
        ("identity probe reproduces every logged probability", identity),
        (f"stripping {PERMITTED!r} turns the fired weapon decisions", lambda: stripped(["--strip", PERMITTED])),
        (
            f"weights at quality {START_QUALITY} turn the same decisions",
            lambda: stripped(["--weights-server", weaker, "--weights-checksum", WEAKER]),
        ),
        (
            "a wrong policy checksum is refused",
            lambda: refused([*paths, "--server", recorded, "--policy-checksum", WEAKER], "serves the weights"),
        ),
        (
            "other weights posing as the traces' own fail the identity probe",
            lambda: refused([*paths, "--server", weaker, "--policy-checksum", WEAKER], "identity probe"),
        ),
    ]


def main() -> int:
    assert START_QUALITY < FIRING_QUALITY, "the weaker stand-in must hold its fire"
    recorded, weaker = _serve(0.7, RECORDED), _serve(START_QUALITY, WEAKER)
    failed = 0
    try:
        with tempfile.TemporaryDirectory() as scratch:
            for name, check in checks(_runs(Path(scratch)), recorded.url, weaker.url):
                problem = check()
                failed += problem is not None
                print(f"{'ok  ' if problem is None else 'FAIL'} {name}" + ("" if problem is None else f": {problem}"))
    finally:
        for server in (recorded, weaker):
            server.stop()
    print("probe end to end: " + ("ok" if not failed else f"{failed} failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
