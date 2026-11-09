"""Exercise Brogue's input boundaries and native recording contract."""

import os
import subprocess
from pathlib import Path

import pytest

from integrations.brogue_ce.engine import BrogueSession


@pytest.mark.integration
def test_native_recording(tmp_path: Path) -> None:
    engine_path = os.environ.get("BROGUE_CE_ENGINE")
    if not engine_path:
        pytest.skip("set BROGUE_CE_ENGINE to the pinned WebBrogue executable")
    engine = Path(engine_path).resolve()
    recording = tmp_path / "match.broguerec"
    with BrogueSession(engine, 7) as first, BrogueSession(engine, 7) as second:
        assert first.observe() == second.observe()
        for key in ("x", "x", "x", "Q", "y"):
            assert first.step(key) == second.step(key)
        assert first.observe().outcome == "quit"
        first.save_recording(recording)
        with pytest.raises(FileExistsError):
            first.save_recording(recording)
    subprocess.run([engine, "-vn", recording], cwd=tmp_path, check=True, timeout=15)
