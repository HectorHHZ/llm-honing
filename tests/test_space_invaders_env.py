"""Space Invaders package, repeat semantics, and complete replay checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from honing.harness import space_invaders as game

np = pytest.importorskip("numpy")


class FakeALE:
    _frameskip = 1

    def __init__(self, *, end_after: int = 4) -> None:
        self.unwrapped = self
        self.end_after = end_after
        self.count = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, int]]:
        self.count = 0
        return np.zeros((2, 2, 3), dtype=np.uint8), {"lives": 3}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, int]]:
        self.count += 1
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[0, 0, 0] = 100 if self.count == 3 else 0
        frame[0, 1, 0] = 200 if self.count == 4 else 0
        end = self.count == self.end_after
        return frame, float(action), end, False, {"lives": 0 if end else 3}

    def get_action_meanings(self) -> list[str]:
        return ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

    def close(self) -> None:
        pass


def test_last_two_frames_pool_and_reward() -> None:
    env = game.MaxPoolRepeat(FakeALE())
    env.reset(seed=19)
    frame, reward, terminated, truncated, info = env.step(1)
    assert reward == 4
    assert frame[0, 0, 0] == 100 and frame[0, 1, 0] == 200
    assert env.last_raw_frame[0, 0, 0] == 0
    assert env.underlying_frames == 4
    assert terminated and not truncated and info["lives"] == 0


def test_repeat_stops_on_first_terminal_frame() -> None:
    env = game.MaxPoolRepeat(FakeALE(end_after=1))
    env.reset()
    _, reward, terminated, truncated, _ = env.step(1)
    assert reward == 1 and terminated and not truncated and env.underlying_frames == 1


def test_complete_episode_replays_and_detects_tampering(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(game, "make_space_invaders", lambda **_: game.MaxPoolRepeat(FakeALE()))
    monkeypatch.setattr(game.importlib.metadata, "version", lambda name: f"fake-{name}")
    trace = tmp_path / "trace.json"
    result = game.run_episode(lambda _frame, _info: 1, seed=61003, output=trace)
    assert result["natural_end"]
    assert game.verify_replay(trace)["verified"]
    record = json.loads(trace.read_text())
    record["steps"][0]["reward"] = 999
    trace.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="diverged"):
        game.verify_replay(trace)


def test_incomplete_episode_is_not_verifiable(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(game, "make_space_invaders", lambda **_: game.MaxPoolRepeat(FakeALE(end_after=100)))
    monkeypatch.setattr(game.importlib.metadata, "version", lambda name: f"fake-{name}")
    trace = tmp_path / "trace.json"
    assert not game.run_episode(lambda _frame, _info: 0, seed=1, output=trace, max_decisions=1)["natural_end"]
    with pytest.raises(ValueError, match="incomplete"):
        game.verify_replay(trace)


def test_installed_module_has_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "honing.harness.space_invaders", "--help"], check=True, capture_output=True, text=True
    )
    assert "play" in result.stdout and "replay" in result.stdout


@pytest.mark.integration
def test_native_and_pooled_preserve_ale_dynamics() -> None:
    pytest.importorskip("ale_py")
    pytest.importorskip("gymnasium")
    native = game.make_space_invaders(pooled=False)
    pooled = game.make_space_invaders(pooled=True)
    seed = 61003
    try:
        raw, _ = native.reset(seed=seed)
        view, _ = pooled.reset(seed=seed)
        assert np.array_equal(raw, view)
        for index in range(1000):
            action = index % 6
            raw, raw_reward, raw_end, raw_truncated, raw_info = native.step(action)
            view, view_reward, view_end, view_truncated, view_info = pooled.step(action)
            assert raw_reward == view_reward
            assert (raw_end, raw_truncated, raw_info["lives"]) == (view_end, view_truncated, view_info["lives"])
            assert np.array_equal(raw, pooled.last_raw_frame)
            assert np.array_equal(native.unwrapped.ale.getRAM(), pooled.unwrapped.ale.getRAM())
            if raw_end or raw_truncated:
                break
        assert raw_end and view_end and not raw_truncated and not view_truncated
    finally:
        native.close()
        pooled.close()
