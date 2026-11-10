"""Optional native Griddly 1.6.7 adapter; base imports load no game packages."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Protocol

from integrations.griddly.catalog import ACTIONS, GAME, GAME_SHA256, VERSIONS


class EngineScenario(Protocol):
    """Read-only game settings supplied by a caller."""

    @property
    def seed(self) -> int: ...

    @property
    def max_steps(self) -> int: ...

    @property
    def level_id(self) -> int: ...


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def runtime_versions() -> dict[str, str]:
    return {
        **{name: importlib.metadata.version(name) for name in VERSIONS},
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


class GriddlyEngine:
    """Advance the real Spiders rules once per action, without auto-reset or safety assistance."""

    def __init__(self, scenario: EngineScenario, *, render: bool = False) -> None:
        try:
            versions = runtime_versions()
            if any(versions[name] != version for name, version in VERSIONS.items()):
                raise RuntimeError("Griddly runtime differs from integrations/griddly/requirements.txt")
            griddly = importlib.import_module("griddly")
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            raise RuntimeError(
                "Install integrations/griddly/requirements.txt in a separate Python 3.10/3.11 environment"
            ) from error
        if griddly.__file__ is None:
            raise RuntimeError("the installed Griddly module has no resource directory")
        path = Path(griddly.__file__).parent / "resources/games" / GAME
        if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != GAME_SHA256:
            raise RuntimeError("The installed native Spiders rules differ from the pin")
        self.scenario = scenario
        self.game_sha256 = GAME_SHA256
        self._render = render
        self._env: Any = griddly.GymWrapper(
            yaml_file=str(path),
            player_observer_type=griddly.gd.ObserverType.VECTOR,
            # The 1.6.7 Windows Vulkan renderer crashes during process teardown.
            # Its upstream Vector2RGB renderer keeps recording independent of Vulkan.
            global_observer_type=griddly.gd.ObserverType.VECTOR,
        )
        mapping = self._env.action_input_mappings["move"]
        if (
            self._env.action_names != ["move"]
            or not mapping["Relative"]
            or set(mapping["InputMappings"]) != {"1", "2", "3", "4"}
        ):
            self._env.close()
            raise RuntimeError("Native Spiders action mapping differs from the pin")
        self._closed, self._ended, self._started = False, False, False
        self._steps = 0
        self._latest: dict[str, Any] | None = None

    def reset(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("the engine is closed")
        # The legacy seed method needs initialized spaces. Reseed before the recorded reset,
        # so randomized initial actions do not depend on a previous episode.
        self._env.reset(level_id=self.scenario.level_id)
        self._env.seed(self.scenario.seed)
        self._env.reset(level_id=self.scenario.level_id)
        self._env.enable_history(True)
        self._steps, self._ended, self._started = 0, False, True
        return self._frame(0.0, False, {})

    def _legal(self) -> list[str]:
        ids = {0}
        for location, names in self._env.game.get_available_actions(1).items():
            if "move" in names:
                ids.update(self._env.game.get_available_action_ids(location, ["move"])["move"])
        return [name for name, action_id in ACTIONS.items() if action_id in ids]

    def _frame(self, reward: float, done: bool, info: dict[str, Any]) -> dict[str, Any]:
        native = json.loads(canonical(self._env.get_state()))
        objects = [
            {key: obj[key] for key in ("Name", "Location", "Orientation", "PlayerId", "Variables")}
            for obj in native["Objects"]
            if not obj["Name"].startswith("_")
        ]
        objects.sort(key=canonical)
        alive = any(obj["Name"] == "gnome" for obj in objects)
        gem = any(obj["Name"] == "gem" for obj in objects)
        status = (
            "won"
            if done and alive and not gem
            else (
                "lost"
                if not alive
                else (
                    "native_terminal"
                    if done
                    else "step_limit" if self._steps >= self.scenario.max_steps else "running"
                )
            )
        )
        self._ended = status != "running"
        frame = {
            "step": self._steps,
            "observation": {
                "grid": native["Grid"],
                "objects": objects,
                "tick": native["GameTicks"],
                "npc_period_ticks": 3,
                "coordinates": "[x,y]; right/down positive; orientation is absolute; actions are relative",
            },
            "legal_actions": [] if self._ended else self._legal(),
            "reward": reward,
            "status": status,
            "terminated": bool(done),
            "truncated": status == "step_limit",
            "native_state": native,
            "events": json.loads(canonical(info.get("History", []))),
        }
        frame["frame_hash"] = digest(frame)
        self._latest = frame
        return frame

    def observe(self) -> dict[str, Any]:
        if self._closed or self._latest is None:
            raise RuntimeError("reset an open engine before observing")
        return json.loads(canonical(self._latest))

    def step(self, action: str) -> dict[str, Any]:
        if self._closed or not self._started or self._ended:
            raise RuntimeError("reset an open, nonterminal engine before stepping")
        if action not in self._legal():
            raise ValueError(f"action is not currently legal: {action}")
        _observation, reward, done, info = self._env.step(ACTIONS[action])
        self._steps += 1
        return self._frame(float(reward), bool(done), info)

    def pixels(self) -> Any:
        if not self._render or self._closed or not self._started:
            raise RuntimeError("vector rendering must be enabled before reset")
        return self._env.render(mode="rgb_array", observer="global").copy()

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._env = None
            self._closed = True

    def __enter__(self) -> GriddlyEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
