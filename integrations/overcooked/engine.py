"""Native Overcooked-AI state, rules, lockstep controls and official rendering.

Importing this module needs only the standard library. Game dependencies load
when an Engine is created. This module does not depend on Honing.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROTOCOL = "overcooked-full-kitchen-v1"
ACTIONS = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
    "STAY": (0, 0),
    "INTERACT": "interact",
}
LEGEND = {
    " ": "floor",
    "X": "counter",
    "O": "onion dispenser",
    "T": "tomato dispenser",
    "D": "dish dispenser",
    "P": "pot",
    "S": "serving counter",
}
LAYOUTS = (
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Scenario:
    layout: str = "cramped_room"
    max_steps: int = 400
    old_dynamics: bool = False
    controlled_player: int = 0
    partner: str = "solo"
    target_deliveries: int = 2
    score_scale: float = 40.0

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 10000:
            raise ValueError("max_steps must be in 1..10000")
        if type(self.old_dynamics) is not bool:
            raise ValueError("old_dynamics must be boolean")
        if type(self.controlled_player) is not int or self.controlled_player not in (0, 1):
            raise ValueError("controlled_player must be 0 or 1")
        if self.partner not in ("role", "solo", "stay", "solo_slow"):
            raise ValueError("partner must be role, solo, stay or solo_slow")
        if type(self.target_deliveries) is not int or not 1 <= self.target_deliveries <= 100:
            raise ValueError("target_deliveries must be in 1..100")
        if type(self.score_scale) not in (int, float) or not math.isfinite(self.score_scale) or self.score_scale <= 0:
            raise ValueError("score_scale must be finite and positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def package_digest(package: Path) -> str:
    """Hash engine code and packaged resources, excluding caches and planner pickles."""
    files = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".layout", ".png", ".json", ".ttf"}:
            data = path.read_bytes()
            if path.suffix in {".py", ".layout", ".json"}:
                data = data.replace(b"\r\n", b"\n")
            files[path.relative_to(package).as_posix()] = hashlib.sha256(data).hexdigest()
    return digest(files)


@lru_cache(maxsize=1)
def runtime_identity() -> dict[str, Any]:
    package = importlib.import_module("overcooked_ai_py")
    location = Path(package.__file__).resolve().parent
    pins = json.loads((ROOT / "pins.json").read_text(encoding="utf-8"))
    actual = package_digest(location)
    if actual != pins.get("native_sha256"):
        raise RuntimeError("Native engine/resources differ from pins.json; run this integration's setup.ps1")
    return {
        "revision": pins["revision"],
        "native_sha256": actual,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("overcooked-ai", "numpy", "scipy", "pygame", "gymnasium")
        },
    }


class Engine:
    """One isolated full-kitchen environment; reading never advances the game."""

    def __init__(self, scenario: Scenario | None = None) -> None:
        scenario = scenario or Scenario()
        self.scenario = scenario
        # Rendering to a surface must also work without a graphical desktop.
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self.runtime = runtime_identity()
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

        self.mdp = OvercookedGridworld.from_layout_name(scenario.layout, old_dynamics=scenario.old_dynamics)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=scenario.max_steps, info_level=0)
        self._visualizer: Any = None
        self._closed = False
        self.reset()

    def reset(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("engine is closed")
        self.env.reset()
        self.score, self.deliveries, self.reward = 0.0, 0, 0.0
        self.delivery_attempts = 0
        self.events: dict[str, Any] = {}
        return self.observe()

    def game_map(self) -> dict[str, Any]:
        return {
            "layout": self.scenario.layout,
            "width": self.mdp.width,
            "height": self.mdp.height,
            "terrain": copy.deepcopy(self.mdp.terrain_mtx),
            "legend": LEGEND.copy(),
            "coordinates": "[x,y], origin upper-left; x right, y down; terrain[y][x]",
            "walkable": [list(p) for p in self.mdp.get_valid_player_positions()],
            "start_positions": [list(p) for p in self.mdp.start_player_positions],
        }

    def rules(self) -> dict[str, Any]:
        from overcooked_ai_py.mdp.overcooked_mdp import Recipe

        return {
            "old_dynamics": self.scenario.old_dynamics,
            "max_ingredients": Recipe.MAX_NUM_INGREDIENTS,
            "recipes": [
                {"ingredients": list(r.ingredients), "cook_ticks": r.time, "value": r.value}
                for r in sorted(self.env.state.all_orders)
            ],
            "order_bonus": self.mdp.order_bonus,
            "actions": {name: list(a) if isinstance(a, tuple) else a for name, a in ACTIONS.items()},
            "instructions": [
                "Both chefs share delivery rewards. Actions resolve together, one native tick per joint action.",
                "A direction moves one floor tile and sets facing; blocked movement can still change facing.",
                "Chefs cannot share a tile or swap positions in one tick. Collisions cancel their movement.",
                "INTERACT affects the adjacent tile in the direction the chef is facing.",
                "Empty hands take ingredients/dishes from dispensers or pick up an object from a counter.",
                "Holding an object: INTERACT places it on an empty counter. " "Ingredients can be added to idle pots.",
                (
                    "A pot with three ingredients starts cooking automatically."
                    if self.scenario.old_dynamics
                    else "With empty hands, INTERACT with an idle pot containing ingredients to start cooking."
                ),
                "A dish is needed to collect ready soup; deliver the soup at S to score. "
                "Cooking uses ticks, not seconds.",
                "All six actions are allowed; an ineffective interaction is a native no-op, not an illegal action.",
                "The episode ends at its fixed step budget; task goals do not end the native episode early.",
            ],
        }

    def observe(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("engine is closed")
        state = self.env.state
        ended = self.env.is_done()
        frame = {
            "protocol": PROTOCOL,
            "step": state.timestep,
            "map": self.game_map(),
            "rules": self.rules(),
            "state": state.to_dict(),
            "pots": {k: [list(p) for p in v] for k, v in self.mdp.get_pot_states(state).items()},
            "score": self.score,
            "deliveries": self.deliveries,
            "delivery_attempts": self.delivery_attempts,
            "reward": self.reward,
            "events": self.events,
            "max_steps": self.scenario.max_steps,
            "status": "step_limit" if ended else "running",
            "terminated": False,
            "truncated": ended,
            "legal_actions": list(ACTIONS) if not ended else [],
        }
        # Convert tuples and NumPy scalar values to transport types once, for stable hashes.
        frame = json.loads(canonical(frame))
        frame["frame_hash"] = digest(frame)
        return frame

    def step(self, actions: list[str] | tuple[str, str]) -> dict[str, Any]:
        if self._closed or self.env.is_done():
            raise RuntimeError("cannot advance a closed or finished game")
        if not isinstance(actions, (tuple, list)) or len(actions) != 2 or any(a not in ACTIONS for a in actions):
            raise ValueError("joint action must contain exactly two native action names")
        _state, reward, _done, info = self.env.step(tuple(ACTIONS[a] for a in actions))
        self.reward = float(reward)
        self.score += self.reward
        # The environment keeps event histories in game_stats rather than step info.
        tick = self.env.state.timestep - 1
        events = {
            key: [tick in per_player for per_player in history]
            for key, history in self.env.game_stats.items()
            if not key.startswith("cumulative_")
        }
        self.events = {k: v for k, v in events.items() if any(v)}
        self.delivery_attempts = sum(len(v) for v in self.env.game_stats["soup_delivery"])
        # Native soup_delivery also fires for disallowed recipes, which earn zero.
        # Only rewarded deliveries satisfy the task; event counts stay diagnostic.
        self.deliveries += sum(float(r) > 0 for r in info["sparse_r_by_agent"])
        return self.observe()

    def png(self, frame: dict[str, Any] | None = None) -> bytes:
        """Render the official sprites; supplied historical frames do not mutate live state."""
        import pygame
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
        from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

        if self._visualizer is None:
            self._visualizer = StateVisualizer(tile_size=90, is_rendering_action_probs=False)
        frame = self.observe() if frame is None else frame
        state = OvercookedState.from_dict(frame["state"])
        surface = self._visualizer.render_state(
            state,
            frame["map"]["terrain"],
            hud_data=StateVisualizer.default_hud_data(
                state,
                score=frame["score"],
                time_left=frame["max_steps"] - frame["step"],
            ),
        )
        output = io.BytesIO()
        pygame.image.save(surface, output, "frame.png")
        return output.getvalue()

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
