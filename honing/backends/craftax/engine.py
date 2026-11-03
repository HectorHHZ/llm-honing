"""Optional native full Craftax adapter. Importing it never loads JAX or starts a game."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from typing import Any

from honing.backends.craftax.catalog import ACHIEVEMENTS, ACTIONS, PROTOCOL
from honing.backends.craftax.scenarios import CraftaxScenario


def runtime_versions() -> dict[str, str]:
    return {
        **{p: importlib.metadata.version(p) for p in ("craftax", "jax", "jaxlib", "numpy")},
        "system": platform.system(),
        "machine": platform.machine(),
    }


def frame_hash(frame: dict[str, Any]) -> str:
    payload = {k: v for k, v in frame.items() if k != "frame_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


class CraftaxEngine:
    """One explicit reset/step/close lifecycle, without auto-resetting a terminal game."""

    def __init__(self, scenario: CraftaxScenario) -> None:
        try:
            import jax
            import numpy as np
            from craftax.craftax.constants import OBS_DIM, Achievement, Action, BlockType, ItemType
            from craftax.craftax.envs.craftax_symbolic_env import get_flat_map_obs_shape, get_map_obs_shape
            from craftax.craftax_env import make_craftax_env_from_name
        except ImportError as error:
            raise RuntimeError(
                "Install integrations/craftax/requirements.txt in a separate Python 3.12 environment"
            ) from error
        if tuple(a.name for a in sorted(Action, key=lambda a: a.value)) != ACTIONS:
            raise ValueError("The installed Craftax action catalogue differs from the pinned full game")
        if {a.name for a in Achievement} != ACHIEVEMENTS:
            raise ValueError("The installed Craftax achievements differ from the pinned full game")
        self.scenario = scenario
        self._jax, self._np = jax, np
        self._obs_dim, self._blocks, self._items = OBS_DIM, BlockType, ItemType
        self._shape, self._flat = get_map_obs_shape(), get_flat_map_obs_shape()
        self._achievements = Achievement
        self._env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
        self._params = self._env.default_params
        self._state: Any = None
        self._rng: Any = None
        self._renderer: Any = None
        self._ended = False
        self._closed = False
        self._steps = 0

    def reset(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("the engine is closed")
        self._rng, key = self._jax.random.split(self._jax.random.PRNGKey(self.scenario.seed))
        obs, self._state = self._env.reset(key, self._params)
        self._steps, self._ended = 0, False
        return self._frame(obs, 0.0, False)

    def _observation(self, obs: Any) -> dict[str, Any]:
        np, state = self._np, self._state
        spatial = np.asarray(obs)[: self._flat].reshape(self._shape)
        block_ids, item_ids = {a.value: a.name for a in self._blocks}, {a.value: a.name for a in self._items}
        blocks = spatial[:, :, : len(self._blocks)]
        items = spatial[:, :, len(self._blocks) : len(self._blocks) + len(self._items)]
        mobs = spatial[:, :, len(self._blocks) + len(self._items) : -1]
        return {
            "tiles": [[block_ids[int(cell.argmax())] if cell.any() else None for cell in row] for row in blocks],
            "items": [[item_ids[int(cell.argmax())] if cell.any() else None for cell in row] for row in items],
            "entities": [
                {"row": int(y), "column": int(x), "class": int(k) // 8, "type": int(k) % 8}
                for y, x, k in zip(*np.nonzero(mobs), strict=True)
            ],
            "player": {
                key.removeprefix("player_"): np.asarray(getattr(state, key)).tolist()
                for key in (
                    "player_level",
                    "player_direction",
                    "player_health",
                    "player_food",
                    "player_drink",
                    "player_energy",
                    "player_mana",
                    "player_xp",
                    "player_dexterity",
                    "player_strength",
                    "player_intelligence",
                )
            },
            "inventory": {key: np.asarray(value).tolist() for key, value in asdict(state.inventory).items()},
            "equipment": {
                key: np.asarray(getattr(state, key)).tolist()
                for key in ("sword_enchantment", "bow_enchantment", "armour_enchantments", "learned_spells")
            },
            "sleeping": bool(state.is_sleeping),
            "resting": bool(state.is_resting),
            "timestep": int(state.timestep),
            "player_cell": [self._obs_dim[0] // 2, self._obs_dim[1] // 2],
            "coordinates": "local [row,column], down/right positive; null tiles are unobserved",
            "entity_classes": ["melee", "passive", "ranged", "mob_projectile", "player_projectile"],
        }

    def _frame(self, obs: Any, reward: float, native_done: bool) -> dict[str, Any]:
        achievements = sorted(a.name for a in self._achievements if bool(self._state.achievements[a.value]))
        complete = set(self.scenario.goals) <= set(achievements)
        dead = float(self._state.player_health) <= 0
        limit = self._steps >= self.scenario.max_steps
        status = (
            "goal_complete"
            if complete
            else "death" if dead else "native_terminal" if native_done else "step_limit" if limit else "running"
        )
        self._ended = status != "running"
        digest = hashlib.sha256()
        # Hash native leaves too, so replay checks more than just the visible map.
        for leaf in self._jax.tree_util.tree_leaves((self._state, self._rng)):
            array = self._np.asarray(leaf)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        frame = {
            "step": self._steps,
            "observation": self._observation(obs),
            "achievements": achievements,
            "reward": reward,
            "status": status,
            "terminated": native_done,
            "truncated": self._ended and not native_done,
            "native_state_hash": digest.hexdigest(),
        }
        frame["frame_hash"] = frame_hash(frame)
        return frame

    def step(self, action: str) -> dict[str, Any]:
        if self._closed or self._state is None or self._ended:
            raise RuntimeError("reset an open, nonterminal engine before stepping")
        if action not in ACTIONS:
            raise ValueError("action must be a name from the full Craftax catalogue")
        self._rng, key = self._jax.random.split(self._rng)
        obs, self._state, reward, done, _info = self._env.step(key, self._state, ACTIONS.index(action), self._params)
        self._steps += 1
        return self._frame(obs, float(reward), bool(done))

    def pixels(self) -> Any:
        """Official pixel renderer, with nearest-neighbor scaling; no replacement graphics."""
        if self._state is None or self._closed:
            raise RuntimeError("reset an open engine before rendering")
        if self._renderer is None:
            from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG
            from craftax.craftax.renderer import make_craftax_pixel_renderer

            self._renderer = self._jax.jit(make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_IMG))
        image = self._np.clip(self._np.asarray(self._renderer(self._state)), 0, 255).astype(self._np.uint8)
        return self._np.repeat(self._np.repeat(image, 4, axis=0), 4, axis=1)

    def close(self) -> None:
        self._state, self._rng, self._renderer, self._closed = None, None, None, True

    def __enter__(self) -> CraftaxEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def protocol(self) -> str:
        return PROTOCOL
