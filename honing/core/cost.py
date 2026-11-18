"""What one move cost, and the sink that episodes and chat calls are charged to while it runs.

A move spends episodes for four purposes (the gate, the harness method, rollouts
and demonstrations), GPU time in training, and tokens and dollars in chat calls.
:class:`MoveCost` holds all of them for one move, in one record.

Episodes are charged where they are run, not reported by each component: every
episode goes through :func:`~honing.core.run_graded`, which calls
:func:`charge_episode`, and a chat model wrapped for metering calls
:func:`charge_chat`. Both charge whatever :class:`CostSink` the campaign has
installed with :func:`charging` for the move in progress, and nothing when none
is installed. The sink lives in a :class:`contextvars.ContextVar`, which
``asyncio`` copies into every task it starts, so episodes run concurrently under
``asyncio.run`` are charged to the move that started them.

An unknown amount is ``None``, never 0: a move whose price is unknown must not
look free.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, get_args

from honing.core.errors import ContractError, require

EpisodePurpose = Literal["gate", "method", "rollout", "demonstration"]
EPISODE_PURPOSES: tuple[EpisodePurpose, ...] = get_args(EpisodePurpose)


def _no_episodes() -> Mapping[str, int]:
    return MappingProxyType(dict.fromkeys(EPISODE_PURPOSES, 0))


def _plus(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left + right


@dataclass(frozen=True)
class MoveCost:
    """Everything one move spent.

    ``episodes`` names every purpose in :data:`EPISODE_PURPOSES`, including the
    ones the move did not spend on. ``gpu_seconds`` is the trainers' elapsed
    time times their GPUs, ``None`` when a trainer ran without reporting it.
    ``api_usd`` is the chat calls' price, ``None`` when any call was unpriced.
    ``usd`` is the move in the campaign's one budget unit, GPU time at a fixed
    GPU-hour price plus ``api_usd``; it is ``None`` when either part is
    unknown or GPU time was spent with no price configured.
    """

    episodes: Mapping[str, int] = field(default_factory=_no_episodes)
    gpu_seconds: float | None = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    api_usd: float | None = 0.0
    usd: float | None = 0.0
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        require(
            set(self.episodes) == set(EPISODE_PURPOSES),
            f"a cost counts episodes for exactly {list(EPISODE_PURPOSES)}, got {sorted(self.episodes)}",
        )
        for purpose, count in self.episodes.items():
            _require_count(count, f"{purpose} episodes")
        _require_count(self.input_tokens, "input_tokens")
        _require_count(self.output_tokens, "output_tokens")
        for name in ("gpu_seconds", "api_usd", "usd"):
            value = getattr(self, name)
            require(value is None or _amount(value), f"{name} must be a finite number >= 0 or None, got {value!r}")
        require(_amount(self.wall_seconds), f"wall_seconds must be a finite number >= 0, got {self.wall_seconds!r}")
        object.__setattr__(self, "episodes", MappingProxyType({p: self.episodes[p] for p in EPISODE_PURPOSES}))

    @property
    def total_episodes(self) -> int:
        return sum(self.episodes.values())

    def plus(self, other: MoveCost) -> MoveCost:
        """The sum of two costs. An unknown part of either makes that part of the sum unknown."""
        return MoveCost(
            episodes={p: self.episodes[p] + other.episodes[p] for p in EPISODE_PURPOSES},
            gpu_seconds=_plus(self.gpu_seconds, other.gpu_seconds),
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            api_usd=_plus(self.api_usd, other.api_usd),
            usd=_plus(self.usd, other.usd),
            wall_seconds=self.wall_seconds + other.wall_seconds,
        )

    def to_record(self) -> dict[str, object]:
        """The cost as JSON-native values, as the commit record and ``summary.json`` hold it."""
        return {
            "episodes": dict(self.episodes),
            "gpu_seconds": self.gpu_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "api_usd": self.api_usd,
            "usd": self.usd,
            "wall_seconds": self.wall_seconds,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MoveCost:
        """Read back what :meth:`to_record` wrote. Refuses a record with missing or unknown keys."""
        keys = {"episodes", "gpu_seconds", "input_tokens", "output_tokens", "api_usd", "usd", "wall_seconds"}
        if set(record) != keys:
            raise ContractError(f"a cost record has keys {sorted(keys)}, got {sorted(record)}")
        episodes = record["episodes"]
        if not isinstance(episodes, Mapping):
            raise ContractError(f"a cost record's episodes must be an object, got {episodes!r}")
        return cls(
            episodes=dict(episodes),
            gpu_seconds=record["gpu_seconds"],  # type: ignore[arg-type]
            input_tokens=record["input_tokens"],  # type: ignore[arg-type]
            output_tokens=record["output_tokens"],  # type: ignore[arg-type]
            api_usd=record["api_usd"],  # type: ignore[arg-type]
            usd=record["usd"],  # type: ignore[arg-type]
            wall_seconds=record["wall_seconds"],  # type: ignore[arg-type]
        )

    def metrics(self) -> dict[str, float]:
        """The known parts as release metrics, each named ``cost_*``. An unknown part is left out, not 0."""
        values: dict[str, float | None] = {f"cost_episodes_{p}": float(n) for p, n in self.episodes.items()}
        values.update(
            cost_gpu_seconds=self.gpu_seconds,
            cost_input_tokens=float(self.input_tokens),
            cost_output_tokens=float(self.output_tokens),
            cost_api_usd=self.api_usd,
            cost_usd=self.usd,
            cost_wall_seconds=self.wall_seconds,
        )
        return {name: value for name, value in values.items() if value is not None}


def _require_count(value: object, name: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{name} must be an integer >= 0, got {value!r}",
    )


def _amount(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


class CostSink(Protocol):
    """Takes the charges made while a move runs. The campaign's meter is one."""

    def charge_episode(self) -> None: ...

    def charge_chat(self, model: str | None, input_tokens: int, output_tokens: int, cost_usd: float | None) -> None:
        """One chat call; ``cost_usd`` is ``None`` when the client could not price it."""
        ...


_SINK: ContextVar[CostSink | None] = ContextVar("honing_cost_sink", default=None)


@contextmanager
def charging(sink: CostSink) -> Iterator[CostSink]:
    """Charge every episode and metered chat call made inside the block to ``sink``."""
    token = _SINK.set(sink)
    try:
        yield sink
    finally:
        _SINK.reset(token)


def charge_episode() -> None:
    """Charge one episode to the installed sink, if any. :func:`~honing.core.run_graded` calls it."""
    sink = _SINK.get()
    if sink is not None:
        sink.charge_episode()


def charge_chat(model: str | None, input_tokens: int, output_tokens: int, cost_usd: float | None) -> None:
    """Charge one chat call to the installed sink, if any."""
    sink = _SINK.get()
    if sink is not None:
        sink.charge_chat(model, input_tokens, output_tokens, cost_usd)
