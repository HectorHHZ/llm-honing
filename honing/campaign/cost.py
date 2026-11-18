"""The campaign's budget, the meter that builds each move's :class:`~honing.core.MoveCost`, and a metered chat model.

A budget is in one unit: dollars at fixed rates (a GPU-hour price plus the chat
calls' price), or wall-clock seconds. The campaign admits no move once the
spend reaches the limit, so a run can overshoot by at most the cost of the last
move it admitted; the overshoot is reported, not hidden.

A dollar budget never treats an unknown price as free. A chat model it cannot
price is refused when the budget is built (:meth:`CampaignBudget.check_chat_model`),
and a reply that still arrives unpriced, or a trainer that does not report its
GPU time, stops the campaign with an error after the move is recorded.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from honing.core import (
    EPISODE_PURPOSES,
    ChatMessage,
    ChatModel,
    ChatReply,
    ContractError,
    EpisodePurpose,
    MoveCost,
    charge_chat,
    require,
)

BudgetUnit = Literal["usd", "wall_seconds"]
_BUDGET_KEYS = {"unit", "limit", "gpu_hour_usd", "chat_usd_per_million_tokens"}


@dataclass(frozen=True)
class ChatRate:
    """A chat model's price in dollars per million input and output tokens."""

    input_usd: float
    output_usd: float

    def __post_init__(self) -> None:
        require(_price(self.input_usd) and _price(self.output_usd), f"chat rates must be finite and >= 0: {self}")

    def usd(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_usd + output_tokens * self.output_usd) / 1_000_000


@dataclass(frozen=True)
class CampaignBudget:
    """How much a campaign may spend, in one unit, and the fixed rates that turn spend into dollars.

    ``gpu_hour_usd`` and ``chat_rates`` are used whatever the unit, so the
    dollars are reported for a wall-clock budget too. A dollar budget requires
    ``gpu_hour_usd``. ``chat_rates`` prices the replies of the models it names
    whose client returns no price (a model served locally, for example).
    """

    unit: BudgetUnit
    limit: float
    gpu_hour_usd: float | None = None
    chat_rates: Mapping[str, ChatRate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(self.unit in get_args(BudgetUnit), f"a budget's unit is one of {list(get_args(BudgetUnit))}")
        require(_price(self.limit) and self.limit > 0, f"a budget's limit must be a finite number > 0: {self.limit}")
        require(self.gpu_hour_usd is None or _price(self.gpu_hour_usd), "gpu_hour_usd must be finite and >= 0")
        if self.unit == "usd" and self.gpu_hour_usd is None:
            raise ContractError("a budget in usd needs gpu_hour_usd, the fixed price of one GPU-hour")

    def spent(self, cost: MoveCost) -> float | None:
        """``cost`` in the budget's unit; ``None`` when its dollars are unknown."""
        return cost.usd if self.unit == "usd" else cost.wall_seconds

    def check_chat_model(self, model: object) -> None:
        """Refuse, under a dollar budget, a chat model whose replies could not be priced."""
        if self.unit != "usd":
            return
        name = getattr(model, "model", None)
        if isinstance(name, str) and name in self.chat_rates:
            return
        prices = getattr(model, "prices_replies", None)
        if callable(prices) and prices():
            return
        raise ContractError(
            f"the budget is in usd, but the chat model {name or type(model).__name__} returns no price; "
            f"set budget.chat_usd_per_million_tokens[{name or '<model>'}]"
        )


def budget_from_config(spec: object) -> CampaignBudget | None:
    """The ``budget`` block of a config, or ``None`` when there is none."""
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise ContractError(f"budget must be an object, got {spec!r}")
    unknown = set(spec) - _BUDGET_KEYS
    if unknown:
        raise ContractError(f"unknown budget keys {sorted(unknown)}; a budget takes {sorted(_BUDGET_KEYS)}")
    if "unit" not in spec or "limit" not in spec:
        raise ContractError("a budget needs a unit and a limit")
    rates = spec.get("chat_usd_per_million_tokens", {})
    if not isinstance(rates, Mapping):
        raise ContractError("budget.chat_usd_per_million_tokens must map a model name to {input, output}")
    chat_rates: dict[str, ChatRate] = {}
    for name, rate in rates.items():
        if not isinstance(rate, Mapping) or set(rate) != {"input", "output"}:
            raise ContractError(f"budget.chat_usd_per_million_tokens[{name}] must be {{input, output}}")
        chat_rates[str(name)] = ChatRate(_number(rate["input"], name), _number(rate["output"], name))
    gpu_hour_usd = spec.get("gpu_hour_usd")
    return CampaignBudget(
        unit=spec["unit"],
        limit=_number(spec["limit"], "limit"),
        gpu_hour_usd=None if gpu_hour_usd is None else _number(gpu_hour_usd, "gpu_hour_usd"),
        chat_rates=chat_rates,
    )


def _number(value: object, name: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"budget value {name} must be a number, got {value!r}")
    return float(value)


def _price(value: float) -> bool:
    return math.isfinite(value) and value >= 0


class CostMeter:
    """Adds up what one move spends. Satisfies :class:`~honing.core.CostSink`.

    The campaign installs the meter with :func:`~honing.core.charging` for the
    whole move and names the purpose of each phase with :meth:`purpose`. An
    episode charged outside every purpose is a bug in the campaign, so it
    raises rather than landing in some default bucket.
    """

    def __init__(self, budget: CampaignBudget | None) -> None:
        self._budget = budget
        self._lock = threading.Lock()
        self._purpose: EpisodePurpose | None = None
        self._episodes: dict[str, int] = dict.fromkeys(EPISODE_PURPOSES, 0)
        self._gpu_seconds: float | None = 0.0
        self._input_tokens = 0
        self._output_tokens = 0
        self._api_usd: float | None = 0.0
        self._unpriced: set[str] = set()
        self._started = time.monotonic()

    @contextmanager
    def purpose(self, purpose: EpisodePurpose) -> Iterator[None]:
        """Charge the episodes run inside the block to ``purpose``."""
        previous, self._purpose = self._purpose, purpose
        try:
            yield
        finally:
            self._purpose = previous

    def charge_episode(self) -> None:
        if self._purpose is None:
            raise ContractError("an episode ran outside the gate, the method, a rollout, or a demonstration")
        with self._lock:
            self._episodes[self._purpose] += 1

    def charge_chat(self, model: str | None, input_tokens: int, output_tokens: int, cost_usd: float | None) -> None:
        if cost_usd is None and self._budget is not None and model is not None and model in self._budget.chat_rates:
            cost_usd = self._budget.chat_rates[model].usd(input_tokens, output_tokens)
        with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            if cost_usd is None:
                self._unpriced.add(model or "an unnamed chat model")
                self._api_usd = None
            elif self._api_usd is not None:
                self._api_usd += cost_usd

    def charge_gpu_seconds(self, seconds: float | None) -> None:
        """A trainer's elapsed time times its GPUs; ``None`` when it did not report it."""
        with self._lock:
            self._gpu_seconds = None if seconds is None or self._gpu_seconds is None else self._gpu_seconds + seconds

    @property
    def unpriced_models(self) -> tuple[str, ...]:
        return tuple(sorted(self._unpriced))

    def close(self) -> MoveCost:
        """The move's cost so far, with the wall clock read now."""
        with self._lock:
            gpu_usd: float | None
            if self._gpu_seconds is None:
                gpu_usd = None
            elif self._gpu_seconds == 0:
                gpu_usd = 0.0
            else:
                rate = None if self._budget is None else self._budget.gpu_hour_usd
                gpu_usd = None if rate is None else self._gpu_seconds / 3600 * rate
            return MoveCost(
                episodes=dict(self._episodes),
                gpu_seconds=self._gpu_seconds,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                api_usd=self._api_usd,
                usd=None if gpu_usd is None or self._api_usd is None else gpu_usd + self._api_usd,
                wall_seconds=time.monotonic() - self._started,
            )


class MeteredChatModel:
    """A :class:`~honing.core.ChatModel` that charges each call's tokens and price to the move in progress.

    ``model`` is the wrapped client's model name, which a budget's chat rates
    are keyed by; ``prices_replies`` is passed through, so the budget can tell
    whether the client prices its own replies.
    """

    def __init__(self, inner: ChatModel) -> None:
        self._inner = inner
        name = getattr(inner, "model", None)
        self.model: str | None = name if isinstance(name, str) else None

    @property
    def inner(self) -> ChatModel:
        return self._inner

    def prices_replies(self) -> bool:
        prices = getattr(self._inner, "prices_replies", None)
        return bool(callable(prices) and prices())

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float) -> ChatReply:
        reply = self._inner.complete(messages, max_tokens=max_tokens, temperature=temperature)
        charge_chat(self.model, reply.input_tokens, reply.output_tokens, reply.cost_usd)
        return reply

    def __getattr__(self, name: str) -> Any:
        # Anything else the wrapped client exposes (settings, for example) stays reachable.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


def cost_summary(costs: Sequence[MoveCost]) -> dict[str, object]:
    """The total of every move's cost, as ``summary.json`` holds it."""
    total = MoveCost()
    for cost in costs:
        total = total.plus(cost)
    return total.to_record()


__all__ = [
    "BudgetUnit",
    "CampaignBudget",
    "ChatRate",
    "CostMeter",
    "MeteredChatModel",
    "budget_from_config",
    "cost_summary",
]
